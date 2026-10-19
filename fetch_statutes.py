"""
fetch_statutes.py

Stage 2 of statute ingestion: reads data/catalog.jsonl (from
crawl_catalog.py), visits each law's detail page to resolve the actual
PDF download link, downloads the PDF, extracts clean text, and writes
normalized JSON records ready for chunking/embedding.

Run:
    pip install requests beautifulsoup4 pdfplumber tenacity tqdm
    python fetch_statutes.py --limit 20                # start small
    python fetch_statutes.py --category "General Laws" # filter by category
    python fetch_statutes.py                            # everything (slow, be deliberate)
"""

import argparse
import json
import re
import time
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw_pdfs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_PATH = DATA_DIR / "catalog.jsonl"
OUT_PATH = DATA_DIR / "statutes.jsonl"
FAILURES_PATH = DATA_DIR / "fetch_failures.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30),
       reraise=True)  # surface the real error (e.g. "404 Not Found"), not a bare RetryError
def fetch(url: str, session: requests.Session) -> requests.Response:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def resolve_pdf_url(detail_url: str, session: requests.Session) -> str | None:
    resp = fetch(detail_url, session)
    soup = BeautifulSoup(resp.text, "html.parser")
    # The download link's <img> has alt text "Download/Print PDF " --
    # anchor on that rather than a brittle CSS selector.
    for a in soup.find_all("a", href=True):
        if "/pdffiles/" in a["href"]:
            href = a["href"]
            return href if href.startswith("http") else f"https://pakistancode.gov.pk{href}"
    return None


def extract_text(pdf_path: Path) -> tuple[str, int]:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    text = "\n".join(text_parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), num_pages


def load_catalog(category: str | None, limit: int | None) -> list[dict]:
    laws = []
    with CATALOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            law = json.loads(line)
            if category and law.get("category") != category:
                continue
            laws.append(law)
    if limit:
        laws = laws[:limit]
    return laws


def already_done(out_path: Path) -> set[str]:
    """Resume support: don't re-fetch laws already written to statutes.jsonl."""
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["detail_url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help='e.g. "General Laws"')
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        raise SystemExit("Run crawl_catalog.py first -- data/catalog.jsonl not found.")

    laws = load_catalog(args.category, args.limit)
    done = already_done(OUT_PATH)
    laws = [law for law in laws if law["detail_url"] not in done]

    print(f"{len(laws)} laws to fetch ({len(done)} already done, skipping those)")

    session = requests.Session()
    failures = []

    with OUT_PATH.open("a", encoding="utf-8") as out_f, \
            FAILURES_PATH.open("a", encoding="utf-8") as fail_f:

        def record_failure(law: dict, reason: str, **where) -> None:
            # Also logged to disk as we go, so an interrupted run still
            # leaves a record of what failed and why. `where` carries the
            # pdf_url / pdf_file once the file has been downloaded.
            failures.append((law["name"], reason))
            fail_f.write(json.dumps({
                "name": law["name"],
                "detail_url": law["detail_url"],
                "reason": reason,
                **where,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            fail_f.flush()

        for law in tqdm(laws, desc="Fetching statutes"):
            try:
                pdf_url = resolve_pdf_url(law["detail_url"], session)
                if not pdf_url:
                    record_failure(law, "no PDF link found on detail page")
                    continue

                pdf_filename = pdf_url.split("/")[-1]
                pdf_path = RAW_DIR / pdf_filename
                if not pdf_path.exists():
                    resp = fetch(pdf_url, session)
                    pdf_path.write_bytes(resp.content)

                text, num_pages = extract_text(pdf_path)
                if len(text) < 100:
                    record_failure(law, "no extractable text -- likely scanned, needs OCR",
                                   pdf_url=pdf_url, pdf_file=pdf_filename)
                    continue

                record = {
                    **law,
                    "pdf_url": pdf_url,
                    "raw_text": text,
                    "num_pages": num_pages,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()  # so a crash mid-run doesn't lose completed work

            except Exception as e:
                record_failure(law, f"{type(e).__name__}: {e}")

            time.sleep(1.5)  # polite pacing -- this is a government server, not a CDN

    print(f"\nOK -> {OUT_PATH}")
    if failures:
        print(f"\n{len(failures)} failed/skipped:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()