"""
fetch_sources.py

Stage 2 for the sources cataloged by crawl_sources.py: for each catalog
row, find the PDF (directly, or via the document's detail page), download
it, extract the text, and write one JSON record per document.

Run:
    python fetch_sources.py --source punjab --limit 20   # start small
    python fetch_sources.py --source punjab              # everything

Per source, under data/<source>/:
    raw_pdfs/            downloaded PDFs
    statutes.jsonl       catalog row + pdf_url, raw_text, num_pages
    fetch_failures.jsonl what was skipped or failed, and why

Same behaviour as fetch_statutes.py: resumable (rerun the same command),
skips PDFs with no text layer rather than writing garbage, and paces
itself -- these are government servers.
"""

import argparse
import hashlib
import html
import json
import re
import time
import zipfile
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from crawl_sources import punjab_article_id
from fetch_statutes import extract_text, fetch

DATA_DIR = Path(__file__).parent / "data"

# How to recognise the PDF link on each source's detail page, and (where
# detail pages link to further documents, e.g. a law's subordinate rules)
# which links are documents in their own right.
SOURCES = {
    "drs": {"pdf": re.compile(r"/pdf/.+\.pdf$", re.I)},
    "punjab": {
        "pdf": re.compile(r"/uploads/.+\.pdf$", re.I),
        "related": re.compile(r"^https://punjabcode\.punjab\.gov\.pk/en/show_article/[^/#]+$"),
        "ident": punjab_article_id,
    },
    # KP publishes a few laws only as Word files
    "kp": {"pdf": re.compile(r"/uploads/.+\.(pdf|docx)$", re.I)},
    "balochistan": {"pdf": re.compile(r"/lawdir/.+\.pdf$", re.I)},
    # Catalog rows already carry pdf_url. Some listing links point at the
    # Sindh Code folder for files that only exist in the older publications
    # folder, so try that when the linked URL is a 404.
    "sindh": {
        "alternates": lambda url: [re.sub(r"/setup/Publications_SindhCode/", "/Setup/Publications/", url, flags=re.I)],
    },
}


def download(pdf_url: str, source: str, session: requests.Session) -> tuple[requests.Response, str]:
    """Fetch the file, falling back to the source's alternate URLs on a 404."""
    try:
        return fetch(pdf_url, session), pdf_url
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        for alternate in SOURCES[source].get("alternates", lambda url: [])(pdf_url):
            if alternate != pdf_url:
                time.sleep(1.0)
                try:
                    return fetch(alternate, session), alternate
                except requests.HTTPError:
                    continue
        raise


def read_detail_page(law: dict, source: str, session: requests.Session) -> tuple[str | None, list[dict]]:
    """Return (pdf_url, related documents not yet in the catalog)."""
    config = SOURCES[source]
    resp = fetch(law["detail_url"], session)
    soup = BeautifulSoup(resp.content, "html.parser")
    pdf_url = None
    related = []
    for tag in soup.find_all(["a", "embed", "iframe"]):
        link = tag.get("href") or tag.get("src")
        if not link:
            continue
        url = urljoin(law["detail_url"], link).split("#")[0]
        if pdf_url is None and config["pdf"].search(urlparse(url).path):
            pdf_url = url
        elif tag.name == "a" and "related" in config and config["related"].search(url):
            name = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
            doc_id = config.get("ident", lambda u: u)(url)
            if name and doc_id != law["id"]:
                related.append({
                    "source": source,
                    "id": doc_id,
                    "name": name,
                    "doc_type": "subordinate legislation",
                    "detail_url": url,
                    "parent": law["name"],
                })
    if pdf_url and urlparse(pdf_url).hostname == urlparse(law["detail_url"]).hostname:
        # some pages link their own PDFs over plain http
        pdf_url = re.sub(r"^http://", "https://", pdf_url)
    return pdf_url, related


def local_name(pdf_url: str) -> str:
    """Stable, filesystem-safe file name; the hash keeps same-named files apart."""
    base = unquote(urlparse(pdf_url).path.rsplit("/", 1)[-1])
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base.rsplit(".", 1)[0]).strip("_")[:80]
    digest = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:10]
    suffix = ".docx" if base.lower().endswith(".docx") else ".pdf"
    return f"{stem}_{digest}{suffix}"


def extract_docx_text(docx_path: Path) -> tuple[str, None]:
    """Text of a Word file, one line per paragraph. Page count isn't stored in a .docx."""
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    paragraphs = []
    for paragraph in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        paragraph = re.sub(r"<w:(tab|br)\b[^>]*/>", " ", paragraph)
        paragraphs.append(html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", paragraph, re.S))))
    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), None


def already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def known_scanned(failures_path: Path) -> set[str]:
    """Documents whose latest recorded failure was 'scanned'.

    Retrying those can't help -- the PDF is already on disk and has no
    text layer -- so resume runs leave them for a separate OCR pass.
    """
    latest = {}
    if failures_path.exists():
        with failures_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    failure = json.loads(line)
                    latest[failure["id"]] = failure["reason"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return {doc_id for doc_id, reason in latest.items() if "likely scanned" in reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(SOURCES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_dir = DATA_DIR / args.source
    catalog_path = source_dir / "catalog.jsonl"
    out_path = source_dir / "statutes.jsonl"
    failures_path = source_dir / "fetch_failures.jsonl"
    raw_dir = source_dir / "raw_pdfs"
    if not catalog_path.exists():
        raise SystemExit(f"Run crawl_sources.py --source {args.source} first -- {catalog_path} not found.")
    raw_dir.mkdir(parents=True, exist_ok=True)

    with catalog_path.open(encoding="utf-8") as f:
        laws = [json.loads(line) for line in f]
    known = {law["id"] for law in laws}
    done = already_done(out_path)
    scanned = known_scanned(failures_path) - done
    todo = [law for law in laws if law["id"] not in done and law["id"] not in scanned]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.source}: {len(todo)} documents to fetch ({len(done)} already done, "
          f"{len(scanned)} already known to be scanned -- skipping those)")

    session = requests.Session()
    queue = deque(todo)
    failures = []

    with out_path.open("a", encoding="utf-8") as out_f, \
            failures_path.open("a", encoding="utf-8") as fail_f, \
            catalog_path.open("a", encoding="utf-8") as catalog_f, \
            tqdm(total=len(queue), desc=f"Fetching {args.source}") as bar:

        def record_failure(law: dict, reason: str, **where) -> None:
            # `where`: pdf_url / pdf_file when the file was downloaded, so a
            # later OCR pass knows which file belongs to the document
            failures.append((law["name"], reason))
            fail_f.write(json.dumps({
                "id": law["id"],
                "name": law["name"],
                "reason": reason,
                **where,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            fail_f.flush()

        while queue:
            law = queue.popleft()
            try:
                pdf_url = law.get("pdf_url")
                if not pdf_url and law.get("detail_url"):
                    pdf_url, related = read_detail_page(law, args.source, session)
                    time.sleep(1.0)
                    for extra in related:
                        # documents only reachable from another document's page
                        if extra["id"] not in known:
                            known.add(extra["id"])
                            catalog_f.write(json.dumps(extra, ensure_ascii=False) + "\n")
                            catalog_f.flush()
                            # a --limit run stays bounded; the next run finds
                            # these in the catalog
                            if extra["id"] not in done and not args.limit:
                                queue.append(extra)
                                bar.total += 1
                if not pdf_url:
                    record_failure(law, "no PDF link found" + (" on detail page" if law.get("detail_url") else " in listing"))
                    continue

                pdf_path = raw_dir / local_name(pdf_url)
                is_docx = pdf_path.suffix == ".docx"
                if not pdf_path.exists():
                    resp, pdf_url = download(pdf_url, args.source, session)
                    if is_docx:
                        start = 0 if resp.content[:2] == b"PK" else -1
                    else:
                        # Some of these servers prepend a stray HTML snippet to
                        # otherwise valid PDFs; drop anything before the header.
                        start = resp.content[:1024].find(b"%PDF")
                    if start < 0:
                        record_failure(law, f"download is not a {'Word file' if is_docx else 'PDF'} (content-type "
                                            f"{resp.headers.get('content-type')}, {len(resp.content)} bytes)")
                        continue
                    # write-then-rename so an interrupted run can't leave a half file
                    tmp_path = pdf_path.with_suffix(".part")
                    tmp_path.write_bytes(resp.content[start:])
                    tmp_path.replace(pdf_path)

                text, num_pages = extract_docx_text(pdf_path) if is_docx else extract_text(pdf_path)
                if len(text) < 100:
                    record_failure(law, "no extractable text -- likely scanned, needs OCR",
                                   pdf_url=pdf_url, pdf_file=pdf_path.name)
                    continue

                record = {
                    **law,
                    "pdf_url": pdf_url,
                    "pdf_file": pdf_path.name,
                    "raw_text": text,
                    "num_pages": num_pages,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                record_failure(law, f"{type(e).__name__}: {e}")

            finally:
                bar.update(1)
                time.sleep(1.5)  # polite pacing -- government servers, not CDNs

    print(f"\nOK -> {out_path}")
    if failures:
        print(f"\n{len(failures)} failed/skipped:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
