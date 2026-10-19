"""Download a varied, resumable corpus from the official Sindh High Court archive."""
import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://caselaw.shc.gov.pk/caselaw/"
SEARCH = BASE + "search-all/search"
GROUPS = {
    "family": ["Family Appeal", "Divorce P."],
    "terrorism": ["Spl.Cr.A.T.A.", "Spl.Anti.Ter.A.", "Spl.Anti.T.Acq.A", "Spl.Anti.Ter.J.A.", "Conf.Case (A.T.A)"],
    "criminal": ["Criminal Appeal", "Cr.Appeal", "Cr.J.A", "Cr.Rev", "Cr.Bail"],
    "civil": ["Civil Revision", "R.A (Civil Revision)", "Civil Appeal", "II.A.", "Regular First Appeal"],
    "labour_service": ["Labour A.", "Labour Appeal", "L.R.A.", "Service Appeal"],
    "tax_customs": ["I.T.R", "Income Tax Reference", "Income Tax Appeal", "Sales Tax Reference", "Custom Reference", "Spl.Cus.A."],
    "commercial": ["Commercial Appeal", "Companies Original", "Judicial Companies Misc.", "Suit", "Suit.B"],
    "constitutional": ["Const. P.", "Writ P.", "Human Rights Petition"],
}
FAMILY_TOPICS = ("(family matter)", "family courts act", "guardian and ward", "muslim family")

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

class Archive:
    def __init__(self, delay, cache):
        self.delay = delay
        self.last_request = 0
        self.cache = cache
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "PakistanCaseCorpus/1.0 (research downloader)"
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "POST"], respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        response = self.request("GET", "https://caselaw.shc.gov.pk/robots.txt")
        self.robots = RobotFileParser()
        if response.status_code == 404:
            self.robots.parse([])
        else:
            response.raise_for_status()
            self.robots.parse(response.text.splitlines())
            if re.search(r"ai-input\s*=\s*no", response.text, re.I):
                raise RuntimeError("The source explicitly disallows AI input/RAG collection.")
        self.delay = max(delay, self.robots.crawl_delay("PakistanCaseCorpus") or 0)
        response = self.request("GET", SEARCH)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        self.categories = [(o["value"], o.get_text(" ", strip=True))
                           for o in soup.select("#STD_CASETYPES option") if o.get("value") != "-1"]
        self.family_topics = [o["value"] for o in soup.select("#ALL_TOPICS_M_SELECT option")
                              if any(k in o.get_text().lower() for k in FAMILY_TOPICS)]
        if not self.categories:
            raise RuntimeError("Archive search structure changed: no categories found.")

    def request(self, method, url, **kwargs):
        if hasattr(self, "robots") and not self.robots.can_fetch("PakistanCaseCorpus", url):
            raise RuntimeError("robots.txt disallows: " + url)
        time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        return self.session.request(method, url, timeout=(15, 90), **kwargs)

    def search(self, category_id, category, group, document_type, topics=None):
        key = hashlib.sha256(json.dumps([category_id, document_type, topics]).encode()).hexdigest()[:20]
        cached = self.cache / (key + ".json")
        if cached.exists():
            rows = json.loads(cached.read_text(encoding="utf-8"))
            return [dict(row, subject_group=group) for row in rows]
        payload = {name: "" for name in (
            "ALL_JUDGES_M_SELECT", "ALL_ADVOCATES_M_SELECT", "ALL_TOPICS_M_SELECT",
            "CASENO", "CASEYEAR", "PARTYSIDE1_NAMES", "ADVOCATENAME", "TAGJUDGES",
            "DATE_ORDER_JUDGMENT", "DATE_ORDER_JUDGMENT2")}
        payload.update(opt="search_judgement", STD_CASETYPES=category_id,
                       STD_DOCUMENTTYPES=document_type, STD_COURTS="-1",
                       STD_BENCHTYPES="-1", STD_JUDGES="-1", CASEGROUP="5", AFR_TF="false")
        if topics:
            payload.pop("ALL_TOPICS_M_SELECT")
            payload["ALL_TOPICS_M_SELECT[]"] = topics
        response = self.request("POST", BASE + "AJAX_PUBLIC.php", data=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, str):
            data = json.loads(data)
        if "msg" not in data:
            raise RuntimeError("Unexpected archive search response")
        soup = BeautifulSoup(data["msg"], "html.parser")
        headers = [th.get_text(" ", strip=True) for th in soup.select("thead th")]
        rows = []
        for tr in soup.select("tbody tr"):
            link = tr.select_one('a[href*="view-file/"]')
            if not link:
                continue
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
            raw = dict(zip(headers, cells))
            reference = raw.get("Case No.", "")
            rows.append({
                "document_id": raw.get("Code", ""),
                "case_key": re.sub(r"\s+", " ", reference).strip().casefold(),
                "case_reference": reference,
                "case_year": raw.get("Case Year", ""),
                "parties": raw.get("Parties", ""),
                "citation": raw.get("Citation", ""),
                "decision_date_raw": raw.get("Order Date", ""),
                "bench": raw.get("Bench", ""),
                "judges": raw.get("Hon'ble Judge/Bench", ""),
                "topics": raw.get("Topic", ""),
                "headnotes": raw.get("Head Notes", ""),
                "approved_for_reporting": raw.get("A.F.R", "").lower() == "yes",
                "court": "Sindh High Court",
                "archive_category": category,
                "subject_group": group,
                "subject_label_basis": "archive topic filter" if topics else "archive case category",
                "document_type": "judgment" if document_type == "1" else "order",
                "source_url": urljoin(BASE, link["href"]),
                "search_url": SEARCH,
                "raw_metadata": raw,
            })
        if not rows and "No Record Found" not in soup.get_text():
            raise RuntimeError("Archive result structure changed: missing download links")
        write_json(cached, rows)
        return rows

def discover(archive, groups, doc_type, seed, errors):
    queues = {}
    for group in groups:
        records = {}
        queries = [(cid, label, None) for cid, label in archive.categories if label in GROUPS[group]]
        if group == "family" and archive.family_topics:
            queries += [(cid, label, archive.family_topics) for cid, label in archive.categories
                        if label in ("Const. P.", "Civil Revision", "R.A (Civil Revision)")]
        for cid, label, topics in queries:
            try:
                results = archive.search(cid, label, group, doc_type, topics)
                print(f"[SEARCH] {group}: {label}: {len(results)} documents", flush=True)
                for row in results:
                    records.setdefault(row["source_url"], row)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                print(f"[SEARCH FAILED] {group}: {label}: {exc}", flush=True)
                append_json(errors, {"stage": "search", "group": group, "category": label,
                                     "error": str(exc), "timestamp": utc_now()})
        rows = list(records.values())
        random.Random(seed).shuffle(rows)
        queues[group] = deque(rows)
    return queues

def download(archive, row, output, max_bytes):
    from pypdf import PdfReader
    response = archive.request("GET", row["source_url"], stream=True)
    with response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("PDF exceeds --max-mb")
            chunks.append(chunk)
        content = b"".join(chunks)
    if not content.lstrip().startswith(b"%PDF-"):
        raise ValueError("Download is not a PDF (possibly an HTML error/login page)")
    digest = hashlib.sha256(content).hexdigest()
    stem = "shc_" + digest[:24]
    pdf = output / "pdfs" / (stem + ".pdf")
    temporary = pdf.with_suffix(".pdf.part")
    temporary.write_bytes(content)
    temporary.replace(pdf)
    pages, extraction_errors = [], []
    reader = PdfReader(pdf)
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            extraction_errors.append({"page": number, "error": str(exc)})
        pages.append({"page": number, "text": text})
    text_path = output / "texts" / (stem + ".json")
    write_json(text_path, {"source_url": row["source_url"], "pages": pages})
    row.update(sha256=digest, pdf_path=pdf.relative_to(output).as_posix(),
               text_path=text_path.relative_to(output).as_posix(), page_count=len(pages),
               text_characters=sum(len(p["text"].strip()) for p in pages),
               pages_needing_ocr=[p["page"] for p in pages if len(p["text"].strip()) < 40],
               extraction_errors=extraction_errors, downloaded_at=utc_now())
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="Total unique cases, including previous runs")
    parser.add_argument("--output", type=Path, default=Path("pakistan_cases"))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--document-type", choices=["judgment", "order"], default="judgment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mb", type=int, default=40)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.delay < 0 or args.max_mb < 1:
        parser.error("limit/max-mb must be positive; delay must be nonnegative")
    if not args.discover_only:
        try:
            import pypdf
        except ImportError:
            parser.error("Install PDF extraction first: uv add pypdf")
    for directory in ("pdfs", "texts", "cache"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    errors = args.output / "errors.jsonl"
    existing, seen_urls, hashes = {}, set(), set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                pdf = args.output / row["pdf_path"]
                if pdf.exists() and hashlib.sha256(pdf.read_bytes()).hexdigest() == row["sha256"]:
                    existing[row["case_key"]] = row
                    seen_urls.add(row["source_url"])
                    hashes.add(row["sha256"])
            except (ValueError, KeyError, OSError):
                continue
    counts = Counter(row["subject_group"] for row in existing.values())
    if len(existing) >= args.limit:
        print(f"Already have {len(existing)} verified unique cases. Increase --limit to collect more.")
        return 0
    archive = Archive(args.delay, args.output / "cache")
    queues = discover(archive, args.groups, "1" if args.document_type == "judgment" else "2", args.seed, errors)
    if args.discover_only:
        print("Discovered documents by group:", {group: len(q) for group, q in queues.items()})
        return 0
    while len(existing) < args.limit:
        active = [group for group, queue in queues.items() if queue]
        if not active:
            break
        group = min(active, key=lambda name: counts[name])
        row = queues[group].popleft()
        if row["source_url"] in seen_urls or row["case_key"] in existing:
            continue
        seen_urls.add(row["source_url"])
        try:
            row = download(archive, row, args.output, args.max_mb * 1024 * 1024)
            if row["sha256"] in hashes:
                continue
            append_json(manifest, row)
            hashes.add(row["sha256"])
            existing[row["case_key"]] = row
            counts[group] += 1
            print(f"[{len(existing)}/{args.limit}] {group}: {row['case_reference']} "
                  f"({row['page_count']} pages)", flush=True)
        except Exception as exc:
            print(f"[DOWNLOAD FAILED] {row['source_url']}: {exc}", flush=True)
            append_json(errors, {"stage": "download", "source_url": row["source_url"],
                                 "error": str(exc), "timestamp": utc_now()})
    summary = {"unique_cases": len(existing), "target": args.limit, "subjects": dict(counts),
               "cases_with_pages_needing_ocr": sum(bool(r.get("pages_needing_ocr")) for r in existing.values()),
               "source_scope": "Sindh High Court only", "timestamp": utc_now()}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if len(existing) < args.limit:
        print("Target not reached. Review errors.jsonl; expand categories or add another court.")
        return 2
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command to resume.")
        raise SystemExit(130)
