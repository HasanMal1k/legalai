"""
crawl_status.py

The official word on which laws are in force, collected from the governments' own sites -- the ground truth the
register of laws in force (register.py) is built from, instead of inferring status from the texts.

    python crawl_status.py drs        # the Ministry of Law & Justice's Document Retrieval System

DRS (drs.molaw.gov.pk) files every federal document under one of five statuses -- Existing, Lapsed, Not Valid,
Repealed, Unlawful -- and lists each status on its own (/documents?status=...). Each list is read page by page
(100 cards a page, one request a second: a government server) into data/register/drs_status.jsonl, one row per
document with its status. A document the site lists under no status is recorded as such.

Never pakistanlawsite.com: its terms forbid it.
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawl_sources import DELAY, DRS_BASE, HEADERS, clean, year_from

OUT = Path(__file__).parent / "data" / "register"
DRS_STATUSES = ["Existing", "Lapsed", "Not Valid", "Repealed", "Unlawful"]
PAGE_SIZE = 100


def get(session: requests.Session, url: str) -> requests.Response:
    for attempt in range(5):
        try:
            response = session.get(url, timeout=90)
            time.sleep(DELAY)
            if response.status_code == 200:
                return response
        except requests.RequestException as error:
            print(f"  {type(error).__name__} on {url}; retrying", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def cards(html: bytes) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for card in soup.find_all("a", class_="document-card", href=True):
        query = parse_qs(urlparse(card["href"]).query)
        doc_id = (query.get("docId") or [None])[0]
        if not doc_id:
            continue
        title, kind = card.find("h3"), card.find("p")
        meta = clean(" | ".join(s.get_text(" ", strip=True) for s in card.select("div.flex.gap-3 span")))
        name = clean(title.get_text(" ", strip=True)) if title else ""
        published = re.search(r"Published:\s*([\d-]+)", meta)
        out.append({"doc_id": doc_id, "name": name, "doc_type": clean(kind.get_text(" ", strip=True)) if kind else None,
                    "year": (re.search(r"Year:\s*(\d{4})", meta) or [None, year_from(name)])[1],
                    "published": published.group(1) if published else None,
                    "detail_url": urljoin(DRS_BASE, card["href"])})
    return out


def crawl_drs(session: requests.Session) -> dict:
    found = {}
    for status in DRS_STATUSES + [None]:
        page, seen = 1, 0
        while True:
            filter_ = f"status={quote(status)}&" if status else ""
            batch = cards(get(session, f"{DRS_BASE}/documents?{filter_}pageNumber={page}&pageSize={PAGE_SIZE}").content)
            new = [c for c in batch if c["doc_id"] not in found or (status and found[c["doc_id"]]["status"] is None)]
            for c in batch:
                if status is None:
                    found.setdefault(c["doc_id"], dict(c, status=None))       # (listed under no status)
                elif c["doc_id"] not in found:
                    found[c["doc_id"]] = dict(c, status=status)
            seen += len(batch)
            print(f"  {status or 'all documents'}: page {page}, {len(batch)} cards, {len(new)} new, {len(found)} documents so far", flush=True)
            if len(batch) < PAGE_SIZE or not new:
                break
            page += 1
    return found


DETAIL_SECTIONS = {"List of Amendments": "amendments", "Sub-Ordinate Legislations": "subordinate", "Previous Laws": "previous",
                   "Primary Laws": "primary"}


def detail(html: bytes) -> dict:
    """A DRS law page: its general details, and the laws it is tied to -- its amendments (each with its own status),
    the subordinate legislation made under it (with the section), the laws it replaced, and for rules or a
    notification, the Act it was made under."""
    soup = BeautifulSoup(html, "html.parser")
    out = {key: [] for key in DETAIL_SECTIONS.values()}
    text = clean(soup.get_text(" ", strip=True))
    date = r"\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}"
    for label, key, value in (("Document Type", "document_type", r"\w+"), ("Number", "number", r"[\w./-]+"),
                              ("Gazette Year", "gazette_year", r"\d{4}"), ("(?<!Document )Type", "gazette_type", r"\w+"),
                              ("Publishing Date", "publishing_date", date), ("Commencement Date", "commencement_date", date)):
        m = re.search(rf"\b{label}:\s*({value})", text)
        out[key] = m.group(1) if m else None
    summary = soup.find(string=re.compile(r"^\s*Document Summary\s*$"))
    if summary:
        nxt = summary.find_parent().find_next(["p", "div"])
        out["summary"] = clean(nxt.get_text(" ", strip=True))[:600] if nxt else None
    for heading in soup.find_all("h5"):
        key = DETAIL_SECTIONS.get(heading.get_text(" ", strip=True))
        if not key or out[key]:
            continue
        # the section's own table, if it has one: an empty section ("No amendments available for this document") is
        # followed by the next section's heading, not a table
        table = None
        for element in heading.next_elements:
            if getattr(element, "name", None) == "table":
                table = element
                break
            if getattr(element, "name", None) == "h5":
                break
        if not table:
            continue
        rows = table.find_all("tr")
        names = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])] if rows else []
        for tr in rows[1:]:
            cells = [clean(c.get_text(" ", strip=True)) for c in tr.find_all("td")]
            if len(cells) < 2:
                continue
            link = tr.find("a", href=True)
            doc = (parse_qs(urlparse(link["href"]).query).get("docId") or [None])[0] if link else None
            out[key].append(dict(zip(names, cells), doc_id=doc))
    return out


def crawl_drs_details(session: requests.Session) -> None:
    """Every DRS document's page, once (resumable: pages already read are skipped)."""
    path = OUT / "drs_details.jsonl"
    done = {json.loads(line)["doc_id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
    docs = [json.loads(line) for line in (OUT / "drs_status.jsonl").open(encoding="utf-8")]
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"{len(done)} pages read before, {len(todo)} to read", flush=True)
    with path.open("a", encoding="utf-8") as out:
        for n, doc in enumerate(todo, 1):
            try:
                row = detail(get(session, doc["detail_url"]).content)
            except RuntimeError as error:
                print(f"  failed: {error}", flush=True)
                continue
            out.write(json.dumps({"doc_id": doc["doc_id"], **row}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 100 == 0:
                print(f"  {n} of {len(todo)} pages", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    if "drs-details" in sys.argv[1:]:
        crawl_drs_details(session)
    if "pakistancode" in sys.argv[1:]:
        # the Pakistan Code's alphabetical list of laws, read afresh (crawl_catalog.py's first pass): the laws the
        # Ministry lists as federal law; the site's search finds others it keeps out of the list
        import string
        from crawl_catalog import crawl_letter
        listed, seen = [], set()
        for letter in string.ascii_uppercase:
            for law in crawl_letter(letter, session):
                if law["detail_url"] not in seen:
                    seen.add(law["detail_url"])
                    listed.append(dict(law, listed=True))
            print(f"  {letter}: {len(listed)} laws so far", flush=True)
        with (OUT / "pakistancode_list.jsonl").open("w", encoding="utf-8") as out:
            for law in listed:
                out.write(json.dumps(law, ensure_ascii=False) + "\n")
        print(f"Pakistan Code: {len(listed)} laws in the alphabetical list")
    if "drs" in sys.argv[1:]:
        rows = crawl_drs(session)
        with (OUT / "drs_status.jsonl").open("w", encoding="utf-8") as out:
            for row in rows.values():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = {}
        for row in rows.values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(f"DRS: {len(rows)} documents, by status {counts}")


if __name__ == "__main__":
    main()
