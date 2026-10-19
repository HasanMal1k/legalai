"""
crawl_sources.py

Stage 1 for the sources beyond pakistancode.gov.pk: build a catalog of
every document each official portal lists. One catalog per source:

    data/<source>/catalog.jsonl

Sources (all government-run public portals):
    drs          drs.molaw.gov.pk       Ministry of Law & Justice "Document
                                        Retrieval System": federal Acts,
                                        Ordinances, President's Orders, Rules,
                                        Regulations as gazetted
    punjab       punjabcode.punjab.gov.pk   Punjab laws and rules
    sindh        sindhlaws.gov.pk           Sindh Acts, Ordinances, Rules
    kp           kpcode.kp.gov.pk           Khyber Pakhtunkhwa laws
    balochistan  balochistancode.gob.pk     Balochistan laws and rules

Every row has "id" (unique within the source), "name", and either a
"detail_url" (page that links to the PDF) or a direct "pdf_url".

Run:
    python crawl_sources.py --source punjab
    python crawl_sources.py --source all

Then: python fetch_sources.py --source <name>

Like crawl_catalog.py this goes slowly on purpose: one request at a
time per site, with a delay between requests.
"""

import argparse
import base64
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawl_catalog import HEADERS

DATA_DIR = Path(__file__).parent / "data"
DELAY = 1.0  # seconds between requests to the same site


def get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """GET with a couple of patient retries, then the polite delay."""
    last_error = None
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=90, **kwargs)
            resp.raise_for_status()
            time.sleep(DELAY)
            return resp
        except requests.RequestException as e:
            last_error = e
            time.sleep(5 * (attempt + 1))
    raise last_error


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def year_from(*texts: str) -> str | None:
    for text in texts:
        years = re.findall(r"\b(1[89]\d{2}|20\d{2})\b", text or "")
        if years:
            return years[-1]
    return None


# --------------------------------------------------------------------------
# DRS (federal, Ministry of Law & Justice)
# --------------------------------------------------------------------------

DRS_BASE = "https://drs.molaw.gov.pk"
DRS_PAGE_SIZE = 100


def crawl_drs(session: requests.Session) -> list[dict]:
    """/documents lists everything, newest-first, in pages of cards."""
    rows = {}
    page = 1
    while True:
        url = f"{DRS_BASE}/documents?pageNumber={page}&pageSize={DRS_PAGE_SIZE}"
        soup = BeautifulSoup(get(session, url).content, "html.parser")
        cards = soup.find_all("a", class_="document-card", href=True)
        new = 0
        for card in cards:
            query = parse_qs(urlparse(card["href"]).query)
            doc_id = (query.get("docId") or [None])[0]
            if not doc_id or doc_id in rows:
                continue
            title = card.find("h3")
            doc_type = card.find("p")
            meta = clean(" | ".join(s.get_text(" ", strip=True) for s in card.select("div.flex.gap-3 span")))
            name = clean(title.get_text(" ", strip=True)) if title else ""
            published = re.search(r"Published:\s*([\d-]+)", meta)
            rows[doc_id] = {
                "source": "drs",
                "id": f"drs-{doc_id}",
                "name": name,
                "doc_type": clean(doc_type.get_text(" ", strip=True)) if doc_type else None,
                "detail_url": urljoin(DRS_BASE, card["href"]),
                "year": (re.search(r"Year:\s*(\d{4})", meta) or [None, year_from(name)])[1],
                "promulgation_date": published.group(1) if published else None,
            }
            new += 1
        print(f"  drs page {page}: {len(cards)} cards, {new} new, {len(rows)} total", flush=True)
        if not new:
            break
        page += 1
    return list(rows.values())


# --------------------------------------------------------------------------
# Generic link-following crawl (Punjab, KP): listing pages link to other
# listing pages and to per-law detail pages.
# --------------------------------------------------------------------------

def crawl_links(session, source, starts, follow_re, detail_re, describe, ident=None, max_pages=4000):
    """Breadth-first over listing pages.

    starts:    [(doc_type label, url), ...] -- the label sticks to every law
               first reached from that start page.
    follow_re: listing-page URLs worth visiting.
    detail_re: per-law detail-page URLs to collect.
    describe:  callable(anchor) -> listing metadata text for that law.
    ident:     callable(detail url) -> stable document id, for sites where
               one document has many URLs. Defaults to the URL itself.
    """
    ident = ident or (lambda url: url)
    queue = deque(starts)
    queued = {url for _, url in starts}
    rows = {}
    pages = 0
    while queue and pages < max_pages:
        label, url = queue.popleft()
        try:
            soup = BeautifulSoup(get(session, url).content, "html.parser")
        except requests.RequestException as e:
            print(f"  ! {source}: giving up on {url}: {e}", flush=True)
            continue
        pages += 1
        new = 0
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0]
            if detail_re.search(href):
                name = clean(a.get_text(" ", strip=True))
                doc_id = ident(href)
                if doc_id in rows or not name:
                    continue
                meta = describe(a)
                rows[doc_id] = {
                    "source": source,
                    "id": doc_id,
                    "name": name,
                    "doc_type": label,
                    "detail_url": href,
                    "meta": meta or None,
                    "year": year_from(name, meta),
                    "listing_url": url,
                }
                new += 1
            elif follow_re.search(href) and href not in queued:
                queued.add(href)
                queue.append((label, href))
        if pages % 25 == 0 or new:
            print(f"  {source}: {pages} pages visited, {len(queue)} queued, "
                  f"{len(rows)} documents (+{new} from {url})", flush=True)
    if queue:
        print(f"  ! {source}: stopped at the {max_pages}-page safety limit with "
              f"{len(queue)} listing pages unvisited", flush=True)
    return list(rows.values())


def punjab_describe(a) -> str:
    # <div class="artlist"><a>NAME</a></div><div class="artdets">dept | Act no | date ...</div>
    holder = a.find_parent("div", class_="artlist")
    details = holder.find_next_sibling("div") if holder else None
    if details and "artdets" in (details.get("class") or []):
        return clean(details.get_text(" ", strip=True))
    return ""


def punjab_article_id(url: str) -> str:
    """Stable id for a Punjab Code show_article URL.

    The site salts the token in these URLs, so every listing page links
    the same law under a different URL. The token is base64 of
    (salt byte, masked byte) pairs; XOR-ing each pair cancels the salt
    and leaves a value that is constant per document.
    """
    token = url.rstrip("/").rsplit("/", 1)[-1].replace("-", "=").replace(".", "+").replace("~", "/")
    raw = base64.b64decode(token + "=" * (-len(token) % 4))
    return "punjab-" + bytes(raw[i] ^ raw[i + 1] for i in range(0, len(raw) - 1, 2)).hex()


def crawl_punjab(session: requests.Session) -> list[dict]:
    base = "https://punjabcode.punjab.gov.pk/en/"
    starts = [
        ("law", base + "get_laws"),
        ("law", base + "alphabetical/all"),
        ("rule", base + "get_rules"),
        ("rule", base + "service_rules"),
        ("rule", base + "general_rules"),
        ("repealed law", base + "list_repealed_chrono"),
        ("amending law", base + "list_amending_chrono"),
        # as-gazetted copies: separate documents from the consolidated ones above
        ("law (gazette copy)", base + "list_law_gazette_chrono"),
        ("rule (gazette copy)", base + "list_rule_gazette_chrono"),
    ]
    follow = re.compile(
        r"^https://punjabcode\.punjab\.gov\.pk/en/"
        r"(alphabetical|departmental|chronological|categorical|articles_by_\w+|"
        r"list_\w+|service_rules\w*|general_rules\w*|rules_by_\w+|\w*repealed\w*|\w*amending\w*)"
        r"(/[\w.~%-]+)*$")
    detail = re.compile(r"^https://punjabcode\.punjab\.gov\.pk/en/show_article/[^/]+$")
    return crawl_links(session, "punjab", starts, follow, detail, punjab_describe, ident=punjab_article_id)


def crawl_kp(session: requests.Session) -> list[dict]:
    base = "https://kpcode.kp.gov.pk/homepage/"
    starts = [("law", base + "list_all_law")]
    follow = re.compile(r"^https://kpcode\.kp\.gov\.pk/homepage/list_all_law(/\d+)+$")
    detail = re.compile(r"^https://kpcode\.kp\.gov\.pk/homepage/lawDetails/\d+$")

    def describe(a) -> str:
        holder = a.find_parent(["li", "tr", "div"])
        text = clean(holder.get_text(" ", strip=True)) if holder else ""
        return text if len(text) < 400 else ""

    return crawl_links(session, "kp", starts, follow, detail, describe)


# --------------------------------------------------------------------------
# Balochistan: each listing page ships its whole list in a hidden field,
# as '@'-separated HTML snippets that the page's JS paginates client-side.
# --------------------------------------------------------------------------

BALOCHISTAN_BASE = "https://balochistancode.gob.pk/"


def crawl_balochistan(session: requests.Session) -> list[dict]:
    rows = {}
    # The alphabetical view returns laws, rules, notifications and policies
    # in one list whichever tab ("opento") is asked for.
    for opento in (1,):
        url = f"{BALOCHISTAN_BASE}laws_rules.aspx?wise=alphabetical&opento={opento}"
        soup = BeautifulSoup(get(session, url).content, "html.parser")
        field = soup.find(id="datafield")
        items = (field.get("value") or "").split("@") if field else []
        heading = None
        new = 0
        for item in items:
            snippet = BeautifulSoup(item, "html.parser")
            a = snippet.find("a", href=True)
            if not a:
                # section headings ("ACTS", "ORDINANCES", ...) are link-less items
                heading = clean(snippet.get_text(" ", strip=True)) or heading
                continue
            detail_url = urljoin(BALOCHISTAN_BASE, a["href"])
            if "docid=" not in detail_url or detail_url in rows:
                continue
            small = snippet.find("small")
            meta = clean(small.get_text(" ", strip=True)) if small else ""
            name = clean(a.get_text(" ", strip=True))
            date = re.search(r"Promulgation Date:\s*([^|]+)", meta)
            rows[detail_url] = {
                "source": "balochistan",
                "id": detail_url,
                "name": name,
                "doc_type": heading.title() if heading else None,
                "detail_url": detail_url,
                "meta": meta or None,
                "year": year_from(name, meta),
                "promulgation_date": clean(date.group(1)) if date else None,
            }
            new += 1
        print(f"  balochistan opento={opento}: {len(items)} items, {new} new documents", flush=True)
    return list(rows.values())


# --------------------------------------------------------------------------
# Sindh: ASP.NET GridView, paged by form postback; rows link straight to PDFs.
# --------------------------------------------------------------------------

SINDH_BASE = "https://sindhlaws.gov.pk/"
SINDH_GRID = "ctl00$ContentPlaceHolder1$GridView1"


def sindh_rows(soup: BeautifulSoup, doc_type: str, page_url: str) -> list[dict]:
    rows = []
    grid = soup.find("table", id="ContentPlaceHolder1_GridView1")
    for tr in grid.find_all("tr") if grid else []:
        links = {clean(a.get_text()): urljoin(page_url, a["href"])
                 for a in tr.find_all("a", href=True) if ".pdf" in a["href"].lower()}
        if not links:
            continue  # header and pager rows
        cells = [clean(td.get_text(" ", strip=True)) for td in tr.find_all("td", recursive=False)]
        plain = [c for c in cells if c and c not in ("English", "Urdu", "Sindhi")]
        name = max(plain, key=len) if plain else ""
        date = plain[-1] if len(plain) > 1 and re.search(r"\d{4}$", plain[-1]) else None
        number = plain[0] if len(plain) > 1 and plain[0] != name and plain[0] != date else None
        pdf_url = links.get("English")
        rows.append({
            "source": "sindh",
            "id": pdf_url or next(iter(links.values())),
            "name": name,
            "doc_type": doc_type,
            "pdf_url": pdf_url,
            "act_or_ordinance_no": number,
            "year": year_from(name, date),
            "promulgation_date": date,
            "other_language_pdfs": {k: v for k, v in links.items() if k != "English"} or None,
        })
    return rows


def crawl_sindh(session: requests.Session) -> list[dict]:
    rows = {}
    for pg, doc_type in (("ACT", "Act"), ("ORDINANCE", "Ordinance"), ("RULE", "Rule")):
        url = f"{SINDH_BASE}Alpha.aspx?pg={pg}"
        soup = BeautifulSoup(get(session, url).content, "html.parser")
        page = 1
        while True:
            found = sindh_rows(soup, doc_type, url)
            new = [r for r in found if r["id"] not in rows]
            for r in new:
                rows[r["id"]] = r
            print(f"  sindh {pg} page {page}: {len(found)} rows, {len(new)} new, {len(rows)} total", flush=True)
            page += 1
            # only go on if the pager actually offers the next page
            if f"'Page${page}'" not in str(soup):
                break
            form = {i["name"]: i.get("value", "") for i in soup.find_all("input", type="hidden") if i.get("name")}
            form.update({"__EVENTTARGET": SINDH_GRID, "__EVENTARGUMENT": f"Page${page}"})
            resp = session.post(url, data=form, headers=HEADERS, timeout=90)
            resp.raise_for_status()
            time.sleep(DELAY)
            soup = BeautifulSoup(resp.content, "html.parser")
    return list(rows.values())


CRAWLERS = {
    "drs": crawl_drs,
    "punjab": crawl_punjab,
    "sindh": crawl_sindh,
    "kp": crawl_kp,
    "balochistan": crawl_balochistan,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[*CRAWLERS, "all"])
    args = parser.parse_args()

    for source in (CRAWLERS if args.source == "all" else [args.source]):
        print(f"Crawling {source} ...", flush=True)
        rows = CRAWLERS[source](requests.Session())
        out_dir = DATA_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "catalog.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"OK: {len(rows)} documents cataloged -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
