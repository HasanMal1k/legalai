"""
crawl_catalog.py

Stage 1 of statute ingestion: crawl pakistancode.gov.pk's alphabetical
listing pages to build a catalog of every federal law, with metadata
(name, category, act number, promulgation date, detail-page URL)
pulled straight from the listing -- no PDF parsing needed yet. The
alphabetical listing leaves some laws out, so a second pass asks the
site's search for each year and adds whatever the listing missed.

Source: pakistancode.gov.pk is the Ministry of Law & Justice's own public
portal ("an initiative ... to pave the path for access to justice in
Pakistan"). No scraping restriction in its terms (unlike PakistanLawSite).
Still: this hits a government server, so we go slow and polite.

Run:
    pip install requests beautifulsoup4 tqdm
    python crawl_catalog.py

Output:
    data/catalog.jsonl  -- one row per law found
"""

import json
import re
import string
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://pakistancode.gov.pk/english/LGu0xAD"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "catalog.jsonl"

# How many pages to try per letter before giving up (site paginates;
# "No Records Found" signals the end -- we stop on that, this is just
# a safety ceiling so a bug can't loop forever).
MAX_PAGES_PER_LETTER = 50

SEARCH = "https://pakistancode.gov.pk/english/sHyuRxF"
# Every law title contains at least one of these letters.
SEARCH_TERMS = ("a", "e")
SEARCH_ROW_CAP = 100
FIRST_YEAR = 1830  # oldest law on the site is from 1838


def parse_promulgation_date(meta_text: str) -> str | None:
    match = re.search(r"Promulgation Date:\s*([^.|]*)", meta_text)
    if not match:
        return None
    date = match.group(1).strip()
    # Normal form is "February 17 1975". When the site has no structured
    # date it prints a bare "December" placeholder, sometimes followed by
    # a free-text date ("December 15th May, 1961") -- drop the placeholder
    # and keep whatever real date is left.
    if not re.fullmatch(r"[A-Za-z]+ \d{1,2},? \d{4}", date):
        date = re.sub(r"^December\s*", "", date)
    return date if re.search(r"\d{4}$", date) else None


def parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []

    # Each law entry is a <div class="accordion-section"> holding two
    # children: an "accordion-section-title" div with the detail-page
    # link, and an "accordion-section-content" div with a line of
    # "Category | Act No | Promulgation Date: ... | Archive N |" text.
    # The same markup repeats in each tab pane (primary legislation,
    # ordinances, amendments, secondary legislation), and a law can
    # appear in more than one pane -- callers dedupe on detail_url.
    for section in soup.find_all("div", class_="accordion-section"):
        title = section.find("div", class_="accordion-section-title")
        a = title.find("a", href=True) if title else None
        # Detail-page links all contain '-sg-', which nav links don't.
        if not a or "-sg-" not in a["href"]:
            continue
        name = a.get_text(strip=True)
        if not name:
            continue

        content = section.find("div", class_="accordion-section-content")
        meta_text = re.sub(r"\s+", " ", content.get_text(" ", strip=True)) if content else ""
        fields = [f.strip() for f in meta_text.split("|")]

        # Usually "<X> Laws", but not always (e.g. "Law of Evidence").
        category = fields[0] if len(fields) > 1 and fields[0] else None
        # The number field is free text on the site ("XX of 1975",
        # "Ordinance No.XXVII of 1981", "Act(VII of 2010) of 2010", ...),
        # so keep it verbatim and only pull the year out of it.
        act_no = fields[1] if len(fields) > 1 and "Promulgation" not in fields[1] else None
        promulgation_date = parse_promulgation_date(meta_text)
        year_matches = re.findall(r"of\s+(\d{4})", act_no or "", re.IGNORECASE)
        if year_matches:
            year = year_matches[-1]
        elif promulgation_date:
            year = promulgation_date[-4:]
        else:
            year = None

        records.append({
            "name": name,
            # hrefs are relative to the listing page's /english/ directory
            "detail_url": urljoin(BASE, a["href"]),
            "category": category,
            "act_or_ordinance_no": act_no or None,
            "year": year,
            "promulgation_date": promulgation_date,
        })

    return records


def crawl_letter(letter: str, session: requests.Session) -> list[dict]:
    all_records = []
    seen_urls = set()
    for page in range(1, MAX_PAGES_PER_LETTER + 1):
        url = f"{BASE}?alp={letter}&page={page}&action=inactive"
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # Can't stop on the literal "No Records Found" text: every page
        # has several tab panes and an empty pane (e.g. amendments)
        # prints that message even when other panes have laws. In
        # practice the site returns a whole letter on page 1 and nothing
        # on page 2, so stop as soon as a page adds no new laws.
        records = parse_listing_page(resp.text)
        new_records = [r for r in records if r["detail_url"] not in seen_urls]
        if not new_records:
            time.sleep(1.0)
            break

        for r in new_records:
            seen_urls.add(r["detail_url"])
        all_records.extend(new_records)

        time.sleep(1.0)  # be polite to a government server
    return all_records


def crawl_search_year(year: int, session: requests.Session) -> list[dict]:
    """Laws the site's search returns for one year.

    The alphabetical listing silently omits some laws (mostly repealed
    ones and President's Orders) that search still returns. Search is a
    substring match on the title and shows at most 100 rows per tab, so
    we ask per year with substrings broad enough to match every title.
    """
    records = []
    for term in SEARCH_TERMS:
        url = f"{SEARCH}?title={term}&cat=&year={year}&type=title&search=1"
        resp = session.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        # Search pages aren't always valid UTF-8, so don't trust resp.text.
        html = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for pane in soup.find_all("div", class_="tab-pane"):
            if len(pane.find_all("div", class_="accordion-section")) >= SEARCH_ROW_CAP:
                print(f"  ! search for {term!r} in {year} hit the {SEARCH_ROW_CAP}-row "
                      f"cap -- results for that year may be incomplete")
        records.extend(parse_listing_page(html))
        time.sleep(1.0)
    return records


def main():
    session = requests.Session()
    all_laws = []
    seen = set()

    for letter in tqdm(string.ascii_uppercase, desc="Crawling letters"):
        try:
            letter_laws = crawl_letter(letter, session)
        except Exception as e:
            print(f"  ! failed on letter {letter}: {e}")
            continue
        for law in letter_laws:
            if law["detail_url"] not in seen:
                seen.add(law["detail_url"])
                all_laws.append(law)

    listed = len(all_laws)
    for year in tqdm(range(FIRST_YEAR, time.localtime().tm_year + 1), desc="Searching years"):
        try:
            year_laws = crawl_search_year(year, session)
        except Exception as e:
            print(f"  ! failed on search year {year}: {e}")
            continue
        for law in year_laws:
            if law["detail_url"] not in seen:
                seen.add(law["detail_url"])
                all_laws.append(law)
    print(f"\n{listed} laws from the alphabetical listing, "
          f"{len(all_laws) - listed} more found only via search")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for law in all_laws:
            f.write(json.dumps(law, ensure_ascii=False) + "\n")

    print(f"\nOK: {len(all_laws)} laws cataloged -> {OUT_PATH}")
    print("Next: run fetch_statutes.py to resolve each detail page's PDF "
          "link and download the actual law text.")


if __name__ == "__main__":
    main()