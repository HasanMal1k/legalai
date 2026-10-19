"""
enrich_metadata.py

Stage 6: a second, deeper metadata pass over every document.

The first pass (generate_metadata.py) read only the opening of each
document. This one gives the model the opening, the END (where "Repeal
and savings" sections sit), and every passage in between that talks
about repeal, lapse, supersession or the Act that rules were made under,
and asks for a richer record:

    data/<source>/metadata_v2.jsonl   one row per document

Nothing else is modified. Resumable.

The model is asked only what the document itself says. Whether a law is
still in force can't be known from its own text, so this stage records
just the in-text evidence ("status_in_text"); finalize_metadata.py
combines that with the site listings and with repeals found in OTHER
documents to produce the actual status.

Needs Ollama running locally (start the server with OLLAMA_NUM_PARALLEL=4):
    python enrich_metadata.py --source kp --limit 40
    python enrich_metadata.py --source all
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from generate_metadata import done_ids
from ocr_scanned import SOURCES, read_jsonl, source_layout

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
NUM_CTX = 8192
HEAD_CHARS = 6500
TAIL_CHARS = 2500
SNIPPET_CHARS = 2400   # total budget for passages picked out of the middle
SNIPPET_RADIUS = 260

SITES = {
    "pakistancode": "Pakistan Code (federal laws, consolidated text)",
    "drs": "Ministry of Law Document Retrieval System (federal documents as gazetted)",
    "punjab": "Punjab Code (laws applicable in Punjab; also hosts some federal laws)",
    "sindh": "Sindh Code (Sindh laws)",
    "kp": "Khyber Pakhtunkhwa Code (KP laws)",
    "balochistan": "Balochistan Code (Balochistan laws, rules and notifications)",
}

# Passages worth showing the model wherever they occur in the document.
SIGNALS = re.compile(
    r"repeal|rescind|cease[sd]? to have effect|shall stand (?:dissolved|abolished)|superse[ds]|"
    r"lapse|expire|shall remain in force|omitted by|substituted by|"
    r"in exercise of (?:the )?powers? conferred|is hereby amended|shall be amended|savings?\b", re.I)

AREAS = ["Constitutional and Administrative", "Criminal", "Civil Procedure and Courts", "Family and Personal",
         "Property and Land", "Taxation and Revenue", "Banking and Finance", "Corporate and Commercial",
         "Labour and Employment", "Civil Service", "Local Government", "Elections", "Police and Security",
         "Defence and Military", "Education", "Health", "Environment and Natural Resources", "Energy",
         "Agriculture and Food", "Transport and Communications", "Media and Information Technology",
         "Religious Affairs", "Social Welfare", "Other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "short_title": {"type": "string"},
        "long_title": {"type": ["string", "null"]},
        "instrument_type": {"type": "string", "enum": ["Act", "Ordinance", "Order", "Rules", "Regulation",
                                                       "Notification", "Other"]},
        "number": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "enacting_authority": {"type": "string", "enum": [
            "Parliament", "President", "Governor-General", "Provincial Assembly", "Governor",
            "Federal Government", "Provincial Government", "Regulatory Authority",
            "Martial Law Administrator", "Pre-1947 Legislature", "Other"]},
        "jurisdiction": {"type": "string", "enum": ["Federal", "Punjab", "Sindh", "Khyber Pakhtunkhwa",
                                                    "Balochistan", "West Pakistan", "Other"]},
        "territorial_extent": {"type": ["string", "null"]},
        "date_enacted": {"type": ["string", "null"]},
        "commencement": {"type": ["string", "null"]},
        "parent_law": {"type": ["string", "null"]},
        "is_amendment": {"type": "boolean"},
        "amends": {"type": "array", "items": {"type": "string"}},
        "repeals": {"type": "array", "items": {"type": "string"}},
        "status_in_text": {"type": "string", "enum": ["no_indication", "stated_repealed", "stated_lapsed_or_expired",
                                                      "time_limited", "stated_superseded"]},
        "status_evidence": {"type": ["string", "null"]},
        "repealed_by": {"type": ["string", "null"]},
        "practice_areas": {"type": "array", "items": {"type": "string", "enum": AREAS}},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "establishes": {"type": "array", "items": {"type": "string"}},
        "creates_offences": {"type": "boolean"},
        "max_penalty": {"type": ["string", "null"]},
        "key_definitions": {"type": "array", "items": {"type": "string"}},
        "administering_body": {"type": ["string", "null"]},
        "one_line": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["short_title", "long_title", "instrument_type", "number", "year", "enacting_authority",
                 "jurisdiction", "territorial_extent", "date_enacted", "commencement", "parent_law",
                 "is_amendment", "amends", "repeals", "status_in_text", "status_evidence", "repealed_by",
                 "practice_areas",
                 "subjects", "establishes", "creates_offences", "max_penalty", "key_definitions",
                 "administering_body", "one_line", "summary"],
}

PROMPT = """You are cataloguing Pakistani legislation for a legal research index. Read the excerpts of ONE legal document below and fill in the JSON record.

Hard rules:
- Use ONLY what the excerpts say. If something is not stated, use null (or an empty list). Never guess and never use outside knowledge.
- Copy titles of other laws exactly as written, with their year.

Fields:
- short_title: the document's own short title, e.g. "The Arbitration Act, 1940".
- long_title: the sentence stating its purpose ("An Act to ..."), if present.
- instrument_type / number / year: e.g. Act, "XX of 1975", 1975. year is the year in the title.
- enacting_authority: who made it. Acts are made by Parliament (or the National Assembly / Majlis-e-Shoora) or by a Provincial Assembly; Ordinances by the President or a Governor (Governor-General before 1956); rules, regulations and notifications by the Federal Government, a Provincial Government or a regulatory authority. Laws from before 1947 are "Pre-1947 Legislature".
- jurisdiction: follows from who made it. Parliament, the President, the Federal Government or a federal regulator -> "Federal". The Assembly, Governor or Government of a province -> that province ("Punjab", "Sindh", "Khyber Pakhtunkhwa" (formerly N.W.F.P.), "Balochistan"). The former province of West Pakistan (1955 to 1970) -> "West Pakistan". Decide from the enacting words and the title, not from the city where the gazette was printed or from places merely mentioned.
- territorial_extent: where it applies, as stated ("the whole of Pakistan", "the Province of Sindh" ...).
- date_enacted: date of assent / promulgation / notification, YYYY-MM-DD.
- commencement: when it comes into force, as stated ("at once", a date, "on such date as the Government may notify").
- parent_law: ONLY for rules, regulations, orders and notifications: the Act or Ordinance under whose powers they were made. For an Act or an Ordinance this is null (the Constitution is never a parent_law).
- is_amendment, amends: other laws that this document amends.
- repeals: other laws that THIS document repeals or supersedes (usually a "Repeal" section near the end). Never list the document itself.
- status_in_text: does the text say that THIS WHOLE DOCUMENT is no longer law?
    stated_repealed: a heading, bracket or footnote says this law itself was repealed, e.g. "[Repealed by Act I of 2019]", or the catalogue entry says "(Repealed by ...)".
    stated_lapsed_or_expired: it says this law has lapsed or expired.
    time_limited: it says it remains in force only for a fixed period.
    stated_superseded: it says this whole law was replaced by a named later law.
    no_indication: anything else. In particular it is no_indication when (a) this document repeals or supersedes ANOTHER law, or (b) footnotes say that a section, clause or word was "substituted", "inserted", "added" or "omitted" by a later Act: those are ordinary amendments and the law is still in force.
- status_evidence: the exact words supporting status_in_text; null for no_indication.
- repealed_by: the law that repealed or replaced this document, if the text names it; otherwise null.
- practice_areas: 1 to 3 from the allowed list.
- subjects: 4 to 8 specific topic keywords.
- establishes: bodies it creates (authorities, boards, tribunals, funds, universities).
- creates_offences, max_penalty: whether it creates criminal offences, and the heaviest penalty stated.
- key_definitions: up to 8 terms it defines.
- administering_body: the ministry, department or authority responsible for it, if stated.
- one_line: one sentence, under 25 words, saying what the document does.
- summary: 3 to 5 plain sentences covering purpose, who it applies to, and its main mechanisms.

WHERE THIS COPY CAME FROM: {site}
CATALOGUE ENTRY (from the website; may be imperfect): {catalogue}

{excerpts}
"""


def excerpts(text: str) -> str:
    """Opening + signal passages from the middle + ending, labelled."""
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return "=== FULL TEXT ===\n" + text
    head, tail = text[:HEAD_CHARS], text[-TAIL_CHARS:]
    middle = text[HEAD_CHARS:-TAIL_CHARS]
    spans = []
    for match in SIGNALS.finditer(middle):
        start, end = max(0, match.start() - SNIPPET_RADIUS), min(len(middle), match.end() + SNIPPET_RADIUS)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    picked, used = [], 0
    for start, end in spans:
        if used + (end - start) > SNIPPET_CHARS:
            break
        picked.append(middle[start:end].strip())
        used += end - start
    parts = ["=== OPENING OF THE DOCUMENT ===\n" + head]
    if picked:
        parts.append("=== PASSAGES FROM THE MIDDLE (not continuous) ===\n" + "\n[...]\n".join(picked))
    parts.append("=== END OF THE DOCUMENT ===\n" + tail)
    return "\n\n".join(parts)


def catalogue_line(row: dict) -> str:
    keep = ("name", "doc_type", "category", "act_or_ordinance_no", "year", "promulgation_date", "meta", "parent")
    return json.dumps({k: row[k] for k in keep if row.get(k)}, ensure_ascii=False)


def enrich(source: str, row: dict) -> dict:
    prompt = PROMPT.format(site=SITES[source], catalogue=catalogue_line(row), excerpts=excerpts(row["raw_text"]))
    options = {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 1400}
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA, json={
                "model": MODEL, "stream": False, "think": False, "format": SCHEMA,
                "messages": [{"role": "user", "content": prompt}],
                "options": options,
            }, timeout=1800)
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])
        except ValueError as e:
            # The answer was cut off mid-JSON: a "Repealing and Amending" Act
            # can list dozens of laws. Give it room, then cap the lists.
            last_error = e
            options = {**options, "num_ctx": 12288, "num_predict": 5000}
            if attempt == 1:
                prompt += "\nList at most 40 laws in each of amends and repeals.\n"
        except (requests.RequestException, KeyError) as e:
            last_error = e
            time.sleep(5 * (attempt + 1))
    raise last_error


def run(source: str, limit: int | None, workers: int) -> int:
    data_dir, key = source_layout(source)
    out_path = data_dir / "metadata_v2.jsonl"
    done = done_ids(out_path, key)
    todo = []
    with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row[key] not in done:
                todo.append(row)
                if limit is not None and len(todo) >= limit:
                    break
    print(f"{source}: {len(todo)} documents to enrich ({len(done)} already done)", flush=True)
    failed = 0
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(enrich, source, row): row for row in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Enriching {source}", unit="doc"):
            row = futures[future]
            try:
                metadata = future.result()
            except Exception as e:
                failed += 1
                print(f"  ! {row.get('name')}: {type(e).__name__}: {e}", flush=True)
                continue  # left undone, so the next run retries it
            out.write(json.dumps({
                key: row[key],
                "source": source,
                "name": row.get("name"),
                "text_source": row.get("text_source", "native"),
                "text_chars": len(row["raw_text"]),
                "metadata": metadata,
                "model": MODEL,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            out.flush()
    print(f"OK -> {out_path}" + (f"  ({failed} failed; rerun to retry)" if failed else ""), flush=True)
    return len(todo)


def count_remaining() -> int:
    """Documents across all sources that still have no metadata_v2 row."""
    left = 0
    for source in SOURCES:
        data_dir, key = source_layout(source)
        done = done_ids(data_dir / "metadata_v2.jsonl", key)
        with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
            left += sum(1 for line in f if json.loads(line)[key] not in done)
    return left


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many documents (across sources)")
    parser.add_argument("--workers", type=int, default=4, help="requests in flight; match OLLAMA_NUM_PARALLEL")
    parser.add_argument("--remaining", action="store_true", help="just print how many documents are left")
    args = parser.parse_args()

    if args.remaining:
        print(count_remaining())
        return

    remaining = args.limit
    for source in (SOURCES if args.source == "all" else [args.source]):
        if remaining is not None and remaining <= 0:
            break
        handled = run(source, remaining, args.workers)
        if remaining is not None:
            remaining -= handled


if __name__ == "__main__":
    main()
