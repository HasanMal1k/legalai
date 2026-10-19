"""
generate_metadata.py

Stage 5: have a local LLM read the start of every document in
statutes.jsonl and write structured metadata for it:

    data/<source>/metadata.jsonl   one row per document, keyed like statutes.jsonl

The documents themselves are never modified. Resumable: rerun the same
command and it carries on from where it stopped.

Needs Ollama running locally with the model pulled:
    ollama pull qwen3.5:9b
For speed, start the server with OLLAMA_NUM_PARALLEL=4 so it batches
requests on the GPU.

Run:
    python generate_metadata.py --source kp --limit 50   # try it
    python generate_metadata.py --source all             # everything
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from ocr_scanned import SOURCES, source_layout

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
MAX_CHARS = 6000   # the title, number, dates and purpose are all near the start
NUM_CTX = 6144     # room for the prompt, 6,000 characters of text (denser in OCR'd documents) and the answer

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "instrument_type": {"type": "string", "enum": ["Act", "Ordinance", "Order", "Rules", "Regulation",
                                                       "Notification", "Other"]},
        "number": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "jurisdiction": {"type": "string", "enum": ["Federal", "Punjab", "Sindh", "Khyber Pakhtunkhwa",
                                                    "Balochistan", "West Pakistan", "Other"]},
        "date_enacted": {"type": ["string", "null"]},
        "is_amendment": {"type": "boolean"},
        "amends": {"type": ["string", "null"]},
        "repeals": {"type": ["string", "null"]},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["title", "instrument_type", "number", "year", "jurisdiction", "date_enacted", "is_amendment",
                 "amends", "repeals", "subjects", "summary"],
}

PROMPT = """You are cataloguing Pakistani legislation. From the text of the legal document below, extract metadata as JSON.

Rules:
- Use only what the text says. If something is not stated, use null. Never guess.
- title: the document's own short title, e.g. "The Arbitration Act, 1940".
- instrument_type: Act, Ordinance, Order, Rules, Regulation, Notification or Other.
- number: the law's number and year as printed, e.g. "XX of 1975"; null if none.
- year: the year in the title.
- jurisdiction: which legislature or government made it.
- date_enacted: date of assent / promulgation / notification as YYYY-MM-DD; null if not stated.
- is_amendment / amends: whether it amends another law, and the title of that law.
- repeals: the title of any law it repeals; null if none.
- subjects: 3 to 6 short topic keywords.
- summary: 2 to 3 plain sentences on what the document does.

DOCUMENT TEXT (may be truncated):
"""


def extract(text: str) -> dict:
    """Metadata for one document; raises if the model can't produce valid JSON."""
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA, json={
                "model": MODEL, "stream": False, "think": False, "format": SCHEMA,
                "messages": [{"role": "user", "content": PROMPT + text[:MAX_CHARS]}],
                "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 700},
            }, timeout=900)
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as e:
            last_error = e
            time.sleep(5 * (attempt + 1))
    raise last_error


def done_ids(path: Path, key: str) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)[key])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def documents(source: str, skip: set[str]):
    """Yield (id, row without the full text, text) for documents still to do."""
    data_dir, key = source_layout(source)
    with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row[key] not in skip:
                yield row[key], row.get("name"), row.get("text_source", "native"), row["raw_text"]


def run(source: str, limit: int | None, workers: int) -> int:
    data_dir, key = source_layout(source)
    out_path = data_dir / "metadata.jsonl"
    done = done_ids(out_path, key)
    todo = list(documents(source, done))
    if limit is not None:
        todo = todo[:limit]
    print(f"{source}: {len(todo)} documents to describe ({len(done)} already done)", flush=True)
    failed = 0
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract, text): (doc_id, name, text_source, len(text))
                   for doc_id, name, text_source, text in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Metadata {source}", unit="doc"):
            doc_id, name, text_source, chars = futures[future]
            try:
                metadata = future.result()
            except Exception as e:
                failed += 1
                print(f"  ! {name}: {type(e).__name__}: {e}", flush=True)
                continue  # left undone, so the next run retries it
            out.write(json.dumps({
                key: doc_id,
                "source": source,
                "name": name,
                "text_source": text_source,
                "metadata": metadata,
                "model": MODEL,
                "input_chars": min(chars, MAX_CHARS),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
            out.flush()
    print(f"OK -> {out_path}" + (f"  ({failed} failed; rerun to retry)" if failed else ""), flush=True)
    return len(todo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[*SOURCES, "all"])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many documents (across sources)")
    parser.add_argument("--workers", type=int, default=4, help="requests in flight; match OLLAMA_NUM_PARALLEL")
    args = parser.parse_args()

    remaining = args.limit
    for source in (SOURCES if args.source == "all" else [args.source]):
        if remaining is not None and remaining <= 0:
            break
        handled = run(source, remaining, args.workers)
        if remaining is not None:
            remaining -= handled


if __name__ == "__main__":
    main()
