"""
chunk_model.py

Second opinion for the laws chunk_statutes.py could not structure (scans
with marginal notes, garbled text layers, numbers the rules cannot follow).
A local model reads the text, a window of lines at a time, and says on which
lines the numbered provisions begin. It only points at lines: nothing it
writes reaches the chunks. chunk_statutes.py checks each pointer against
the numbering and then cuts the original text there.

Needs Ollama running locally (start the server with OLLAMA_NUM_PARALLEL=4):
    python chunk_model.py --only "bait-ul-mal" --show     # try it on one law
    python chunk_model.py --source punjab --minutes 100    # as much as fits in 100 minutes
    python chunk_model.py                                  # every law reported with a problem
then run chunk_statutes.py again.

Writes data/<source>/chunk_model.jsonl, one row per law, each as soon as it
is read; laws already there are skipped, so it can be stopped and restarted.
Within a source the shortest laws are read first.
"""

import argparse
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests

from chunk_statutes import clean, laws
from ocr_scanned import SOURCES, read_jsonl, source_layout

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
WINDOW = 120    # lines shown to the model at a time ...
OVERLAP = 8     # ... each window starting this many lines before the last one ended
WORTH_ASKING = ("no numbered provisions found", "numbers are missing", "the contents list", "table of contents could not be followed")

SCHEMA = {
    "type": "object",
    "properties": {"provisions": {"type": "array", "items": {
        "type": "object",
        "properties": {"line": {"type": "integer"}, "number": {"type": "string"}, "heading": {"type": ["string", "null"]}},
        "required": ["line", "number", "heading"],
    }}},
    "required": ["provisions"],
}

PROMPT = """Below is part of a Pakistani law, one line of the document per row, each row starting with its line number in square brackets. The text may come from a poor scan: letters and digits can be wrong, and a marginal note may be mixed into a line.

Find every line on which a numbered provision of the law BEGINS: a section of an Act or Ordinance, a rule, a regulation, an article, or a numbered paragraph of an order or notification. These are numbered 1, 2, 3 ... in order through the law (sometimes 2A, 12B).

For each one give:
- line: the number in square brackets of the row where the provision begins
- number: the provision's own number, as digits with any letter (for example 7 or 12A). If the scan garbled it, give the number it must be from the order.
- heading: the provision's heading or marginal note if one is printed, otherwise null

Do NOT list: sub-sections such as (1) (2), clauses such as (a) (b) (i), items of a list or table inside a provision, entries of a table of contents, entries of a schedule, page numbers, page headings, or amendments quoted inside a provision.
If no provision begins in these rows, return an empty list.

{rows}"""


def ask(rows: str) -> list[dict]:
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA, json={
                "model": MODEL, "stream": False, "think": False, "format": SCHEMA,
                "messages": [{"role": "user", "content": PROMPT.replace("{rows}", rows)}],
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1500 + 1500 * attempt},
            }, timeout=1800)
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])["provisions"]
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(3)
    raise RuntimeError(f"model gave no usable answer: {last_error}")


def point(raw: str, give_up_at: float = float("inf"), threads: int = 1) -> dict:
    """The model's pointers for one law: [{line, number, heading, text}]; `text` is the line pointed at.
    `threads` reads that many windows of the law at once (for the few very long laws)."""
    lines, _ = clean(raw)

    def read(first: int, last: int) -> list[dict]:
        """Pointers for lines first .. last-1. A window the model cannot answer (it runs on past the
        length allowed, usually in a long table) is read again as two halves; a small one is left out."""
        if time.time() > give_up_at:
            raise TimeoutError("not finished when the time allowed ran out; it will be read on the next run")
        try:
            answer = ask("\n".join(f"[{n}] {lines[n].text[:220]}" for n in range(first, last)))
        except RuntimeError:
            if last - first <= 30:
                return []
            middle = (first + last) // 2
            return read(first, middle) + read(middle, last)
        found_here = []
        for found in answer:
            number = re.sub(r"[^0-9A-Za-z]", "", str(found.get("number", ""))).upper()
            if isinstance(found.get("line"), int) and first <= found["line"] < last and re.fullmatch(r"\d{1,3}[A-Z]{0,3}", number):
                found_here.append({"line": found["line"], "number": number, "heading": found.get("heading"),
                                   "text": lines[found["line"]].text[:120]})
        return found_here

    windows = [(first, min(len(lines), first + WINDOW)) for first in range(0, len(lines), WINDOW - OVERLAP)]
    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            answers = list(pool.map(lambda window: read(*window), windows))
    else:
        answers = [read(*window) for window in windows]
    starts = {start["line"]: start for answer in answers for start in answer}
    return {"lines": len(lines), "starts": [starts[n] for n in sorted(starts)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, nargs="+", help="these sources only, in this order (default: all)")
    parser.add_argument("--only", help="only laws whose title matches this regular expression, whether reported or not")
    parser.add_argument("--limit", type=int, help="stop after this many laws")
    parser.add_argument("--minutes", type=float, help="start no new law after this many minutes (those in hand are finished)")
    parser.add_argument("--max-chars", type=int, help="leave laws longer than this for a later run")
    parser.add_argument("--workers", type=int, default=4, help="laws in hand at once; workers x inner should match OLLAMA_NUM_PARALLEL")
    parser.add_argument("--inner", type=int, default=1, help="windows of one law read at once (use --workers 1 --inner 4 for very long laws)")
    parser.add_argument("--show", action="store_true", help="print what the model pointed at")
    parser.add_argument("--remaining", action="store_true", help="just print how many laws are left")
    args = parser.parse_args()
    only = re.compile(args.only, re.I) if args.only else None

    started, done_now, left = time.time(), 0, 0
    deadline = started + 60 * args.minutes if args.minutes else float("inf")
    for source in args.source or SOURCES:
        if time.time() >= deadline:
            break
        data_dir, key = source_layout(source)
        out_path = data_dir / "chunk_model.jsonl"
        done = {row["work_id"] for row in read_jsonl(out_path)} if out_path.exists() else set()
        reported = {row["work_id"] for row in read_jsonl(data_dir / "chunk_report.jsonl")
                    if any(problem in text for text in row["problems"] for problem in WORTH_ASKING)}
        todo = [(record, raw) for record, raw in laws(source, only)
                if record["work_id"] not in done and (only or record["work_id"] in reported)]
        left += len(todo)
        if args.remaining:
            continue
        # short laws first: more laws are finished per hour, and the few giants come last
        todo.sort(key=lambda item: len(item[1]))
        if args.max_chars:
            todo = [item for item in todo if len(item[1]) <= args.max_chars]
        if args.limit is not None:
            todo = todo[:max(0, args.limit - done_now)]

        def work(item):
            record, raw = item
            try:
                return record, point(raw, deadline + 600, args.inner), None
            except Exception as e:     # one law failing must not stop the run
                return record, None, str(e)

        with ThreadPoolExecutor(args.workers) as pool, out_path.open("a", encoding="utf-8") as out:
            waiting, flying = iter(todo), set()
            while True:
                # keep `workers` laws in flight; each is written the moment it is finished
                while len(flying) < args.workers and time.time() < deadline:
                    item = next(waiting, None)
                    if item is None:
                        break
                    flying.add(pool.submit(work, item))
                if not flying:
                    break
                finished, flying = wait(flying, return_when=FIRST_COMPLETED)
                for future in finished:
                    record, pointed, error = future.result()
                    if error:
                        print(f"  FAILED {source} {record['title'][:60]}: {error}", flush=True)
                        continue
                    done_now += 1
                    out.write(json.dumps({"work_id": record["work_id"], "title": record["title"], **pointed}, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"  {source:12s} {record['title'][:60]:60s} {pointed['lines']:6d} lines  {len(pointed['starts']):4d} provisions pointed at", flush=True)
                    if args.show:
                        for start in pointed["starts"]:
                            print(f"      [{start['line']:5d}] {start['number']:5s} {str(start['heading'])[:40]:40s} | {start['text'][:70]}")
    if args.remaining:
        print(f"{left} laws left")
    else:
        print(f"{done_now} laws read in {(time.time() - started) / 60:.0f} minutes" + ("; time allowed is up" if time.time() >= deadline else ""))


if __name__ == "__main__":
    main()
