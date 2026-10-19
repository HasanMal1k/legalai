"""
eval/resolve.py

Turns the targets in eval/questions.py into the chunk ids that count as a correct answer,
and prints what each target resolved to, so every question can be checked by eye against
the provisions it points at. Writes eval/eval_set.jsonl.

    python eval/resolve.py            # resolve and print the check sheet
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from ocr_scanned import SOURCES, source_layout  # noqa: E402
from questions import QUESTIONS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
# provisions of the main text, not an Order's rules or a schedule's items with the same number
ORDER = re.compile(r"^Order [IVXLC]+\b")
ANNEXED = re.compile(r"^(?:\w+ )?Schedule\b|^Appendix|^Annex", re.I)


def in_main_text(path: list[str]) -> bool:
    return not (path and ANNEXED.match(path[0])) and not any(ORDER.match(part) for part in path)


def matches(chunk: dict, target: dict) -> bool:
    if not re.search(target["law"], chunk["title"], re.I):
        return False
    if target.get("source") and chunk["source"] != target["source"]:
        return False
    path = " > ".join(chunk["path"])
    if "n" in target:
        if chunk["number"] not in target["n"]:
            return False
        if "path" in target:
            return bool(re.search(target["path"], path))
        return in_main_text(chunk["path"])
    if "heading" in target:
        return bool(re.search(target["heading"], chunk["heading"] or "", re.I))
    if "text" in target:
        return bool(re.search(target["text"], chunk["text"], re.I))
    return False


def main():
    chunks = []
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if chunk["unit"] != "card":
                    chunks.append({k: chunk[k] for k in ("chunk_id", "source", "title", "status", "unit", "number", "heading", "path", "part", "text")})
    print(f"chunks searched: {len(chunks):,}")

    rows, problems = [], []
    for question in QUESTIONS:
        relevant, seen = [], []
        for target in question["targets"]:
            # a chunk holding only a heading (a contents line left in the text) answers nothing
            hits = [chunk for chunk in chunks if len(chunk["text"]) >= 120 and matches(chunk, target)]
            if not hits:
                problems.append(f"{question['id']}: nothing found for {target}")
            relevant += [chunk["chunk_id"] for chunk in hits]
            seen += hits
        rows.append({**{k: question[k] for k in ("id", "lang", "kind", "group", "q")}, "relevant": sorted(set(relevant)),
                     "laws": sorted({f"{c['source']}: {c['title']}" for c in seen})})
        print(f"\n{question['id']} [{question['lang']}/{question['kind']}] {question['q'][:150]}")
        shown = set()
        for chunk in seen:
            key = (chunk["source"], chunk["title"], chunk["number"])
            if key in shown:
                continue
            shown.add(key)
            print(f"    {chunk['source']:12s} {chunk['status'][:8]:8s} {chunk['title'][:44]:44s} {chunk['unit'][:4]} {chunk['number'] or '-':5s} "
                  f"{(chunk['heading'] or '')[:40]:40s} | {chunk['text'][:80]!r}")
    with (HERE / "eval_set.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} questions | by language {dict(Counter(r['lang'] for r in rows))} | by kind {dict(Counter(r['kind'] for r in rows))}")
    print(f"relevant chunks per question: median {sorted(len(r['relevant']) for r in rows)[len(rows) // 2]}")
    print("PROBLEMS:" if problems else "every target resolved", *problems, sep="\n  ")


if __name__ == "__main__":
    main()
