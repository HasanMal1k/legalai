"""
eval/find.py

Look up provisions in the chunked corpus while writing evaluation questions.

    python eval/find.py "companies act, 2017 :: oppress"  "punjab labour code :: child|adolescent"

Each argument is "<title regex> :: <regex on the chunk's heading or text>" ("h:<regex>" for
the heading only); prints the matching provisions (first part of each) with their number,
heading and opening words.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_scanned import SOURCES, source_layout  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
specs = []
for arg in sys.argv[1:]:
    title, _, pattern = arg.partition("::")
    pattern = pattern.strip()
    heading_only = pattern.startswith("h:")
    specs.append((arg, re.compile(title.strip(), re.I), re.compile(pattern[2:] if heading_only else pattern or ".", re.I), heading_only))
found = {arg: [] for arg, *_ in specs}
for source in SOURCES:
    data_dir, _ = source_layout(source)
    with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["unit"] == "card":
                continue
            for arg, title, pattern, heading_only in specs:
                if title.search(chunk["title"]) and (pattern.search(chunk["heading"] or "") or (not heading_only and pattern.search(chunk["text"]))):
                    found[arg].append(chunk)
for arg, chunks in found.items():
    laws = sorted({(c["source"], c["title"], c["status"]) for c in chunks})
    print(f"\n=== {arg}  ({len(chunks)} chunks in {len(laws)} laws)")
    for source, title, status in laws[:6]:
        print(f"  [{source} | {status}] {title}")
        for c in [c for c in chunks if c["title"] == title and c["source"] == source and c["part"] == 1][:8]:
            path = (" > ".join(c["path"]))[-40:]
            print(f"     {c['unit']} {c['number'] or '-':6s} {(c['heading'] or '')[:46]:46s} {path:40s} | {c['text'][:90]!r}")
