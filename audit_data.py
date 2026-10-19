"""
audit_data.py

Measures what is wrong with the indexed data, law by law, so that clean-up works from numbers
and not from the cases that happen to be noticed. Read-only; writes data/audit/*.jsonl.

    python audit_data.py

Checks:
  scan      share of words in a law's text that are garbled (seen in fewer than 5 documents of the
            whole corpus: real words recur, a scan's misreadings do not)
  copies    laws with more than one copy in the index (the same family): how alike the copies
            are and which is the cleaner
  title     the law's own short-title clause ("may be cited as the X Act, 2019") against the title
            in the metadata
  chunks    chunks that hold only a heading, chunks whose label does not match the number their
            text begins with, and provisions numbered twice in one law
  chars     stray characters: private-use code points, the replacement character, mojibake
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import dedupe
from finalize_metadata import norm_title, same_title
from ocr_scanned import SOURCES, source_layout

OUT = Path(__file__).parent / "data" / "audit"
WORD = re.compile(r"[A-Za-z]{3,}")
SHORT_TITLE = re.compile(r"(?:may\s+be\s+(?:cited|called)|shall\s+be\s+(?:cited|called))\s+(?:as\s+)?(?:the\s+)?"
                         r"([A-Z][^.;:\n]{3,150}?(?:Act|Ordinance|Order|Rules|Regulations?|Code))[,\s]*((?:1[89]|20)\d{2})", re.I)
PRIVATE = re.compile(r"[-]")
MOJIBAKE = re.compile(r"Ã.|â€|Â\s")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    laws = {}
    for source in SOURCES:
        data_dir, key = source_layout(source)
        meta = {}
        for line in (data_dir / "metadata_final.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["index_default"]:
                meta[row[key]] = row
        for line in (data_dir / "statutes.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row[key] in meta:
                record = meta[row[key]]
                record["_text"] = row["raw_text"]
                laws[record["work_id"]] = record
    print(f"indexed laws: {len(laws):,}")

    # ---- scan quality: words that recur across the corpus are real
    doc_freq = Counter()
    for record in laws.values():
        doc_freq.update({w.lower() for w in WORD.findall(record["_text"])})
    for record in laws.values():
        words = [w.lower() for w in WORD.findall(record["_text"])]
        record["_garble"] = sum(doc_freq[w] < 5 for w in words) / max(1, len(words))
    bands = Counter("over 25%" if r["_garble"] > .25 else "10-25%" if r["_garble"] > .10 else "5-10%" if r["_garble"] > .05 else "under 5%"
                    for r in laws.values())
    print("\nSCAN: share of garbled words per law:", dict(bands))
    print("   by text source (laws over 10%):", dict(Counter(r["text_source"] for r in laws.values() if r["_garble"] > .10)))
    print("   by site (laws over 10%):", dict(Counter(r["source"] for r in laws.values() if r["_garble"] > .10)))
    worst = sorted(laws.values(), key=lambda r: -r["_garble"])[:6]
    for r in worst:
        print(f"      {r['_garble']:4.0%} {r['source']:12s} {r['text_source']:7s} {r['title'][:60]} | {r['_text'][300:380]!r}")

    # ---- several copies of one law in the index
    families = defaultdict(list)
    for record in laws.values():
        families[record["family_id"]].append(record)
    multi = [members for members in families.values() if len(members) > 1]
    rows = []
    kinds = Counter()
    for members in multi:
        members.sort(key=lambda r: (r["authority_rank"], r["_garble"]))
        best = members[0]
        best_print = dedupe.fingerprint(best["_text"])
        for other in members[1:]:
            shared, contained = dedupe.overlap(best_print, dedupe.fingerprint(other["_text"]))
            kind = ("near-identical" if shared >= .85 else "same law, different text" if shared >= .3 else "related only")
            garbled = other["_garble"] > .10 and other["_garble"] > best["_garble"] + .05
            kinds[(kind, "worse scan" if garbled else "")] += 1
            rows.append({"family": best["family_id"], "best": [best["source"], best["title"], best["status"], round(best["_garble"], 3)],
                         "other": [other["source"], other["title"], other["status"], round(other["_garble"], 3), other["jurisdiction"]],
                         "shared": round(shared, 2), "kind": kind, "worse_scan": garbled})
    with (OUT / "copies.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nCOPIES: families with several indexed copies: {len(multi):,}; extra copies: {len(rows):,}")
    for (kind, worse), n in kinds.most_common():
        print(f"   {n:5d}  {kind}{' + ' + worse if worse else ''}")

    # ---- the law's own short title against the metadata title
    mismatch = []
    for record in laws.values():
        found = SHORT_TITLE.search(record["_text"][:30000])
        if not found:
            continue
        own = f"{found.group(1)}, {found.group(2)}"
        if not same_title(norm_title(own), norm_title(record["title"])):
            mismatch.append({"source": record["source"], "title": record["title"], "own": re.sub(r"\s+", " ", own),
                             "year_differs": str(record["year"]) != found.group(2), "work_id": record["work_id"]})
    with (OUT / "titles.jsonl").open("w", encoding="utf-8") as out:
        for row in mismatch:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nTITLES: short-title clause differs from the metadata title: {len(mismatch):,} "
          f"(year differs in {sum(r['year_differs'] for r in mismatch):,})")
    for row in mismatch[:8]:
        print(f"      {row['source']:12s} {row['title'][:55]:55s} <> own: {row['own'][:60]}")

    # ---- chunks
    stubs = mislabelled = 0
    numbered_twice = Counter()
    labels = defaultdict(Counter)
    bad_examples = []
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        for line in (data_dir / "chunks.jsonl").open(encoding="utf-8"):
            chunk = json.loads(line)
            if chunk["unit"] == "card":
                continue
            text = chunk["text"]
            if len(text) < 120:
                stubs += 1
            if chunk["number"] and chunk["part"] == 1:
                labels[chunk["work_id"]][(" > ".join(chunk["path"][:1]), chunk["unit"], chunk["number"])] += 1
                digits = re.match(r"\d+", chunk["number"]).group(0)
                if not re.search(rf"(?<!\d){digits}(?!\d)", text[:60]):
                    mislabelled += 1
                    if len(bad_examples) < 6:
                        bad_examples.append(f"{chunk['source']} {chunk['title'][:40]} | {chunk['unit']} {chunk['number']} | {text[:70]!r}")
    for work, counts in labels.items():
        numbered_twice[work] = sum(n - 1 for n in counts.values() if n > 1)
    print(f"\nCHUNKS: heading-only stubs {stubs:,} | labelled with a number their text does not begin with {mislabelled:,} | "
          f"laws with a provision number used twice {sum(1 for n in numbered_twice.values() if n):,}")
    for example in bad_examples:
        print("      " + example)

    # ---- stray characters
    private = sum(bool(PRIVATE.search(r["_text"])) for r in laws.values())
    replacement = sum("�" in r["_text"] for r in laws.values())
    mojibake = sum(bool(MOJIBAKE.search(r["_text"])) for r in laws.values())
    print(f"\nCHARS: laws with private-use characters {private:,} | with the replacement character {replacement:,} | with mojibake {mojibake:,}")
    print("   most common private-use characters:", Counter(c for r in laws.values() for c in PRIVATE.findall(r["_text"])).most_common(6))

    with (OUT / "laws.jsonl").open("w", encoding="utf-8") as out:
        for record in laws.values():
            out.write(json.dumps({"work_id": record["work_id"], "source": record["source"], "title": record["title"],
                                  "garble": round(record["_garble"], 3), "text_source": record["text_source"],
                                  "numbered_twice": numbered_twice.get(record["work_id"], 0)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
