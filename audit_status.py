"""Audit of the in-force tag: every way it can contradict itself or the sites it came from."""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"D:\Hasan\Scrapper")
import finalize_metadata as f
import ocr_scanned as o

rows = []
for s in o.SOURCES:
    d, key = o.source_layout(s)
    for l in (d / "metadata_final.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        r["_id"] = r[key]
        rows.append(r)
index = [r for r in rows if r["index_default"]]
print(f"documents {len(rows)} | in index {len(index)} | status in index: {dict(Counter(r['status'] for r in index))}")


def kind_of(basis: str) -> str:
    for label, start in [("site says repealed", "the website's entry"), ("Punjab Code repealed list", "listed under repealed"),
                         ("note at head of document", "note at the head"), ("repeal note quoted by model", "the document carries a repeal note"),
                         ("another law's own words", "repealed by "), ("same law on a current code", "same law as"),
                         ("federal law on provincial code", "a federal law still carried")]:
        if basis.startswith(start):
            return label
    return basis[:40]


repealed = [r for r in rows if r["status"] == "repealed"]
print("\n1. REPEALED, by what the tag rests on (all documents / in index):")
c_all, c_idx = Counter(kind_of(r["status_basis"][0]) for r in repealed), Counter(kind_of(r["status_basis"][0]) for r in repealed if r["index_default"])
for k, n in c_all.most_common():
    print(f"     {n:4d} / {c_idx[k]:4d}  {k}")

print("\n2. THE SAME LAW WITH DIFFERENT TAGS (same family, one copy repealed and another in force), in index:")
families = defaultdict(list)
for r in index:
    families[r["family_id"]].append(r)
clash = [m for m in families.values() if {"repealed", "in_force"} <= {r["status"] for r in m}]
print(f"   families: {len(clash)}")
for m in clash[:40]:
    print("   - " + " | ".join(f"{r['source']}:{r['status']}({r['jurisdiction'][:8]}) {r['title'][:38]}" for r in m))

print("\n3. SAME TITLE AND YEAR, DIFFERENT TAGS, whatever the family (in index, repealed vs in force):")
by_title = defaultdict(list)
for r in index:
    if r["year"]:
        by_title[(f.norm_title(r["title"]), r["year"])].append(r)
clash2 = [m for m in by_title.values() if {"repealed", "in_force"} <= {r["status"] for r in m} and len({r["family_id"] for r in m}) > 1]
print(f"   groups: {len(clash2)}")
for m in clash2[:25]:
    print("   - " + " | ".join(f"{r['source']}:{r['status']}({r['jurisdiction'][:8]}) {r['title'][:38]}" for r in m))

print("\n4. PAKISTAN CODE laws tagged unknown (on the site, no repeal note, not in the alphabetical listing):")
unk = [r for r in index if r["source"] == "pakistancode" and r["status"] == "unknown"]
print(f"   {len(unk)}:", "; ".join(f"{r['title'][:45]} ({r['year']})" for r in unk[:40]))

print("\n5. IN FORCE, but another law in the corpus names it in a repeal that was not accepted (in index):")
leads = [r for r in index if r["status"] == "in_force" and any(x["relation"] == "possibly_affected_by" for x in r["related"])]
why = Counter(x["note"][:60] for r in leads for x in r["related"] if x["relation"] == "possibly_affected_by")
print(f"   {len(leads)} laws |", dict(why))

print("\n6. IN FORCE with a later law of the same name in the same jurisdiction also in force (older one probably replaced), in index:")
stems = defaultdict(list)
for r in index:
    if r["year"] and r["instrument_type"] in ("Act", "Ordinance") and r["status"] == "in_force" and not r["is_amendment"]:
        stems[(r["jurisdiction"], f.base_title(r["title"]).replace(" ordinance", "").replace(" act", ""))].append(r)
pairs = [sorted(m, key=lambda r: r["year"]) for m in stems.values() if len({r["year"] for r in m}) > 1]
print(f"   groups: {len(pairs)}")
for m in pairs[:45]:
    print("   - " + " | ".join(f"{r['source']} {r['title'][:44]}" for r in m))
