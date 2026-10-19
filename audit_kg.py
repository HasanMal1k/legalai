"""
audit_kg.py

The check behind "nothing served as in force has been repealed": every place in the whole corpus where
the name of a law in the current tier stands next to repealing words, set against what the knowledge
graph made of that place. Whatever the graph cannot account for is written out to be read:

    data/kg/audit_unexplained.jsonl     one row per (current law, document, passage)

A passage is accounted for when the graph has a relation from that document to that law (a partial or
territorial repeal, an amendment, a mention it settled), or when the document is the law itself or
another copy of it. Run after finalize_metadata.py:   python audit_kg.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ocr_scanned import SOURCES, source_layout

DATA = Path(__file__).parent / "data"
FILLER = {"the", "of", "and", "for", "act", "ordinance", "order", "rules", "regulation", "regulations", "code", "pakistan", "west",
          "punjab", "sindh", "khyber", "pakhtunkhwa", "balochistan", "province", "provincial", "federal", "islamabad", "capital",
          "territory", "with", "from", "under"}
REPEAL = re.compile(r"(?i)\b(?:is|are|stand|stands|shall\s+be|hereby)\s+(?:hereby\s+)?(?:stand\s+)?repealed\b|\brepeal(?:s|ed)?\s+(?:of|the)\b")


def words_of(norm: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", norm) if w not in FILLER]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    records, texts = [], {}
    for source in SOURCES:
        data_dir, key = source_layout(source)
        for line in (data_dir / "statutes.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            texts[(source, row[key])] = row["raw_text"]
        for line in (data_dir / "metadata_final.jsonl").open(encoding="utf-8"):
            record = json.loads(line)
            record["_key"] = record[key]
            records.append(record)
    current = [r for r in records if r["tier"] == "current"]

    # distinctive words -> current laws; a passage is a candidate for a law when it holds all its words and its year
    by_word = defaultdict(set)
    wanted = {}
    for n, record in enumerate(current):
        norm = re.sub(r"[^a-z0-9 ]+", " ", record["title"].lower())
        words = words_of(norm)
        if len(words) < 2 or not record["year"]:
            continue          # names of one distinctive word ("The Contract Act") match far too much to check this way
        wanted[n] = (set(words), str(record["year"]))
        for w in words:
            by_word[w].add(n)

    # what the graph already says about (document, law) pairs
    explained = set()
    for line in (DATA / "kg" / "edges.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e["type"] in ("repeals", "amends"):
            explained.add((e["from"]["source"], e["from"]["id"], e["to"]["work_id"]))
    for record in records:
        for x in record.get("related") or []:
            if x["relation"] in ("lead_settled", "partly_repealed_by", "repealed_in", "conditionally_repealed_by"):
                explained.add(("title", x["title"], x["source"], record["work_id"]))
    title_of = {(r["source"], r["_key"]): r for r in records}

    unexplained, counted = [], Counter()
    for (source, doc_id), raw in texts.items():
        doc = title_of.get((source, doc_id))
        if not doc:
            continue
        flat = re.sub(r"\s+", " ", raw)
        for hit in REPEAL.finditer(flat):
            window = flat[max(0, hit.start() - 350):hit.end() + 150]
            low = re.sub(r"[^a-z0-9 ]+", " ", window.lower())
            present = set(low.split())
            candidates = Counter()
            for w in present:
                for n in by_word.get(w, ()):
                    candidates[n] += 1
            for n, hits in candidates.items():
                words, year = wanted[n]
                if hits < len(words) or year not in low:
                    continue
                law = current[n]
                if law["work_id"] == doc.get("work_id") or law["family_id"] == doc.get("family_id"):
                    continue          # the law's own text, or another copy of it
                counted["passages"] += 1
                if (source, doc_id, law["work_id"]) in explained or ("title", doc["title"], source, law["work_id"]) in explained:
                    counted["accounted for by the graph"] += 1
                    continue
                unexplained.append({"law": law["title"], "law_source": law["source"], "in": doc["title"], "in_source": source,
                                    "in_year": doc["year"], "passage": window})
    # what the clause reader makes of each passage the graph has no relation for
    import kg
    from finalize_metadata import norm_title
    kinds = Counter()
    to_read = []
    kind_of = {(r["title"], r["source"]): r["instrument_type"] for r in records}
    reviewed_file = DATA / "kg" / "audit_reviewed.json"
    reviewed = {(x["law"], x["in"][:60]) for x in json.loads(reviewed_file.read_text(encoding="utf-8"))["reviewed"]} if reviewed_file.exists() else set()
    for row in unexplained:
        if (row["law"], row["in"][:60]) in reviewed:
            kinds["read by hand: not a repeal of it (audit_reviewed.json)"] += 1
            continue
        verdict = classify(row, kg, norm_title)
        if verdict == "the law is the subject of a whole repeal":
            verdict = benign(row, kind_of.get((row["law"], row["law_source"])), kg) or verdict
        row["verdict"] = verdict
        kinds[verdict] += 1
        if verdict == "the law is the subject of a whole repeal":
            to_read.append(row)
    out = DATA / "kg" / "audit_unexplained.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in to_read:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    laws = {(u["law"], u["law_source"]) for u in to_read}
    print(f"current laws checked: {len(wanted)} of {len(current)} (the rest have names of one distinctive word)")
    print(f"passages naming a current law beside repealing words: {counted['passages']}; accounted for by the graph: "
          f"{counted['accounted for by the graph']}; the rest, as the clause reader reads them: {dict(kinds)}")
    print(f"to read by hand: {len(to_read)} passages about {len(laws)} laws -> {out}")


PROVINCIAL = ("punjab", "sindh", "kp", "balochistan")


def benign(row: dict, kind: str | None, kg) -> str | None:
    """Two readings that are right not to touch the law: a different law of the same name (the Ordinance an Act
    replaced, the rules of the same name), and another province repealing its own copy."""
    if row["in_source"] in PROVINCIAL and row["law_source"] in PROVINCIAL and row["in_source"] != row["law_source"]:
        return "another province's repeal of its own copy"
    name_words = [w for w in re.findall(r"[A-Za-z]{4,}", row["law"]) if w.lower() not in FILLER]
    text = kg.clean(row["passage"])
    last = name_words[-1] if name_words else None
    if last and kind:
        for m in re.finditer(re.escape(last) + r"\)?\s*,?\s*(Act|Ordinance|Order|Rules|Regulations?|Statutes?)\b", text, re.I):
            printed = m.group(1).lower().rstrip("s")
            if printed[:5] != kind.lower()[:5]:
                return "a different law of the same name (its Ordinance, rules or statutes)"
    return None


def classify(row: dict, kg, norm_title) -> str:
    """What a passage says of the law: not its subject, a partial repeal, or a whole repeal the graph did not record."""
    text = kg.clean(row["passage"])
    wanted = norm_title(row["law"])
    year = re.search(r"\b(1[89]\d{2}|20\d{2})\b", row["law"])
    for verb in kg.VERB.finditer(text):
        start = kg.clause_start(text, verb.start())
        subject = text[start:verb.start()]
        flat = norm_title(subject)
        if kg.name_covered(wanted, flat) < 0.8 or (year and year.group(1) not in subject):
            continue
        # where in the subject the law is named: the words before it decide
        words = [w for w in re.findall(r"[A-Za-z]{4,}", row["law"]) if w.lower() not in FILLER]
        first = min((m.start() for w in words[:2] for m in re.finditer(re.escape(w), subject, re.I)), default=0)
        before = subject[:first]
        rest = verb.group(0) + text[verb.end():verb.end() + 250]
        if kg.NOT_THE_SUBJECT.search(before[-150:]):
            return "named, but not as what is repealed"
        if kg.PARTIAL_BEFORE.search(before[-220:] + " ") or kg.PARTIAL_BEFORE.search(re.sub(r"\s*(?:the\s*)?$", " the ", before[-220:])) \
                or kg.PARTIAL_AFTER.search(subject[first:]) or kg.LIMITS.search(subject[first:] + " " + rest[:200]):
            return "provisions of it, or a limited repeal"
        if re.search(r"amendment", row["in"], re.I) and re.search(r"\bin\s+the\s+$", before[-60:], re.I):
            return "provisions of it, or a limited repeal"
        return "the law is the subject of a whole repeal"
    return "not the subject of a repealing sentence"


if __name__ == "__main__":
    main()
