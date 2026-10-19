"""
dedupe.py

Works out which documents are the same law, and which one copy of each
to index. Called by finalize_metadata.py; it adds these fields to every
row of metadata_final.jsonl and deletes nothing:

    work_id         same value for every copy/version of one law
    is_primary      the one copy of that law to index
    version         consolidated | as_enacted | exact_copy | near_copy |
                    other_version | translation
    duplicate_of    {source, id, title} of the primary (non-primaries only)
    similarity_to_primary   0..1, how much of the text is shared
    siblings        the same law as carried by OTHER jurisdictions
    family_id       same value for a law and all its siblings
    authority_rank  1 (current official text) to 4 (repealed); 2 for archive originals and for
                    federal laws as a provincial code carries them
    index_default   whether to put it in the default search index
    index_exclusion why not, when index_default is false

Three relationships, treated differently:

  exact copy      identical text (the same file uploaded twice). Merged.
  version         the same law in another form: the consolidated text on
                  an official code versus the original as gazetted.
                  Grouped; the consolidated text is primary.
  sibling         the same law as carried by different provinces (a West
                  Pakistan law on both the Punjab and Sindh codes). NOT
                  merged -- each province has amended its copy separately --
                  only linked.

Two documents are the same law when title, year and instrument type
match (finalize_metadata.same_title) within the same scope, and either
their law numbers agree or their texts overlap. A title match with a
different number, or with unrelated text and no number to go on, is not
merged; it is written to data/dedup_review.jsonl for a person to look at.
"""

import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from ocr_scanned import looks_english

DATA_DIR = Path(__file__).parent / "data"

FEDERAL_SOURCES = {"pakistancode", "drs"}
ARCHIVE_DOC_TYPES = {"law (gazette copy)", "rule (gazette copy)"}
SAME_TEXT = 0.85     # shared share of text above which two copies are the same version
RELATED_TEXT = 0.30  # ... and above which they are clearly the same law in another version
CONTAINED = 0.70     # one text sits almost entirely inside the other (a gazette holding several laws)
SAME_NOTICE = 0.92   # stricter for notifications and rules, which share boilerplate
SAMPLE = 8           # long texts: keep one word-5-gram in eight, chosen by hash
SAMPLE_ABOVE = 3000  # ... once a text has more grams than this
MIN_TEXT = 400       # characters; anything shorter is a cover page or a placeholder

NUMBERED_LAWS = {"Act", "Ordinance", "Regulation"}  # instruments identified by "No. X of YEAR"

# Whole categories of a site left out of the index, by (source, the site's own category).
# Gazette Part III is "other notifications": about half are ministers' leave,
# appointments and portfolio changes. Dropped at the user's decision (2026-10-12);
# delete the entry to bring them back.
OUT_OF_SCOPE = {
    ("drs", "Notification (Part-III)"): "a Gazette Part III notification (mostly administrative): out of scope",
}

ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def law_number(value: str | None) -> tuple[int, int] | None:
    """("XX of 1975") -> (20, 1975); None when there is no usable number."""
    match = re.search(r"\b([ivxlcdm]+|\d+)\s+of\s+(\d{4})\b", (value or "").lower())
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        number = int(raw)
    else:
        number = 0
        for index, char in enumerate(raw):
            current = ROMAN[char]
            number += -current if index + 1 < len(raw) and ROMAN[raw[index + 1]] > current else current
    return (number, int(match.group(2))) if number else None


def fingerprint(text: str) -> tuple[int, frozenset[int]]:
    """(sampling rate, hashed word-5-grams). Short texts keep every gram, so that
    two one-page notifications differing by a name are not mistaken for copies;
    long texts keep one gram in SAMPLE."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = {zlib.crc32(" ".join(words[i:i + 5]).encode()) for i in range(max(0, len(words) - 4))}
    if len(grams) <= SAMPLE_ABOVE:
        return 1, frozenset(grams)
    return SAMPLE, frozenset(g for g in grams if g % SAMPLE == 0)


EMPTY = (1, frozenset())


def overlap(a: tuple[int, frozenset], b: tuple[int, frozenset]) -> tuple[float, float]:
    """(share of the combined text that is common, share of the shorter text found in the longer)."""
    rate = max(a[0], b[0])  # compare both at the coarser sampling rate
    grams_a = a[1] if a[0] == rate else {g for g in a[1] if g % rate == 0}
    grams_b = b[1] if b[0] == rate else {g for g in b[1] if g % rate == 0}
    if not grams_a or not grams_b:
        return 0.0, 0.0
    common = len(grams_a & grams_b)
    return common / len(grams_a | grams_b), common / min(len(grams_a), len(grams_b))


def scope_of(record: dict) -> str:
    """Whose statute book this copy belongs to. Copies are only merged within one scope."""
    if record["source"] not in FEDERAL_SOURCES:
        # A provincial code's copy is that province's version, even of a federal or
        # pre-1947 central Act: provinces amend those for themselves (Court-fees Act,
        # 1870). It joins the federal text only if the texts match -- see step 2c.
        return record["source"]
    return "federal" if record["jurisdiction"] == "Federal" else "federal-archive:" + record["jurisdiction"]


def is_archive(record: dict) -> bool:
    doc = record["_doc"]
    return record["source"] == "drs" or doc["catalog"].get("doc_type") in ARCHIVE_DOC_TYPES


def preference(record: dict) -> tuple:
    """Sort key: the best copy to index comes first."""
    text_rank = {"native": 0, "ocr": 1, "machine-translation": 2}.get(record["text_source"], 1)
    # a federal law's home is Pakistan Code, even when a provincial code also hosts it
    away_from_home = record["jurisdiction"] == "Federal" and record["source"] not in FEDERAL_SOURCES
    # a translation is never preferred over the law it translates
    return (record["text_source"] == "machine-translation", is_archive(record), away_from_home, text_rank, record["needs_review"], -len(record["_doc"]["text"]), str(record["_id"]))


def authority_rank(record: dict) -> int:
    """1 (best) to 4: how a hit from this copy should rank beside another copy of the same law.

    1  the current text on an official code, not known to be out of force
    2  the original as gazetted (an archive copy), not known to be out of force
    3  superseded, lapsed or merely amending
    4  repealed
    """
    if record["status"] == "repealed":
        return 4
    if record["status"] not in ("in_force", "unknown"):
        return 3
    if is_archive(record):
        return 2
    # a federal law as a provincial code carries it: such copies lag behind (the Punjab Code's Penal
    # Code still gives the old ages in sections 82 and 83), so the Pakistan Code's text comes first
    if record["source"] not in FEDERAL_SOURCES and record["jurisdiction"] == "Federal":
        return 2
    return 1


def unusable_text(record: dict) -> str | None:
    """Why this document's text is not worth indexing, or None when it is."""
    text = record["_doc"]["text"]
    if len(text.strip()) < MIN_TEXT:
        return "too little text (a cover page or placeholder)"
    if record["text_source"] != "machine-translation" and not looks_english(text):
        return "the text is not readable English"
    solid = [char for char in text if not char.isspace()]
    # a failed scan; long texts are left alone (a schedule of coordinates is mostly digits too)
    if len(text) < 20000 and sum(char.isascii() and char.isalpha() for char in solid) < 0.5 * len(solid):
        return "the text is mostly symbols (a failed scan)"
    return None


KIND_WORD = re.compile(r"\b(?:act|ordinance|order|rules|regulations?|code)\b")


def kindless_same(a: str, b: str) -> bool:
    """The same name, one side printed without its kind: the Qanun-e-Shahadat's own short title is "the
    Qanun-e-Shahadat, 1984"; the Pakistan Code calls it "Qanun-e-Shahadat Order, 1984". (Two names that both
    carry a kind are left to same_title, which keeps "X Ordinance" and "X Act" apart.)"""
    if bool(KIND_WORD.search(a)) == bool(KIND_WORD.search(b)):
        return False
    strip = lambda t: re.sub(r"\s+", " ", KIND_WORD.sub(" ", t)).strip()
    return strip(a) == strip(b)


def assign_works(records: list[dict], same_title, norm, title_year) -> dict:
    """Add the dedup fields to every record. Returns counts for the summary.

    same_title, norm and title_year are finalize_metadata's title helpers.
    """
    for index, record in enumerate(records):
        doc = record["_doc"]
        record["_i"] = index
        record["_id"] = doc["id"]
        words = re.sub(r"[^a-z0-9]+", " ", doc["text"].lower()).strip()
        record["_hash"] = hashlib.sha1(words.encode()).hexdigest() if len(words) >= 400 else None
        record["_number"] = law_number(record.get("number")) or law_number(doc["catalog"].get("act_or_ordinance_no"))
        record["_scope"] = scope_of(record)

    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    # 1. identical text within a scope is the same document, whatever it is called
    by_hash = defaultdict(list)
    for index, record in enumerate(records):
        if record["_hash"]:
            by_hash[(record["_scope"], record["_hash"])].append(index)
    for members in by_hash.values():
        for other in members[1:]:
            union(members[0], other)

    # 2. candidates: same scope, same year, titles that name the same law
    buckets = defaultdict(list)
    for index, record in enumerate(records):
        if record["year"]:
            buckets[(record["_scope"], record["year"], record["_norm"][:4])].append(index)
    candidate_pairs = []
    for members in buckets.values():
        for position, i in enumerate(members):
            for j in members[position + 1:]:
                a, b = records[i], records[j]
                if a["instrument_type"] == b["instrument_type"] and (same_title(a["_norm"], b["_norm"]) or kindless_same(a["_norm"], b["_norm"])):
                    candidate_pairs.append((i, j))

    needed = {i for pair in candidate_pairs for i in pair}
    prints = {i: fingerprint(records[i]["_doc"]["text"]) for i in needed}

    review = []
    for i, j in candidate_pairs:
        a, b = records[i], records[j]
        shared, contained = overlap(prints[i], prints[j])
        if a["instrument_type"] not in NUMBERED_LAWS:
            # Notifications, rules and orders reuse titles and file numbers freely
            # ("No. 2-1/2017-Min-I" heads many different notifications), and share
            # boilerplate. Only the same text makes them the same document.
            if shared >= SAME_NOTICE:
                union(i, j)
            elif shared >= 0.6:
                review.append((a, b, shared, "same title and similar text, but not a law with a number to confirm"))
            continue
        numbers_known = a["_number"] and b["_number"]
        if numbers_known and a["_number"] != b["_number"]:
            # Usually two different laws (two amending Acts in one year). But the same
            # text means one law: a misread number, or an Ordinance re-promulgated.
            amending = a["is_amendment"] or b["is_amendment"]
            if shared >= SAME_TEXT or (shared >= 0.5 and not amending):
                union(i, j)
            elif {a["source"], b["source"]} == {"pakistancode", "drs"} and (a["_norm"] == b["_norm"] or kindless_same(a["_norm"], b["_norm"])) \
                    and not amending:
                # the gazette original of a law the Pakistan Code carries under exactly the same name, its number
                # misread on one side ("XXVII of 1965" / "XXVIII of 1965"): no two laws of one year share a name
                # unless one amends the other
                union(i, j)
                for record in (a, b):
                    record["_number_conflict"] = True
            else:
                review.append((a, b, shared, "same title but different law numbers"))
            continue
        if shared >= RELATED_TEXT or contained >= CONTAINED or numbers_known:
            union(i, j)
        elif {a["source"], b["source"]} == {"pakistancode", "drs"} and (a["_norm"] == b["_norm"] or kindless_same(a["_norm"], b["_norm"])):
            # the gazette original of a law the Pakistan Code carries under exactly the same title:
            # often an old scan whose text no longer resembles the consolidated one
            union(i, j)
        else:
            review.append((a, b, shared, "same title, unrelated text, and no law number to confirm"))

    # 2b. a translation is linked to the law it translates by the website's name for it
    originals = defaultdict(list)
    for index, record in enumerate(records):
        if record["text_source"] != "machine-translation":
            originals[(record["_scope"], record["year"])].append(index)
    for index, record in enumerate(records):
        tagged = re.search(r"\[?\s*urdu version\s*\]?", record["name"], re.I)
        if record["text_source"] == "machine-translation" and tagged:
            wanted = norm(record["name"][:tagged.start()])
            for other in originals.get((record["_scope"], title_year(record["name"])), []):
                if same_title(wanted, records[other]["_norm"]):
                    record["_translation"] = True
                    union(index, other)
                    break

    # 2c. a federal law hosted unchanged on a provincial code is the same document
    federal = defaultdict(list)
    for index, record in enumerate(records):
        if record["_scope"] == "federal" and record["year"]:
            federal[(record["year"], record["instrument_type"])].append(index)
    for index, record in enumerate(records):
        if record["source"] in FEDERAL_SOURCES or record["jurisdiction"] != "Federal" or not record["year"]:
            continue
        for other in federal.get((record["year"], record["instrument_type"]), []):
            if same_title(record["_norm"], records[other]["_norm"]):
                for i in (index, other):
                    if i not in prints:
                        prints[i] = fingerprint(records[i]["_doc"]["text"])
                if overlap(prints[index], prints[other])[0] >= SAME_TEXT:
                    union(index, other)
                    break

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(index)

    # 3. pick the copy to index and label the rest
    for members in groups.values():
        ordered = sorted((records[i] for i in members), key=preference)
        primary = ordered[0]
        # includes the primary's own id: different documents can share a title and year
        key = f"{primary['_scope']}|{primary['_norm']}|{primary['year']}|{primary['source']}|{primary['_id']}"
        work_id = "w-" + hashlib.sha1(key.encode()).hexdigest()[:12]
        primary_print = prints.get(primary["_i"])
        for record in ordered:
            record["work_id"] = work_id
            record["work_copies"] = len(members)
            record["is_primary"] = record is primary
            if record is primary:
                record["version"] = "as_enacted" if is_archive(record) else "consolidated"
                record["duplicate_of"] = None
                record["similarity_to_primary"] = None
                continue
            shared, contained = overlap(primary_print or EMPTY, prints.get(record["_i"], EMPTY))
            if record["_hash"] and record["_hash"] == primary["_hash"]:
                version, shared = "exact_copy", 1.0
            elif record["text_source"] == "machine-translation":
                version = "translation"
            elif shared >= SAME_TEXT:
                version = "near_copy"
            elif is_archive(record) and not is_archive(primary):
                version = "as_enacted"
            else:
                version = "other_version"
            record["version"] = version
            record["duplicate_of"] = {"source": primary["source"], "id": primary["_id"], "title": primary["title"]}
            record["similarity_to_primary"] = round(shared, 2)

    # 4. the same law in other scopes: linked, never merged
    by_law = defaultdict(list)
    for record in records:
        if record["is_primary"] and record["year"]:
            by_law[(record["year"], record["instrument_type"], record["_norm"][:4])].append(record)
    for record in records:
        record["siblings"] = []
    for members in by_law.values():
        for position, a in enumerate(members):
            for b in members[position + 1:]:
                if a["_scope"] != b["_scope"] and (same_title(a["_norm"], b["_norm"]) or kindless_same(a["_norm"], b["_norm"])):
                    a["siblings"].append({"source": b["source"], "id": b["_id"], "jurisdiction": b["jurisdiction"]})
                    b["siblings"].append({"source": a["source"], "id": a["_id"], "jurisdiction": a["jurisdiction"]})

    # 4b. a family is one law in all the forms it is in force somewhere: the works
    # linked as siblings. Search results can be grouped by it; nothing is merged.
    family = {record["work_id"]: record["work_id"] for record in records}

    def root(work: str) -> str:
        while family[work] != work:
            family[work] = family[family[work]]
            work = family[work]
        return work

    by_id = {(record["source"], record["_id"]): record for record in records}
    for record in records:
        for sibling in record["siblings"]:
            a, b = root(record["work_id"]), root(by_id[(sibling["source"], sibling["id"])]["work_id"])
            family[max(a, b)] = min(a, b)
    for record in records:
        record["family_id"] = "f-" + root(record["work_id"])[2:]
        record["authority_rank"] = authority_rank(record)

    # 4c. copies that would sit in the index beside a better one and add nothing to it
    family_members = defaultdict(list)
    for record in records:
        if record["is_primary"]:
            family_members[record["family_id"]].append(record)
    for members in family_members.values():
        if len(members) < 2:
            continue
        on_codes = [r for r in members if r["source"] not in FEDERAL_SOURCES]
        clean_federal = [r for r in members if r["source"] == "pakistancode" and r.get("_garble", 0) < 0.05]
        for record in members:
            if record["source"] == "drs" and on_codes:
                mine = fingerprint(record["_doc"]["text"])
                if any(overlap(mine, fingerprint(other["_doc"]["text"]))[0] >= SAME_TEXT for other in on_codes):
                    record["_same_as_sibling"] = True
            if record["source"] not in FEDERAL_SOURCES and record["jurisdiction"] == "Federal" \
                    and record.get("_garble", 0) > 0.10 and clean_federal:
                record["_poor_federal_copy"] = True

    # 5. what goes in the default index
    for record in records:
        if not record["is_primary"]:
            reason = "another copy of the same text" if record["version"] in ("exact_copy", "near_copy") \
                else "a translation of an indexed law" if record["version"] == "translation" \
                else "an earlier or other version of an indexed law"
        elif (record["source"], record["_doc"]["catalog"].get("doc_type")) in OUT_OF_SCOPE:
            reason = OUT_OF_SCOPE[(record["source"], record["_doc"]["catalog"].get("doc_type"))]
        elif record["is_amendment"]:
            # by its title, whatever its status: an amending law that was itself
            # later repealed is still only an amending law
            reason = "an amending instrument: its content is carried in the law it amended"
        elif record.get("_same_as_sibling"):
            reason = "the same text as its copy on a provincial code, which is indexed"
        elif record.get("_poor_federal_copy"):
            reason = "a poor scan of a federal law whose clean Pakistan Code text is indexed"
        else:
            reason = unusable_text(record)
        record["index_default"] = reason is None
        record["index_exclusion"] = reason

    with (DATA_DIR / "dedup_review.jsonl").open("w", encoding="utf-8") as out:
        for a, b, shared, why in review:
            out.write(json.dumps({
                "why": why, "text_shared": round(shared, 2),
                "a": {"source": a["source"], "id": a["_id"], "title": a["title"], "number": a.get("number"), "chars": len(a["_doc"]["text"])},
                "b": {"source": b["source"], "id": b["_id"], "title": b["title"], "number": b.get("number"), "chars": len(b["_doc"]["text"])},
            }, ensure_ascii=False) + "\n")

    return {
        "works": len(groups),
        "works_with_several_copies": sum(len(m) > 1 for m in groups.values()),
        "non_primary_by_version": dict(Counter(r["version"] for r in records if not r["is_primary"]).most_common()),
        "primary_by_version": dict(Counter(r["version"] for r in records if r["is_primary"]).most_common()),
        "index_default": sum(r["index_default"] for r in records),
        "index_exclusion": dict(Counter(r["index_exclusion"] for r in records if r["index_exclusion"]).most_common()),
        "works_with_siblings": sum(bool(r["siblings"]) for r in records if r["is_primary"]),
        "families_of_several_works": sum(n > 1 for n in Counter(r["family_id"] for r in records if r["is_primary"]).values()),
        "index_by_authority_rank": dict(sorted(Counter(r["authority_rank"] for r in records if r["index_default"]).items())),
        "pairs_for_review": len(review),
    }
