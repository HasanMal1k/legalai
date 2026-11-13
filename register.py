"""
register.py

The register of Pakistan's laws in force, law by law, built on the governments' own word -- and the knowledge
graph over it. Read-only on the corpus; writes data/register/.

    python crawl_status.py drs pakistancode drs-details     # the official lists and DRS relations, first
    python register.py

One entry per law per jurisdiction: a federal law once, however many codes print a copy of it; a provincial law
once for its province; a West Pakistan law once for each province whose code carries its version (and once for
Islamabad, whose version the Pakistan Code carries). Amending laws are not entries: their changes are part of the
laws they amend (they are edges of the graph); nor are administrative Part III notifications.

Evidence, official first:
  Pakistan Code     the Ministry of Law & Justice's alphabetical list of federal laws (pakistancode_list.jsonl)
  DRS               the Ministry's Document Retrieval System: every federal document filed as Existing, Lapsed,
                    Not Valid, Repealed or Unlawful (drs_status.jsonl); its pages list amendments, previous laws,
                    and the Act a rule or notification was made under (drs_details.jsonl)
  provincial codes  the Punjab, Khyber Pakhtunkhwa and Balochistan Codes' listings of current law (as metadata_final
                    records them); the Sindh Code lists laws but never says which are current
  the corpus        repeals in another law's own words, Ordinances lapsed or enacted (finalize_metadata.py, kg.py)

How the official word was measured against what the corpus had proven (2026-10-20): of federal laws proven current,
95% are "Existing" on DRS; of laws proven repealed, 27% are still "Existing" there -- DRS is slow to mark repeals --
and "Not Valid" is given to laws plainly in force (the General Clauses Act, 1897), so it is not read as a status.
Hence the levels:

  in force            an official code lists it as current, nothing repeals it, and no official source says otherwise
  in force (DRS)      the Ministry's DRS files it as Existing, it is not on the Pakistan Code's list, nothing in the
                      corpus repeals it, and it is not an Ordinance old enough to have lapsed
  not confirmed       Sindh laws nothing repeals (the Sindh Code does not say what is current)
  disputed            official sources disagree with each other or with a repeal found in a law's own words
  not established     no source says either way
  not in force        repealed, superseded, lapsed or struck down, on official word or a repealing law's own words
"""

import bisect
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from ocr_scanned import SOURCES, source_layout

OUT = Path(__file__).parent / "data" / "register"
TODAY = date(2026, 10, 20)
PROVINCE = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
CODE_NAME = {"pakistancode": "the Pakistan Code", "punjab": "the Punjab Code", "kp": "the Khyber Pakhtunkhwa Code",
             "balochistan": "the Balochistan Code", "sindh": "the Sindh Code", "drs": "the DRS archive"}
OUT_OF_SCOPE = ("a Gazette Part III notification", "the text is not readable", "the text is mostly symbols",
                "too little text")
PERMANENT_ORDINANCE_YEARS = set(range(1977, 1986)) | set(range(1999, 2004))     # Articles 270A, 270AA
NOT_IN_FORCE = {"repealed": "repealed", "superseded": "replaced by an Act", "lapsed_presumed": "an Ordinance presumed lapsed"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def owner(record: dict) -> str:
    """The jurisdiction a law belongs to: a West Pakistan law to the province whose code carries its version."""
    jurisdiction = record.get("jurisdiction") or "Federal"
    if jurisdiction == "West Pakistan":
        return PROVINCE.get(record["_source"]) or ("Islamabad Capital Territory" if record["_source"] == "pakistancode" else "West Pakistan")
    return jurisdiction


def drs_id(record: dict) -> str | None:
    key = str(record.get("id") or "")
    return key[4:] if record["_source"] == "drs" and key.startswith("drs-") else None


def lapsed_by_age(record: dict) -> bool:
    """An Ordinance old enough to have lapsed (120 days, extendable once) unless enacted or saved by the Constitution."""
    year = int(record.get("year") or 0) if str(record.get("year") or "").isdigit() else 0
    return record.get("instrument_type") == "Ordinance" and year >= 1986 and year not in PERMANENT_ORDINANCE_YEARS \
        and (TODAY.year - year) * 365 > 240 and record.get("jurisdiction") != "West Pakistan"


def listed_on_code(record: dict) -> bool:
    basis = (record.get("status_basis") or [""])[0]
    return basis.startswith(("listed as current law on the", "kept up to date on the Pakistan Code"))


# annual and amending money laws: their changes are carried in the laws they change, like any amending law
# a law whose whole work is to repeal or dissolve others: spent once in force, however a code lists it
REPEALING = re.compile(r"\((?:[^)]*\s)?repeal\)|\brepeal\s+(?:act|ordinance)\b|\(dissolution\)", re.I)
# (the word next to "Act": "The Finance (Supplementary) Act", not "The House Building Finance Corporation Act" or "The Sindh
# Public Finance Administration Act"; a validation law stands on its own, nothing carries it)
SPENT_KIND = re.compile(r"\b(?:finance|appropriation|tax\s+laws|removal\s+of\s+difficult\w*)\)?(?:\s*\([^)]*\))*(?:\s+supplementary)?,?\s+"
                        r"(?:act|ordinance)\b", re.I)
REPEAL_NEAR = re.compile(r"repeal|omitted|shall cease|ceased to have effect")
# the Pakistan Code's list keeps some repealed laws, saying so in the name: "Companies Ordinance, 1984 (Repealed by
# Act XIX of 2017)" -- listed to say the law is gone; a note with a term ("valid 25-03-10", "when notified") settles nothing
REPEAL_NOTE = re.compile(r"\(\s*(?:repealed|repeal|rep\.)\s+by\b", re.I)
CONDITIONAL_NOTE = re.compile(r"\bvalid\b|\bup\s*to\b|when notified", re.I)


def law_key(title: str, year=None) -> tuple[str, str]:
    """A law's name for matching citations: words only, no "the", no year."""
    name = re.sub(r"[,\s]*\(?(?:1[89]|20)\d{2}\)?\s*$", "", str(title or ""))
    name = re.sub(r"[^a-z0-9]+", " ", name.lower())
    name = re.sub(r"^\s*the\s+", " ", name)
    return re.sub(r"\s+", " ", name).strip(), str(year or "")


class Citations:
    """The texts of the current laws of 2015 or later, as lowercase words: which of them name a law (its name and
    year side by side, "insurance ordinance 2000") outside any repealing sentence -- evidence it was alive when
    they were made."""

    def __init__(self, records: list[dict]):
        # (laws, not compendia: the Esta-Code reprints old rules, it does not rely on them)
        citing = {(r["_source"], r["_key"]): r for r in records if r.get("tier") == "current" and r.get("index_default")
                  and not re.search(r"\besta[- ]?code\b|compendium|compilation", r.get("title") or "", re.I)
                  and str(r.get("year") or "0").isdigit() and int(r.get("year") or 0) >= 2015}
        parts, self.starts, self.laws, at = [], [], [], 0
        for source in SOURCES:
            data_dir, key = source_layout(source)
            for row in read(data_dir / "statutes.jsonl"):
                law = citing.get((source, row[key]))
                if law:
                    words = " " + re.sub(r"[^a-z0-9]+", " ", row["raw_text"].lower()) + " "
                    parts.append(words)
                    self.starts.append(at)
                    self.laws.append(law)
                    at += len(words)
        self.blob = "".join(parts)

    def of(self, title: str, year, family: str) -> list[tuple[int, str]]:
        name, year = law_key(title, year)
        if len(name.split()) < 2 or not year:
            return []
        needle, out, at = f" {name} {year} ", {}, 0
        while (at := self.blob.find(needle, at)) >= 0:
            law = self.laws[bisect.bisect_right(self.starts, at) - 1]
            if law["family_id"] != family and not REPEAL_NEAR.search(self.blob[max(0, at - 300):at + len(needle) + 200]):
                out[law["work_id"]] = (int(law["year"]), law["title"])
            at += len(needle)
        return sorted(out.values(), reverse=True)


RECENT_PRINCIPAL = 2018     # a principal Act or Order this recent, which nothing in a corpus to 2026 repeals
AMENDED_SINCE = 2010        # an amendment this recent, filed as Existing on DRS: the law is being kept up
CITED_SINCE = 2015          # a current law this recent names it outside any repealing sentence
LEVELS = ["in force", "in force (in part)", "in force (evidence)", "not confirmed", "probably in force", "to be confirmed", "disputed", "not established",
          "probably not in force", "continued in the provinces", "not in force"]


def drs_year(text: str | None) -> int:
    m = re.search(r"(1[89]\d\d|20\d\d)", text or "")
    return int(m.group(1)) if m else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    code_list = read(OUT / "pakistancode_list.jsonl")
    pc_repealed = {r["detail_url"]: r["name"] for r in code_list if REPEAL_NOTE.search(r["name"])}
    pc_listed = {r["detail_url"] for r in code_list} - set(pc_repealed)
    drs_status = {r["doc_id"]: r for r in read(OUT / "drs_status.jsonl")}
    details = {r["doc_id"]: r for r in read(OUT / "drs_details.jsonl")}
    records = []
    for source in SOURCES:
        data_dir, key = source_layout(source)
        for row in read(data_dir / "metadata_final.jsonl"):
            row["_source"], row["_key"] = source, row[key]
            records.append(row)
    in_corpus_drs = {drs_id(r) for r in records if drs_id(r)}
    citations = Citations(records)
    by_stem = defaultdict(list)          # DRS's instruments by name without year: a later one of the same name replaces
    for doc_id, doc in drs_status.items():
        if str(doc.get("year") or "").isdigit():
            by_stem[law_key(doc["name"])[0]].append((int(doc["year"]), doc_id))

    def later_versions(title: str, year: int) -> list[str]:
        name = law_key(title)[0]
        if len(name.split()) < 3 or not year:
            return []
        return [d for y, d in sorted(by_stem.get(name, []), reverse=True) if y > year]
    print(f"citations: {len(citations.laws)} current laws of {CITED_SINCE} or later read", flush=True)

    # ---- entries: the records of one law in one jurisdiction
    entries = defaultdict(list)
    for record in records:
        if (record.get("index_exclusion") or "").startswith(OUT_OF_SCOPE):
            continue
        entries[(record["family_id"], owner(record))].append(record)
    for doc_id, doc in drs_status.items():          # DRS documents the corpus never fetched: the official word alone
        if doc_id not in in_corpus_drs:
            entries[(f"drs-{doc_id}", "Federal")].append({"_source": "drs-only", "_key": f"drs-{doc_id}", "title": doc["name"],
                                                           "year": doc["year"], "instrument_type": doc["doc_type"], "id": f"drs-{doc_id}",
                                                           "jurisdiction": "Federal", "family_id": f"drs-{doc_id}", "work_id": f"drs-{doc_id}"})

    register, amending_nodes = [], []
    for (family, where), members in entries.items():
        real = [r for r in members if r["_source"] != "drs-only"]
        main_records = [r for r in members if r.get("index_default")] or members
        main = min(main_records, key=lambda r: (r.get("authority_rank") or 9, r["_source"] != "pakistancode"))
        drs_docs = [drs_id(r) for r in members if drs_id(r)] + [r["id"][4:] for r in members if r["_source"] == "drs-only"]
        detail_rows = [details[d] for d in drs_docs if d in details]
        title = main.get("title") or ""
        # (a Finance Act a code lists as current law stays a law: provincial ones levy taxes that go on)
        listed_somewhere = any((r["_source"] == "pakistancode" and r["_key"] in pc_listed) or
                               (r["_source"] in ("punjab", "kp", "balochistan") and listed_on_code(r)) for r in members)
        amending = bool(real) and all(r.get("status") == "amending_instrument" or r.get("is_amendment") for r in real) \
            or any(d.get("document_type") == "Amendment" for d in detail_rows) or (bool(SPENT_KIND.search(title)) and not listed_somewhere) \
            or bool(REPEALING.search(title))
        node = {"register_id": f"{family}|{where}", "jurisdiction": where, "title": title, "year": main.get("year"),
                "kind": main.get("instrument_type"), "work_ids": sorted({r.get("work_id") for r in members if r.get("work_id")}),
                "drs_docs": drs_docs}
        if amending:
            amending_nodes.append(dict(node, level="amending or annual law"))
            continue
        drs_says = sorted({drs_status[d]["status"] for d in drs_docs if d in drs_status and drs_status[d]["status"]})
        on_pc_list = any(r["_source"] == "pakistancode" and r["_key"] in pc_listed for r in members)
        on_code = [r for r in members if r["_source"] in ("punjab", "kp", "balochistan") and listed_on_code(r)]
        status, proof = main.get("status"), (main.get("confidence") or {}).get("status")
        noted = [pc_repealed[r["_key"]] for r in members if r["_source"] == "pakistancode" and r["_key"] in pc_repealed]
        evidence = []
        if on_pc_list:
            evidence.append("listed on the Pakistan Code")
        elif noted:
            evidence.append(f"the Pakistan Code's list names it \"{noted[0]}\"")
        elif any(r["_source"] == "pakistancode" for r in members):
            evidence.append("on the Pakistan Code, found only through its search (not in its list)")
        evidence += [f"listed as current on {CODE_NAME[r['_source']]}" for r in on_code]
        if drs_says:
            evidence.append("DRS: " + " / ".join(drs_says))
        basis = (main.get("status_basis") or [""])[0]
        if basis and not basis.startswith("listed as current"):
            evidence.append(basis[:220])
        # a federal law's authority is the Pakistan Code: a provincial code's listing of its copy speaks for it only
        # where the Pakistan Code does not carry the law at all (as confidence.py rules)
        on_pc = any(r["_source"] == "pakistancode" for r in members)
        official_current = on_pc_list or (bool(on_code) and not (where == "Federal" and on_pc))
        drs_existing, drs_ended = "Existing" in drs_says, [s for s in drs_says if s in ("Repealed", "Lapsed", "Unlawful")]
        named_in = [x for x in main.get("related") or [] if x["relation"] in ("possibly_affected_by", "possibly_repealed_by", "repealed_federally")]
        leads = bool(named_in)
        if named_in:
            evidence.append(f"named in a repeal the corpus could not confirm: {named_in[0]['title']}")
        # the official sources that call it current, each on its own: the Pakistan Code's list, a provincial code, DRS
        vouching = int(on_pc_list) + len({r["_source"] for r in on_code}) + int("Existing" in drs_says)

        # positive evidence beyond a listing: amended lately, relied on lately, or a recent principal law
        amended = [a for d in detail_rows for a in d.get("amendments", [])
                   if a.get("Status") == "Existing" and drs_year(a.get("Commencement Date") or a.get("Publishing Date")) >= AMENDED_SINCE]
        cited = citations.of(title, main.get("year"), family) if not official_current else []
        year = int(main["year"]) if str(main.get("year") or "").isdigit() else 0
        recent = year >= RECENT_PRINCIPAL and main.get("instrument_type") in ("Act", "Order", "President Order")
        support = []
        if amended:
            support.append(f"amended as late as {max(drs_year(a.get('Commencement Date') or a.get('Publishing Date')) for a in amended)} "
                           f"({amended[-1].get('Document Title')}, filed as Existing)")
        if cited:
            support.append(f"named by {len(cited)} current law(s) of {CITED_SINCE} or later, e.g. {cited[0][1]} ({cited[0][0]})")
        if recent:
            support.append(f"a principal law of {year}; nothing in the corpus (to 2026) repeals it")

        if where == "West Pakistan":
            level = "continued in the provinces"
            evidence.append("a law of the old province of West Pakistan: it continues as each province's own version")
        elif noted and not on_pc_list and not CONDITIONAL_NOTE.search(noted[0]):
            level = "not in force"
        elif status in NOT_IN_FORCE and proof in ("verified", "probable") and main["_source"] != "drs-only":
            level = "not in force"
            if official_current or drs_existing:
                evidence.append("(an official list still carries it: " + ("the code lists it" if official_current else "DRS: Existing") + ")")
        elif main.get("tier") != "current" and official_current and not on_pc_list and where == "Federal" \
                and not re.search(r"^(?:Act|Ordinance|Order|Regulation)$", str(main.get("instrument_type"))):
            # a federal rule a provincial code lists: the Pakistan Code never hosts rules, so the provincial word stands alone
            level = "probably in force"
            evidence.append("a federal rule or notification, listed as current only by a provincial code")
        elif main.get("tier") != "current" and official_current and leads and vouching < 2:
            # one official list against a repealing law that names it (the Securities Act, 2015 and the Listed Companies
            # (Substantial Acquisition ...) Ordinance, 2002, which the Pakistan Code still lists): not settled
            level = "disputed"
        elif main.get("tier") == "current" or (official_current and status == "in_force"):
            level = "in force"
            if drs_ended and on_pc_list:
                # DRS files it as ended where the Pakistan Code lists it: DRS is wrong for some (the Trade Marks
                # Ordinance, 2001, amended in 2023) and records territorial repeals (the Child Marriage Restraint Act,
                # 1929, repealed for Islamabad only): explained by a territorial repeal found in a law's words, the
                # code's listing stands; otherwise the law is disputed until checked
                territorial = [x for x in main.get("status_exceptions") or []]
                if territorial:
                    evidence.append("DRS: " + "/".join(drs_ended) + " -- for part of the country only: repealed for the " +
                                    "; ".join(f"{x['territory']} by {x['by']}" for x in territorial))
                else:
                    level = "disputed"
                    evidence.append(f"the Pakistan Code lists it but DRS files it as {'/'.join(drs_ended)}: to be checked")
        elif where == "Sindh" and main.get("tier") == "unconfirmed":
            level = "not confirmed"
        elif drs_ended and not official_current:
            level = "not in force"
        elif drs_existing and not official_current:
            later = [d for d in later_versions(title, year) if d not in drs_docs]
            if leads:
                # named in a repeal the corpus could not confirm, not carried by the Pakistan Code: the official
                # compilation leaving it out and a repealing law naming it point the same way
                level = "probably not in force" if not on_pc else "disputed"
                evidence.append("named in a repeal that could not be confirmed" + ("; the Pakistan Code does not carry it" if not on_pc else ""))
            elif later and not amended:
                level = "not established"
                newest = drs_status[later[0]]
                evidence.append(f"a later instrument of the same name is on DRS ({newest['name']}): this one is probably replaced")
            elif any(lapsed_by_age(r) for r in main_records) and not amended:
                level = "not established"
                evidence.append("an Ordinance of a year whose Ordinances lapse unless enacted: DRS's 'Existing' is not taken for it")
            elif support:
                level = "in force (evidence)"
                evidence += support
            else:
                level = "to be confirmed"
                evidence.append("DRS files it as Existing, but nothing else vouches for it: DRS is slow to mark repeals and lapses")
        else:
            level = "not established"
        # a repeal the Pakistan Code notes for a term or on a condition ("w.e.f 17-09-2007 upto 25-03-2010", "when notified"):
        # the corpus's own verdict stands, but nothing lifts the law above "to be confirmed"; and a document filed as a
        # draft is the text of a draft, whatever is in force
        if noted and not on_pc_list and CONDITIONAL_NOTE.search(noted[0]):
            evidence.append("the repeal the Pakistan Code notes is for a term or on a condition: to be checked")
            if level in ("in force", "in force (evidence)", "probably in force"):
                level = "to be confirmed"
        if re.match(r"\W*draft\b", title, re.I) and level in ("in force", "in force (evidence)"):
            level = "probably in force"
            evidence.append("the document is filed as a draft: the instrument may be in force, this text may not be its final one")
        # rules a code still lists, made under an Act since repealed with no savings clause found that keeps its rules
        # (kg.py R5: the Punjab Manufactured Drugs Rules, 1932 under the Dangerous Drugs Act, 1930): the listing may be stale
        orphaned = [x for x in main.get("related") or [] if x["relation"] == "parent_repealed"]
        if orphaned and level in ("in force", "in force (evidence)", "probably in force"):
            level = "disputed"
            evidence.append(f"made under {orphaned[0]['title']}, since repealed, and no savings clause keeping its rules was found")
        served = [r for r in members if r.get("tier") in ("current", "unconfirmed")]
        text = ("served" if served else "historical text only" if any(r.get("tier") == "historical" for r in members)
                else "quarantined" if any(r.get("tier") == "quarantine" for r in members)
                else "not indexed" if real else "not in the corpus")
        drs_detail = detail_rows[0] if detail_rows else {}
        links = sorted({r["_key"] for r in members if r["_source"] == "pakistancode" and isinstance(r.get("_key"), str)} |
                       {drs_status[d]["detail_url"] for d in drs_docs if d in drs_status})[:4]
        register.append(dict(node, number=main.get("number") or drs_detail.get("number"), level=level, evidence=evidence,
                             text_in_app=text, tier=main.get("tier"), quarantine_reasons=main.get("quarantine_reasons") or [],
                             amendments=[a.get("Document Title") for a in drs_detail.get("amendments", [])],
                             previous_laws=[(p.get("Title"), p.get("Status")) for p in drs_detail.get("previous", [])],
                             made_under=[p.get("Title") or p.get("Document Title") for p in drs_detail.get("primary", [])],
                             last_amended=main.get("last_amended"), official_links=links))

    # ---- one federal law filed twice under two names (DRS's "The National Accountability Bureau Ordinance, 1999" and
    # the Pakistan Code's "The National Accountability Ordinance, 1999"; a "Bill, 2025" for its Act): same kind, same
    # year, and one name's words inside the other's, but for two words at most -> one entry, the official one
    decoration = {"of", "and", "the", "for", "na", "act", "bill", "royal", "indian"}     # ("The Indian Succession Act": Pakistan dropped the word)
    # a word that makes another law: the Deputy Chairman's salaries are not the Chairman's, Regulation No. 41 not No. 69
    qualifiers = {"deputy", "chief", "second", "third", "additional", "supplementary", "special", "amendment", "amending",
                  "continuance", "revival", "trials", "west", "east", "federal", "provincial", "interim", "temporary",
                  "transitional", "punjab", "sindh", "khyber", "pakhtunkhwa", "balochistan", "baluchistan", "islamabad",
                  "repeal", "validation", "administrator", "taking", "over", "no", "part", "procedure", "general"}

    def words(title: str) -> frozenset:
        joined = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", str(title))       # "Take-overs" = "Takeovers"
        return frozenset(w for w in law_key(joined)[0].split() if w not in decoration)

    def same_law(a: dict, b: dict) -> bool:
        printed = [re.search(r"((?:1[89]|20)\d{2})\W*$", str(x["title"])) for x in (a, b)]
        if all(printed) and printed[0].group(1) != printed[1].group(1):       # the Presidential Cabinet Orders of 1971 and 1972
            return False
        mine, theirs = words(a["title"]), words(b["title"])
        # the same words, however few (same kind, same year): "The General Clauses Act, 1897" filed twice on DRS, "The
        # Indian Contract Act, 1872" for "The Contract Act, 1872"
        if mine and mine == theirs:
            return True
        if len(mine) < 3 or len(theirs) < 3 or {w for w in mine if w.isdigit()} != {w for w in theirs if w.isdigit()}:
            return False
        if mine == theirs:
            return True
        # one word more, between an official entry and DRS's copy ("National Accountability [Bureau] Ordinance")
        extra = mine ^ theirs
        return len(extra) == 1 and not (extra & qualifiers) and bool(a["official_links"]) != bool(b["official_links"]) \
            and (mine <= theirs or theirs <= mine)

    # which of two names for one law speaks for it: official word first, then the strongest evidence
    strength = {"in force": 0, "not in force": 0, "disputed": 1, "in force (evidence)": 2, "probably in force": 3,
                "to be confirmed": 4, "not established": 5, "probably not in force": 6}
    groups = defaultdict(list)
    for e in register:
        if e["jurisdiction"] == "Federal" and e["level"] in strength:
            # (by the year the name prints, where it prints one: DRS files "The Protection of Communal Properties of
            # Minorities Ordinance, 2001" under 2002)
            printed = re.search(r"((?:1[89]|20)\d{2})\W*$", str(e["title"]))
            groups[(str(e["kind"]), printed.group(1) if printed else str(e["year"]))].append(e)
    merged, gone = 0, set()
    for members in groups.values():
        # (the Pakistan Code's entry before a DRS filing, a law the corpus holds before one it never fetched)
        members.sort(key=lambda e: (strength[e["level"]], not any("pakistancode.gov.pk" in x for x in e["official_links"]),
                                    e["text_in_app"] == "not in the corpus", not e["official_links"]))
        for n, e in enumerate(members):
            if id(e) in gone:
                continue
            for other in members[n + 1:]:
                if id(other) in gone or not same_law(e, other):
                    continue
                e["work_ids"] = sorted(set(e["work_ids"]) | set(other["work_ids"]))
                e["drs_docs"] = sorted(set(e["drs_docs"]) | set(other["drs_docs"]))
                e["evidence"].append(f"also filed as \"{other['title']}\" ({other['level']}: "
                                     f"{'; '.join(x for x in other['evidence'] if x.startswith(('DRS:', 'on the Pakistan Code'))) or 'no status'})")
                gone.add(id(other))
                merged += 1
    register = [e for e in register if id(e) not in gone]
    # a federal Act or Ordinance older than RECENT_PRINCIPAL that only DRS files, and that the Pakistan Code -- the
    # Ministry's compilation of federal law, its list and its search both read -- does not carry: spent, repealed or
    # never federal, most likely; not taken for law in force
    for e in register:
        if e["jurisdiction"] == "Federal" and e["level"] == "to be confirmed" and e["kind"] in ("Act", "Ordinance") \
                and not any("Pakistan Code" in x for x in e["evidence"]):
            e["level"] = "probably not in force"
            e["evidence"].append("the Pakistan Code does not carry it, in its list or its search: most likely spent or repealed")
    print(f"one law under two names: {merged} entries folded into the official one", flush=True)

    # ---- the knowledge graph: the register's laws and the amending laws, joined by what the texts and DRS say
    nodes = register + amending_nodes
    entry_of_work, entry_of_drs = {}, {}
    for e in nodes:
        for w in e["work_ids"]:
            entry_of_work.setdefault(w, e["register_id"])
        for d in e["drs_docs"]:
            entry_of_drs.setdefault(d, e["register_id"])
    entry_of_id = {}
    for r in records:
        e = entry_of_work.get(r.get("work_id"))
        if e:
            entry_of_id[str(r["_key"])] = e
    drs_by_name = defaultdict(list)
    for doc_id, doc in drs_status.items():
        drs_by_name[law_key(doc["name"], doc["year"])].append(doc_id)
    edges = Counter()
    for e in read(Path(__file__).parent / "data" / "kg" / "edges.jsonl"):
        a = entry_of_work.get(e["from"].get("work_id"))
        to = e.get("to") or {}
        b = entry_of_work.get(to.get("work_id")) or entry_of_id.get(str(to.get("id")))
        if e["type"] == "copy_of" or not a or not b or a == b:
            continue
        kind = e["type"] if e["type"] != "repeals" else f"repeals ({e.get('extent')})"
        edges[(a, kind, b, "the law's own words")] += 1
    for doc_id, d in details.items():
        here = entry_of_drs.get(doc_id)
        if not here:
            continue
        for row in d.get("amendments", []):
            for other in drs_by_name.get(law_key(row.get("Document Title"), drs_year(row.get("Document Title")) or ""), []):
                if entry_of_drs.get(other) and entry_of_drs[other] != here:
                    edges[(entry_of_drs[other], "amends", here, "DRS")] += 1
        for row in d.get("previous", []):
            if entry_of_drs.get(row.get("doc_id")) and entry_of_drs[row["doc_id"]] != here:
                edges[(here, "replaces", entry_of_drs[row["doc_id"]], "DRS")] += 1
        for row in d.get("primary", []):
            if entry_of_drs.get(row.get("doc_id")) and entry_of_drs[row["doc_id"]] != here:
                edges[(here, "made_under", entry_of_drs[row["doc_id"]], "DRS")] += 1

    # ---- what was settled by hand: on the web (data/register/web_checks.json: the verdict, a note, the sources), and
    # by reading the repealing law's own words (text_checks.json: the verdict, the clause, the law it was read in);
    # a check naming a jurisdiction settles that jurisdiction's law only
    for file, said in (("web_checks.json", "checked on the web (2026-10-20)"), ("text_checks.json", "read in the repealing law (2026-10-20)")):
        checks = json.loads((OUT / file).read_text(encoding="utf-8")) if (OUT / file).exists() else []
        for check in checks:
            name, settled = law_key(check["title"])[0], 0
            for e in register:
                if law_key(e["title"])[0] == name and (not re.search(r"\d{4}", check["title"]) or str(e["year"]) in check["title"]) \
                        and check.get("jurisdiction", e["jurisdiction"]) == e["jurisdiction"]:
                    e["level"], settled = check["level"], settled + 1
                    sources = check.get("sources") or ([check["read_in"]] if check.get("read_in") else [])
                    e["evidence"].append(f"{said}: {check['note']}" + (f" -- {', '.join(sources)}" if sources else ""))
            if not settled:
                print(f"  {file}: no entry matches {check['title']!r}", flush=True)

    # ---- rules, regulations and notifications DRS files as Existing, made under a law that is in force, with nothing
    # of the same name after them: probably in force (they fall with their Act, and are mostly replaced by name)
    level_of = {e["register_id"]: e["level"] for e in register}
    parents = defaultdict(set)
    for (a, kind, b, said) in edges:
        if kind == "made_under":
            parents[a].add(b)
    for e in register:
        # (rules and regulations only: a notification -- a price fixed, an exemption, an appointment -- is an act done,
        # mostly replaced under another name, not a standing law)
        if e["level"] == "to be confirmed" and re.search(r"rule|regulation", str(e["kind"]), re.I):
            alive = [p for p in parents.get(e["register_id"], ()) if level_of.get(p) in ("in force", "in force (evidence)")]
            if alive:
                e["level"] = "probably in force"
                e["evidence"].append(f"made under a law in force ({alive[0].split('|')[0]} -- see the graph), and no later instrument of its name")

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "register.jsonl").open("w", encoding="utf-8") as out:
        for e in register:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (OUT / "graph_nodes.jsonl").open("w", encoding="utf-8") as out:
        for e in nodes:
            out.write(json.dumps({k: e[k] for k in ("register_id", "jurisdiction", "title", "year", "kind", "level")}, ensure_ascii=False) + "\n")
    with (OUT / "graph_edges.jsonl").open("w", encoding="utf-8") as out:
        for (a, kind, b, said), n in edges.items():
            out.write(json.dumps({"from": a, "type": kind, "to": b, "evidence": said}, ensure_ascii=False) + "\n")
    order = {level: n for n, level in enumerate(LEVELS)}
    fields = ["jurisdiction", "kind", "title", "year", "number", "level", "text_in_app", "evidence", "amendments", "official_links"]
    for name, keep in (("in_force.csv", ("in force", "in force (in part)", "in force (evidence)", "not confirmed")), ("probably_in_force.csv", ("probably in force",)), ("to_be_confirmed.csv", ("to be confirmed",)),
                       ("disputed.csv", ("disputed",)), ("not_established.csv", ("not established",))):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for e in sorted((e for e in register if e["level"] in keep), key=lambda e: (e["jurisdiction"], order[e["level"]], str(e["title"]).lower())):
                writer.writerow({k: (" | ".join(map(str, e[k])) if isinstance(e[k], list) else e[k]) for k in fields})
    summary = {
        "built": TODAY.isoformat(), "laws": len(register), "amending or annual laws (graph nodes, not on the list)": len(amending_nodes),
        "by level": {level: n for level in LEVELS if (n := sum(e["level"] == level for e in register))},
        "by jurisdiction and level": {f"{j} | {l}": n for (j, l), n in sorted(Counter((e["jurisdiction"], e["level"]) for e in register).items())},
        "in force, by text in the app": dict(Counter(e["text_in_app"] for e in register if e["level"].startswith("in force"))),
        "in force, by kind": dict(Counter(e["kind"] for e in register if e["level"].startswith("in force")).most_common()),
        "graph": {"nodes": len(nodes), "edges by kind": dict(Counter(f"{k} [{s}]" for (_, k, _, s) in edges).most_common())},
        "drs pages read": len(details),
    }
    (OUT / "summary.json").write_text(json.dumps({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in summary.items()},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
