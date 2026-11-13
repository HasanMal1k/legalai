"""
finalize_metadata.py

Stage 7: combine everything known about each document into one final
metadata record, and work out whether the law is still in force.

    data/<source>/metadata_final.jsonl   one row per document
    data/metadata_summary.json           counts, for a quick look

Inputs per source: catalog.jsonl (what the website says), statutes.jsonl,
metadata.jsonl (first LLM pass) and metadata_v2.jsonl (second, deeper
pass). Documents the second pass hasn't reached yet fall back to the
first pass, so this can be run at any time and rerun later. No LLM, no
network: it takes about a minute.

STATUS. Whether a law is in force can't be read off its own text, and a
model must not answer from memory. So "status" is derived from evidence,
and every record says which evidence ("status_basis"):

    repealed              the site or the document says so, or another
                          document in this corpus repeals it
    superseded            an Ordinance later enacted as an Act found here,
                          or a notification replaced by a later one
    lapsed_presumed       an Ordinance of a kind that expires unless
                          enacted (see ORDINANCE_LAPSES) with no Act found
    amending_instrument   an amending law: its changes live on in the law
                          it amended; not a standalone law to cite
    in_force              listed as current law by an official code, with
                          nothing above against it
    unknown               no evidence either way

"status_confidence" is high / medium / low. A repeal by one province of a
law inherited from West Pakistan only affects that province's copy.
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

import confidence
import dedupe
import kg
from ocr_scanned import SOURCES, read_jsonl, source_layout

DATA_DIR = Path(__file__).parent / "data"

SOURCE_JURISDICTION = {"pakistancode": "Federal", "drs": "Federal", "punjab": "Punjab", "sindh": "Sindh",
                       "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
PROVINCE_WORDS = {
    "Punjab": r"\bpunjab\b",
    "Sindh": r"\bsindh?\b",
    "Khyber Pakhtunkhwa": r"khyber|pakhtunkhwa|n\.?\s?-?w\.?\s?f\.?\s?p|north[- ]west frontier",
    "Balochistan": r"bal[ou]chistan",
    "West Pakistan": r"west pakistan",
}
# Sources whose listing is a statement of current law (as opposed to an
# archive of everything ever gazetted).
CURRENT_CODES = {"pakistancode": "Pakistan Code", "punjab": "Punjab Code", "kp": "Khyber Pakhtunkhwa Code",
                 "balochistan": "Balochistan Code"}
# Ordinances expire (120 days federal, 90 days provincial) unless enacted as
# an Act -- except those made in the two periods whose ordinances were
# given permanent effect by Articles 270A and 270AA of the Constitution,
# and pre-1973 ones, which continued as existing laws under Article 268.
PERMANENT_ORDINANCE_YEARS = set(range(1977, 1986)) | set(range(1999, 2004))
FIRST_LAPSING_YEAR = 1973

PAKISTAN_CODE_LISTED = 1037  # catalog rows from its alphabetical listing; later rows were found only via search
MATCH_THRESHOLD = 93  # title similarity needed to say two titles name the same law
WORD = re.compile(r"(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])")     # ordinary words, as printed in lower case
re_number = re.compile(r"\d+%?")
SITE_PREFIX = re.compile(r"^\s*NA\s*-\s*")  # the archive's own marker in front of 54 names; not part of the title
PARTIAL = re.compile(r"^(?:sub-?)?(?:sections?|clauses?|rules?|regulations?|articles?|paragraphs?|schedules?|"
                     r"chapters?|parts?|the rules made|so much of)\b", re.I)


def norm_title(title: str) -> str:
    """Comparable form of a law's title: no article, numbering, punctuation or spelling variants."""
    t = (title or "").lower()
    # the law's number in brackets: "(XV of 2010)", "(W.P. Ord. No. XX of 1961)" -- but not words of the name
    # itself, "(Ejectment of Unauthorized Occupants)", "(No. 2)"
    t = re.sub(r"\([^()]*\bof\s*(?:1[89]|20)\d{2}[^()]*\)", " ", t)
    t = re.sub(r"\((?:repealed|omitted)\b[^)]*\)?", " ", t)      # "(Repealed by Act I of 2019)"; "(Repeal)" is part of a name
    t = re.sub(r"n\.?\s?-?w\.?\s?f\.?\s?p\.?|north[- ]west frontier province", "khyber pakhtunkhwa", t)
    t = re.sub(r"\bsind\b", "sindh", t).replace("baluchistan", "balochistan")
    t = re.sub(r"^(\s*(?:the\s+)?)w\.\s?p\.\s", r"\1west pakistan ", t)      # "The W.P. Shops and Establishment Ordinance"
    t = re.sub(r"\bwild\s+life\b", "wildlife", t)
    # pay scales named three ways in service rules: "(B-16 & above)", "(Grade 16 and above)", "(BPS-16 and above)"
    t = re.sub(r"\b(?:basic\s+pay\s+scale|bps|grade|b)\s*[-–]?\s*(\d{1,2})\b", r"bps \1", t)
    t = re.sub(r"\bbps (\d{1,2})\s*(?:-|–|to)\s*(?:bps\s*-?\s*)?(\d{1,2})\b", r"bps \1 to \2", t)
    t = re.sub(r"['’‘`]", "", t)                                             # "Prisoner's", "Workers’"
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"^\s*(?:\d+\s+)?the\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()


KIND_WORDS = ("act", "ordinance", "order", "rules", "regulation", "regulations", "code", "notification")
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
FILLER = {"the", "of", "and", "for", "in", "to", "on", "a", "an", "by"}
MARKERS = ("amendment", "repeal", "validation", "extension", "temporary", "supplementary", *ORDINALS)
# digits, or a well-formed roman numeral ("xv", not "civil")
NUMBER = re.compile(r"\d+|(?=[ivxlcdm])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})")


def same_title(a: str, b: str) -> bool:
    """Do two normalised titles name the same law?

    Close spelling is not enough: "X (Amendment) Act" and "X (Second
    Amendment) Act", or "X Ordinance" and the "X Act" that replaced it,
    differ by one word and are different laws. So the words that tell
    laws apart must agree exactly, and the rest must be near-identical.
    """
    words_a, words_b = set(a.split()), set(b.split())
    for group in (KIND_WORDS, MARKERS):
        if {w for w in group if w in words_a} != {w for w in group if w in words_b}:
            return False
    # every number must match: "S.R.O. 29(1)/2006" is not "S.R.O. 229(1)/2006",
    # and "Act No. II of 1871" is not "Act No. XXII of 1871"
    if {w for w in words_a if NUMBER.fullmatch(w)} != {w for w in words_b if NUMBER.fullmatch(w)}:
        return False
    if fuzz.ratio(a, b) < MATCH_THRESHOLD:
        return False
    # and every word needs a counterpart: in a long title one different word
    # ("Punjab Arms (Amendment) ..." / "Punjab Drugs (Amendment) ...") barely moves
    # the overall score, but it is a different law. Spelling slips still pass.
    content_a, content_b = words_a - FILLER, words_b - FILLER
    return all(any(fuzz.ratio(w, v) >= 80 for v in content_b) for w in content_a) \
        and all(any(fuzz.ratio(w, v) >= 80 for v in content_a) for w in content_b)


def title_year(title: str) -> int | None:
    years = re.findall(r"\b(1[89]\d{2}|20\d{2})\b", title or "")
    return int(years[-1]) if years else None


def base_title(title: str) -> str:
    """Title without its year, for comparing an Ordinance with the Act that replaced it."""
    return re.sub(r"\b(?:1[89]\d{2}|20\d{2})\b", " ", norm_title(title)).strip()


def jurisdiction_for(source: str, model_value: str | None, title: str) -> tuple[str, str]:
    """Jurisdiction and how it was decided. The website is the anchor; the model only refines it."""
    home = SOURCE_JURISDICTION[source]
    lowered = (title or "").lower()
    named = [j for j, pattern in PROVINCE_WORDS.items() if re.search(pattern, lowered)]
    if home == "Federal":
        # a federal archive: provincial only if the title itself names the province
        if model_value in named:
            return model_value, "title names it; model agrees"
        return "Federal", "federal source"
    if model_value == home:
        return home, "provincial source; model agrees"
    # "The West Pakistan Suppression of Prostitution Ordinance, 1961" on a province's code is a law of the old
    # province of West Pakistan that the province inherited, whatever the model called it
    if "West Pakistan" in named and home not in named and model_value in ("Federal", "West Pakistan", None):
        return "West Pakistan", "the title names West Pakistan: a West Pakistan law carried on this province's code"
    if model_value == "West Pakistan" and ("West Pakistan" in named or (title_year(title) or 0) in range(1955, 1971)):
        return "West Pakistan", "West Pakistan law carried on this province's code"
    if model_value == "Federal" and home not in named:
        return "Federal", "federal law hosted on a provincial code (model; title does not name the province)"
    return home, "provincial source"


def clean_list(values, own_norm: str) -> list[str]:
    seen, out = set(), []
    for value in values or []:
        value = re.sub(r"\s+", " ", str(value)).strip()
        key = norm_title(value)
        # blanks, repeats, and the document naming itself -- but not its predecessor of the same
        # name: "Local Government Act, 2013" in the 2019 Act is the law it repeals (same_title
        # tells years apart; plain similarity did not, and threw those away)
        if not key or key in seen or same_title(key, own_norm):
            continue
        seen.add(key)
        out.append(value)
    return out


REPEAL_WORDS = re.compile(r"\b(?:is|are|shall\s+stand|stands?|shall\s+be)\s+(?:hereby\s+)?repealed\b|\bhereby\s+repeals?\b", re.I)
LIST_INTRO = re.compile(r"(?:following|namely)[^.]{0,200}?repeal|repeal[^.]{0,200}?(?:following|namely)", re.I | re.S)
# words that make a repeal partial: only some provisions, or only in some place
LIMITED = re.compile(r"to the extent|in (?:its|their) application|so far as|in so far|as applicable to|"
                     r"in relation to the|except\b|other than", re.I)
PROVISION_OF = re.compile(r"(?:sub-?)?(?:sections?|clauses?|rules?|articles?|schedules?|chapters?|parts?|paragraphs?)"
                          r"\s+[\w\-, ()]{1,60}?\s+of\s+(?:the\s+)?$", re.I)


# The Sindh code prints the marginal note inside the line: "is hereby Repeal and saving. repealed",
# "shall Repeal and stand repealed". MARGIN lets such a note sit among the verb's words.
MARGIN = r"(?:Repeals?\b[^;:]{0,45}?\s+)?"
REPEAL_VERB = rf"(?i:(?:is|are|stands?|shall\s+{MARGIN}stand|shall\s+{MARGIN}be)\s+(?:hereby\s+)?{MARGIN}repealed)"
# What may stand between a law's title and the verb: its number in brackets, "hereinafter ...",
# and the names of other laws repealed in the same breath. Names only -- Capitalised words and the
# small words inside titles -- never a clause of its own ("shall apply"), so this part is matched
# with case as printed. One flat run of words: nested repeats here made the pattern take minutes.
NAME_WORD = (r"(?:[A-Z][\w'’]*-\s[a-z]+|[A-Z][\w'’\-]*|of|and|the|for|to|in|on|by|&|\([^()]{0,90}\)|(?:1[89]|20)\d{2}"   # "Adver- tisements": a word split at the line end
             r"|\d{1,2}(?=\[))")                                                                                        # the note mark of "3[the Bahawalpur ... Act, 1949]"
AFTER_TITLE = (r"\d{0,2}(?:\s*,)?(?:\s*\d{1,2}(?=[\s,]))?(?:\s*\([^()]{0,90}\)){0,2}"                             # a footnote mark; "(XXI of 1974)"; "(hereinafter ...)"
               r"(?:\s*,?\s*(?i:hereinafter\s+(?:referred\s+to\s+as|called))\s+[^,.;]{3,60})?"
               r"(?:\s*,?\s*(?i:issued|notified|published|made)\s+(?i:vide|under|by|in)\s+(?:(?!\.\s+[A-Z])[^;:]){3,120}?)?"   # "issued vide S.R.O. 43(I)/2016 dated ..."
               r"(?:\s*,)?"
               rf"(?:\s*(?:,|;|and\b)(?:[\s,;\[\]]+{NAME_WORD})*[\]]?)?"                                                    # (square brackets mark inserted words)
               r"(?:\s*,?\s*(?i:and|along\s+with|together\s+with|including)\s+(?i:all\s+)?(?i:the\s+)?"                      # "and all amendments made therein"
               r"(?i:amendments?|rules|notifications|orders|instructions)\b(?:(?!\.\s+[A-Z])[^;:]){0,100}?)?")
IS_REPEALED = re.compile(rf"^{AFTER_TITLE}\s*,?\s*{REPEAL_VERB}\b")
LIST_OF_REPEALS = re.compile(rf"(?i:following)\s+(?:[A-Za-z,]+\s+){{0,6}}?{REPEAL_VERB}|{REPEAL_VERB}\s*,?\s*(?i:namely)")
LIST_ITEM = re.compile(r"(?:\(\w{1,4}\)|\b\d{1,2}\.|[;:—–-]|\band)\s*$", re.I)


IN_THE_PROVINCE = re.compile(r",?\s*in\s+(?:its|their)\s+application\s+to\s+(?:the\s+)?(?:Province(?:\s+of\s+(?:the\s+)?[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)?|"
                             r"Punjab|Sindh?|Bal[ou]chistan|Khyber\s+Pakhtunkhwa|North-West\s+Frontier\s+Province|West\s+Pakistan)\s*,?", re.I)


PROVINCE_OF_CODE = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
PROVINCIAL_NAME = re.compile(r"\b(?:west\s+pakistan|punjab|sindh?|bal[ou]chistan|khyber|n\.?-?w\.?-?f\.?-?p\.?|north-west\s+frontier|"
                             r"bahawalpur|karachi|khairpur|quetta|lahore)\b", re.I)
IN_WEST_PAKISTAN = re.compile(r",?\s*in\s+(?:its|their)\s+application\s+to\s+(?:the\s+)?(?:Province\s+of\s+)?West\s+Pakistan\s*,?", re.I)


def repeal_in_text(text: str, title: str, provincial: bool = False, purpose: bool = False) -> str | None:
    """The words in `text` that repeal the whole of the law called `title`, or None.

    `provincial`: the repealing law is a province's. "in its application to the Province" is then
    the whole of what that province can repeal, not a limit; the tag goes on that province's copy.
    `purpose`: the repealing law is itself a "(Repeal) Act"; its long title then counts too.

    Strict on purpose: a wrong "repealed" on the General Clauses Act is far worse than a
    missed one. The title must be the subject of the repealing words --

        "The X Act, 1974 (XXI of 1974), is hereby repealed"        (other titles may share the verb)
        "The following laws are hereby repealed- (a) the X Act, 1974; (b) ..."

    -- and not merely stand in a sentence that repeals something: "sections 3 and 4 of the X Act
    are repealed", "... is repealed and the X Act shall apply", "the words are repealed wherever
    the X Act is in force". Nothing may limit it to a place or a part ("in its application to").
    """
    words = re.findall(r"[A-Za-z0-9]+", re.sub(r"\([^()]*\bof\s*(?:1[89]|20)\d{2}[^()]*\)", " ", title, flags=re.I))
    words = [w for w in words if w.lower() != "the"]
    if len(words) < 3:
        return None
    # "the" may stand anywhere in the title as printed: "Representation of the People Act"
    pattern = re.compile(r"\W+(?:the\W+)?".join(map(re.escape, words)), re.I)
    for match in pattern.finditer(text):
        lead = re.sub(r"\s+", " ", text[max(0, match.start() - 6000):match.start()])     # (a list of 26 repealed laws is long)
        tail = re.sub(r"\s+", " ", text[match.end():match.end() + 900])
        # West Pakistan was the whole of today's Pakistan; a province repeals for itself
        tail = (IN_THE_PROVINCE if provincial else IN_WEST_PAKISTAN).sub(" ", tail, count=1)
        # the title must open its clause: after punctuation, a list mark, "and", "that", or the
        # section's own heading -- not after "of the", "under the", "wherever the", which only refer to it
        opening = re.sub(r"(?<![A-Za-z])the\s*$", "", lead, flags=re.I).rstrip()
        opening = re.sub(r"(?<![A-Za-z])the\s+whole\s+of$", "", opening, flags=re.I).rstrip()
        if purpose and re.search(r"\b(?:Act|Ordinance|Order)\s+to\s+repeal$", opening, re.I):
            return (opening[-40:] + " " + match.group(0)).strip()        # the long title of a "(Repeal) Act": "An Act to repeal the X Act"
        if opening and opening[-1].isalpha() and not re.search(r"\b(?:that|Repeals?|Savings?)$", opening):
            # after "and", what stands before it must be another law with its year ("the A Act, 1960
            # (I of 1960) and the X Act are repealed") -- not "On the commencement of this Act and the
            # X Act, 2023, the B Ordinance shall be repealed", where X is not repealed at all
            if not re.search(r"\b(?:Act|Ordinance|Order|Rules|Regulations?|Code),?\s*(?:1[89]|20)\d{2}\d{0,2}(?:\s*\([^()]{0,90}\))?\s*,?\s*and$", opening):
                continue
        direct = IS_REPEALED.match(tail)
        if direct:
            stop = re.search(r"[.;](?:\s|$)", tail[direct.end():])
            rest = tail[direct.end():direct.end() + (stop.start() if stop else 200)]
            if not LIMITED.search(tail[:direct.end()] + rest):
                return (opening[-60:] + " " + match.group(0) + tail[:direct.end()]).strip()[-300:]
            continue
        # "(a) the X Act, 1960;" inside "The following laws are hereby repealed, namely:-"
        heads = list(LIST_OF_REPEALS.finditer(lead))
        if not heads or not LIST_ITEM.search(opening):
            continue
        head = heads[-1]
        intro = lead[max(0, lead.rfind(". ", 0, head.start())):head.end()]
        between = lead[head.end():]
        stop = re.search(r"[.;](?:\s|$)", tail)
        item = tail[:stop.start() if stop else 200]
        if LIMITED.search(intro + " " + item) or re.search(r"\bschedule\b|\bamendments?\b|\bprovisions\b", intro, re.I):
            continue
        if re.search(r"\.\s+\d{1,3}[A-Z]?\.\s*[A-Z]", between):     # a new section began in between
            continue
        return (lead[head.start():head.end()] + " ... " + match.group(0)).strip()[-300:]
    return None


LAW_WORD = r"(?:[A-Z][A-Za-z'’\-]*\.?|\([^()]{1,50}\)|of|and|the|for|to|in|on|by|&)"
NAMED_LAW = re.compile(rf"({LAW_WORD}(?:[ ,\n]+{LAW_WORD}){{1,20}}?[ ,\n]+(?:Act|Ordinance|Order|Regulations?|Rules|Code))"
                       r"[ ,\n]+((?:1[89]|20)\d{2})\b")


def repeal_candidates(text: str) -> list[str]:
    """Titles of laws named near this document's own repealing words ("... is hereby repealed",
    "the following laws are hereby repealed"). Only candidates: each still has to match a law in
    the corpus and pass repeal_in_text. This is what makes the in-force tag independent of the
    model's list of repeals, which is often empty for laws that plainly repeal others."""
    found = {}
    for match in REPEAL_WORDS.finditer(text):
        window = text[max(0, match.start() - 700):match.end() + 2500]   # a list of repealed laws follows its introduction
        for law in NAMED_LAW.finditer(window):
            name, year = re.sub(r"\s+", " ", law.group(1)), law.group(2)
            # the words caught in front of the title vary ("Act, the Trade Organisations Ordinance"):
            # offer the title from each "the" onwards as well
            starts = [0] + [m.end() for m in re.finditer(r"\bthe\s+", name, re.I)]
            for start in starts:
                title = re.sub(r"^(?:\([a-z0-9]{1,4}\)\s*)+", "", name[start:]).strip(" ,")
                if len(title.split()) >= 2:
                    found.setdefault(norm_title(f"{title}, {year}"), f"{title}, {year}")
    return list(found.values())


def load_source(source: str) -> list[dict]:
    data_dir, key = source_layout(source)
    catalog = {row[key]: (index, row) for index, row in enumerate(read_jsonl(data_dir / "catalog.jsonl"))}
    first = {row[key]: row["metadata"] for row in read_jsonl(data_dir / "metadata.jsonl")}
    second = {row[key]: row["metadata"] for row in read_jsonl(data_dir / "metadata_v2.jsonl")}
    docs = []
    with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row[key]
            position, cat = catalog.get(doc_id, (None, {}))
            docs.append({
                "key": key, "id": doc_id, "source": source, "name": row.get("name") or "",
                "text_source": row.get("text_source", "native"), "pdf_file": row.get("pdf_file"),
                "num_pages": row.get("num_pages"), "opening": row["raw_text"][:500],
                # full text: to check what a law says it repeals, and to compare copies
                "text": row["raw_text"],
                "catalog": cat, "catalog_position": position,
                "pass1": first.get(doc_id), "pass2": second.get(doc_id),
                # what a second reading of the scan showed (repair_pages.py)
                "text_check": row.get("text_check"),
            })
    return docs


SHORT_TITLE = re.compile(r"(?:may\s+be\s+(?:cited|called)|shall\s+be\s+(?:cited|called))\s+(?:as\s+)?(?:the\s+)?"
                         r"([A-Z][^.;:\n]{3,150}?(?:Act|Ordinance|Order|Rules|Regulations?|Code))[,\s]*((?:1[89]|20)\d{2})", re.I)


def own_short_title(text: str) -> tuple[str, int] | None:
    """(name, year) from the law's own short-title clause, footnote marks taken out."""
    found = SHORT_TITLE.search(text[:30000])
    if not found:
        return None
    name = re.sub(r"\d*\s*\[\s*\*?\s*\]|\d+\s*\*|\*|\d+\[|\]", " ", found.group(1))
    return re.sub(r"\s+", " ", name).strip(" ,"), int(found.group(2))


def without_years(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\b(?:1[89]|20)\d{2}\b", " ", title)).strip()


def build_record(doc: dict) -> dict:
    """Merge the two passes and the catalogue into one cleaned record (status comes later)."""
    p1, p2 = doc["pass1"] or {}, doc["pass2"] or {}
    best = p2 or p1
    title = best.get("short_title") or best.get("title") or ""
    if title.strip().lower() in ("", "null", "none", "n/a", "unknown", "notification"):
        title = doc["name"]  # the model found no title of its own in the text
    title = SITE_PREFIX.sub("", title)
    own = norm_title(title)
    kind = best.get("instrument_type") or "Other"

    disagreements = []
    # Gazette PDFs often hold several laws; the model may have described the
    # wrong one. When its title is nothing like the website's name for the
    # document, trust the website and flag the record.
    listed = norm_title(SITE_PREFIX.sub("", doc["name"]))
    if title_year(doc["name"]) and set(listed.split()) & set(KIND_WORDS) and fuzz.token_set_ratio(listed, own) < 70:
        disagreements.append("title_vs_catalog")
        title, own = SITE_PREFIX.sub("", doc["name"]), listed
    if p1 and p2:
        if fuzz.token_sort_ratio(norm_title(p1.get("title")), own) < 85:
            disagreements.append("title")
        for field in ("instrument_type", "year", "date_enacted"):
            if p1.get(field) and p2.get(field) and p1[field] != p2[field]:
                disagreements.append(field)
        if p1.get("number") and p2.get("number") and norm_title(p1["number"]) != norm_title(p2["number"]):
            disagreements.append("number")
    # The law's own short-title clause ("This Act may be called the X Act, 1956") beats a title
    # read off a gazette's date. Applied only to the same name with an earlier year: a later
    # one is more often a misreading, and another name is usually another law in the same PDF.
    corrected_from, year_from_text = None, None
    cited = own_short_title(doc["text"])
    listed_year = best.get("year") or title_year(title)
    if cited and listed_year and without_years(norm_title(cited[0])) == without_years(own) and cited[1] < listed_year <= cited[1] + 15:
        # the name as listed stays (it is better spelt than the scanned clause); only its year changes
        year_from_text = cited[1]
        fixed = re.sub(r"\b(?:1[89]|20)\d{2}\b(?!.*\b(?:1[89]|20)\d{2}\b)", str(cited[1]), title) if title_year(title) else f"{title}, {cited[1]}"
        if fixed != title:
            corrected_from, title, own = title, fixed, norm_title(fixed)
            disagreements.append("title_vs_text")
    year = year_from_text or best.get("year") or title_year(title)
    catalog_year = doc["catalog"].get("year")
    if catalog_year and year and int(catalog_year) != year and title_year(title) != year:
        disagreements.append("year_vs_catalog")

    jurisdiction, basis = jurisdiction_for(doc["source"], best.get("jurisdiction"), title + " " + doc["name"])
    record = {
        doc["key"]: doc["id"],
        "source": doc["source"],
        "name": doc["name"],
        "title": title,
        "title_corrected_from": corrected_from,
        "long_title": p2.get("long_title"),
        "instrument_type": kind,
        "number": best.get("number"),
        "year": year,
        "jurisdiction": jurisdiction,
        "jurisdiction_basis": basis,
        "enacting_authority": p2.get("enacting_authority"),
        "territorial_extent": p2.get("territorial_extent"),
        "date_enacted": best.get("date_enacted"),
        "commencement": p2.get("commencement"),
        # only subordinate legislation has a parent law
        "parent_law": p2.get("parent_law") if kind not in ("Act", "Ordinance") else None,
        # by its own title; the model's view (it amends something) is kept separately
        "is_amendment": bool(re.search(r"\bamend(?:ment|ing)\b", title + " " + doc["name"], re.I)),
        "amends_other_laws": bool(best.get("is_amendment")),
        "amends": clean_list(p2.get("amends") if p2 else [p1.get("amends")] if p1.get("amends") else [], own),
        "repeals": clean_list(p2.get("repeals") if p2 else [p1.get("repeals")] if p1.get("repeals") else [], own),
        # only the model's first choice: it pads the second and third slots with unrelated areas
        "practice_area": (p2.get("practice_areas") or [None])[0],
        "subjects": best.get("subjects") or [],
        "establishes": p2.get("establishes") or [],
        "creates_offences": p2.get("creates_offences"),
        "max_penalty": p2.get("max_penalty"),
        "key_definitions": p2.get("key_definitions") or [],
        "administering_body": p2.get("administering_body"),
        "one_line": p2.get("one_line"),
        "summary": best.get("summary"),
        "text_source": doc["text_source"],
        "text_check": doc.get("text_check"),
        "num_pages": doc["num_pages"],
        "pdf_file": doc["pdf_file"],
        "passes": [n for n, p in (("first", p1), ("second", p2)) if p],
        "pass_disagreements": disagreements,
        "needs_review": len(disagreements) >= 2 or bool({"instrument_type", "title_vs_catalog"} & set(disagreements))
                        or not best,
    }
    record["_norm"] = own
    record["_doc"] = doc
    return record


def explicit_repeal(record: dict) -> tuple[str, str] | None:
    """Evidence, from the site or the document itself, that this law has been repealed."""
    doc = record["_doc"]
    p2 = doc["pass2"] or {}
    note = re.search(r"\(?\s*repeal(?:ed)?\s+by\s+[^)]{3,80}\)?", doc["name"], re.I)
    if note:
        return "the website's entry for it says: " + note.group(0).strip(), "high"
    if doc["catalog"].get("doc_type") == "repealed law":
        return "listed under repealed laws on the Punjab Code", "high"
    # the head of the document: its first lines, and on the Punjab Code the lines after "TEXT",
    # where the law itself begins once the table of contents is over
    head = doc["opening"]
    body = re.search(r"\nTEXT\s*\n", doc["text"][:20000])
    if body:
        head += "\n" + doc["text"][body.end():body.end() + 400]
    opening = re.search(r"[\[(]\s*(?:repealed|rep\.)\s+by\s+[^\])]{3,100}[\])]", head, re.I)
    if opening:
        return "note at the head of the document: " + opening.group(0), "high"
    evidence = p2.get("status_evidence") or ""
    is_repealer = re.search(r"\brepeal(?:ing)?\b", record["title"], re.I) or re.search(r"hereby\s+repeal", evidence, re.I)
    if p2.get("status_in_text") == "stated_repealed" and p2.get("repealed_by") and not is_repealer \
            and fuzz.ratio(norm_title(p2["repealed_by"]), record["_norm"]) < MATCH_THRESHOLD \
            and re.search(r"repeal", evidence, re.I) and whole_law_note(evidence):
        return f"the document carries a repeal note ({evidence[:120]!r})", "medium"
    return None


PART_OF_A_LAW = re.compile(r"\b(?:sections?|sec\.|s\.\s?\d|ss\.|sub-?section|clause|schedule|sch\.|chapter|part\s+[ivx\d]|proviso|"
                           r"para(?:graph)?|article|art\.|rule|entry|footnote)\b|^\W*\d+[A-Z]?\s?[.\-–]", re.I)
# Statute-revision Acts strike out spent sections and schedules; they do not end a law that is still printed.
REVISION_ACT = re.compile(r"repealing|revision\s+and\s+declaration|adaptation\s+of|\bA\.\s?O\.", re.I)


def whole_law_note(evidence: str) -> bool:
    """Does a repeal note quoted from inside a document speak of the whole law?

    Most such notes mark one repealed section or schedule of a law that is very much in force
    ("2. [Repeal of Acts.] Rep. by the Repealing Act, 1873" in the Canal and Drainage Act;
    "SCHEDULE V.-[Enactments repealed.] Rep. by ..." in the Christian Marriage Act). Only the
    words in front of "repealed" tell: what follows names the repealing law's own section.
    """
    found = re.search(r"rep(?:ealed|\.)", evidence, re.I)
    if not found or REVISION_ACT.search(evidence) or PART_OF_A_LAW.search(evidence[:found.start()]) \
            or re.search(r"\bto the extent\b|\bexcept\b|\bin so far as\b|\bin (?:its|their) application\b", evidence, re.I):
        return False
    # Only two forms say it of the law itself: the footnote printed on a law's title ("Since
    # repealed by the X Act, 1973, s. 19") and a sentence with the law as its subject ("This Act
    # has been repealed by ..."). A bare "[Repealed by Act X of 1878]" from somewhere inside the
    # text marks a section more often than not, and a note at the very top is caught separately.
    return bool(re.match(r"\W*since\s*repealed\s*by", evidence, re.I)
                or re.search(r"\b(?:this|the\s+said)\s+(?:act|ordinance|order|regulation|law)\b[^.]{0,40}?\b(?:has\s+been|is|was|stands?)\s+repealed", evidence, re.I))


# the codes that print a law as amended to date (DRS keeps the gazette text as first published)
UP_TO_DATE_CODES = ("pakistancode", "punjab", "kp", "balochistan")
LIMITING = re.compile(r"\b(?:except|save\s+(?:for|as)|to\s+the\s+extent|in\s+so\s+far|insofar|so\s+far\s+as)\b", re.I)


def limited_on_code(repealer: dict, title: str, copies: list[dict]) -> tuple[str, str] | None:
    """The repealing law as amended, on a code that keeps it up to date, limiting the repeal its first text made:
    "(f) the Conduct of General Elections Order, 2002 (Chief Executive's Order No.7 of 2002) [except Articles 1, 7B and
    7C]" (the exception inserted in 2017; DRS's gazette copy of the Elections Act, 2017 has none). The limiting words
    have to be there, right after the name -- a copy that merely could not be read proves nothing.
    -> (the code, its words) or None."""
    words = [w for w in re.findall(r"[A-Za-z]+|\d{4}", title)]
    if words and words[0].lower() == "the":
        words = words[1:]
    if len(words) < 3:
        return None
    name = re.compile(r"\W+".join(map(re.escape, words)), re.I)
    for copy in copies:
        if copy is repealer or copy["source"] not in UP_TO_DATE_CODES or repealer["source"] in UP_TO_DATE_CODES:
            continue
        text = copy["_doc"]["text"]
        for m in name.finditer(text):
            # up to the end of the item: the next ";" or the next "(g) the ..."
            after = re.split(r";|\(\s*[a-z]{1,3}\s*\)\s+(?:the\s+)?[A-Z]", text[m.end():m.end() + 200])[0]
            if LIMITING.search(after) and repeal_in_text(text, title, provincial=copy["jurisdiction"] != "Federal") is None:
                return copy["source"], re.sub(r"\s+", " ", text[m.start():m.end() + len(after)])[-220:]
    return None


def assign_status(records: list[dict]) -> None:
    by_year = defaultdict(list)
    copies_of = defaultdict(list)          # a law's copies on every site, by name and year
    for record in records:
        copies_of[(norm_title(record["title"]), record["year"])].append(record)
        for title in {record["title"], record["name"]}:
            year = title_year(title)
            if year:
                by_year[year].append((norm_title(title), record))

    def same_law(title: str, repealer: dict) -> list[dict]:
        """Corpus records that `title` (as written in `repealer`) refers to, within the repealer's reach."""
        year, wanted = title_year(title), norm_title(title)
        found = {}
        for candidate_norm, candidate in by_year.get(year, []) if year else []:
            # not itself, nor another copy of itself (gazette copy / consolidated copy)
            if candidate is repealer or candidate["_norm"] == repealer["_norm"] or not same_title(wanted, candidate_norm):
                continue
            # an Ordinance is replaced by the Act of the same name, never the other way round
            if repealer["instrument_type"] == "Ordinance" and candidate["instrument_type"] == "Act" \
                    and base_title(repealer["title"]).replace(" ordinance", "") == base_title(candidate["title"]).replace(" act", ""):
                continue
            if repealer["jurisdiction"] == "Federal":
                # A federal law ends a law for the Federation: the copies on the federal sites
                # (the Pakistan Code also carries West Pakistan laws as they apply in Islamabad).
                # A copy on a province's code is that province's to end -- an Islamabad food
                # safety Act does not repeal the Pure Food Ordinance in the Punjab.
                in_reach = candidate["source"] == "pakistancode" or (candidate["source"] == "drs" and candidate["jurisdiction"] == "Federal")
            else:
                # a province can only repeal a law as it applies in that province
                in_reach = candidate["source"] == repealer["source"]
            if in_reach:
                found[id(candidate)] = candidate
        return list(found.values())

    # 1. repealed by another document in the corpus -- but only when that
    #    document's own words repeal the whole law (see repeal_in_text)
    repealed_by = defaultdict(list)   # target -> [(repealer, its words)]
    mentioned_by = defaultdict(list)  # target -> [(repealer, why it doesn't count)]
    for record in records:
        # "Repealing and Amending" Acts list dozens of laws with a column saying how
        # much of each is repealed, often one section; a flat list can't be trusted
        mass = bool(re.search(r"\brepealing\b", record["title"], re.I)) or len(record["repeals"]) > 15
        named = list(record["repeals"])
        listed_by_model = {norm_title(title) for title in named}
        if not mass:
            # the model's list is a hint, not the source: read the document's own repeal clauses too
            named += [title for title in repeal_candidates(record["_doc"]["text"]) if norm_title(title) not in listed_by_model]
        for title in named:
            if PARTIAL.match(title):
                continue  # a repeal of some sections is an amendment, not the end of the law
            targets = same_law(title, record)
            if not targets:
                continue
            words = None if mass else repeal_in_text(record["_doc"]["text"], title, provincial=record["jurisdiction"] != "Federal",
                                                     purpose=bool(re.search(r"\(\s*repeal\s*\)|\brepeal\)", record["title"], re.I)))
            limited = limited_on_code(record, title, copies_of[(norm_title(record["title"]), record["year"])]) if words else None
            if limited:
                # the first text repealed it whole; the law as amended keeps part of it
                for target in targets:
                    mentioned_by[id(target)].append((record, f"the up-to-date text on the {limited[0]} code limits this repeal: \"{limited[1]}\""))
                continue
            for target in targets:
                # a law cannot be repealed by one made before it: the title was only mentioned
                if words and record["year"] and target["year"] and record["year"] < target["year"]:
                    continue
                if words:
                    repealed_by[id(target)].append((record, words))
                elif mass or norm_title(title) in listed_by_model:
                    # (a title merely printed near a repealing sentence is no lead worth keeping)
                    mentioned_by[id(target)].append(
                        (record, "listed in a general repealing Act, which may repeal only part of it" if mass
                         else "named by it, but its text does not plainly repeal the whole law"))

    # 2. an Ordinance later enacted as an Act
    acts = defaultdict(list)
    for record in records:
        if record["instrument_type"] == "Act" and record["year"] and not record["is_amendment"]:
            acts[base_title(record["title"]).replace(" act", "")].append(record)

    for record in records:
        kind, year = record["instrument_type"], record["year"]
        explicit = explicit_repeal(record)
        repealers = repealed_by.get(id(record), [])
        status, basis, confidence = "unknown", ["no evidence found either way"], "low"
        related = []

        if explicit:
            status, basis, confidence = "repealed", [explicit[0]], explicit[1]
        if repealers:
            related = [{"relation": "repealed_by", "title": r["title"], "source": r["source"], "words": words}
                       for r, words in repealers[:5]]
            first, words = repealers[0]
            line = f"repealed by {first['title']} ({first['source']}), which says: \"{words[-200:]}\""
            basis = ([basis[0]] if status == "repealed" else []) + [line]
            status, confidence = "repealed", "high"
        # leads that did not pass the check: kept for a human to look at, never used for status
        related += [{"relation": "possibly_affected_by", "title": r["title"], "source": r["source"], "note": why}
                    for r, why in mentioned_by.get(id(record), [])[:5]]

        if status == "unknown" and kind == "Ordinance" and year and not record["is_amendment"]:
            stem = base_title(record["title"]).replace(" ordinance", "")
            successors = [a for a in acts.get(stem, []) if 0 <= a["year"] - year <= 3
                          and (a["source"] == record["source"] or a["jurisdiction"] == record["jurisdiction"])]
            if successors:
                status, confidence = "superseded", "medium"
                basis = [f"enacted as {successors[0]['title']} ({successors[0]['source']})"]
                related = [{"relation": "enacted_as", "title": a["title"], "source": a["source"]}
                           for a in successors[:3]] + related

        if status == "unknown" and record["is_amendment"]:
            status, confidence = "amending_instrument", "high"
            basis = ["its title says it is an amending law: its effect is carried in the law it amended"]

        if status == "unknown" and kind == "Ordinance" and year and year >= FIRST_LAPSING_YEAR \
                and year not in PERMANENT_ORDINANCE_YEARS and record["jurisdiction"] != "West Pakistan":
            if record["source"] in CURRENT_CODES and record["_doc"]["catalog"].get("doc_type") != "law (gazette copy)":
                status, confidence = "in_force", "low"
                basis = [f"an Ordinance of a kind that normally expires, but still listed as current on the "
                         f"{CURRENT_CODES[record['source']]}"]
            else:
                status, confidence = "lapsed_presumed", "low"
                basis = [f"an Ordinance of {year}: ordinances expire within months unless enacted as an Act, "
                         f"and no such Act was found in this corpus"]

        if status == "unknown" and record["source"] in CURRENT_CODES:
            doc = record["_doc"]
            gazette_copy = doc["catalog"].get("doc_type") in ("law (gazette copy)", "rule (gazette copy)")
            # Pakistan Code keeps some laws on the site but out of its listing of laws
            # (found only through search); those are not vouched for as current.
            unlisted = record["source"] == "pakistancode" and (doc["catalog_position"] or 0) >= PAKISTAN_CODE_LISTED
            if unlisted:
                basis = ["on the Pakistan Code site but not in its alphabetical list of laws (found through its search): "
                         "the site does not vouch for it as current, and no repeal of it was found"]
            elif not gazette_copy:
                status, confidence = "in_force", "medium"
                basis = [f"listed as current law on the {CURRENT_CODES[record['source']]}; nothing found repealing it"]
        # Where the tag holds. A province's code speaks for that province only: the Factories Act,
        # 1934 is repealed in the Punjab (by its Labour Code) and in force for the Federation.
        applies_in = PROVINCE_OF_CODE.get(record["source"]) or ("Federation" if record["jurisdiction"] == "Federal" else record["jurisdiction"])
        record.update(status=status, status_confidence=confidence, status_basis=basis, status_applies_in=applies_in, related=related)

    # 3. archive copies inherit the status of the same law in a current code
    by_title = defaultdict(list)
    for record in records:
        if record["source"] in CURRENT_CODES and record["status"] in ("in_force", "repealed"):
            by_title[record["_norm"]].append(record)
    for record in records:
        if record["status"] == "unknown":
            twins = [t for t in by_title.get(record["_norm"], []) if t["jurisdiction"] == record["jurisdiction"]
                     or record["jurisdiction"] == "Federal" and t["source"] == "pakistancode"]
            if twins:
                twin = twins[0]
                record.update(status=twin["status"], status_confidence="medium" if twin["status"] == "in_force" else twin["status_confidence"],
                              status_basis=[f"same law as '{twin['title']}' on the {CURRENT_CODES[twin['source']]}: "
                                            + twin["status_basis"][0]])
        # A federal law that a provincial code still lists as current, while the Pakistan Code says it
        # is repealed (the Companies Ordinance, 1984 on the Punjab Code). Which is right depends on
        # whether the subject is federal or the province's own, which nothing here can tell: so the
        # tag says so, instead of repeating either site.
        elif record["status"] == "in_force" and record["jurisdiction"] == "Federal" and record["source"] not in ("pakistancode", "drs") \
                and not PROVINCIAL_NAME.search(record["title"]):
            twins = [t for t in by_title.get(record["_norm"], []) if t["source"] == "pakistancode" and t["status"] == "repealed"]
            if twins:
                record.update(status="unknown", status_confidence="low",
                              status_basis=[f"the sites disagree: the {CURRENT_CODES[record['source']]} lists it as current law, but for the "
                                            f"Federation it is repealed ({twins[0]['status_basis'][-1][:200]}). It survives in the province "
                                            f"only if the subject is the province's own"])
                record["related"] = record["related"] + [{"relation": "repealed_federally", "title": twins[0]["title"], "source": "pakistancode"}]


def main():
    started = time.time()
    records = []
    for source in SOURCES:
        records += [build_record(doc) for doc in load_source(source)]
    # how much of each text is garbled: ordinary (lower-case) words seen in fewer than 5
    # documents of the whole corpus. Real words recur; a scan's misreadings do not. Capitalised
    # words are left out, so that a schedule of village names is not taken for a bad scan.
    seen_in = Counter()
    for record in records:
        seen_in.update(set(WORD.findall(record["_doc"]["text"])))
    for record in records:
        words = WORD.findall(record["_doc"]["text"])
        record["_garble"] = sum(seen_in[w] < 5 for w in words) / max(1, len(words))
        record["garbled_words"] = round(record["_garble"], 3)
    assign_status(records)
    dedup = dedupe.assign_works(records, same_title, norm_title, title_year)
    # the knowledge graph: every repeal, amendment and parent law read from the laws' own words, and the
    # rules that settle status from them (kg.py; written to data/kg/)
    graph = kg.build(records, same_title, norm_title, title_year)
    for record in records:
        record["authority_rank"] = dedupe.authority_rank(record)
    print("knowledge graph:", json.dumps(graph))
    tiers = confidence.assign(records, same_title, norm_title, own_short_title)

    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "documents": len(records),
               "deduplication": dedup,
               "with_second_pass": sum("second" in r["passes"] for r in records),
               "needs_review": sum(r["needs_review"] for r in records), "by_source": {}}
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        mine = [r for r in records if r["source"] == source]
        with (data_dir / "metadata_final.jsonl").open("w", encoding="utf-8") as out:
            for record in mine:
                out.write(json.dumps({k: v for k, v in record.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
        summary["by_source"][source] = {
            "documents": len(mine),
            "status": dict(Counter(r["status"] for r in mine).most_common()),
            "jurisdiction": dict(Counter(r["jurisdiction"] for r in mine).most_common()),
        }
        print(f"{source:13s} {len(mine):5d}  " + "  ".join(f"{k}={v}" for k, v in summary["by_source"][source]["status"].items()))
    summary["status"] = dict(Counter(r["status"] for r in records).most_common())
    summary["status_confidence"] = dict(Counter((r["status"], r["status_confidence"]) for r in records).most_common())
    summary["status_confidence"] = {f"{s} ({c})": n for (s, c), n in Counter((r["status"], r["status_confidence"]) for r in records).most_common()}
    (DATA_DIR / "metadata_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(records)} documents; second pass available for {summary['with_second_pass']}; "
          f"{summary['needs_review']} flagged for review  ({time.time() - started:.0f}s)")
    print("status:", summary["status"])
    print(f"deduplication: {dedup['works']} distinct laws from {len(records)} documents; "
          f"{dedup['index_default']} in the default index; {dedup['pairs_for_review']} pairs left for review "
          f"(data/dedup_review.jsonl)")
    print("  copies set aside:", dedup["non_primary_by_version"])
    print("  left out of the index:", dedup["index_exclusion"])
    summary["tiers"] = tiers
    summary["quarantine_reasons"] = dict(Counter(reason.split(":")[0] + ": " + re_number.sub("N", reason.split(": ", 1)[1])
                                                 for r in records for reason in r["quarantine_reasons"]).most_common())
    (DATA_DIR / "metadata_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tiers:", tiers)
    print("held in quarantine because:", summary["quarantine_reasons"])


if __name__ == "__main__":
    main()
