"""
kg.py

The corpus as a knowledge graph, and the rules that reason over it (KAG: the app answers from
facts in the graph, each carrying the words of the law that prove it).

Called by finalize_metadata.py once status and deduplication are done. Writes data/kg/:

    nodes.jsonl      one row per law (work): copies, family, jurisdiction, kind, number, year,
                     status and where that status comes from
    edges.jsonl      one row per relation, each with the sentence it was read from:
                       repeals      extent whole | part | territory | conditional
                       amends       an amending law -> the law it amends
                       made_under   rules, regulations, notifications -> the Act whose powers they use
                       enacted_as   an Ordinance -> the Act that replaced it
                       copy_of      a copy set aside for the indexed copy of the same law (dedupe.py)
                       sibling_of   the same law in another jurisdiction's statute book (dedupe.py)
    conflicts.jsonl  facts that contradict each other, kept for review rather than settled by guess
    summary.json

The rules, applied after the relations are read:

  R1 repeal     a later law within reach repeals the whole of it -> repealed. Verified when the
                clause is plain and names it by title, and the repealer is not an Ordinance that may
                have lapsed or a Bill; probable when read off a schedule, matched by number only, or
                made by such an Ordinance.
  R2 territory  repealed for one territory only ("to the extent of the Islamabad Capital Territory")
                -> still in force elsewhere; the territory goes into status_exceptions.
  R3 part       some provisions repealed -> an amendment, nothing more.
  R4 conflict   an official code lists it as current but only a probable repeal says otherwise
                -> status unknown, with both sides in the basis. Neither side is picked.
  R5 parent     rules or regulations made under an Act that is repealed -> recorded; they live on
                only through the repealing law's savings clause.
  R6 order      a law is never repealed by one made before it.
  R10 everywhere repealed for every territory, one by one (the Trusts Act, 1882: by the Islamabad Capital
                Territory and each province in 2020) -> repealed; verified when every one of those repeals is.
  R7 leads      repeal leads left by finalize_metadata (a law the metadata pass said repeals this
                one, but whose words could not be matched) are settled by what that law's text says:
                a repeal (R1-R4), or only a partial repeal, a mention, an amendment or an override.

Reach: a federal law repeals the federal copies of a law; a province's law repeals that
province's copy; a West Pakistan law reaches every copy (it was the law of all of today's
Pakistan), but its effect on a copy that an official code still lists as current is a conflict
(R4), since such laws were often re-enacted or adapted later.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

KG_DIR = Path(__file__).parent / "data" / "kg"
FEDERAL_SOURCES = ("pakistancode", "drs")
CURRENT_CODES = ("pakistancode", "punjab", "kp", "balochistan")
PROVINCE_OF = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
EVERY_TERRITORY = {"Islamabad Capital Territory", *PROVINCE_OF.values()}     # where a federal law runs (rule R10)
# a word that makes another law of a name ("the X (Amendment) Rules" are not "the X Rules"): never the one word a clause
# may add to a law's name (Graph.resolve)
NEAR_NAME_QUALIFIERS = {"amendment", "amending", "second", "third", "fourth", "additional", "supplementary", "special", "deputy",
                        "chief", "interim", "temporary", "transitional", "west", "east", "federal", "provincial", "repeal", "validation",
                        "no", "part", "procedure", "general", "revised", "new", "former", "senior", "junior", "assistant",
                        "punjab", "sindh", "khyber", "pakhtunkhwa", "balochistan", "islamabad", "pakistan", "national", "district"}
YEAR = r"(?:1[89]\d{2}|20\d{2})"

# ---- reading the text ---------------------------------------------------------------------------

# OCR slips seen in repeal clauses: "hercby", "here by", "rcpealed", "rel ealed"
OCR_FIXES = [(re.compile(r"\bhe\s?r[ce]\s?b[yv]\b|\bherehy\b", re.I), "hereby"),
             (re.compile(r"\br[ce]p[ce]a[l1]\s?[ec]d\b|\brel\s?ealed\b|\brepeal\s+ed\b|\bRepcaled\b", re.I), "repealed")]
# Marginal notes the Balochistan and Sindh codes print inside the sentence: "The Balochistan Board of
# Intermediate and Savings. Secondary Education Ordinance, 1977 ... is hereby repealed"
MARGIN_NOTE = re.compile(r"(?<=\s)(?:Repeals?\s+and\s+)?[Ss]avings?\s*[.,]\s+(?=[A-Za-z(])|(?<=\s)Repeal\s*[.,]\s+(?=[a-z])")
VERB = re.compile(r"\b(?:is|are|shall\s+(?:stand|be)|stands?|hereby\s+stands?)\s+(?:hereby\s+)?(?:stand\s+)?repealed\b"
                  r"|\bhereby\s+repeals?\b", re.I)
KIND = r"(?:Act|Ord[a-z]{4,7}|Order|Regulations?|Rules|Code)"
REF = re.compile(rf"\b({KIND})\b\.?\s*,?\s*({YEAR})(?:\d{{1,2}}(?!\d))?(?!\d)")
TOKEN = re.compile(r"\([^()]{0,90}\)|[A-Za-z][A-Za-z'’\-]*\.?|&")
SMALL = {"of", "and", "the", "for", "to", "in", "on", "by", "&", "or", "with", "from", "at"}
GENERIC = {"provisions", "provision", "whole", "words", "following", "said", "same", "enactments", "enactment", "laws", "law", "rest",
           "remaining", "schedule", "entries", "repealed", "existing", "principal", "aforesaid", "relevant", "corresponding"}
NUMBER_NOTE = re.compile(r"\s*,?\s*\(([^()]{0,80})\)")
NUMBER_IN = re.compile(rf"(?:No\.?\s*,?\s*)?\b([IVXLCDMl1]{{1,9}}|\d{{1,3}})\s*(?:of|0f)\s*({YEAR})", re.I)
NUMBER_ONLY = re.compile(rf"\b(Act|Ordinance|Ord\.|Regulation|P\.\s?O\.|President'?s\s+Order|Order)\s+(?:No\.?\s*)?([IVXLCDM]{{1,9}}|\d{{1,3}})\s+of\s+({YEAR})")
ABBREVIATIONS = {"no", "nos", "ord", "sind", "art", "arts", "sec", "secs", "ss", "vol", "pt", "cl", "sch", "govt", "dr", "mr",
                 "st", "viz", "ibid", "pres", "reg", "regs", "notfn", "notif"}

PARTIAL_BEFORE = re.compile(r"\b(?:sub-?\s?sections?|sections?|ss?\.|clauses?|sub-?clauses?|articles?|rules?|regulations?|paragraphs?|"
                            r"chapters?|parts?|schedules?|entries|entry|provisos?|explanations?|provisions(?!\s+of\s+(?:the\s+)?$)|the\s+words|items?)\b"
                            r"[^;]{0,220}?\bof\s+(?:the\s+)?$|\bIn\s+the\s+$"
                            # "the following enactments and the amendments made thereby in the Land Acquisition Act are hereby repealed"
                            r"|\bamendments?\s+(?:\w+\s+){0,3}?(?:made|effected)\s+(?:\w+\s+){0,2}?(?:in|to)\s+(?:the\s+)?$", re.I)
# provisions named after the law, before the verb: "In the X Act, 1990, sections 3 and 4 are hereby repealed"
PARTIAL_AFTER = re.compile(r"\b(?:sub-?\s?sections?|sections?|clauses?|sub-?clauses?|articles?|paragraphs?|chapters?|schedules?|"
                           r"entries|provisos?)\s*[\dIVXLC(]", re.I)
TERRITORY = re.compile(r"(?:to\s+the\s+extent\s+of|in\s+(?:its|their)\s+application\s+to|in\s+relation\s+to|for)\s+(?:the\s+)?"
                       r"(Islamabad(?:\s+Capital\s+Territory)?|Federal\s+Capital(?:\s+Area)?|Province(?:\s+of\s+(?:the\s+)?[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)?|"
                       r"Punjab|Sindh?|Bal[ou]chistan|Khyber\s+Pakhtunkhwa|North-West\s+Frontier\s+Province|West\s+Pakistan|"
                       r"(?:Provincially\s+Administered\s+)?Tribal\s+Areas|Karachi)", re.I)
LIMITS = re.compile(r"to\s+the\s+extent|to\s+(?:that|such)\s+extent|so\s+f[ao]r\s+as|in\s+so\s+f[ao]r|insofar|in\s+(?:its|their)\s+application|\bexcept\b|other\s+than|"
                    r"as\s+applicable\s+to|in\s+relation\s+to|with\s+respect\s+to|cease\s+to\s+have\s+application|as\s+relates\s+to", re.I)
CONDITION = re.compile(r"immediately\s+after\s+the|(?:after|upon|on)\s+(?:the\s+)?dissolution|on\s+such\s+date\s+as|from\s+such\s+date\s+as|"
                       r"in\s+an\s+area\s+in\s+which|as\s+and\s+when|with\s+effect\s+from\s+such|on\s+the\s+date\s+notified", re.I)
SCHEDULED = re.compile(r"(?:specified|mentioned|set\s+out|enumerated|listed|contained)\s+in\s+(?:column\s+\w+\s+of\s+)?"
                       r"(?:Part\s+[IVX\d]+\s+of\s+)?the\s+(?:\w+\s+)?Schedule", re.I)
# "The following laws are hereby repealed" -- not "the rules made under the following notifications are hereby repealed"
LIST_INTRO = re.compile(r"\bfollowing\s+(?:[\w\-]+\s+){0,3}?(?:laws?|enactments?|Acts?|Ordinances?|Orders?|Regulations?|Rules|statutes?|"
                        r"instruments?)\b", re.I)
NOT_A_LIST_OF_LAWS = re.compile(r"made\s+under|issued\s+under|notifications?", re.I)
# an Ordinance's lapse, described: "the date the X Ordinance stands repealed under Article 128 of the Constitution"
LAPSE = re.compile(r"^\W*under\s+(?:clause\s+\(\w+\)\s+of\s+)?Article\s+\d+", re.I)
# a law that comes into force only when the government notifies it may never have repealed anything
DEFERRED = re.compile(r"come\s+into\s+force\s+on\s+such\s+\W?(?:date|day)s?(?:\s+or\s+dates)?\s+as\s+(?:the\s+(?:Federal\s+|Provincial\s+)?"
                      r"Government|it|he|the\s+President|the\s+Governor)\s+may", re.I)
PERMANENT_ORDINANCE_YEARS = set(range(1977, 1986)) | set(range(1999, 2004))   # Articles 270A, 270AA (finalize_metadata)
SUBORDINATE = ("Rules", "Notification", "Other")
PRIMARY = ("Act", "Ordinance", "Order")
PROVINCE_PREFIX = re.compile(r"^(?:west pakistan|punjab|sindh|khyber pakhtunkhwa|pakhtunkhwa|balochistan|indian|pakistan)\s+")
LIST_MARK = re.compile(r";|\.\s(?=\()|\(\s*(?:[ivxlcdm]{1,5}|[a-z]{1,2}|\d{1,3})\s*\)\s", re.I)
# references that only qualify the clause: "Subject to the provisions of section 4 of the General Clauses Act"
NOT_THE_SUBJECT = re.compile(r"\b(?:subject\s+to|under|notwithstanding|conferred\s+by|made\s+under|referred\s+to\s+in|issued\s+under|"
                             r"framed\s+under|in\s+exercise\s+of|by\s+virtue\s+of|as\s+amended\s+by|as\s+defined\s+in|within\s+the\s+meaning\s+of|"
                             # "On the commencement of this Act and the X Act, 2023, the B Ordinance shall stand repealed": X is not
                             # repealed; but "On coming into force of these Regulations, the X Regulations shall stand repealed": X is
                             r"commencement\s+of|coming\s+into\s+(?:force|operation)\s+of|enactment\s+of|promulgation\s+of)\b"
                             r"[^,;]{0,120}$"
                             # a clause that only supposes the repeal: "when the Defence of Pakistan Ordinance stands repealed",
                             # "from the date the X Ordinance stands repealed"
                             r"|\b(?:when|where|if|until|unless|after|before|once|whether|wherever)\s+(?:the\s+)?$"
                             r"|\b(?:date|day)\s+(?:on\s+which\s+)?(?:the\s+)?$"
                             # a law's own short title, and definitions: "This Act may be called the X Act", "'Agency' means ..."
                             r"|\bmay\s+be\s+(?:called|cited)\s+(?:as\s+)?(?:the\s+)?$|\bmeans\b[^,;]{0,80}$", re.I)
# a repeal clause quoted inside an amending law ("for section 72 the following shall be substituted: '72. Repeal ...'")
QUOTED_PROVISION = re.compile(r"shall\s+be\s+(?:substituted|inserted|added)|following\s+(?:new\s+)?(?:section|sub-section|clause)", re.I)
# the next section: its number right after a full stop ("... repealed. 58. Savings"), not "sections 1, 3 and 36. (2)"
# (and the next law printed in the same gazette: "... (W. P. Ordinance I of 1961). ACT No. XXIX OF 1977 An Act to provide ...")
NEXT_SECTION = re.compile(r"(?:(?<=[.;:])|(?<=[.;:]\s))\d{1,3}[A-Z]?\.\s*(?:\(\d+\)\s*)?[A-Z\[]|\bNotwithstanding\b|\bSCHEDULE\b"
                          r"|\b(?:ACT|ORDINANCE|Act|Ordinance)\s+No\.?\s*[IVXLCDM\d]+\s+(?:OF|of)\s+(?:1[89]|20)\d{2}\b|\bAN?\s+(?:ACT|ORDINANCE)\b|"
                          r"\bAn\s+(?:Act|Ordinance)\s+(?:to|further)\b|\bShort\s+title\b|\bmay\s+be\s+called\b")
WHOLE_CELL = re.compile(r"^\W{0,4}(?:the\s+)?whole\b(?!\s+of\b)(?!\W{0,3}except)|^\W{0,4}so\s+much\s+as\s+(?:has\s+not\s+been\s+repealed|is\s+unrepealed|remains)", re.I)


def roman(value: str) -> int | None:
    value = value.strip()
    if value.isdigit():
        return int(value)
    value = value.upper().replace("L1", "LI").replace("1", "I") if re.fullmatch(r"[IVXLCDMl1]+", value) else value.upper()
    value = value.replace("l", "I")
    if not re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", value) or not value:
        return None
    numbers = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(value):
        n = numbers[ch]
        total += -n if i + 1 < len(value) and numbers[value[i + 1]] > n else n
    return total


def law_number(value) -> tuple[int, int] | None:
    """(number, year) from "XIV of 2015", "Act No. XIV of 2015", "(W.P. Ord. No. XX of 1961)"."""
    found = NUMBER_IN.search(str(value or ""))
    if not found:
        return None
    number = roman(found.group(1))
    return (number, int(found.group(2))) if number else None


# a marginal note inside the verb itself (Sindh code): "is hereby Repeal and savings repealed", "shall Repeal and stand repealed"
MARGIN_IN_VERB = re.compile(r"\b(is|are|hereby|shall)\s+Repeals?\b[^;:]{0,30}?\s+(?=(?:hereby\s+)?(?:stand\s+|be\s+)?repealed)")
SCANNED_YEAR = re.compile(r"\b[lI1][89lO][\dlO]{2}\b")


def clean(text: str) -> str:
    """The text as the clause reader wants it: OCR slips in the repealing words mended, marginal notes
    taken out of the sentence, words split at the line end joined, spaces collapsed."""
    for pattern, repl in OCR_FIXES:
        text = pattern.sub(repl, text)
    text = SCANNED_YEAR.sub(lambda m: m.group(0).replace("l", "1").replace("I", "1").replace("O", "0")
                            if re.search(r"[lIO]", m.group(0)) else m.group(0), text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z])- (?=[a-z])", r"\1", text)            # "Adver- tisements"
    text = re.sub(r"(?<=\d-[A-Z])of\b|(?<=\d[A-Z])of\b|(?<=\d)of\b", " of", text)   # "section l5-Aof the", "(XXI of2005)"
    # the codes' footnote marks: "2[Balochistan] Finance Act", "the West Pakistan7 Finance Ordinance"
    text = re.sub(r"\b\d{1,2}\[|[\[\]]", "", text)
    text = re.sub(r"(?<=[a-z])\d{1,2}(?=[\s,])", "", text)
    text = MARGIN_IN_VERB.sub(r"\1 ", text)
    return MARGIN_NOTE.sub("", text)


def clause_start(text: str, end: int, limit: int = 1000) -> int:
    """Where the sentence holding position `end` begins: the last full stop, semicolon or colon
    outside brackets that is not an abbreviation ("No.", "W.P.", "Ord.")."""
    unmatched, low = [], max(0, end - limit)     # positions of ")" whose "(" is still to come
    for i in range(end - 1, low, -1):
        ch = text[i]
        # a section's number after a full stop ("... in force. 152. Repeal of ...") begins a new sentence, whatever
        # brackets an OCR slip left open ("Ordinance X)O I of 1960")
        if ch == "." and text[i + 1:i + 2] == " " and re.search(r"[.;:]\s+\d{1,3}[A-Z]?$", text[max(0, i - 10):i]):
            return i + 1
        if unmatched and unmatched[0] - i > 90:
            unmatched.clear()  # a bracket the scan lost: stop treating what precedes it as inside it
        if ch == ")":
            # a law number whose "(" the scan dropped ("Ordinance, 1961 CXVIII of 1961)") opens nothing
            if not (re.search(rf"[IVXLCDM\d]+\s+of\s*{YEAR}$", text[max(0, i - 25):i]) and "(" not in text[max(0, i - 30):i]):
                unmatched.append(i)
        elif ch == "(":
            if unmatched:
                unmatched.pop()
        elif not unmatched and ch in ".;:" and text[i + 1:i + 2].isspace():
            before = re.search(r"([A-Za-z]+)$", text[max(0, i - 12):i])
            word = before.group(1) if before else ""
            if word.lower() in ABBREVIATIONS or len(word) == 1:
                continue
            return i + 1
    return low


def title_before(text: str, pos: int) -> tuple[str, int]:
    """The law's name printed just before its kind word at `pos` ("... Ordinance, 1966"), and where it begins."""
    left = text[max(0, pos - 400):pos]
    base = max(0, pos - 400)
    tokens = list(TOKEN.finditer(left))
    kept, edge = [], len(left)
    for position in range(len(tokens) - 1, -1, -1):
        token = tokens[position]
        gap = left[token.end():edge]
        if gap.strip(" ,") or (gap.count(",") and not token.group(0)[:1].isupper()):
            break
        # a comma inside a name is followed by another of its capitalised words ("Generation, Transmission and ..."),
        # one between names by "the" or "and" ("these Regulations, the Competition ... Regulations"; "relates to the
        # Chairman, and the Chairman and Speaker ... Ordinance")
        if gap.count(",") and kept and (kept[-1].group(0).lower() in SMALL or not kept[-1].group(0)[:1].isupper()):
            break
        word = token.group(0)
        if word.startswith("("):
            # a law number ends the name before it; "(f)", "(2)", "(xvi)" are list marks, not part of a name
            if re.search(rf"\bof\s+{YEAR}\b|\bNo\b", word) or re.fullmatch(r"\(\s*(?:[ivxlcdm]{1,5}|[a-z]{1,2}|\d{1,3})\s*\)", word, re.I):
                break
        elif not (word[:1].isupper() or word.lower() in SMALL):
            # a word of the name printed without its capital: right after "The" ("The salary of Judges ... Order"),
            # or between two capitalised words ("Tibb and Homeopathic employees (Regulation of Services) Act") --
            # but never "The provisions of", "The whole of", nor more than two of them
            previous = tokens[position - 1] if position else None
            after_the = previous and previous.group(0) == "The" and not left[previous.end():token.start()].strip()
            between = previous and previous.group(0)[:1].isupper() and kept and (kept[-1].group(0)[:1].isupper() or kept[-1].group(0).startswith("("))
            lowered = sum(1 for t in kept if t.group(0)[:1].islower() and t.group(0).lower() not in SMALL)
            if word.lower() in GENERIC or lowered >= 2 or not (after_the or between):
                break
        kept.append(token)
        edge = token.start()
        if len(kept) >= 32:
            break
    kept.reverse()   # now in reading order
    # a table row prints the law's number in front of its name ("1918 X The Usurious Loans Act"): the name
    # starts at its own "The"; and the small words caught in front of a name are not part of it
    capital_the = [i for i, t in enumerate(kept) if t.group(0) == "The"]
    if capital_the and capital_the[0] > 0:
        kept = kept[capital_the[0]:]
    while kept and (re.fullmatch(r"[IVXLCDM]+\.?|[A-Za-z]\.?", kept[0].group(0)) or kept[0].group(0).lower() in SMALL - {"the"}):
        kept = kept[1:]
    if not kept or not any(t.group(0)[:1].isupper() and t.group(0).lower() != "the" for t in kept):
        return "", pos
    start = kept[0].start()
    return re.sub(r"\s+", " ", left[start:]).strip(" ,"), base + start


def references(text: str) -> list[dict]:
    """Laws named in `text`: {"title", "year", "kind", "start", "end", "number"}."""
    found = []
    for match in REF.finditer(text):
        name, start = title_before(text, match.start())
        if not name:
            continue
        kind = match.group(1)
        kind = "Ordinance" if kind.lower().startswith("ord") and kind.lower() != "order" else kind
        end = match.end()
        note = NUMBER_NOTE.match(text, end)
        number, prefix = None, None
        if note:
            number = law_number(note.group(1))
            prefix = jurisdiction_named(note.group(1))
            end = note.end()
        found.append({"title": f"{name} {kind}, {match.group(2)}", "year": int(match.group(2)), "kind": kind,
                      "start": start, "end": end, "number": number, "prefix": prefix,
                      # a leading "the" belongs with the words before the name when reading them ("... made thereby in the")
                      "lead": name[:4] if name[:4].lower() == "the " else "", "loose": False})
    # the codes, whose names do not end in their kind and are often cited without a year
    taken = [(r["start"], r["end"]) for r in found]
    for match in CODES.finditer(text):
        if any(start <= match.start() < end or match.start() <= start < match.end() for start, end in taken):
            continue
        name = re.sub(r"\s+", " ", match.group(1))
        key = next(k for k in CODE_YEARS if k in name.lower())
        year = int(match.group(3)) if match.group(3) else CODE_YEARS[key]
        title = CODE_TITLES[key] + ("" if key == "penal" else f", {year}")
        end = match.end()
        note = NUMBER_NOTE.match(text, end)
        if note:
            end = note.end()
        # the name starts at its "the", as the other names do (so "Sections 86 and 87 of the" reads as before it)
        start = match.start() - 4 if text[max(0, match.start() - 4):match.start()].lower() == "the " else match.start()
        found.append({"title": title, "year": year, "kind": "Act" if key != "qanun" else "Order", "start": start, "end": end,
                      "number": law_number(note.group(1)) if note else None, "prefix": None,
                      "lead": "the " if start < match.start() else "",
                      # "the Penal Code" in passing, without its year or number, may name it but never repeals it
                      "loose": not (match.group(2) or match.group(3) or note)})
    return sorted(found, key=lambda r: r["start"])


# a code's name -- but not inside an amending law's ("The Code of Civil Procedure (Punjab Amendment) Ordinance, 2020")
CODES = re.compile(rf"\b(Code\s+of\s+Civil\s+Procedure|Code\s+of\s+Criminal\s+Procedure|(?:Pakistan\s+)?Penal\s+Code|Qanun-e-Shahadat(?:\s+Order)?)"
                   rf"(?!\s*\((?:[^()]*\s)?Amendment|\s+Amendment|\s*(?:\([^()]*\))?\s*,?\s*(?:Act|Ordinance)\b)"
                   rf"(?:\s*,?\s*(\(?(?:Act\s+)?(?:No\.?\s*)?\b(?:V|XLV|X)\b\s+of\s+(?:1908|1898|1860|1984)\)?))?(?:\s*,?\s*({YEAR}))?", re.I)
CODE_YEARS = {"civil": 1908, "criminal": 1898, "penal": 1860, "qanun": 1984}
CODE_TITLES = {"civil": "The Code of Civil Procedure", "criminal": "The Code of Criminal Procedure", "penal": "The Pakistan Penal Code",
               "qanun": "The Qanun-e-Shahadat Order"}


JURISDICTION_WORDS = [(re.compile(r"\bW\.?\s?P\.|\bWest\s+Pakistan", re.I), "West Pakistan"),
                      (re.compile(r"\bPunjab\b", re.I), "Punjab"), (re.compile(r"\bSindh?\b", re.I), "Sindh"),
                      (re.compile(r"N\.?\s?-?\s?W\.?\s?F\.?\s?P|Khyber|Pakhtunkhwa", re.I), "Khyber Pakhtunkhwa"),
                      (re.compile(r"Bal[ou]chistan|\bBaln\b", re.I), "Balochistan"),
                      (re.compile(r"Bahawalpur|Khairpur|Bombay|Bengal|Madras|Oudh|Burma", re.I), "elsewhere")]


def jurisdiction_named(note: str) -> str | None:
    """The statute book a law number belongs to, when its note says ("W.P. Ordinance No. IV of 1963", "Sind Act XV of 1932")."""
    for pattern, jurisdiction in JURISDICTION_WORDS:
        if pattern.search(note):
            return jurisdiction
    return None


def prefix_fits(prefix: str | None, candidate: dict, repealer: dict) -> bool:
    """Can a law numbered in `prefix`'s statute book be `candidate`?"""
    if prefix is None:
        # an unmarked number is the repealing legislature's own series
        return candidate["jurisdiction"] == repealer["jurisdiction"] or candidate["source"] == repealer["source"] != "drs"
    if prefix == "elsewhere":
        return False
    if prefix == "West Pakistan":
        return candidate["jurisdiction"] == "West Pakistan" or candidate["source"] in ("punjab", "kp", "balochistan", "sindh")
    return candidate["jurisdiction"] == prefix or (candidate["source"] == repealer["source"] and candidate["source"] != "drs"
                                                   and candidate["jurisdiction"] in (prefix, "West Pakistan"))


# ---- the graph ----------------------------------------------------------------------------------

class Graph:
    def __init__(self, records, same_title, norm_title, title_year):
        self.records, self.same_title, self.norm_title = records, same_title, norm_title
        self.by_year = defaultdict(list)
        self.by_number = defaultdict(list)
        for record in records:
            for title in {record["title"], record["name"]}:
                year = title_year(title) or record["year"]
                if year:
                    self.by_year[year].append((norm_title(title), record))
            number = law_number(record.get("number"))
            if number:
                self.by_number[number].append(record)
        self.edges, self.conflicts = [], []

    # which laws a reference names
    def resolve(self, ref: dict, repealer: dict, reach: bool = True) -> list[tuple[dict, str]]:
        """[(record, how)] for the corpus copies `ref` names that `repealer` could reach (any copy, with reach=False).
        how: title | number."""
        wanted = self.norm_title(ref["title"])
        found = {}
        for candidate_norm, candidate in self.by_year.get(ref["year"], []):
            if id(candidate) not in found and (self.same_title(wanted, candidate_norm) or self.adapted_name(wanted, candidate_norm)):
                found[id(candidate)] = (candidate, "title")
        if not found and ref.get("number"):
            # the name misprinted or garbled, but its number given: the number has to agree, and most of the
            # law's own name has to be there
            for candidate in self.by_number.get(ref["number"], []):
                if candidate["year"] == ref["year"] and kind_matches(ref["kind"], candidate["instrument_type"]) \
                        and name_covered(candidate["_norm"], wanted) >= 0.7 and prefix_fits(ref.get("prefix"), candidate, repealer):
                    found[id(candidate)] = (candidate, "number")
        if not found and reach:
            # one ordinary word more in the clause than in the law's own name ("The Balochistan Civil Service (Executive
            # Branch) Service Rules, 1984" for "... (Executive Branch) Rules, 1984"): the only law of that kind and year the
            # repealer can reach whose name it is but for that word -- read, never verified (standing)
            # (words counted, not just seen: here the added word is one the name already has)
            mine = Counter(re.findall(r"[a-z]+|\d+", wanted))
            near = []
            for candidate_norm, candidate in self.by_year.get(ref["year"], []):
                theirs = Counter(re.findall(r"[a-z]+|\d+", candidate_norm))
                extra = mine - theirs
                if len(theirs) >= 4 and not theirs - mine and sum(extra.values()) == 1 and not set(extra) & NEAR_NAME_QUALIFIERS \
                        and candidate is not repealer and kind_matches(ref["kind"], candidate["instrument_type"]) \
                        and self.reachable(repealer, candidate):
                    near.append(candidate)
            if len({c.get("work_id") or id(c) for c in near}) == 1:
                found.update({id(c): (c, "near_name") for c in near})
        if not reach:
            return [(c, how) for c, how in found.values() if c is not repealer and c.get("work_id") != repealer.get("work_id")]
        return [(c, how) for c, how in found.values() if self.reachable(repealer, c) or self.named_in_its_book(ref, c, repealer)]

    @staticmethod
    def named_in_its_book(ref: dict, candidate: dict, repealer: dict) -> bool:
        """A clause that cites the law by its own province's number ("the Defence Housing Authority Lahore Ordinance,
        1999 (Punjab Ordinance LI of 1999)") reaches that province's copy, whoever made the clause."""
        province = PROVINCE_OF.get(candidate["source"])
        # only a law that could end it there: the Federation's, West Pakistan's, or that province's own -- a Balochistan
        # law repealing the "Town Improvement Act, 1922 (Punjab Act IV of 1922)" it inherited ends Balochistan's copy only
        able = repealer["source"] in FEDERAL_SOURCES or repealer["jurisdiction"] in ("West Pakistan", province) \
            or PROVINCE_OF.get(repealer["source"]) == province
        same_number = bool(ref.get("number")) and ref["number"] == law_number(candidate.get("number")) and candidate is not repealer
        if able and province and ref.get("prefix") == province and same_number:
            return True
        # a federal clause citing the federal series ("The Baluchistan Constabulary Act, 1975 (XLI of 1975)") reaches a
        # province's copy carrying that very number, whatever the province labelled it (read, never verified: standing)
        return bool(province and same_number and ref.get("prefix") is None and repealer["source"] in FEDERAL_SOURCES
                    and repealer["jurisdiction"] == "Federal")

    def beyond_reach(self, ref: dict, repealer: dict) -> list[dict]:
        """Copies of the named law that `repealer` cannot end: they still get a note when it repeals them where it can."""
        reached = {id(c) for c, _ in self.resolve(ref, repealer)}
        return [c for c, _ in self.resolve(ref, repealer, reach=False) if id(c) not in reached]

    def adapted_name(self, a: str, b: str) -> bool:
        """The same law under a name the adaptation orders changed: "The West Pakistan Money-lenders Ordinance" became
        "The Punjab Money-Lenders Ordinance"; "The Indian Medical Degrees Act" became "The Medical Degrees Act"."""
        pa, pb = PROVINCE_PREFIX.match(a), PROVINCE_PREFIX.match(b)
        if not (pa or pb):
            return False
        if pa and pb and pa.group(0) != pb.group(0) and "west pakistan " not in (pa.group(0), pb.group(0)):
            return False      # the Sindh X Act is not the Punjab X Act
        return self.same_title(PROVINCE_PREFIX.sub("", a), PROVINCE_PREFIX.sub("", b))

    @staticmethod
    def reachable(repealer: dict, candidate: dict) -> bool:
        """Whose copies a law can repeal: its own legislature's. A law on a federal site reaches the federal copies;
        one on a province's code reaches that code's copies -- even a federal or West Pakistan law hosted there,
        since its federal twin, where the corpus has it, speaks for the Federation itself."""
        if candidate is repealer or candidate.get("work_id") == repealer.get("work_id") or candidate["_norm"] == repealer["_norm"]:
            return False
        source, jurisdiction = repealer["source"], repealer["jurisdiction"]
        if source in FEDERAL_SOURCES:
            if jurisdiction in ("Federal", "West Pakistan"):
                if candidate["source"] in PROVINCE_OF and candidate["jurisdiction"] == "Federal" and jurisdiction == "Federal":
                    # a province's copy of a federal law, repealed by the Federation: whether it ends there too depends on
                    # whose subject the law is (since 2010 a federal repeal of a devolved subject reaches only Islamabad) --
                    # such edges are read, never verified (see standing), so the province's listing meets them as a conflict
                    return True
                return candidate["source"] == "pakistancode" or (candidate["source"] == "drs" and candidate["jurisdiction"] in ("Federal", jurisdiction))
            return candidate["source"] == "drs" and candidate["jurisdiction"] == jurisdiction
        return candidate["source"] == source or (candidate["source"] == "drs" and candidate["jurisdiction"] == jurisdiction
                                                 and jurisdiction not in ("Federal", "West Pakistan"))

    @staticmethod
    def acceptable(repealer: dict, target: dict) -> bool:
        # R6: never by an earlier law
        if repealer["year"] and target["year"] and repealer["year"] < target["year"]:
            return False
        # rules, notifications and manuals (the Esta Code reprints other laws' repeal sections) do not repeal Acts
        kind = repealer["instrument_type"]
        subordinate = kind in SUBORDINATE or (kind == "Regulation" and re.search(r"\bRegulations\b", repealer["title"]))
        if subordinate and (target["instrument_type"] in PRIMARY or (target["instrument_type"] == "Regulation"
                                                                     and not re.search(r"\bRegulations\b", target["title"]))):
            return False
        return True

    # repeal clauses
    def repeal_edges(self, record: dict) -> list[dict]:
        raw = record["_doc"]["text"]
        if not re.search(r"(?i)r[ce]p[ce]a[l1]", raw):
            return []
        text = clean(raw)
        out, seen_clauses = [], set()
        provincial = record["jurisdiction"] != "Federal"
        for verb in VERB.finditer(text):
            start = clause_start(text, verb.start())
            key = text[start:verb.end()][-160:]
            if key in seen_clauses:
                continue
            seen_clauses.add(key)
            # a repeal clause an amending law writes into another law ("for section 72 the following shall be
            # substituted: '72. Repeal ...'") is that law's, not this one's
            lead_in = text[max(0, start - 300):start + 3]
            quoted = QUOTED_PROVISION.search(lead_in)
            if quoted and (record["is_amendment"] or re.search(r"[“\"‘']", lead_in[quoted.end():])):
                continue
            if record["is_amendment"] and QUOTED_PROVISION.search(text[start:verb.start()]):
                continue
            rest = text[verb.end():]
            stop = re.search(r"[.;](?:\s|$)", rest[:600])
            tail = rest[:stop.start()] if stop else rest[:250]
            if LAPSE.match(tail):
                continue   # "stands repealed under Article 128 of the Constitution": an Ordinance's lapse, not this law's repeal
            out += self.read_clause(record, text, start, verb, tail, provincial)
        return out

    def read_clause(self, record, text, start, verb, tail, provincial):
        subject = text[start:verb.start()]
        words = (subject + verb.group(0) + tail).strip()
        edges = []
        if SCHEDULED.search(subject + " " + tail):
            return self.schedule_edges(record, text, words, tail)
        listed = bool(LIST_INTRO.search(subject)) and not NOT_A_LIST_OF_LAWS.search(subject) and not references(subject)
        if listed:
            region = text[verb.end():verb.end() + 4000]
            follow = NEXT_SECTION.search(region, 20)
            region = region[:follow.start() if follow else 3000]
            refs = references(region)
            items = []
            for i, ref in enumerate(refs):
                # each item on its own: from the last list mark ("(2)", "(b)", ";") before the name, to the next one after it
                raw_before = region[(refs[i - 1]["end"] if i else 0):ref["start"]]
                marks = list(LIST_MARK.finditer(raw_before))
                before = raw_before[marks[-1].end():] if marks else raw_before
                # the list is over when an item is not introduced by a list mark, or runs on (the next law in a gazette)
                if len(before) > 120 or (i and not marks and not re.search(r",|\band\b", raw_before)):
                    break
                own = region[ref["end"]:ref["end"] + 200]
                stop = LIST_MARK.search(own)
                items.append((ref, before, own[:stop.start()] if stop else own))
            refs = refs[:len(items)]
            # what limits every item: the words introducing the list ("The following Rules in their application to the
            # Government servants ... are hereby repealed") and those between the verb and the list
            first = LIST_MARK.search(tail)
            qualifier = subject + " " + tail[:min(first.start() if first else len(tail), refs[0]["start"] if refs else len(tail))]
        else:
            refs = references(subject)
            items = [(ref, subject[(refs[i - 1]["end"] if i else 0):ref["start"]], "") for i, ref in enumerate(refs)]
            qualifier = (subject[refs[-1]["end"]:] if refs else "") + " " + tail
        resolved_any, skipped = False, []
        for i, (ref, before, own_tail) in enumerate(items):
            before = before + ref["lead"]
            if ref["loose"] or NOT_THE_SUBJECT.search(before):
                skipped.append(ref)
                continue
            # what follows the name, up to the next name or the verb: "the X Act, 1990, sections 3 and 4", ", in so
            # far as it relates to the Chairman, and" -- unless it leads into the next name ("..., sub-sections (2) to
            # (6) of section 10 of the Lahore Development Authority Act")
            after = (subject[ref["end"]:(refs[i + 1]["start"] if i + 1 < len(refs) else len(subject))] if not listed else own_tail)
            own_words = "" if PARTIAL_BEFORE.search(after) else after
            if PARTIAL_AFTER.search(own_words):
                extent, territory = "part", None
            else:
                extent, territory = extent_of(before, own_words + " " + qualifier + " " + own_tail, record, provincial)
            for target, how in self.resolve(ref, record):
                if not self.acceptable(record, target):
                    continue
                resolved_any = True
                if record["source"] in FEDERAL_SOURCES and target["source"] in PROVINCE_OF \
                        and not (self.named_in_its_book(ref, target, record) and ref.get("prefix")):
                    how = "federal_to_province"
                edges.append(edge("repeals", record, target, words, extent=extent, territory=territory, how=how,
                                  form="list" if listed else "clause"))
            # a province ending, for itself, a law the Federation also carries: the federal copy is noted as no longer
            # in force there (its status elsewhere is unchanged)
            if extent == "whole" and record["source"] in PROVINCE_OF:
                for target in self.beyond_reach(ref, record):
                    if target["source"] == "pakistancode" and self.acceptable(record, target):
                        edges.append(edge("repeals", record, target, words, extent="territory", territory=PROVINCE_OF[record["source"]],
                                          how="province", form="list" if listed else "clause"))
        if not resolved_any and not listed:
            # "... the Sarhad Development Authority established under the X Act, 1972, shall stand dissolved and as a
            # corollary thereof the said Act shall stand repealed": the law named earlier in the sentence
            said = re.search(r"\bthe\s+said\s+(Act|Ordinance|Order|Regulation)\s*,?\s*$", subject)
            if said and skipped:
                ref = next((r for r in reversed(skipped) if kind_matches(said.group(1), r["kind"]) or r["kind"] == said.group(1)), None)
                if ref:
                    extent, territory = extent_of("", tail, record, provincial)
                    for target, how in self.resolve(ref, record):
                        if self.acceptable(record, target):
                            resolved_any = True
                            edges.append(edge("repeals", record, target, words, extent=extent, territory=territory, how="said", form="clause"))
        if not resolved_any and not listed:
            # a name too broken to read, or printed without its year ("The Balochistan Kachhi Abadis (Regularization and
            # Development) (Act No. II of 1987), is hereby repealed"): the law's number, with most of its name before it
            for match in NUMBER_ONLY.finditer(subject):
                number = (roman(match.group(2)), int(match.group(3)))
                head = subject[:match.start()]
                if not number[0] or NOT_THE_SUBJECT.search(head):
                    continue
                name = self.norm_title(head[-200:])
                prefix = jurisdiction_named(head[-25:] + match.group(0))
                # provisions named before the law ("Sections 86 and 87 of the Code of Civil Procedure, 1908 (Act V of 1908)")
                marks = list(LIST_MARK.finditer(head[-200:]))
                lead = head[-200:][marks[-1].end():] if marks else head[-200:]
                # ... or after it ("In the Pakistan Penal Code (Act XLV of 1860),-- (a) section 366 ... shall stand repealed")
                follows = subject[match.end():match.end() + 300]
                if PARTIAL_AFTER.search(lead) or PARTIAL_BEFORE.search(lead) or (PARTIAL_AFTER.search(follows) and not PARTIAL_BEFORE.search(follows)):
                    extent, territory = "part", None
                else:
                    extent, territory = extent_of(head[-200:], subject[match.end():] + " " + tail, record, provincial)
                for target in self.by_number.get(number, []):
                    if kind_matches(match.group(1), target["instrument_type"]) and name_covered(target["_norm"], name) >= 0.7 \
                            and prefix_fits(prefix, target, record) and self.reachable(record, target) and self.acceptable(record, target):
                        edges.append(edge("repeals", record, target, words, extent=extent, territory=territory, how="number", form="clause"))
        return edges

    def schedule_edges(self, record, text, words, tail):
        """'The enactments specified in the Schedule are hereby repealed to the extent mentioned in the fourth
        column': each law listed in the Schedule, with the extent printed beside it."""
        condition = "conditional" if CONDITION.search(words) else None
        # a scanned table's columns run into one another ("The Pakistan Penal Code | The whole. Act, 1025" is the row
        # of an amending Act of 1925): a schedule is read only off a clean text
        if record.get("garbled_words", 1) > 0.05:
            return []
        heads = [m.start() for m in re.finditer(r"\bS\s?C\s?H\s?E\s?D\s?U\s?L\s?E\b|\bSchedule\b(?=\s*\(|\s*\[|\s*-)", text)]
        if not heads:
            return []
        start = heads[-1] if len(heads) == 1 else next((h for h in heads if re.search(r"(?i)repeal", text[h:h + 400])), heads[-1])
        region = text[start:start + 40000]
        edges = []
        for ref in references(region):
            if ref["loose"]:
                continue
            cell = re.sub(r"\s+", " ", region[ref["end"]:ref["end"] + 140])
            cell = re.sub(rf"^\W*(?:\(?[IVXLCDM]+\s+of\s+{YEAR}\)?\.?\s*)", "", cell)
            extent = condition or ("whole" if WHOLE_CELL.search(cell) else "part")
            for target, how in self.resolve(ref, record):
                if not self.acceptable(record, target):
                    continue
                edges.append(edge("repeals", record, target, f"{words[:200]} ... SCHEDULE: {ref['title']} {cell[:60]}",
                                  extent=extent, territory=None, how=how, form="schedule"))
        return edges

    def amend_edges(self, record: dict) -> list[dict]:
        """An amending law -> the law it amends: the law named in its first pages whose name is its own,
        without "(Amendment)"."""
        if not record["is_amendment"]:
            return []
        stem = re.sub(r"\bamendments?\b|\b(?:1[89]|20)\d{2}\b|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b|"
                      r"\b(?:punjab|sindh|west pakistan|khyber pakhtunkhwa|balochistan)\s+(?=amendment)", " ", self.norm_title(record["title"]))
        stem = re.sub(r"\s+", " ", stem).strip()
        text = record["_doc"]["text"][:8000]
        for ref in references(text):
            name = re.sub(r"\b(?:1[89]|20)\d{2}\b", " ", self.norm_title(ref["title"]))
            name = re.sub(r"\s+", " ", name).strip()
            if "amendment" in name or not name:
                continue
            if fuzz.ratio(stem, name) >= 88:
                targets = [t for t, _ in self.resolve(ref, record)]
                if targets:
                    words = re.sub(r"\s+", " ", text[max(0, ref["start"] - 80):ref["end"] + 80])
                    return [edge("amends", record, t, words) for t in targets]
        return []

    def made_under_edges(self, record: dict) -> list[dict]:
        if record["instrument_type"] in ("Act", "Ordinance"):
            return []
        text = record["_doc"]["text"][:5000]
        power = re.search(r"(?i)powers?\s+conferred\s+(?:up)?on\s+(?:it|him|them)\s+(?:by|under)|powers?\s+conferred\s+(?:by|under)|"
                          r"in\s+exercise\s+of\s+the\s+powers?", text)
        if not power:
            return []
        for ref in references(text[power.end():power.end() + 500]):
            targets = [t for t, _ in self.resolve(ref, record, reach=False)]
            # the parent is the copy in the same statute book when there is one
            own = [t for t in targets if t["source"] == record["source"]] or [t for t in targets if t["source"] in FEDERAL_SOURCES] or targets
            if own:
                words = re.sub(r"\s+", " ", text[power.start():power.end() + 300])
                return [edge("made_under", record, own[0], words)]
        return []

    def mentions(self, law: dict, target: dict) -> str | None:
        """The words where `law` names `target`, when it does."""
        text = law["_doc"]["text"]
        for ref in references(text):
            if any(t is target for t, _ in self.resolve(ref, law, reach=False)):
                return re.sub(r"\s+", " ", text[max(0, ref["start"] - 120):ref["end"] + 160])
        return None


def kind_matches(ref_kind: str, kind: str) -> bool:
    ref_kind = ref_kind.lower()
    if ref_kind.startswith("ord") and ref_kind != "order":
        return kind == "Ordinance"
    if ref_kind in ("p.o.", "p. o.", "order") or "order" in ref_kind:
        return kind == "Order"
    if ref_kind.startswith("regulation"):
        return kind == "Regulation"
    return kind.lower() == ref_kind.rstrip(".")


NAME_FILLER = {"the", "of", "and", "for", "act", "ordinance", "order", "rules", "regulation", "regulations", "code"}


def name_covered(name: str, text: str) -> float:
    """Share of the words of a law's name (`name`, normalised) found in `text` (normalised), spelling slips allowed."""
    wanted = {w for w in name.split() if len(w) > 2 and not w.isdigit() and w not in NAME_FILLER}
    present = set(text.split())
    if not wanted:
        return 0.0
    return sum(any(fuzz.ratio(w, v) >= 80 for v in present) for w in wanted) / len(wanted)


def extent_of(before: str, qualifier: str, repealer: dict, provincial: bool) -> tuple[str, str | None]:
    if PARTIAL_BEFORE.search(before):
        return "part", None
    territory = TERRITORY.search(qualifier)
    if territory:
        place = re.sub(r"\s+", " ", territory.group(1)).strip()
        # a province's repeal "in its application to the Province" is all it can repeal; and West Pakistan
        # was the whole of today's Pakistan
        own = (provincial and (place.lower().startswith("province") or place.lower().startswith(repealer["jurisdiction"].lower()[:5]))) \
            or "west pakistan" in place.lower()
        if own and not re.search(r"to\s+the\s+extent\s+of\s+(?:section|matter|the\s+matters)", qualifier, re.I):
            return "whole", None
        if place.lower().startswith("islamabad") or place.lower().startswith("federal capital"):
            return "territory", "Islamabad Capital Territory"
        if not re.search(r"to\s+the\s+extent\s+of\s+(?:section|matter|the\s+matters|any)", qualifier, re.I):
            return "territory", place
    if CONDITION.search(qualifier):
        return "conditional", None
    if LIMITS.search(qualifier):
        return "part", None
    return "whole", None


def edge(kind: str, source: dict, target: dict, words: str, **extra) -> dict:
    return {"type": kind, "from": source, "to": target, "words": re.sub(r"\s+", " ", words).strip()[-400:], **extra}


# ---- the rules ----------------------------------------------------------------------------------

def listed_current(record: dict) -> bool:
    return record["status"] == "in_force" and bool(record["status_basis"]) and "listed as current law on the" in record["status_basis"][0]


def unsettled(repealer: dict) -> str | None:
    """Why a law's repeals may never have taken effect, or None when they did."""
    if DEFERRED.search(repealer["_doc"]["text"][:6000]):
        return "the repealing law comes into force only on a date the government notifies"
    if re.search(r"\bBill\b", repealer["title"]) and not re.search(r"\bAct\b", repealer["title"]):
        return "the repealing document may be a Bill"
    if repealer["instrument_type"] == "Ordinance":
        year = repealer["year"] or 0
        # an Ordinance's repeals stand if it became permanent (before 1973; the periods of Articles 270A and
        # 270AA), was enacted as an Act, or is still current law
        if not (year < 1973 or year in PERMANENT_ORDINANCE_YEARS or repealer["status"] == "superseded"
                or listed_current(repealer) or repealer["jurisdiction"] == "West Pakistan"):
            return "the repealing Ordinance may have lapsed"
    return None


def standing(e: dict) -> tuple[str, str | None]:
    """How far the repeal `e` can be trusted: (verified | probable, why not verified)."""
    if e["form"] == "schedule":
        return "probable", "read off the repealing law's schedule"
    if e["how"] == "number":
        return "probable", "named only by its number and part of its name"
    if e["how"] == "said":
        return "probable", "named only as 'the said Act'"
    if e["how"] == "near_name":
        return "probable", "named with one word more than the law's own name"
    if e["how"] == "federal_to_province":
        return "probable", "a federal law repealing a province's copy of a federal law, which survives if the subject is the province's"
    why = unsettled(e["from"])
    return ("probable", why) if why else ("verified", None)


def apply_rules(graph: Graph) -> dict:
    changes = Counter()
    by_target = defaultdict(list)
    amended_by = defaultdict(list)
    for e in graph.edges:
        by_target[id(e["to"])].append(e)
        if e["type"] == "amends":
            amended_by[e["to"].get("work_id")].append(e)
    for record in graph.records:
        incoming = by_target.get(id(record), [])
        repeals = [e for e in incoming if e["type"] == "repeals"]
        whole = [e for e in repeals if e["extent"] == "whole"]
        if whole:
            # R8: a law amended (by a law named after it) after a law was made was not wholly repealed by it --
            # whatever a clause seemed to say. Unless the repealer came into force only later, by notification
            amended = last_amended(record, amended_by, own_only=True)
            contradicted = [e for e in whole if amended and e["from"]["year"] and amended > e["from"]["year"]
                            and not DEFERRED.search(e["from"]["_doc"]["text"][:6000])]
            for e in contradicted:
                graph.conflicts.append({"rule": "R8", "law": record["title"], "source": record["source"],
                                        "against": f"read as repealed by {e['from']['title']} ({e['from']['year']}), but amended in {amended}",
                                        "words": e["words"][-200:]})
                changes["R8 repeal contradicted by a later amendment"] += 1
            whole = [e for e in whole if e not in contradicted]
        exceptions, noted = [], set()
        for e in repeals:
            # one note per territory and repealing law (the corpus often holds two copies of the repealer)
            key = (e["territory"], re.sub(r"\W+", "", e["from"]["title"].lower()))
            if e["extent"] == "territory" and e["how"] != "federal_to_province" and key not in noted:
                noted.add(key)
                exceptions.append({"territory": e["territory"], "by": e["from"]["title"], "source": e["from"]["source"], "words": e["words"]})
        if exceptions:
            record["status_exceptions"] = exceptions
            record["related"] = record["related"] + [{"relation": "repealed_in", "territory": x["territory"], "title": x["by"],
                                                      "source": x["source"], "words": x["words"][-200:]} for x in exceptions[:3]]
            changes["R2 repealed for a territory only"] += 1
        # R10: repealed for the Islamabad Capital Territory and for each province, one by one -- nothing of it is left
        if record["jurisdiction"] == "Federal" and EVERY_TERRITORY <= {x["territory"] for x in exceptions} \
                and record["status"] not in ("repealed", "amending_instrument"):
            best = {}
            for e in repeals:
                if e["extent"] == "territory" and e["how"] != "federal_to_province" and e["territory"] in EVERY_TERRITORY:
                    if e["territory"] not in best or (standing(e)[0] == "verified" and standing(best[e["territory"]])[0] != "verified"):
                        best[e["territory"]] = e
            weak = [(place, standing(e)[1]) for place, e in best.items() if standing(e)[0] != "verified"]
            line = "repealed in every part of Pakistan, territory by territory: " + "; ".join(
                f"for {place} by {e['from']['title']} (\"{e['words'][-110:]}\")" for place, e in sorted(best.items()))
            record["_before_kg"] = {"status": record["status"], "basis": record["status_basis"][:1]}
            latest = max(best.values(), key=lambda e: e["from"]["year"] or 0)
            if not weak:
                record["status_basis"] = [line] + ([f"(the {record['source']} site still lists it: {record['status_basis'][0]})"]
                                                   if listed_current(record) else [])
                record.update(status="repealed", status_confidence="high")
                changes["R10 repealed in every territory (verified)"] += 1
            elif listed_current(record):
                graph.conflicts.append({"rule": "R10", "law": record["title"], "source": record["source"], "listed": record["status_basis"][0],
                                        "against": line, "why_unconfirmed": "; ".join(f"{p}: {w}" for p, w in weak)})
                record["status_basis"] = [f"conflict: {record['status_basis'][0]}, but it may be {line} (not confirmed for "
                                          + "; ".join(f"{p}: {w}" for p, w in weak) + ")"]
                record.update(status="unknown", status_confidence="low")
                changes["R10 conflict with an official listing"] += 1
            else:
                record["status_basis"] = [f"probably {line}"]
                record.update(status="repealed", status_confidence="medium")
                changes["R10 repealed in every territory (probable)"] += 1
            if record["status"] == "repealed":
                record["related"] = [{"relation": "repealed_by", "title": latest["from"]["title"], "source": latest["from"]["source"],
                                      "words": latest["words"][-200:]}] + record["related"]
        partial = [e for e in repeals if e["extent"] in ("part", "conditional")]
        if partial:
            record["related"] = record["related"] + [{"relation": "partly_repealed_by" if e["extent"] == "part" else "conditionally_repealed_by",
                                                      "title": e["from"]["title"], "source": e["from"]["source"], "words": e["words"][-200:]}
                                                     for e in partial[:3]]
        named = whole or [e for e in repeals if e["extent"] == "part"]
        if named and record["status"] == "repealed" and not any(x["relation"] == "repealed_by" for x in record["related"]):
            # already known to be repealed (the site says so): the graph still records which law did it -- even through a
            # clause with a saving ("The Companies Ordinance, 1984 ... shall stand repealed, except Part VIIIA")
            best = sorted(named, key=lambda e: (standing(e)[0] != "verified", -(e["from"]["year"] or 0)))[0]
            record["related"] = [{"relation": "repealed_by", "title": best["from"]["title"], "source": best["from"]["source"],
                                  "words": best["words"][-200:]}] + record["related"]
        if not whole or record["status"] == "repealed":
            continue
        if record["status"] == "amending_instrument":
            continue   # an amending law stays one, repealed or not
        best = sorted(whole, key=lambda e: (standing(e)[0] != "verified", -(e["from"]["year"] or 0)))[0]
        grade, why = standing(best)
        if grade == "verified" and listed_current(record) and best["from"]["status"] == "repealed":
            # the repealing law has itself been repealed, and the code still lists this one: it may have been revived
            grade, why = "probable", "the repealing law has itself since been repealed, and the code still lists this law"
        record["_before_kg"] = {"status": record["status"], "basis": record["status_basis"][:1]}
        line = f"repealed by {best['from']['title']} ({best['from']['source']}), which says: \"{best['words'][-220:]}\""
        if grade == "verified":
            record["status_basis"] = [line] + ([f"(the {record['source']} site still lists it: {record['status_basis'][0]})"]
                                               if listed_current(record) else [])
            record.update(status="repealed", status_confidence="high")
            record["related"] = [{"relation": "repealed_by", "title": best["from"]["title"], "source": best["from"]["source"],
                                  "words": best["words"][-200:]}] + record["related"]
            changes["R1 repealed (verified)"] += 1
        elif listed_current(record):
            # R4: an official code says current, a repeal that cannot be confirmed says otherwise -- neither is chosen
            graph.conflicts.append({"rule": "R4", "law": record["title"], "source": record["source"], "listed": record["status_basis"][0],
                                    "against": line, "why_unconfirmed": why})
            record["status_basis"] = [f"conflict: {record['status_basis'][0]}, but it may be {line} (not confirmed: {why})"]
            record.update(status="unknown", status_confidence="low")
            record["related"] = [{"relation": "possibly_repealed_by", "title": best["from"]["title"], "source": best["from"]["source"],
                                  "words": best["words"][-200:]}] + record["related"]
            changes["R4 conflict with an official listing"] += 1
        else:
            record["status_basis"] = [f"probably {line} ({why})"]
            record.update(status="repealed", status_confidence="medium")
            record["related"] = [{"relation": "repealed_by", "title": best["from"]["title"], "source": best["from"]["source"],
                                  "words": best["words"][-200:]}] + record["related"]
            changes["R1 repealed (probable)"] += 1
    return dict(changes)


# an amendment noted in a consolidated text: "1Subs. by the Finance Act, 2023 (XXVIII of 2023)", "Ins. by Act No. XXV of 2017, s. 6"
AMENDMENT_NOTE = re.compile(rf"\b(?:Subs|Ins|Added|Omitted|Amended|Renumbered|Substituted|Inserted)\.?,?\s+(?:and\s+\w+\.?\s+)?(?:by|vide)\b"
                            rf"[^.;]{{0,140}}?\b({YEAR})\b", re.I)


# laws that touch every law on a code (renaming the province, adapting old laws) and say nothing about this one's life
RENAMING = re.compile(r"Khyber\s+Pakhtunkhwa\s+Act\s*(?:No\.?)?\s*IV\s+of\s+2011|Act\s+No\.?\s*IV\s+of\s+2011", re.I)
GENERAL_AMENDER = re.compile(r"adaptation|revision\s+and\s+declaration|repealing\s+and\s+amending|laws\s+\(amendment\)|"
                             r"Khyber\s+Pakhtunkhwa\s+Act\s+No\.?\s*IV\s+of\s+2011|A\.\s?O\.|President'?s\s+Order|Order\s+No", re.I)


def last_amended(record: dict, amended_by: dict, own_only: bool = False) -> int | None:
    """The latest year the law is known to have been amended: from the notes in its own consolidated text,
    and from the amending laws in the corpus that name it (amends edges). Notes by laws that touch every
    law on a code (renaming, adaptation) never count. With own_only, only amendments by a law named after
    this one ("the Cantonments (Amendment) Act, 2023") count: those prove it was alive, where a Finance Act's
    sweep may reach a law the code merely still prints."""
    years = []
    distinctive = [w for w in re.findall(r"[a-z]{4,}", record["_norm"]) if w not in NAME_FILLER | {"pakistan", "west", "punjab", "sindh",
                                                                                                  "khyber", "pakhtunkhwa", "balochistan"}]
    for m in AMENDMENT_NOTE.finditer(record["_doc"]["text"]):
        note = m.group(0)
        # the province-renaming Act is printed in laws long repealed, so it proves nothing; the revision and adaptation
        # laws amended only laws then in force, so they count -- except where a law named after this one is required
        if RENAMING.search(note) or (own_only and GENERAL_AMENDER.search(note)):
            continue
        if own_only and sum(w in note.lower() for w in distinctive) < min(2, len(distinctive) or 1):
            continue
        years.append(int(m.group(1)))
    years += [e["from"]["year"] for e in amended_by.get(record.get("work_id"), []) if e["from"]["year"]]
    years = [y for y in years if y >= (record["year"] or 0)]
    return max(years) if years else None


def repeal_words_near(law: dict, target: dict) -> bool:
    """Does `law`'s text put repealing words next to the distinctive words of `target`'s name? Then a lead
    the reader could not settle may be a repeal it failed to parse, and stays open."""
    words = [w for w in re.findall(r"[a-z]{4,}", target["_norm"]) if w not in NAME_FILLER | {"pakistan", "west", "punjab", "sindh",
                                                                                             "khyber", "pakhtunkhwa", "balochistan"}]
    if len(words) < 2:
        words = re.findall(r"[a-z]{4,}", target["_norm"])
    text = law["_doc"]["text"].lower()
    for hit in re.finditer(r"repeal", text):
        window = text[max(0, hit.start() - 300):hit.start() + 300]
        if sum(w in window for w in words) >= min(2, len(words)):
            return True
    return False


def settle_leads(graph: Graph) -> dict:
    """R7: every repeal lead finalize_metadata left on a law, settled by the lead law's own words.
    R8: and by time -- a law amended after the lead law was made was not repealed by it."""
    by_title = defaultdict(list)
    for record in graph.records:
        by_title[(record["title"], record["source"])].append(record)
    out_edges = defaultdict(list)
    amended_by = defaultdict(list)
    for e in graph.edges:
        out_edges[(id(e["from"]), id(e["to"]))].append(e)
        if e["type"] == "amends":
            amended_by[e["to"].get("work_id")].append(e)
    outcome = Counter()
    for record in graph.records:
        leads = [x for x in record["related"] if x["relation"] == "possibly_affected_by"]
        if not leads:
            continue
        kept = [x for x in record["related"] if x["relation"] != "possibly_affected_by"]
        for lead in leads:
            law = next(iter(by_title.get((lead["title"], lead["source"]), [])), None)
            found = out_edges.get((id(law), id(record)), []) if law else []
            if any(e["type"] == "repeals" and e["extent"] == "whole" for e in found):
                verdict, words = "repeal", found[0]["words"]
            elif any(e["type"] == "repeals" for e in found):
                e = next(e for e in found if e["type"] == "repeals")
                verdict, words = {"part": "partial repeal", "territory": f"repeal for {e['territory']} only",
                                  "conditional": "repeal on a condition"}[e["extent"]], e["words"]
            elif any(e["type"] == "amends" for e in found):
                verdict, words = "amendment", found[0]["words"]
            else:
                words = graph.mentions(law, record) if law else None
                verdict = "mention" if words else "unconfirmed"
            if verdict == "unconfirmed" and law:
                amended = last_amended(record, amended_by)
                if amended and law["year"] and amended > law["year"]:
                    verdict, words = f"refuted: the law was amended in {amended}, after {law['year']}", None
                elif law.get("garbled_words", 1) <= 0.05 and not repeal_words_near(law, record):
                    verdict = "not supported: its text has no repealing words near this law's name"
            outcome[verdict.split(":")[0]] += 1
            settled = {**lead, "relation": "lead_settled" if verdict != "unconfirmed" else "possibly_affected_by", "verdict": verdict}
            if words:
                settled["words"] = words[-240:]
            # a general repealing Act's list, against a law an official code lists as current: the code wins
            if verdict == "unconfirmed" and "general repealing Act" in (lead.get("note") or "") and listed_current(record):
                settled["relation"], settled["verdict"] = "lead_settled", "general repealing Act; not found in its legible text, and the official code lists the law as current"
                outcome["unconfirmed, but listed as current against a general repealing Act"] += 1
            kept.append(settled)
        record["related"] = kept
    return dict(outcome)


RECENTLY = 2019   # a consolidated text amended this late is being kept up to date by the code that hosts it
# a savings clause that keeps the repealed law's rules alive: "any rules, regulations or schemes made ... shall ... remain in
# force", "rules made under the repealed Act shall be deemed to have been made under this Act"
SAVES_RULES = re.compile(r"\brules\b[^.;]{0,300}?(?:(?:shall|to)\s+(?:\w+\s+){0,6}?(?:continue|remain)\s+in\s+force|(?:shall\s+be\s+)?deemed\s+to\s+have\s+been\s+"
                         r"(?:made|framed|issued)\s+under|shall\s+have\s+effect\s+as\s+if|continue\s+to\s+(?:be\s+in\s+force|have\s+effect))", re.I)


def kept_current(graph: Graph) -> dict:
    """R9: a law the Pakistan Code hosts but leaves out of its alphabetical list (found only through its search),
    whose consolidated text there carries amendments made in RECENTLY or later, with no repeal or open lead
    against it anywhere in the corpus -> in force. The Ministry does not keep amending a dead law's text."""
    amended_by = defaultdict(list)
    repealed = set()
    for e in graph.edges:
        if e["type"] == "amends":
            amended_by[e["to"].get("work_id")].append(e)
        if e["type"] == "repeals" and e["extent"] == "whole":
            repealed.add(id(e["to"]))
    changed = Counter()
    for record in graph.records:
        if record["source"] != "pakistancode" or record["status"] != "unknown" or id(record) in repealed \
                or not (record["status_basis"] or [""])[0].startswith("on the Pakistan Code site but not in its alphabetical list"):
            continue
        if any(x["relation"] in ("possibly_affected_by", "possibly_repealed_by") for x in record["related"]):
            continue
        year = last_amended(record, amended_by)
        if year and year >= RECENTLY:
            record["_before_kg"] = {"status": record["status"], "basis": record["status_basis"][:1]}
            record.update(status="in_force", status_confidence="medium",
                          status_basis=[f"kept up to date on the Pakistan Code: its text there carries amendments made as late as {year} "
                                        f"(it is found through the site's search, not its alphabetical list); nothing found repealing it"])
            changed["R9 in force: amended recently on the Pakistan Code"] += 1
    return dict(changed)


def link_succession(graph: Graph) -> dict:
    """Which law replaced which: `replaced_by` on a repealed or superseded law, `replaces` on the law that ended it
    (on every copy of that law's work), so an answer from the current law can show what the law was before.
    Also `last_amended`: the latest year a law is known to have been amended (notes in its text, amending laws)."""
    amended_by = defaultdict(list)
    for e in graph.edges:
        if e["type"] == "amends":
            amended_by[e["to"].get("work_id")].append(e)
    by_title = {(r["title"], r["source"]): r for r in graph.records}
    by_work = defaultdict(list)
    for record in graph.records:
        by_work[record.get("work_id")].append(record)
        record["last_amended"] = last_amended(record, amended_by)
        record["replaced_by"], record["replaces"] = None, []
    linked = Counter()
    for record in graph.records:
        if record["status"] not in ("repealed", "superseded"):
            continue
        link = next((x for x in record["related"] if x["relation"] in ("repealed_by", "enacted_as")), None)
        successor = by_title.get((link["title"], link["source"])) if link else None
        if not successor or successor.get("work_id") == record.get("work_id"):
            continue
        how = "repealed" if link["relation"] == "repealed_by" else "enacted as an Act"
        record["replaced_by"] = {"title": successor["title"], "work_id": successor.get("work_id"), "year": successor["year"], "how": how}
        for copy in by_work[successor.get("work_id")]:
            if all(p["work_id"] != record.get("work_id") for p in copy["replaces"]) and len(copy["replaces"]) < 12:
                copy["replaces"].append({"title": record["title"], "work_id": record.get("work_id"), "year": record["year"], "how": how})
        linked[how] += 1
    return dict(linked)


# ---- running it ---------------------------------------------------------------------------------

def build(records: list[dict], same_title, norm_title, title_year) -> dict:
    graph = Graph(records, same_title, norm_title, title_year)
    for record in records:
        graph.edges += graph.repeal_edges(record)
        graph.edges += graph.amend_edges(record)
        graph.edges += graph.made_under_edges(record)
    changes = apply_rules(graph)
    leads = settle_leads(graph)
    changes.update(kept_current(graph))

    # R5: subordinate laws whose parent Act is repealed -- they fall with it, unless the repealing law saved them
    parents = Counter()
    by_title = {(r["title"], r["source"]): r for r in records}
    for e in graph.edges:
        if e["type"] == "made_under" and e["to"]["status"] == "repealed" and e["from"]["status"] == "in_force":
            repealed_by = next((x for x in e["to"]["related"] if x["relation"] == "repealed_by"), None)
            repealer = by_title.get((repealed_by["title"], repealed_by["source"])) if repealed_by else None
            saving = SAVES_RULES.search(repealer["_doc"]["text"]) if repealer else None
            if saving:
                e["from"]["related"] = e["from"]["related"] + [{"relation": "parent_repealed_saved", "title": e["to"]["title"],
                                                                "source": e["to"]["source"], "saved_by": repealer["title"],
                                                                "words": re.sub(r"\s+", " ", saving.group(0))[-240:]}]
                parents["made under a repealed Act, continued by the repealing law's savings clause"] += 1
                continue
            e["from"]["related"] = e["from"]["related"] + [{"relation": "parent_repealed", "title": e["to"]["title"], "source": e["to"]["source"]}]
            graph.conflicts.append({"rule": "R5", "law": e["from"]["title"], "source": e["from"]["source"],
                                    "against": f"made under {e['to']['title']}, which is repealed ({e['to']['status_basis'][0][:200]})"})
            parents["made under a repealed Act"] += 1

    succession = link_succession(graph)
    write(graph, records)
    summary = {"edges": dict(Counter((e["type"], e.get("extent")) if e["type"] == "repeals" else (e["type"], None) for e in graph.edges)),
               "changes": changes, "leads": leads, "r5": dict(parents), "succession": succession, "conflicts": len(graph.conflicts)}
    summary["edges"] = {f"{k[0]}{' ' + k[1] if k[1] else ''}": v for k, v in summary["edges"].items()}
    (KG_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def write(graph: Graph, records: list[dict]) -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)

    def ref(r: dict) -> dict:
        return {"work_id": r.get("work_id"), "title": r["title"], "source": r["source"], "id": r["_doc"]["id"]}

    with (KG_DIR / "edges.jsonl").open("w", encoding="utf-8") as out:
        for e in graph.edges:
            row = {"type": e["type"], "from": ref(e["from"]), "to": ref(e["to"]), "words": e["words"]}
            row.update({k: v for k, v in e.items() if k not in ("type", "from", "to", "words")})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        for record in records:
            for x in record["related"]:
                if x["relation"] == "enacted_as":
                    out.write(json.dumps({"type": "enacted_as", "from": ref(record), "to": {"title": x["title"], "source": x["source"]}},
                                         ensure_ascii=False) + "\n")
            if record.get("duplicate_of"):
                out.write(json.dumps({"type": "copy_of", "from": ref(record), "to": record["duplicate_of"], "version": record.get("version")},
                                     ensure_ascii=False) + "\n")
            for sibling in record.get("siblings") or []:
                out.write(json.dumps({"type": "sibling_of", "from": ref(record), "to": sibling}, ensure_ascii=False) + "\n")
    with (KG_DIR / "conflicts.jsonl").open("w", encoding="utf-8") as out:
        for c in graph.conflicts:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    # every status the rules changed, for review
    with (KG_DIR / "status_changes.jsonl").open("w", encoding="utf-8") as out:
        for record in records:
            if record.get("_before_kg"):
                out.write(json.dumps({"title": record["title"], "source": record["source"], "index": record.get("index_default"),
                                      "before": record["_before_kg"], "after": {"status": record["status"], "basis": record["status_basis"][:2]}},
                                     ensure_ascii=False) + "\n")
    works = defaultdict(list)
    for record in records:
        works[record.get("work_id")].append(record)
    with (KG_DIR / "nodes.jsonl").open("w", encoding="utf-8") as out:
        for work_id, copies in works.items():
            primary = next((c for c in copies if c.get("is_primary")), copies[0])
            out.write(json.dumps({
                "work_id": work_id, "family_id": primary.get("family_id"), "title": primary["title"], "kind": primary["instrument_type"],
                "year": primary["year"], "number": primary.get("number"), "jurisdiction": primary["jurisdiction"],
                "status": primary["status"], "status_basis": primary["status_basis"][:1], "status_exceptions": primary.get("status_exceptions"),
                "replaced_by": primary.get("replaced_by"), "replaces": primary.get("replaces"), "last_amended": primary.get("last_amended"),
                "copies": [{"source": c["source"], "id": c["_doc"]["id"], "version": c.get("version")} for c in copies],
            }, ensure_ascii=False) + "\n")
