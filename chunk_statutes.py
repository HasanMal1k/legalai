"""
chunk_statutes.py

Cuts every law marked for the index (index_default in metadata_final.jsonl)
into the pieces that are embedded and searched. Rules only, no model: the
whole corpus takes minutes and can be re-cut at any time.

    python chunk_statutes.py                         # every source
    python chunk_statutes.py --source punjab
    python chunk_statutes.py --only "penal code" --sample    # also write readable samples

How a law is cut:

  1. Clean. Page-number lines and gazette running headers are dropped.
     Where the PDF prints "Page 20 of 179" the page of every line is kept,
     and the footnotes at the foot of each page are lifted out of the text
     into a notes field on the chunks that cite them.
  2. Table of contents. If the law opens with one, it is read as the list
     of provisions to expect (number and heading) and left out of the text.
  3. Provisions. A line starting "12A. " begins a provision when the number
     is the next one expected: from the contents when there is one, else by
     counting upwards. Numbered lists inside a provision restart or run
     backwards and are passed over. Part / Chapter / Order headings become
     the provision's path. Schedules and appendices are cut by size.
  4. Size. One provision is one chunk. A provision over SPLIT_ABOVE
     characters is cut at its sub-sections, then clauses, then sentences.
     A law of WHOLE_BELOW characters or less stays in one piece. A longer
     law with no numbering the rules can follow is cut into passages.
  5. Copies. A provincial code's copy of a federal law is not served where
     the Pakistan Code's text is plainly later and the copy carries no
     amendment of its province (the Punjab Code's Penal Code still has
     section 375 "repealed by the Zina Ordinance"), nor ever a provincial
     copy of the Constitution.

The words of a chunk are always the law's own: lines are joined and cut,
never rewritten. The only generated chunk is the one "card" per law
(title, summary, list of headings), marked unit = "card".

Writes, per source:
    data/<source>/chunks.jsonl          one row per chunk
    data/<source>/chunk_report.jsonl    one row per law: how it was cut, and
                                        where the rules struggled ("problems")
and data/chunk_summary.json.
"""

import argparse
import bisect
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from ocr_scanned import SOURCES, read_jsonl, source_layout

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_DIR = DATA_DIR / "chunk_samples"

WHOLE_BELOW = 3000   # characters: a law this short is one chunk
SPLIT_ABOVE = 2200   # a provision longer than this is cut into parts ...
TARGET = 1500        # ... of about this size
SMALL = 200          # a part shorter than this is joined to its neighbour
LONG_LINE = 600      # lines longer than this (no line breaks in the source) are wrapped at sentences
CARD_MAX = 1800

# Kinds of law expected to have numbered provisions; when none are found the law is reported.
STRUCTURED = {"Act", "Ordinance", "Rules", "Regulation", "Order", "Code", "Constitution"}
DEFAULT_UNIT = {"Act": "section", "Ordinance": "section", "Code": "section", "Rules": "rule",
                "Regulation": "regulation", "Order": "article", "Constitution": "article"}
SELF_REFERENCE = {
    "section": r"\bthis section\b|\bsub-?section\b",
    "article": r"\bthis Article\b",
    "rule": r"\bthis rule\b|\bsub-?rule\b|\bthese rules\b",
    "regulation": r"\bthis regulation\b|\bsub-?regulation\b|\bthese regulations\b",
    "paragraph": r"\bthis paragraph\b|\bsub-?paragraph\b",
}

PAGE_MARK = re.compile(r"^(?:page\s*(\d{1,4})\s*of\s*\d{1,4}|page\s*[|:#-]?\s*(\d{1,4})|(\d{1,4})\s*\|\s*page|"
                       r"[-–—]\s*(\d{1,4})\s*[-–—])$", re.I)
# "1Ins. by Act LVI of 2021, s. 3.", "7S.121A, ins. by ...", "2A. O., 1961, Art. 2" -- but not "121A. Conspiracy"
# (a section) nor "3RD SCHEDULE". The first note's number is sometimes printed as a lower-case L.
FOOTNOTE = re.compile(r"^(\d{1,3}|l)(?=[A-Z](?:[a-z.]|\s))(?![A-Z]{1,3}\.\s(?!O\.))")
MARKER = re.compile(r"(?<![\w.,/(])(\d{1,4})\s?(?=\[)")         # the "1" of "1[375A. Gang rape"
LEADING_MARKER = re.compile(r"^(\d{1,3})(?=[A-Z][a-z])")         # the "1" of "1Act No. XLV of 1860"
# what never appears inside a block of footnotes
BODY_LINE = re.compile(r"^\[?\s*(?:\d{1,3}\s?[A-Z]{0,3}\.\s+[“\"]?[A-Z][a-z]{2,}|\(\d{1,3}[A-Z]{0,2}\)\s|\([a-z]{1,3}\)\s|CHAPTER\b|PART\s+[IVX\d])")

# "12. ", "12A. ", "365 A. ", "Section 12. ", and with what the printers put in front: "7[121A. ", "[4. ", "*273. ",
# or behind: "14. 2[(1) Notwithstanding ..." (the note mark of an amendment)
PROVISION = re.compile(r"^(?:\d{1,4}\s*\[|[\[*†])?\s*(?:(?:Section|Sec\.|Article|Art\.|Rule|Regulation|Para(?:graph)?)\s+)?"
                       r"(\d{1,3})\s?-?\s?([A-Z]{0,3})\s?[.:]\s*(?=$|[^\d\s.]|\d{1,3}\[)(.*)$")
# a number in any dress -- "(2E) ", "[26A, ", "4*[294B. ", "1272. " (note 1, Article 272) -- which
# counts only when the heading the contents give that number follows it
LOOSE_NUMBER = re.compile(r"(\d{1,6})\s?-?\s?([A-Z]{0,3})[\])]?\s*[.,:]?\s*")
# never a provision of the law itself: quoted text -- the old wording a note gives ("“74. Tax year ..." after the section
# that replaced it), a section an amending law inserts -- and a note about one ("Section 93 omitted by the Finance Act")
QUOTED = re.compile(r"^[\s\[\d]{0,4}[“\"‘]")
NOTE_ABOUT = re.compile(r"^\W*\d{0,3}\s*(?:sections?|clauses?|rules?|articles?|sub-sections?)\s+\d{1,4}\s?-?[A-Z]{0,3}\b(?!\s?[.:])", re.I)
# nor a line of a list: dot leaders ("1. First Schedule ………. Forms."), or an amendment note standing alone where a
# provision would ("3. Subs vide the Khyber Pakhtunkhwa Act IV of 2011.")
LEADERS = re.compile(r"(?:\.\s?){5,}|…\s?…|(?:…\s?\.){2,}")
# the law's number as the KP Code prints it under the title, "(KHYBER PAKHTUNKHWA ACT NO. XXXVII OF 2014)",
# "2[KHYBER PAKHTUNKHWA] ACT NO. I OF 1985.", and its publication note "(First published after having received ...)"
LAW_NUMBER = re.compile(r"^\W{0,3}(?:\d{1,2}\[)?\s*(?:[A-Za-z.\]]+\s+){0,3}(?:ACT|ORDINANCE|REGULATION)\s+N[Oo]\.?\s|^\W{0,3}First\s+published\b", re.I)
ENACTING = re.compile(r"\b(?:is|are)\s+hereby\s+enacted\b|^\W*(?:Preamble\.?\s*)?WHEREAS\b|\bBe\s+it\s+enacted\b|\bNOW,?\s+THEREFORE\b", re.I)
# the title above a second list after the contents: "THE FIRST STATUTES (See Section 54)", "SCHEDULES", "LIST OF FORMS"
LIST_TITLE = re.compile(r"^\W*(?:THE\s+)?(?:(?:FIRST|SECOND|THIRD)\s+)?(?:SCHEDULES?|STATUTES|CONTENTS|LIST\s+OF|FORMS)\b", re.I)
AMENDMENT_ONLY = re.compile(r"^\W*(?:Subs(?:tituted)?|Ins(?:erted)?|Added|Renumbered)\.?\s+(?:vide|by)\b", re.I)
# what reads as an amendment note wherever it stands ("Cl. 7th ins. By ...", "The words ... subs. by ...")
FOOTNOTE_LIKE = re.compile(r"\b(?:subs|ins|add|omitted|rep|cl)\.?\s+(?:by|vide|ibid)\b|\b(?:substituted|inserted|added|omitted|repealed)\s+(?:by|vide)\b"
                           r"|^For\s+Statement\s+of\s+Objects", re.I)
# (a number without its full stop in front of an omission note is a footnote too, not an entry of a contents table)
FOOTNOTE_TEXT = re.compile(r"^\W*(?:Subs(?:tituted)?|Ins(?:erted)?|Added|Renumbered|Omitted|Deleted|Rep(?:ealed)?)\.?\s+(?:vide|by)\b", re.I)
GAZETTE = re.compile(r"GA[ZS]E[IT]{1,3}E", re.I)                 # as scanned: "Gasette", "GAZEITE"
# the Gazette's own part titles, printed above every law it carries
MASTHEAD = re.compile(r"(?:PART\s*[-–]?\s*[IVX1l]+\s*)?(?:Acts,\s*Ordinances,\s*President|Statutory\s+Notifications|"
                      r"Notifications\s+and\s+Orders|PUBLISHED\s+BY\s+AUTHORITY|EXTRAORDINARY$|REGISTERED\s+No)", re.I)
MARGIN = re.compile(r"^(?!.*\b(?:sections?|rules?|clauses?|articles?|paragraphs?|No|Schedule|item|page)\b.*?\d)"
                    r"[A-Za-z][A-Za-z,.'\- ]{0,28}\s(\d{1,3})\s?([A-Z]{0,2})\.\s+(?=[(\[A-Z“\"])(.*)$")
TABLE_ENTRY = re.compile(r"^(\d{1,3})\s?([A-Z]{0,3})\.?\s+([A-Z\[*“\"].*?)(?:\s+\d{1,4}(?:\s?-\s?\d{1,4})?)?$")
RUNNING_TEXT = re.compile(r"\bshall\b|^\[?\((?:\d{1,2}|[a-z])\)\s")   # what a table of contents never holds
# ... except a lettered sub-heading of its own: "(a) Possession of Immoveable Property" (the Specific Relief Act)
CONTENTS_SUBHEAD = re.compile(r"^\[?\((?:[a-z]{1,3}|\d{1,2})\)\s+[A-Z][^;:,]{2,70}$")
END_NOTES = re.compile(r"^LEGAL REFERENCES?\.?$")                # the Customs Act lists its amendment notes under this
SYMBOL_FONT = str.maketrans({"": "-", "": "*", "": "•", "": "o", "": "…", "": "↔",
                             "": "§", "": "•", "": "→", "": " ", "": "•", "": "•"})
PRIVATE_USE = re.compile(r"[-]")
RUNNING_HEAD = re.compile(r"[A-Za-z].*_{5,}$")                   # "Chapter II – Charge of Tax__________"
NO_WORDS = re.compile(r"[\s._\-—–=…]+")                          # a rule, a row of dots
# after the number: "Heading.— text", or straight into sub-section (1)
SHAPED = re.compile(r"^[\[“\"]?[A-Z][^.]{1,160}\.\s?[—–\-_─:]|^[—–\-_─:. ]*\[?\(1\)")
HEADING_END = re.compile(r"\.\s?[—–\-_─‒―\xad]{1,4}|:\s?[—–\-_]{1,4}|\s[—–]{1,2}\s?(?=\(|[A-Z])|\.\s*(?=\[?\(1\))")
# a contents entry's dot leaders and page number, however printed: "Definitions…………..7-28", "Omitted...... …..… .45"
LEADER_TAIL = re.compile(r"(?:\s*[.…·„]){2,}[\s.…·„]*(?:\d{1,4}(?:\s?[-–]\s?\d{1,4})?)?\s*$")
BODY_WORDS = re.compile(r"\b(?:shall|must|should|followings?|hereby|namely|as under)\b", re.I)    # a sentence, not a heading
# a lettered sub-heading of a contents list ("B.__ Arrest without warrant", "E.—Special rules"): never part of an entry
SUBHEAD = re.compile(r"^[A-Z]\s?\.\s?[_—–-]+\s*\S")
# the note that inserted a provision, printed after it ("Sec.489-F ins by Ordinance LXXII of 95")
INSERTED_NOTE = re.compile(r"\b(?:ins|inserted|subs|substituted|added)\.?\s+(?:by|vide)\b", re.I)

NUMERAL = r"(?:[IVXLC]+|\d{1,3})(?:[-\s]?[A-Z])?(?![A-Za-z])"
ORDINALS = "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH"
HEADINGS = [
    ("schedule", re.compile(rf"^(?:THE\s+)?(?:({ORDINALS})\s+SCHEDULE|SCHEDULE(?:\s*[-–—.]?\s*({NUMERAL}))?)(?![A-Za-z])\s*[.:—–_-]*\s*(.*)$", re.I)),
    ("appendix", re.compile(rf"^(APPENDIX|ANNEXURE|ANNEX)(?![A-Za-z])(?:\s*[-–—.]?\s*({NUMERAL}|[A-Z](?![A-Za-z])))?\s*[.:—–_-]*\s*(.*)$", re.I)),
    ("part", re.compile(rf"^(PART)\s*[-–—.]?\s*({NUMERAL})\s*[.:—–_-]*\s*(.*)$", re.I)),
    ("chapter", re.compile(rf"^(CHAPTER)\s*[-–—.]?\s*({NUMERAL})\s*[.:—–_-]*\s*(.*)$", re.I)),
    ("order", re.compile(r"^(ORDER)\s+([IVXLC]+(?:-?[A-Z])?)(?![A-Za-z])\s*[.:—–_-]*\s*(.*)$")),
]
RANK = {"schedule": 0, "appendix": 0, "part": 1, "chapter": 2, "order": 2}
CONTENTS = re.compile(r"^(?:TABLE\s+OF\s+)?C\s?O\s?N\s?T\s?E\s?N\s?T\s?S\.?$|^ARRANGEMENT\s+OF\s+(?:SECTIONS|RULES|ARTICLES|REGULATIONS)", re.I)
OPENING = re.compile(r"^\[?\d{0,2}\s*(?:(?:ACT|ORDINANCE|REGULATION|PRESIDENT[’']S\s+ORDER|P\.\s?O\.)\s+N[Oo]\.?|"
                     r"AN?\s+(?:ACT|ORDINANCE|ORDER|REGULATION)\b|WHEREAS\b|Preamble\b|NOTIFICATION\b|"
                     r"\[\s*\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4}|\(\s*[IVXLC]+\s+of\s+\d{4}\s*\))", re.I)

# the law's own first words, which no contents list prints ("Preamble." and "(IV of 1879)" it may)
OPENS_LAW = re.compile(r"^\[?\d{0,2}\s*(?:(?:ACT|ORDINANCE|REGULATION|PRESIDENT[’']S\s+ORDER|P\.\s?O\.)\s+N[Oo]\.?|"
                       r"AN?\s+(?:ACT|ORDINANCE|ORDER|REGULATION)\s+(?:to|for|further)\b|WHEREAS\b)", re.I)

SUB = re.compile(r"^\[?\s*\((\d{1,3}[A-Za-z]{0,2})\)")           # (1)  (2A)  [(3)
CLAUSE = re.compile(r"^\[?\s*\(([a-z]{1,4})\)")                  # (a)  (aa)  (ii)
ITEM = re.compile(r"^\[?\s*(\d{1,3}[A-Z]{0,2})\.\s")             # 1.  12A.
ENDS_SENTENCE = re.compile(r"[.;:—–]\]?[”\"]?$")
STARTS_BLOCK = re.compile(r"^\[?\s*(?:\(\w{1,4}\)|\d{1,3}[A-Z]{0,3}\.\s|[“\"]|Explanation|Illustrations?\b|Provided\b|Exception\b)")
LOWERCASE_WORDS = {"of", "and", "the", "to", "in", "for", "or", "by", "on", "a", "an", "at", "with", "from", "as"}


class Line:
    __slots__ = ("text", "page", "notes")

    def __init__(self, text: str, page: int | None = None, notes: list[str] | None = None):
        self.text, self.page, self.notes = text, page, notes or []


def squash(text: str) -> str:
    """Letters and digits only, for comparing two headings."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tidy_caps(text: str) -> str:
    """An ALL-CAPITALS heading in ordinary capitalisation; anything else untouched."""
    text = text.strip(" .:—–-_")
    letters = [char for char in text if char.isalpha()]
    if not letters or sum(char.isupper() for char in letters) < 0.8 * len(letters):
        return text
    words = text.lower().split()
    return " ".join(word if n and word in LOWERCASE_WORDS else word[:1].upper() + word[1:] for n, word in enumerate(words))


def in_capitals(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return len(letters) >= 3 and sum(char.isupper() for char in letters) >= 0.7 * len(letters)


def length(lines: list[Line]) -> int:
    return sum(len(line.text) + 1 for line in lines)


def sentence_like(text: str) -> bool:
    """A line of running text, as a table of contents never prints one."""
    return bool(RUNNING_TEXT.search(text)) and not (CONTENTS_SUBHEAD.match(text) and not re.search(r"\bshall\b", text))


# ---------------------------------------------------------------- 1. clean

def is_furniture(text: str) -> bool:
    """Running headers and footers of the Gazette, which interrupt the text on every page."""
    if len(text) > 120:
        return False
    if GAZETTE.search(text) and re.search(r"EXTRA|EXT\.", text, re.I) and (
            re.match(r"\d{1,5}\s", text) or re.search(r"\s\d{1,5}$", text)
            or re.search(r"P\s?A\s?R\s?T\s*[-–]?\s*[IVX1l]+\s*\]|\[\s*P\s?A\s?R\s?T|^P\s?A\s?R\s?T\s", text, re.I)):
        return True
    if MASTHEAD.match(text):
        return True
    return bool(re.match(r"Price\s*:?\s*Rs|\[\s*\d+\s*\(\s*\d{2,4}\s*\)\s*/?\s*Ex\.?\s*Gaz|\(\d{3,5}\)$", text, re.I))


def wrap(text: str) -> list[str]:
    """A very long line (a source with no line breaks) cut at sentence ends."""
    if len(text) <= LONG_LINE:
        return [text]
    out, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+(?=[A-Z(\[“\"])", text):
        while len(sentence) > 2 * LONG_LINE:                      # no sentence ends at all: cut at a space
            cut = sentence.rfind(" ", 0, LONG_LINE) + 1 or LONG_LINE
            sentence, piece = sentence[cut:], sentence[:cut].strip()
            if current:
                out.append(current)
            current = piece
        if current and len(current) + len(sentence) > LONG_LINE:
            out.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    return out + [current] if current else out


def page_numbers(marks: list[int], pages: list[list[str]]) -> list[int | None]:
    """The page number of each run of lines between page marks, or None when the marks can't be trusted."""
    if not marks or (len(marks) > 1 and sum(b == a + 1 for a, b in zip(marks, marks[1:])) < 0.8 * (len(marks) - 1)):
        return [None] * len(pages)
    if marks[0] == 1 and pages[0]:              # the mark closes its page ("Page 1 of 9" at the foot)
        return marks + [marks[-1] + 1]
    return [marks[0] - 1] + marks               # the mark opens its page


def note_number(match: re.Match) -> int:
    return 1 if match.group(1) == "l" else int(match.group(1))


def split_footnotes(page: list[str], spaced: bool) -> tuple[list[str], dict[int, str]]:
    """(the page's own text, its footnotes by number). Footnotes are the block at the
    foot of the page that starts "1Ins. by ..." and holds nothing that looks like law.

    `spaced` is the layout of the tax manuals: the note's number stands apart from its
    text ("342 Substituted ...") or on a line of its own. A number at the start of a
    line is too common to trust there, so the first note must be cited in the page above.
    """
    for start, text in enumerate(page):
        apart = re.match(r"(\d{1,4}|l)(?:\s+[A-Za-z“\"(]|$)", text) if spaced else None
        if not (FOOTNOTE.match(text) or apart):
            continue
        tail = page[start:]
        if any(BODY_LINE.match(t) for t in tail):
            continue
        if apart and not FOOTNOTE.match(text):
            first = 1 if apart.group(1) == "l" else int(apart.group(1))
            if not any(re.search(rf"(?<![\w.,/(]){first}\s?[\[(]", t) for t in page[:start]):
                continue
        notes, current, rising = {}, None, True
        for t in tail:
            match = FOOTNOTE.match(t)
            apart = re.match(r"(\d{1,4}|l)(?:\s+(?=[A-Za-z“\"(])|$)", t) if spaced else None
            if match:
                rising = rising and (current is None or note_number(match) > current)
                current = note_number(match)
                notes[current] = t[match.end():].strip()
            elif apart and (current is None or note_number(apart) == current + 1):
                current = note_number(apart)
                notes[current] = t[apart.end():].strip()
            else:
                notes[current] = (notes[current] + " " + t).strip()
        if rising:
            return page[:start], notes
    return page, {}


def running_heads(texts: list[str]) -> set[str]:
    """Lines printed at the top of every page: the law's own title, over and over, each
    time next to a page number."""
    counts = Counter(text for text in texts if 8 <= len(text) <= 80 and any(char.isalpha() for char in text))
    heads = set()
    for text, count in counts.items():
        if count >= 8:
            beside = sum(1 for n, t in enumerate(texts) if t == text and (
                (n and re.fullmatch(r"\d{1,4}", texts[n - 1])) or (n + 1 < len(texts) and re.fullmatch(r"\d{1,4}", texts[n + 1]))))
            if beside >= 0.6 * count:
                heads.add(text)
    return heads


def clean(raw: str) -> tuple[list[Line], int]:
    """(the law's lines without page furniture or footnotes, characters set aside as footnotes)."""
    # the PDF's symbol-font characters in plain form; any other private-use code point goes
    raw = PRIVATE_USE.sub("", raw.translate(SYMBOL_FONT))
    texts = [text for text in (re.sub(r"\s+", " ", line).strip() for line in raw.split("\n")) if text and not NO_WORDS.fullmatch(text)]
    heads = running_heads(texts)
    if texts and len(texts[0]) >= 12 and texts.count(texts[0]) >= 4:
        # the title repeated down the document is a page heading; keep it where the contents and the law open
        seen, title = 0, texts[0]
        texts = [text for text in texts if text != title or (seen := seen + 1) <= 2]
    pages, current, marks = [], [], []
    for text in texts:
        mark = PAGE_MARK.match(text)
        if mark:
            marks.append(int(next(group for group in mark.groups() if group)))
            pages.append(current)
            current = []
        elif text in heads or (RUNNING_HEAD.search(text) and len(text) < 120):
            # a running head opens a page; the number on the line before it is the page number
            if current and re.fullmatch(r"\d{1,4}", current[-1]):
                current.pop()
            marks.append(None)
            pages.append(current)
            current = []
        elif is_furniture(text):
            # "PART I" alone above the Gazette's part title is part of the masthead, not of the law
            if MASTHEAD.match(text) and current and re.fullmatch(r"PART\s*[-–]?\s*[IVX1l]+", current[-1], re.I):
                current.pop()
        else:
            current.extend(wrap(text))
    pages.append(current)
    numbers_alone = None in marks
    if numbers_alone:
        marks = []
        pages = [page[1:] if page and n and re.fullmatch(r"\d{1,4}", page[0]) else page for n, page in enumerate(pages)]

    lines, footnote_chars = [], 0
    for number, page in zip(page_numbers(marks, pages), pages):
        joined = []
        for text in page:   # a footnote mark left alone on its line belongs to the "[" that follows
            if joined and re.fullmatch(r"\d{1,3}", joined[-1]) and text.startswith("["):
                joined[-1] += text
            else:
                joined.append(text)
        body, notes = split_footnotes(joined, numbers_alone) if len(pages) > 1 else (joined, {})
        footnote_chars += sum(len(note) for note in notes.values())
        cited_here, pending, first = set(), [], len(lines)
        for text in body:
            cited = pending
            pending = []

            def lift(match):
                if int(match.group(1)) in notes:
                    cited.append(notes[int(match.group(1))])
                    cited_here.add(int(match.group(1)))
                    return ""
                return match.group(0)

            if notes and re.fullmatch(r"\d{1,3}(?: \d{1,3}){0,3}", text) and all(int(n) in notes for n in text.split()):
                # note marks the PDF set on a line of their own: they belong to the line below
                pending = cited + [notes[int(n)] for n in text.split()]
                cited_here.update(int(n) for n in text.split())
                continue
            if notes:
                text = LEADING_MARKER.sub(lift, MARKER.sub(lift, text))
            lines.append(Line(text, number, cited))
        # a note whose mark was not found in the text still belongs to its page
        loose = pending + [note for n, note in notes.items() if n not in cited_here]
        if loose and len(lines) > first:
            lines[-1].notes = lines[-1].notes + loose
    return lines, footnote_chars


# ---------------------------------------------------------------- 2. structure

def provision_of(text: str) -> tuple[tuple[int, str], str] | None:
    """((12, "A"), "Heading.- text") for a line starting "12A. Heading.- text"."""
    match = PROVISION.match(text)
    if not match:
        return None
    rest = match.group(3).lstrip(" —–-_─\xad")
    return (int(match.group(1)), match.group(2)), rest


def margin_provision(text: str) -> tuple[tuple[int, str], str] | None:
    """The same for a scanned page whose marginal note runs into the line: "Short title 1. (1) This Act"."""
    match = MARGIN.match(text)
    return ((int(match.group(1)), match.group(2)), match.group(3)) if match else None


def heading_of(text: str, previous: str) -> tuple[str, str, str] | None:
    """(kind, label, name) for "CHAPTER XVI", "PART I - GENERAL", "THE FIRST SCHEDULE" ..."""
    text = re.sub(r"^\d{0,3}\s*\[\s*", "", text).rstrip(" ]")
    if len(text) > 90 or not text[:1].isupper() or text.endswith((",", ";")):
        return None
    for kind, pattern in HEADINGS:
        match = pattern.match(text)
        if not match:
            continue
        word, numeral, name = match.group(1), match.group(2), match.group(3).strip()
        if kind == "schedule":
            label = f"{word.title()} Schedule" if word else f"Schedule {numeral}" if numeral else "Schedule"
        else:
            label = f"{word.title()} {numeral}" if numeral else word.title()
        # "Part II of the Schedule shall ...": a sentence, not a heading
        if name and not (name[0] in "([" or (name[0].isupper() and len(name) <= 70 and " shall " not in name)):
            return None
        # in ordinary type it must stand alone, not be the end of a wrapped sentence
        if not in_capitals(text) and (text.endswith(".") or (previous and not ENDS_SENTENCE.search(previous) and not in_capitals(previous))):
            return None
        return kind, label, "" if name[:1] in ("(", "[") else name
    return None


class Contents:
    """What the table of contents says the law holds."""

    def __init__(self):
        self.order: list[tuple[int, str]] = []
        self.position: dict[tuple[int, str], int] = {}
        self.heading: dict[tuple[int, str], str] = {}
        self.unnumbered: set[str] = set()
        self.chars = 0

    def matches(self, key: tuple[int, str], rest: str) -> bool:
        """Does the text after a number open with the heading the contents give that number?"""
        heading = squash(self.heading.get(key, ""))[:50]
        if len(heading) < 6:        # "Fund", "Oath": too short to compare loosely
            return len(heading) >= 3 and squash(rest[:len(heading) + 12]).startswith(heading)
        return fuzz.partial_ratio(heading, squash(rest[:len(heading) + 40])) >= 80

    def opens(self, key: tuple[int, str], rest: str) -> bool:
        """Does the text after a number begin with the heading the contents give it (word for word, give or take)?"""
        heading = squash(self.heading.get(key, ""))[:40]
        return len(heading) >= 3 and fuzz.ratio(heading, squash(rest)[:len(heading)]) >= 85


def find_contents(lines: list[Line], provisions: list, headings: list) -> tuple[int, Contents | None]:
    """(index of the first line after the table of contents, the contents). (0, None) without one.

    A table of contents is a run of numbered lines with almost nothing between
    them, after which the numbering starts again with real text behind it.
    """
    marker = next((n for n, line in enumerate(lines[:80]) if CONTENTS.match(line.text)), None)
    # a footnote at the foot of a contents page ("1 Substituted vide P. O. No. 4 of 1975.") is no entry, and does not
    # start the numbering again (it cut the KP Code's Motor Vehicles Ordinance's contents short at section 19); an
    # entry may itself say "113A. Omitted by the Finance Act, 2016." (the Income Tax Ordinance's contents)
    provisions = [None if found and AMENDMENT_ONLY.match(found[1]) else found for found in provisions]
    if marker is not None:
        # a contents page set out as a table has no full stop after the number: "3A  Directorate General ...  10"
        quiet, entries, latest = 0, 0, (0, "")
        for n in range(marker + 1, len(lines)):
            # the table ends where the law opens: its preamble, or a "1." written the ordinary way
            if entries >= 3 and (OPENING.match(lines[n].text) or (provisions[n] and provisions[n][0][0] == 1)):
                break
            entry = None if provisions[n] else TABLE_ENTRY.match(lines[n].text)
            # ("100F Omitted by the Finance Act," carries on the count; "1 Substituted vide ..." after entry 19 is a note)
            if entry and FOOTNOTE_TEXT.match(entry.group(3)) and (int(entry.group(1)), entry.group(2)) <= latest:
                entry = None
            if provisions[n] or entry:
                latest = provisions[n][0] if provisions[n] else (int(entry.group(1)), entry.group(2))
            if entry:
                provisions[n] = ((int(entry.group(1)), entry.group(2)), entry.group(3))
                entries += 1
            quiet = 0 if provisions[n] or headings[n] else quiet + 1
            if quiet > 15:
                break
    numbered = [i for i, found in enumerate(provisions) if found]
    if len(numbered) < 6:
        return 0, None
    # the law's own opening words ("ACT NO. XXVI OF 1925", "An Act to ...", "WHEREAS ...", "NOW, THEREFORE ...") end
    # any list before them, whatever the numbers after them: a line of the Carriage of Goods by Sea Act's preamble
    # read as entry 15, the Road Transport Board Ordinance's "51 . Short title" (note 5 against section 1)
    opens = [k for k, line in enumerate(lines) if OPENS_LAW.match(line.text) or ENACTING.search(line.text)]
    runs, run = [], [numbered[0]]
    for i in numbered[1:]:
        number, before = provisions[i][0][0], provisions[run[-1]][0][0]
        k = bisect.bisect_right(opens, run[-1])
        opened = not runs and len(run) >= 3 and k < len(opens) and opens[k] < i     # (the first run only: a recital
        # further down, "Whereas in the Marwat District ...", does not cut the body in two)
        # the numbering starts again: at 1 or 2, or (a note mark printed against the number) well below half
        if opened or (provisions[i][0] < provisions[run[-1]][0] and (number <= 2 or (number <= 20 and number * 2 <= before))):
            runs.append(run)
            run = [i]
        else:
            run.append(i)
    runs.append(run)
    if len(runs) < 2:
        return 0, None

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.text) + 1)

    def is_listing(members: list[int]) -> bool:
        sizes = [offsets[b] - offsets[a] for a, b in zip(members, members[1:])]
        return len(members) >= 3 and statistics.median(sizes) <= 160

    def keys(members: list[int]) -> set:
        return {provisions[i][0] for i in members}

    announced = marker is not None and marker < runs[0][0]
    if not is_listing(runs[0]):
        return 0, None
    def second_list(n: int) -> bool:
        """Is run n a list of headings announced as one right after the contents ("SCHEDULE", "THE FIRST STATUTES
        (See Section 54)" above it)? Footnotes printed after the contents are not, nor is the preamble they precede."""
        members = runs[n]
        span = lines[members[0]:members[-1] + 1]
        if members[0] - runs[n - 1][-1] > 12 or len(members) < 4 or any(FOOTNOTE_LIKE.search(line.text) for line in span):
            return False
        if sum(sentence_like(line.text) for line in span) > 0.05 * len(span):
            return False
        above = range(max(runs[n - 1][-1] + 1, members[0] - 3), members[0])
        return any(LIST_TITLE.match(lines[k].text) or (headings[k] and RANK[headings[k][0]] == 0) for k in above)

    body = None
    for n in range(1, len(runs)):
        replay = len(keys(runs[0]) & keys(runs[n])) / min(len(keys(runs[0])), len(keys(runs[n])))
        if not is_listing(runs[n]) or (announced and replay >= 0.6):
            # a second list of headings straight after the contents -- the Schedules with their titles ("1. First
            # Schedule ...... Forms"), the contents of the First Statutes a university Act annexes ("1. Faculties.") --
            # is not the law: its body is the run after it, which is no list and names the provisions the contents give
            if is_listing(runs[n]) and second_list(n) and n + 1 < len(runs) and not is_listing(runs[n + 1]) \
                    and len(keys(runs[0]) & keys(runs[n + 1])) >= 0.5 * min(len(keys(runs[0])), len(keys(runs[n + 1]))):
                continue
            # without the word "CONTENTS" above it, the list must at least name the same provisions
            body = n if announced or replay >= 0.5 else None
            break
    if body is None:
        return 0, None

    last_entry, first_section = runs[body - 1][-1], runs[body][0]
    title = squash(lines[0].text)
    # the title as the head of the document prints it, note marks aside ("THE 1[KHYBER PAKHTUNKHWA] FINANCE ACT"):
    # the KP Code's first line is its page number, "1 | P a ge", and the title follows it
    plain = lambda text: squash(re.sub(r"\d{1,2}\[|\]", "", text))
    heads = {plain(lines[k].text) for k in range(min(runs[0][0], 8)) if len(plain(lines[k].text)) > 12}
    start = last_entry + 1

    def furniture_only(upto: int) -> bool:
        """Only notes and page furniture between the contents and line `upto`: a title found further on (a running
        head, "Cloud First Policy - Government of Balochistan") must not carry the start past text of the law."""
        return sum(len(line.text) for line in lines[last_entry + 1:upto]
                   if not (FOOTNOTE_TEXT.match(re.sub(r"^\d{1,3}\.?\s*", "", line.text)) or len(line.text) < 25)) <= 200

    for i in range(last_entry + 1, first_section):      # the law proper opens with its title, number or preamble
        if OPENING.match(lines[i].text) or (len(title) > 12 and squash(lines[i].text).startswith(title)) \
                or ((LAW_NUMBER.match(lines[i].text) or plain(lines[i].text) in heads) and furniture_only(i)):
            start = i
            break

    # everything before `start` is left out of the chunks, so be sure: a table of contents is
    # headings, not sentences
    listed = lines[runs[0][0]:start]
    # (under the word CONTENTS a few headings do say "shall": "8. Fish that shall not be taken.")
    # (one such heading never sinks a short list under the word CONTENTS: "3. Rulers shall cease to exercise ...")
    if sum(sentence_like(line.text) for line in listed) > (max(1, 0.12 * len(listed)) if announced else 0.05 * len(listed)):
        return 0, None
    # and the law's own enacting words are never part of it: a "table" that stretches over them took in the preamble
    # and the body (the Balochistan Finance Ordinance, 1969: 13 entries, then a stray "349." from a tariff 180 lines on)
    if any(ENACTING.search(line.text) for line in listed):
        return 0, None

    contents = Contents()
    contents.chars = offsets[start]
    # a contents page that gives page numbers ends most entries with one
    paged = sum(bool(re.search(r"\s\d{1,4}$", provisions[i][1])) for i in runs[0]) >= 0.7 * len(runs[0])
    for position, i in enumerate(runs[0]):
        key, rest = provisions[i]
        if key in contents.position:
            continue
        # an entry printed with its dot leaders is complete: what follows is the next entry or a sub-heading (a page
        # number alone is no sign -- some contents print it on the first line of an entry that wraps)
        ended = bool(LEADER_TAIL.search(rest))
        rest = LEADER_TAIL.sub("", re.sub(r"\s\d{1,4}$", "", rest) if paged else rest)
        end = runs[0][position + 1] if position + 1 < len(runs[0]) else start
        for j in range(i + 1, min(end, i + 3)):          # a heading wrapped onto the next line
            # (a short entry was not wrapped: what follows it is a sub-heading of the contents)
            if ended or len(lines[j - 1].text) < 70 or headings[j] or len(rest) > 150 or ENDS_SENTENCE.search(rest) \
                    or (in_capitals(lines[j].text) and not in_capitals(rest)) or SUBHEAD.match(lines[j].text):
                break
            ended = bool(LEADER_TAIL.search(lines[j].text))
            rest += " " + LEADER_TAIL.sub("", lines[j].text)
        contents.position[key] = len(contents.order)
        contents.order.append(key)
        contents.heading[key] = tidy_caps(rest)
    for i in range(runs[0][0], start):
        if not provisions[i] and not headings[i] and 4 <= len(lines[i].text) <= 80 and squash(lines[i].text) != title:
            contents.unnumbered.add(squash(lines[i].text))
    return start, contents


def longest_rising(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The longest selection of (line, place) pairs, in line order, whose places keep rising."""
    tails, holders, links = [], [], []
    for n, (_, place) in enumerate(pairs):
        k = bisect.bisect_left(tails, place)
        if k == len(tails):
            tails.append(place)
            holders.append(n)
        else:
            tails[k], holders[k] = place, n
        links.append(holders[k - 1] if k else None)
    chain, n = [], holders[-1] if holders else None
    while n is not None:
        chain.append(pairs[n])
        n = links[n]
    return chain[::-1]


def follows(last: tuple[int, str] | None, key: tuple[int, str], reach: int = 8) -> bool:
    """Can `key` be the provision after `last`?  12 -> 12A -> 12B -> 13; small jumps allowed."""
    if last is None:
        return key[0] <= 2
    if key[0] == last[0]:
        return key[1] > last[1]
    return last[0] < key[0] <= last[0] + reach and key[1] in ("", "A")


def walk(lines: list[Line], provisions: list, headings: list, start: int, contents: Contents | None, word: str,
         trusted: frozenset[int] = frozenset()) -> list[dict]:
    """Group the lines from `start` into units: the preamble, each provision, each schedule.
    `trusted` are lines a reader (chunk_model.py) pointed at: taken on its word if the number fits the count."""
    provisions = list(provisions)
    for i in range(start, len(lines)):
        # a line of a list ("1. First Schedule ……. Forms.") or a note ("3. Subs vide ...") begins no provision
        if provisions[i] and (LEADERS.search(lines[i].text) or AMENDMENT_ONLY.match(provisions[i][1])):
            provisions[i] = None
    for i in range(start, len(lines) - 1):
        # a number alone on its line: what follows it is on the next line
        if provisions[i] and not provisions[i][1] and not provisions[i + 1] and not headings[i + 1]:
            provisions[i] = (provisions[i][0], lines[i + 1].text)
    # A table of contents inside the body (rules bound in behind their Act bring their own): five or
    # more numbered lines close together, headings without a sentence among them, whose numbers
    # come again further down. Its lines stay in the text but begin nothing. Only applied to a
    # reading by chunk_model.py: on the rules' own pass it cost real provisions (short ones in a
    # row), and a reading is only used when it finds more provisions than the rules did.
    order = [i for i in range(start, len(lines)) if provisions[i]] if trusted else []
    groups = [[i] for i in order[:1]]
    for i in order[1:]:
        if i - groups[-1][-1] <= 4:     # an entry may wrap, and Part headings stand between entries
            groups[-1].append(i)
        else:
            groups.append([i])
    for group in groups:
        listed = {provisions[i][0] for i in group}
        span = lines[group[0]:group[-1] + 1]
        if len(group) >= 5 and sum(bool(RUNNING_TEXT.search(line.text)) for line in span) <= 0.05 * len(span) \
                and len(listed & {provisions[i][0] for i in order if i > group[-1]}) >= 0.5 * len(listed):
            for i in group:
                provisions[i] = None
    trusted = frozenset(i for i in trusted if provisions[i])
    headed = False
    anchors: dict[int, int] = {}        # line -> place in the contents, for provisions printed with their heading
    if contents:
        # the contents' own headings find provisions whose number is oddly printed: "(2E) Removal of ..."
        notes_on = set()        # "Section 113A omitted by the Finance Act, 2016. The omitted section read ...": the
        for i in range(start, len(lines)):      # omitted section's place, but only where no provision of the law is
            text = lines[i].text
            if provisions[i] and contents.matches(*provisions[i]):
                continue
            if QUOTED.match(text) or LEADERS.search(text):
                continue
            if NOTE_ABOUT.match(text):
                notes_on.add(i)
            for loose in LOOSE_NUMBER.finditer(text[:18]) if any(char.isdigit() for char in text[:12]) else ():
                digits, rest = loose.group(1), text[loose.end():]
                keys = [(int(digits[drop:]), loose.group(2)) for drop in range(min(4, len(digits))) if len(digits) - drop <= 3]
                # a number in brackets is mostly a sub-section ("(4) An agreement to sell becomes a sale ..."): it is a
                # provision only when its heading opens what follows, as in "(2E) Removal of Chairperson and members"
                bracketed = text[:loose.start()].rstrip().endswith("(")
                found = next((key for key in keys if key in contents.position and
                              (contents.opens(key, rest) if bracketed else contents.matches(key, rest))), None)
                if found:
                    provisions[i] = (found, rest)
                    break
        # The numbers that are followed by their own heading, taken in the order of the
        # contents, are the law's skeleton. One out of order is something else: definition
        # (46) of section 2 that reads like the heading of section 46, a schedule's item 4A.
        named = [(i, contents.position[provisions[i][0]]) for i in range(start, len(lines))
                 if provisions[i] and provisions[i][0] in contents.position and contents.matches(*provisions[i])]
        # the law's own provisions first; a note on an omitted one only where it fits between them (the note on
        # section 93 is printed after section 94 begins, and must not cost section 94 its place)
        anchors = dict(longest_rising([pair for pair in named if pair[0] not in notes_on]))
        lines_so_far = sorted(anchors)
        for i, position in named:
            if i in notes_on:
                k = bisect.bisect_left(lines_so_far, i)
                if (k == 0 or anchors[lines_so_far[k - 1]] < position) and (k == len(lines_so_far) or position < anchors[lines_so_far[k]]):
                    anchors[i] = position
                    lines_so_far.insert(k, i)
        # does the body print the headings at all? (Some Sindh laws keep them in the margin only.)
        headed = len(anchors) >= 0.5 * len(contents.order)
    anchor_lines = sorted(anchors)
    numbered = [i for i in range(start, len(lines)) if provisions[i]]
    place = {i: n for n, i in enumerate(numbered)}
    schedule_lines = [j for j in range(start, len(lines)) if headings[j] and RANK[headings[j][0]] == 0]

    def ahead(i: int, count: int = 30) -> list:
        # not past a schedule heading: a schedule's own list ("4. Restitution of conjugal rights.") says nothing of
        # the provisions still to come
        k = bisect.bisect_right(schedule_lines, i)
        stop = schedule_lines[k] if k < len(schedule_lines) else len(lines)
        # nor a note printed in the text ("1. Cl. 7th ins. By Sind Repealing ... Act"): a "1." to come that is a
        # footnote does not mean section 1 is still to come
        return [provisions[j] for j in numbered[place[i] + 1:place[i] + 1 + count]
                if j < stop and not FOOTNOTE_LIKE.search(provisions[j][1][:160])]

    trail: list[tuple[str, str]] = []       # (kind, "Chapter XVI: Of Offences ...")
    units: list[dict] = []
    zone = "main"                           # main | order (numbering restarts under each Order) | schedule
    last = main_last = None
    expected = 0                            # how far down the contents we are
    accepted = 0
    in_notes = False
    annex_word = "paragraph"

    def open_unit(kind: str, number: str | None = None, key=None) -> dict:
        unit = {"kind": kind, "number": number, "key": key, "path": [label for _, label in trail], "lines": [],
                "zone": zone, "heading": None}
        units.append(unit)
        return unit

    def stray(i: int, key, rest: str) -> bool:
        """A number that fits the count but is a list item: skipped numbers turn up just after it, or a
        better-formed line with the same number does."""
        if last is not None and key[0] > last[0] + 1 and any(last[0] < other[0][0] < key[0] for other in ahead(i)):
            return True
        if not SHAPED.match(rest):
            for other in ahead(i):
                if other[0] < key:          # the numbering has started again: another Order, another list
                    break
                if other[0] == key and SHAPED.match(other[1]):
                    return True
        return False

    def repeats(i: int) -> bool:
        """The number of the provision just begun, printed again as a provision of its own straight after the note that
        inserted the first: the Punjab Code prints both sections 489-F (prize bonds, 1995; dishonoured cheques, 2002)."""
        key, rest = provisions[i]
        return bool(key == last and units and units[-1]["key"] == key and len(units[-1]["lines"]) >= 2 and i
                    and INSERTED_NOTE.search(lines[i - 1].text) and SHAPED.match(rest) and HEADING_END.search(rest[:200]))

    def accept(i: int) -> bool:
        nonlocal last, expected
        key, rest = provisions[i]
        if repeats(i):
            return True
        if zone == "main" and contents and key in contents.position:
            position = contents.position[key]
            if i not in anchors:
                # without its heading, a number must fall between the provisions on either side of it ...
                after = bisect.bisect_right(anchor_lines, i)
                ceiling = anchors[anchor_lines[after]] if after < len(anchor_lines) else len(contents.order)
                if not expected <= position < ceiling or contents.matches(key, rest):
                    return False
                # ... and, where the body does print headings, be the very next one expected,
                # or nearly so and set out like a provision
                if headed and position != expected and not (SHAPED.match(rest) and position <= expected + 2):
                    return False
                if any(other[0] in contents.order[expected:position] for other in ahead(i)):
                    return False
            expected = position + 1
        elif i in trusted:
            if not follows(last, key, reach=15):
                return False
        elif not follows(last, key) or stray(i, key, rest):
            return False
        last = key
        return True

    def restarts(i: int) -> bool:
        """A pointed-at "1" followed by a pointed-at "2", well into the law: the scheme or rules annexed to it."""
        later = [j for j in numbered[place[i] + 1:place[i] + 6] if j in trusted]
        return bool(i in trusted and last and last[0] >= 3 and provisions[i][0] == (1, "")
                    and later and provisions[later[0]][0][0] == 2)

    def resumes(i: int) -> bool:
        """After a schedule: is this line the law's next provision after all?"""
        key, rest = provisions[i]
        if contents:
            return i in anchors
        before = last if in_notes else main_last
        return bool(before and key == (before[0] + 1, "") and re.match(r"^[\[“\"]?[A-Z][^.]{1,160}\.\s?[—–\-_─]", rest))

    current = open_unit("preamble")
    i = start
    while i < len(lines):
        line = lines[i]
        heading = headings[i]
        # the tail of a table of contents says "SCHEDULE" before any provision has begun
        if heading and not (contents and not accepted and RANK[heading[0]] == 0):
            kind, label, name = heading
            step = 1
            # the heading's name is on the next line, and may itself run on to a second
            while i + step < len(lines) and step <= 2 and not provisions[i + step] and not headings[i + step]:
                following = lines[i + step].text
                if len(following) > 100 or following[0] in "([" or (name and not (in_capitals(name) and in_capitals(following))):
                    break
                if name and contents and squash(following) in contents.unnumbered:      # a sub-heading with a line of its own in the contents
                    break
                if not in_capitals(following) and (kind == "schedule" or len(following) > 60 or ENDS_SENTENCE.search(following)):
                    break
                name, step = (name + " " + following).strip(), step + 1
            trail[:] = [entry for entry in trail if RANK[entry[0]] < RANK[kind]]
            trail.append((kind, label + (": " + tidy_caps(name) if name else "")))
            in_notes = False
            if RANK[kind] == 0:
                if zone == "main":
                    main_last = last
                zone = "schedule"
                current = open_unit("schedule")
            else:
                if kind == "order":
                    if zone == "main":
                        main_last = last
                    zone, last = "order", None
                current = None
            i += step
            continue
        if END_NOTES.match(line.text):
            # a numbered list of notes: only a provision under its own heading ends it
            in_notes = True
            current = open_unit("notes")
        elif provisions[i]:
            if zone == "schedule" and accepted and resumes(i):
                zone, last = "main", main_last
                trail[:] = [entry for entry in trail if RANK[entry[0]] > 0]
            if zone != "schedule" and restarts(i):
                if zone == "main":
                    main_last = last
                zone, last = "annex", None
                if not any(RANK[kind] == 0 for kind, _ in trail):
                    trail[:] = [("schedule", "Schedule")]
                # what the annexed text calls its own provisions (rules bound in behind their Act)
                ahead_text = " ".join(following.text for following in lines[i:i + 400])
                counts = {name: len(re.findall(pattern, ahead_text)) for name, pattern in SELF_REFERENCE.items()}
                annex_word = max(counts, key=counts.get) if max(counts.values()) >= 2 else "paragraph"
            again = zone != "schedule" and repeats(i)
            if zone != "schedule" and (resumes(i) if in_notes else True) and accept(i):
                key = provisions[i][0]
                accepted += 1
                in_notes = False
                current = open_unit({"order": "rule", "annex": annex_word}.get(zone, word), f"{key[0]}{key[1]}", key)
                current["repeat"] = again       # its heading is its own, not the contents' entry for the number
        if current is None:     # text straight after a heading, before any provision
            current = open_unit("passage")
        current["lines"].append(line)
        i += 1

    # sub-headings without a number ("Of Rape"): known from the contents, they sit
    # at the end of the provision before them and belong in the path of those after
    sub, base = None, None
    for unit in units:
        if unit["path"] != base:
            sub, base = None, list(unit["path"])
        popped = None
        while contents and unit["lines"] and squash(unit["lines"][-1].text) in contents.unnumbered \
                and (len(unit["lines"]) > 1 or unit["kind"] == "passage"):
            popped = unit["lines"].pop().text
        if sub and unit["kind"] not in ("preamble", "schedule"):
            unit["path"] = unit["path"] + [sub]
        if popped:
            sub = tidy_caps(popped)
    return [unit for unit in units if unit["lines"]]


LONG_HEADING = 120      # characters: a contents heading longer than this is checked for a run-on into the next entry
NEXT_NUMBER = re.compile(r"(?<=[a-z.)\]])\s+\d{1,3}(?:\s?-?\s?[A-Z]{1,2})?[.,]?\s+(?=[A-Z])")    # "... inferred. 189 When ..."
SUB_ENTRY = re.compile(r"\s\([a-z]{1,3}\)\s+(?=[A-Z])")                                         # "... after death (b) Estates ..."
CITES = re.compile(r"\b(?:sections?|rules?|articles?|clauses?|regulations?|paragraphs?|paras?|schedules?|chapters?|parts?|"
                   r"orders?|items?|forms?|act|ordinance|no|of|and|to|under)\.?$", re.I)


def merged_at(heading: str, following: str) -> int | None:
    """Where a long contents entry runs into what comes after it, if it shows: the next provision's number, a
    lettered sub-entry, the opening words of the next provision, or its own first words printed again."""
    # (a number that a heading cites -- "Provision of section 11, Act III of 1852" -- is not the next entry's)
    numbers = (m for m in NEXT_NUMBER.finditer(heading) if not CITES.search(heading[:m.start()]))
    number, sub = next(numbers, None), SUB_ENTRY.search(heading)
    cuts = [m.start() for m, least in ((number, 15), (sub, 5)) if m and m.start() >= least]    # ("Other Leave (a) For ...")
    heading = re.sub(r"\s+\d{1,4}\s*$", "", heading)        # (a page or provision number left at its end)
    words = [m.start() for m in re.finditer(r"(?<=\s)\S", heading)]
    target = squash(following)[:200]
    for start in words:
        tail = squash(heading[start:])
        if start >= 15 and len(tail) >= 15 and target.startswith(tail):
            cuts.append(start)
            break
    first = squash(" ".join(heading.split()[:4]))
    for start in words:
        if start >= 15 and len(first) >= 12 and squash(heading[start:]).startswith(first):
            cuts.append(start)
            break
    return min(cuts) if cuts else None


# a sentence ending after a word -- not after an initial ("C.P. Fund") or an abbreviation ("Act No. II")
SENTENCE_BREAK = re.compile(r"(?<!\b[A-Za-z])(?<!\bNo)(?<!\bNos)(?<!\bArt)(?<!\bSec)(?<!\bGovt)(?<!\bOrd)(?<!\bLtd)(?<!\bPvt)"
                            r"(?<!\bCo)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bSt)(?<!\bVol)\.\s+(?=[A-Z(\[“\"])")
# a heading's own separator, printed without a full stop: "Application for registration of design——(1) An application",
# "Compliance with Other Laws - Nothing in these rules", "Responsibilities towards ... Proxy Voting: An Asset Management"
DASH_BREAK = re.compile(r"\s?[—–]{1,3}\s?-?(?=\S)|\s-\s|:\s+(?=[A-Z‘“(])")
SUB_SECTION = re.compile(r"\s\(\d{1,2}\)\s*(?=[A-Z])")           # "Short Title, Extent and Commencement (1) This Act shall"
RUN_ON = [NEXT_NUMBER, SUB_ENTRY, SUB_SECTION, DASH_BREAK, SENTENCE_BREAK]   # the next provision's number first: "... Risalpur. 229."
# how a rule begins, not a heading: "The Officer-in-Charge shall ...", "No person ...", "Subject to the provisions ..."
# (not "Where", "When", "No": "Where value of subject matter must be stated", "No decree to be reversed ..." are headings)
SENTENCE_START = re.compile(r"^(?:The|A|An|Any|Every|In|Subject|After|Notwithstanding|Following|Nothing|All|Such)\b")


def trim_run_on(head: str) -> str:
    """A printed heading that runs on into the provision's own text ("Training. The Authority shall formulate ...
    11. Maintenance of website", "PROCUREMENT OF CSR PROJECT/SCHEME (i) Schemes approved ... shall be executed"): cut
    where the text begins -- only where what follows reads as text, or the heading is too long to be one ("Payment to
    Govt. Servants" keeps its full stop)."""
    # every sign judged on the whole heading; the earliest that qualifies is where the text begins
    cuts = [m.start() for pattern in RUN_ON for m in pattern.finditer(head)
            if m.start() >= 3 and (pattern is not NEXT_NUMBER or not CITES.search(head[:m.start()]))
            and (BODY_WORDS.search(head[m.end():]) or len(head) > 100)]
    return head[:min(cuts)] if cuts else head


def choose_heading(listed: str | None, printed: str | None, following: str) -> str | None:
    """The contents' heading for a provision, unless it is long and the law shows it ran on into the next entry:
    then the heading the provision prints in front of its text, if it prints one, else the entry cut where the
    run-on begins. ("Administrator General not precluded ... after death (b) Estates of persons subject to the
    Army Act" is s. 14's heading and the sub-heading after it.)"""
    if not listed or len(listed) <= LONG_HEADING:
        return listed or printed
    # the printed heading must be the same heading (a footnote line printed above a section is not)
    if printed and len(printed) < len(listed) and not BODY_WORDS.search(printed) \
            and fuzz.ratio(squash(printed), squash(listed)[:len(squash(printed))]) >= 80:
        return printed
    cut = merged_at(listed, following)
    if cut is None:
        return listed
    return re.sub(r"\s\d{1,4}$", "", listed[:cut]).rstrip(" .,;:-–—")


def heading_from_text(unit: dict) -> str | None:
    """The heading a provision carries in front of its text: "Short title, extent and commencement"."""
    opening = " ".join(line.text for line in unit["lines"][:3])[:260]
    found = provision_of(opening)
    if not found:
        return None
    rest = found[1]
    if re.match(r"\[?\(1\)", rest):
        return None
    end = HEADING_END.search(rest)
    if end and 3 <= end.start() <= 180:
        head = rest[:end.start()]
        # "... must be endorsed with one of the followings:-" is a sentence leading into a list, not a heading -- unless
        # a heading stands before a dash in it: "Prohibition against discrimination,-No person shall ... following:-"
        if rest[end.start()] == ":":
            # (a full stop after a word or a year, not after an initial: "C.P. Fund" is one name)
            inner = re.search(rf"\s?[—–]|[.,]\s?-|{SENTENCE_BREAK.pattern}|:\s*(?=\()", head)
            if inner and inner.start() >= 3 and not BODY_WORDS.search(head[:inner.start()]):
                head = head[:inner.start()]     # "Levy of stallages, rents and fees. A Cantonment Administration may ...:—"
            elif BODY_WORDS.search(head):
                return None             # a sentence: no heading rather than a sentence for one
        return finish_heading(head)
    sentence = re.match(r"([A-Z][^.]{3,120})\.\s+[A-Z(]", rest)   # "2. Definitions. In this Act ..."
    # (the same checks: "30. Other Leave" / "(a) For the purpose of long leave, every department should ...")
    return finish_heading(sentence.group(1)) if sentence else None


# (a dash, not the hyphen inside "Bait-ul-Mal")
MARGIN_TAIL = re.compile(r"(?:[—–]{1,3}|-{2,}|-\s|_{2,})\s*([A-Z][^—–_]{2,60})$")


def tidy_heading(heading: str | None) -> str | None:
    """Last touches: dot leaders with a scan's noise after them ("Contempt of Court... a a we a", "... . . 277"), and
    the Pakistan Code's "__" printed for a dash ("Abetment of offence punishable with imprisonment—if offence be not
    committed")."""
    if not heading:
        return heading
    heading = re.sub(r"(?:\s*[.…·„]){3,}[^.…]{0,15}$", "", heading)
    heading = re.sub(r"\s*_{2,}\s*", "—", heading)
    return heading.strip(" -—–") or None


def finish_heading(head: str) -> str | None:
    """A heading candidate, run-on trimmed; None if it is a sentence of the law's own ("The Press and Publications
    Ordinance, 1960 ... are hereby repealed") -- though a heading may say "shall" ("Procedure where Court desires that
    its own decree shall be executed by another Court") -- unless the Sindh Code's marginal note ends it: "The
    functions of the Commission shall be— Functions of the Commission"."""
    head = trim_run_on(head).strip(" [“\"\xad").rstrip(",;:.~ -—–")
    note = MARGIN_TAIL.search(head)
    if note and BODY_WORDS.search(head[:note.start()]) and not BODY_WORDS.search(note.group(1)):
        return note.group(1).strip()        # ("No person shall____ Restrictions on hunting")
    if BODY_WORDS.search(head) and (SENTENCE_START.match(head) or len(head) > 120):
        return None
    return head or None


def unit_word(record: dict, text: str) -> str:
    """What this law calls its numbered provisions: by its kind, unless its own words say otherwise
    (the Constitution is an "Act" of Articles). Judged on the opening third: schedules speak of rules."""
    usual = DEFAULT_UNIT.get(record["instrument_type"], "paragraph")
    opening = text[:max(20000, len(text) // 3)]
    counts = {word: len(re.findall(pattern, opening)) for word, pattern in SELF_REFERENCE.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 3 and counts[best] > 2 * counts.get(usual, 0) else usual


# ---------------------------------------------------------------- 3. size

def span_of(labels: list[str]) -> str | None:
    if not labels:
        return None
    return f"({labels[0]})" if len(labels) == 1 else f"({labels[0]}) to ({labels[-1]})"


def by_size(lines: list[Line]) -> list[list[Line]]:
    """Last resort: cut after the line that reaches TARGET."""
    out, group = [], []
    for line in lines:
        group.append(line)
        if length(group) >= TARGET:
            out.append(group)
            group = []
    return out + [group] if group else out


def pieces(lines: list[Line], levels: list, depth: int = 0, label: str | None = None) -> list[tuple[list[Line], str | None]]:
    """Cut lines into pieces of at most SPLIT_ABOVE characters, at the highest-ranking
    marker that divides them: sub-sections before clauses before sentences."""
    if length(lines) <= SPLIT_ABOVE:
        return [(lines, label)]
    if depth == len(levels):
        return [(group, label) for group in by_size(lines)]
    marker = levels[depth]
    segments: list[tuple[str | None, list[Line]]] = []
    for n, line in enumerate(lines):
        if marker is None:                              # sentence level: a new segment after each full stop
            starts = n > 0 and bool(ENDS_SENTENCE.search(lines[n - 1].text))
            found = None
        else:
            match = marker.match(line.text) if n else None      # the first line is the opening, whatever it starts with
            starts, found = bool(match), match.group(1) if match else None
        if starts or not segments:
            segments.append((found, [line]))
        else:
            segments[-1][1].append(line)
    if len(segments) < 2:
        return pieces(lines, levels, depth + 1, label)

    out, group, labels = [], [], []

    def flush():
        if group:
            out.append((group[:], span_of(labels) or label))
            group.clear()
            labels.clear()

    for found, members in segments:
        size = length(members)
        if size > SPLIT_ABOVE:                          # too long on its own: cut it at the next kind of marker
            flush()
            out.extend(pieces(members, levels, depth + 1, f"({found})" if found else label))
            continue
        if group and (length(group) + size > SPLIT_ABOVE or length(group) >= TARGET):
            flush()
        group.extend(members)
        if found:
            labels.append(found)
    flush()

    merged: list[tuple[list[Line], str | None]] = []
    for part, part_label in out:                        # a few stray words are not a chunk of their own
        if merged and (length(part) < SMALL or length(merged[-1][0]) < SMALL) and length(merged[-1][0]) + length(part) <= SPLIT_ABOVE + 400:
            merged[-1] = (merged[-1][0] + part, merged[-1][1] or part_label)
        else:
            merged.append((part, part_label))
    return merged


def flow(lines: list[Line]) -> str:
    """The lines as running text: a line the PDF merely wrapped is joined to the one
    before; a new sub-section, clause or paragraph keeps its own line. Tables stay as they are."""
    solid = "".join(line.text for line in lines).replace(" ", "")
    if solid and sum(char.isdigit() for char in solid) > 0.25 * len(solid):
        return "\n".join(line.text for line in lines)
    out = []
    for line in lines:
        if out and not ENDS_SENTENCE.search(out[-1]) and not STARTS_BLOCK.match(line.text):
            out[-1] += " " + line.text
        else:
            out.append(line.text)
    return "\n".join(out)


def opening_words(lines: list[Line]) -> str:
    """The provision's own opening, repeated in front of its later parts so they can be read alone."""
    words = lines[0].text
    for line in lines[1:]:
        if SUB.match(line.text) or CLAUSE.match(line.text) or len(words) > 200:
            break
        words += " " + line.text
    return words if len(words) <= 260 else words[:260].rsplit(" ", 1)[0] + " ..."


# ---------------------------------------------------------------- 4. one law

LOOKALIKE = str.maketrans({"l": "1", "I": "1", "|": "1", "!": "1", "L": "1", "O": "0", "o": "0"})


def shows_number(number: int, text: str) -> bool:
    """Is this number printed at the head of the line? Allows what a scan does to digits ("I6." for
    16, "|." for 1), a marginal note in front, and a note mark run into the number ("1272." for 272)."""
    wanted = str(number)
    for token in re.split(r"[\s\[(]+", text[:45]):
        token = token.rstrip(".,:;)]}—–-_'\"”")
        for core in {token, re.sub(r"[A-Z]{1,3}$", "", token)}:      # with and without the "A" of 12A
            if 1 <= len(core) <= 5 and re.fullmatch(r"[0-9lI|!LOo]+", core):
                digits = core.translate(LOOKALIKE)
                if digits == wanted or (digits.endswith(wanted) and len(digits) - len(wanted) <= 2):
                    return True
    return False


def edge_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def margin_note_cuts(unit: dict, heading: str) -> list[tuple[int, str, int]] | None:
    """The Sindh and Balochistan Codes' texts run each provision's marginal note into its opening lines, a few
    words at the start or the end of each line:
        "1. (1) This Ordinance may be called the Sind Rented Short title and" / "Premises Ordinance, 1979. commencement."
        "Powers of the 4. (1) The Director of Fisheries, or such other officer" / "director to as he may empower ..."
    When the heading the contents give is found that way -- every word, in order, at an edge of consecutive lines
    from the first, the note's first word capitalised -- those words are the label, not the law's text: the cuts
    that take them off, as (line, "start" | "end", tokens). None when anything is less certain."""
    wanted = edge_words(heading)
    lines = unit["lines"]
    if not wanted or len(wanted) > 40 or not lines:
        return None
    found = provision_of(lines[0].text)
    if found and edge_words(found[1])[:len(wanted)] == wanted:
        return None                     # printed the usual way: "5. Agreement between landlord and tenant.— (1) ..."

    def take(tokens: list[str], p: int, from_end: bool) -> tuple[int, int]:
        """(tokens, words) of the longest run at one edge of the line that is the note's next words."""
        best = (0, 0)
        for k in range(1, min(len(tokens), 10) + 1):
            run = tokens[-k:] if from_end else tokens[:k]
            words = edge_words(" ".join(run))
            if words and words == wanted[p:p + len(words)] and (p or run[0].lstrip("[“\"(")[:1].isupper()):
                best = (k, len(words))
        return best

    cuts, p = [], 0
    for n, line in enumerate(lines[:14]):
        if p == len(wanted):
            break
        tokens = line.text.split()
        if n == 0:
            front = 0
            margin = MARGIN.match(line.text) if not found else None
            if margin:                  # in front of the number: all of it is the note's
                before = line.text[:margin.start(1)].split()
                k, w = take(before, p, False)
                if not k or k != len(before):
                    return None
                cuts.append((n, "start", k))
                p, front = p + w, k
                if p == len(wanted):
                    break
            k, w = take(tokens[front:], p, True)
            if k and len(tokens) - front - k >= 2:
                cuts.append((n, "end", k))
                p += w
            elif not cuts:
                return None             # the note begins beside the provision's first line, or it is not there
            continue
        start, end = take(tokens, p, False), take(tokens, p, True)
        if not start[0] and not end[0]:
            return None                 # a line without its piece of the note: not a marginal note
        side, (k, w) = ("start", start) if start[1] >= end[1] else ("end", end)
        cuts.append((n, side, k))
        p += w
    return cuts if p == len(wanted) else None


def lift_cuts(unit: dict, cuts: list[tuple[int, str, int]]) -> None:
    lines = unit["lines"]
    for n, side, k in cuts:
        tokens = lines[n].text.split()
        lines[n].text = " ".join(tokens[k:] if side == "start" else tokens[:-k])
    unit["lines"] = [line for line in lines if line.text.strip()]


def unglue_note_mark(unit: dict) -> None:
    """A footnote mark printed against a provision's number ("719." for section 19 with note 7,
    "1272." for Article 272) is taken off the start of its text; the label is already right."""
    first = unit["lines"][0]
    number, suffix = str(unit["key"][0]), unit["key"][1]
    glued = re.match(rf"^(\[?\s*)\d{{1,2}}({number}\s?-?\s?{re.escape(suffix)}\s?[.:,])", first.text)
    if glued:
        first.text = glued.group(1) + glued.group(2) + first.text[glued.end():]


TINY = 120          # characters: a chunk smaller than this is looked at again
OMITTED = re.compile(r"\[\s*(?:\*\s*){2,}\]|^\W*\d*[A-Z]*\W*\s*[\[(]?\s*(?:omitted|repealed|deleted)\b|(?:\*\s*){3,}"
                     # the word straight after the number and its marks: "47-52. [omitted]", "5[49. Deleted]",
                     # "16 and 17. [Omitted.]", "5A. 8[Repealed]", "5[18B OMITTED.]", "156. [Repeals.]"
                     # (a live "15. Repeals.— The X Act is hereby repealed" is not: only the bracketed "[Repeals.]")
                     r"|^[\W\d]*(?:(?:and|to)\s+[\W\d]+)?[A-Z]{0,3}\W*(?:\d{1,4}\s?[A-Z]{0,3}\W*)?(?:omitted|repealed|deleted)\b"
                     r"|^[\W\d]*\[\s*repeals?\.?\s*\]", re.I)
BRACKETED = re.compile(r"^\W{0,2}\(\s*[0-9lI|]{1,4}\s*[A-Za-z]{0,2}\s*\)")
MASTHEAD_SCRAP = re.compile(r"GAZETTE|EXTRAORDINARY|REGISTERED|PUBLISHED\s+BY\s+AUTHORITY|^\W*(?:ISLAMABAD|LAHORE|KARACHI|QUETTA|PESHAWAR|RAWALPINDI)\b", re.I)


def tidy_small(chunks: list[dict]) -> list[dict]:
    """Chunks too small to stand on their own:
      * an omitted or repealed provision ("7. [Omitted]") stays -- it answers "what does section 7
        say?" -- but is marked stub = "omitted" so it can be left out of the embedding;
      * a provision of a single short line whose number has a full chunk elsewhere in the law is a
        contents line left in the text, and is dropped;
      * a scrap of a gazette's masthead is dropped;
      * any other small piece without a number ("(See section 5)", "Division I") is put in front
        of the chunk that follows it, so its words stay with the text they introduce."""
    def where(chunk: dict) -> tuple:
        """A provision's address: its number within its Order (CPC rules), else within the law."""
        order = next((re.match(r"Order [IVXLC]+[A-Z]?", p).group(0) for p in chunk["path"] if re.match(r"Order [IVXLC]+", p)), "")
        return chunk["unit"], chunk["number"], order

    def heading_only(chunk: dict) -> bool:
        return bool(chunk["number"]) and chunk["parts"] == 1 and len(chunk["text"]) < 160 and "\n" not in chunk["text"] \
            and not re.search(r"\.\s+\S.*\.", chunk["text"])          # no sentence after the heading

    # Four or more heading-only provisions in a row are a contents list printed inside the text
    # (the Punjab Code puts one at the head of every Order of the CPC): one chunk, marked.
    grouped, run = [], []
    for chunk in chunks + [None]:
        if chunk is not None and heading_only(chunk) and (not run or run[-1]["path"] == chunk["path"]):
            run.append(chunk)
            continue
        if len(run) >= 4:
            grouped.append({**run[0], "unit": "contents", "number": None, "heading": "Contents", "span": None,
                            "text": "\n".join(c["text"] for c in run), "stub": "contents", "_lead": None})
        else:
            grouped.extend(run)
        run = [chunk] if chunk is not None and heading_only(chunk) else []
        if chunk is not None and not run:
            grouped.append(chunk)
    chunks = grouped

    full = {where(c) for c in chunks if c["number"] and len(c["text"]) >= TINY}
    out, carry = [], None
    for chunk in chunks:
        small = len(chunk["text"]) < TINY
        if chunk.get("stub") == "contents":
            out.append(chunk)
            continue
        if small and OMITTED.search(chunk["text"]):
            chunk["stub"] = "omitted"
            out.append(chunk)
            continue
        if small and chunk["number"] and chunk["parts"] == 1 and "\n" not in chunk["text"] and where(chunk) in full:
            continue
        if small and not chunk["number"]:
            if MASTHEAD_SCRAP.search(chunk["text"]):
                continue
            if chunk["unit"] == "preamble":     # a short title-and-number block still stands on its own
                out.append(chunk)
                continue
            carry = chunk if carry is None else {**carry, "text": carry["text"] + "\n" + chunk["text"], "notes": carry["notes"] + chunk["notes"]}
            continue
        if carry is not None:
            chunk = {**chunk, "text": carry["text"] + "\n" + chunk["text"], "notes": list(dict.fromkeys(carry["notes"] + chunk["notes"])),
                     "page_start": min(p for p in (carry["page_start"], chunk["page_start"]) if p is not None) if (carry["page_start"] or chunk["page_start"]) else None}
            carry = None
        out.append(chunk)
    if carry is not None:
        if out:
            out[-1] = {**out[-1], "text": out[-1]["text"] + "\n" + carry["text"], "notes": out[-1]["notes"] + carry["notes"]}
        else:
            out.append(carry)
    return out


def locate(pointers: dict, lines: list[Line]) -> dict[int, tuple[tuple[int, str], str | None]]:
    """{line index: (number, heading)} for chunk_model.py's pointers. Each is checked against the text:
    the line must be the one it was made from (the cleaning may have changed since; then it is looked
    for nearby), and the number must be printed on it. On a scan the reader cannot follow it numbers
    line after line in order; those pointers show no number and are dropped here."""
    where = {}
    for start in pointers["starts"]:
        near = sorted(range(max(0, start["line"] - 60), min(len(lines), start["line"] + 61)), key=lambda n: abs(n - start["line"]))
        index = next((n for n in near if lines[n].text[:120] == start["text"]), None)
        number = re.fullmatch(r"(\d{1,3})([A-Z]{0,3})", start["number"])
        # a number in brackets at the head of the line is a sub-section or a definition ("(9) “Aircraft Operator”
        # means ..."), which the reader took for section 9 of the Balochistan Sales Tax on Services Act
        if index is not None and BRACKETED.match(lines[index].text):
            continue
        if index is not None and number and shows_number(int(number.group(1)), lines[index].text):
            where[index] = ((int(number.group(1)), number.group(2)), start["heading"])
    # still a pointer on nearly every line: a list or table read as provisions
    order = sorted(where)
    if len(order) >= 15 and statistics.median(b - a for a, b in zip(order, order[1:])) <= 1:
        return {}
    return where


def cut(record: dict, raw: str, pointers: dict | None = None) -> tuple[list[dict], dict]:
    """(chunks, report) for one law. `pointers` is chunk_model.py's row for it, if there is one."""
    lines, footnote_chars = clean(raw)
    lines_at = {id(line): n for n, line in enumerate(lines)}
    total = length(lines)
    report = {"title": record["title"], "instrument_type": record["instrument_type"], "text_source": record["text_source"],
              "chars": total, "method": "whole", "contents_entries": 0, "provisions": 0, "problems": []}
    if not lines:
        report["problems"].append("no text left after cleaning")
        return [], report
    if len(raw) > 2 * WHOLE_BELOW and footnote_chars > 0.35 * len(raw):     # (short old Acts really are mostly notes)
        report["problems"].append("over a third of the text was read as footnotes")

    word = unit_word(record, raw)
    units, contents = [{"kind": "document", "number": None, "key": None, "path": [], "lines": lines, "zone": "main", "heading": None}], None
    listing = None      # the table of contents, kept for its headings even where the provisions were found otherwise
    if total > WHOLE_BELOW:
        provisions = [provision_of(line.text) for line in lines]
        headings = [heading_of(line.text, lines[n - 1].text if n else "") for n, line in enumerate(lines)]
        start, contents = find_contents(lines, provisions, headings)
        if contents and contents.chars > 0.4 * total:      # far too long for a table of contents: a misreading
            start, contents = 0, None
        if contents is None and record["instrument_type"] in STRUCTURED:
            # the Balochistan Code prints the marginal note in front of the number ("Short title and 1. (1) This
            # Ordinance ..."): its body is numbered only that way, and unseen, the contents list was read as the law
            # and its "FIRST SCHEDULE." entry took in the whole body
            beside = [found or margin_provision(line.text) for found, line in zip(provisions, lines)]
            margin_start, margin_contents = find_contents(lines, beside, headings)
            if margin_contents:
                start, contents, provisions = margin_start, margin_contents, beside
        if contents and contents.chars > 0.4 * total:      # far too long for a table of contents: a misreading
            start, contents = 0, None
        listing = contents
        units = walk(lines, provisions, headings, start, contents, word)
        if contents:
            # a garbled table of contents must not hide provisions that plain counting finds
            counted = walk(lines, provisions, headings, start, None, word)
            found = sum(unit["key"] is not None and unit["zone"] == "main" for unit in units)
            if found < 0.6 * len(contents.order) and sum(unit["key"] is not None for unit in counted) > 1.5 * found:
                units, contents = counted, None
                report["problems"].append("the table of contents could not be followed; provisions found by counting")

        def carried(found: list[dict]) -> bool:
            """Do numbered provisions hold the body of the law?"""
            numbered = [unit for unit in found if unit["key"] is not None]
            outside = sum(length(unit["lines"]) for unit in found if unit["zone"] != "schedule" and unit["kind"] != "notes")
            return len(numbered) >= 3 and sum(length(unit["lines"]) for unit in numbered) >= 0.5 * outside

        def count(found: list[dict]) -> int:
            return sum(unit["key"] is not None for unit in found)

        # (a walk the contents guide that finds under half of what they list also gets the second reading: the
        # Balochistan Public Safety Regulation's ten bracketed "[Repealed]" sections carried the law on their own)
        followed_poorly = contents is not None and count(units) < 0.5 * len(contents.order)
        if (not carried(units) or followed_poorly) and record["instrument_type"] in STRUCTURED:
            # old gazettes print the marginal note in front of the number: "Short title 1. (1) This Act ..."
            beside = [found or margin_provision(line.text) for found, line in zip(provisions, lines)]
            retry = walk(lines, beside, headings, start, None, word)
            numbers = {unit["key"][0] for unit in retry if unit["key"] is not None}
            if carried(retry) and len(numbers) >= 4 and max(numbers) - min(numbers) + 1 - len(numbers) <= 0.25 * len(numbers) \
                    and (not carried(units) or count(retry) > count(units)):
                units, contents = retry, None
        if pointers:
            # chunk_model.py's reading: used where it finds more provisions than the rules did
            where = locate(pointers, lines)
            aided = [(where[n][0], lines[n].text) if n in where else found for n, found in enumerate(provisions)]
            # from where the law opens: a table of contents whose headings could not be followed is still a
            # table of contents, and read from line 0 its entries count as provisions (Sind Rented Premises, 1979)
            retry = walk(lines, aided, headings, start, None, word, frozenset(where))
            # (against the rules' own count only where their cut holds the law: contents lines counted as provisions do not)
            if carried(retry) and (not carried(units) or count(retry) > count(units)):
                units, contents = retry, None
                report["read_by_model"] = True
                for unit in units:      # its heading is used only when those very words open the provision
                    said = where.get(lines_at.get(id(unit["lines"][0])), (None, None))[1] if unit["key"] is not None else None
                    said = re.split(r"\s*[-–—:.]?\s*\[?\(1\)", said)[0][:120] if said else None    # the heading, not the text after it
                    if said and len(squash(said)) >= 4 and squash(said) in squash(" ".join(line.text for line in unit["lines"][:3])):
                        unit["heading"] = said.strip(" .:—–-")
        numbered = [unit for unit in units if unit["key"] is not None]
        main = [unit for unit in numbered if unit["zone"] == "main"]
        if carried(units):
            report["method"] = "provisions"
            report["provisions"] = len(numbered)
            if contents:
                report["contents_entries"] = len(contents.order)
                if len(main) < 0.85 * len(contents.order):
                    report["problems"].append(f"the contents list {len(contents.order)} provisions; {len(main)} were found")
            elif main:
                numbers = {unit["key"][0] for unit in main}
                missing = max(numbers) - min(numbers) + 1 - len(numbers)
                if missing > max(2, 0.15 * len(numbers)):
                    report["problems"].append(f"{missing} numbers are missing between provisions {min(numbers)} and {max(numbers)}")
        else:
            # no numbering to follow: one run of text, cut into passages
            report["method"] = "passages"
            units = [{"kind": "passage", "number": None, "key": None, "path": [], "lines": lines[start:], "zone": "main", "heading": None}]
            if record["instrument_type"] in STRUCTURED and total > 2 * WHOLE_BELOW:
                report["problems"].append("no numbered provisions found")

    # marginal notes run into the text (Sindh and Balochistan Codes) are lifted off it -- but only in a law set out
    # that way throughout: most provisions whose heading the contents give show it, and some notes run over lines
    known = {id(unit): listing.heading.get(unit["key"]) for unit in units
             if listing and unit["key"] is not None and unit["zone"] == "main" and not unit.get("repeat") and listing.heading.get(unit["key"])}
    plans = {id(unit): plan for unit in units if id(unit) in known and (plan := margin_note_cuts(unit, known[id(unit)]))}
    if len(plans) >= 2 and len(plans) >= 0.5 * len(known) and sum(len({n for n, _, _ in plan}) > 1 for plan in plans.values()) >= 2:
        for unit in units:
            if id(unit) in plans:
                lift_cuts(unit, plans[id(unit)])
        report["margin_notes_lifted"] = len(plans)
    else:
        plans = {}

    chunks = []
    for n, unit in enumerate(units):
        if unit["key"] is not None:
            listed = (contents.heading.get(unit["key"]) if contents and unit["zone"] == "main" and not unit.get("repeat") else None) \
                or (known[id(unit)] if id(unit) in plans else None)
            following = next((" ".join(line.text for line in later["lines"][:3]) for later in units[n + 1:] if later["key"] is not None), "")
            following = re.sub(r"^\W*\d{1,4}\s?-?\s?[A-Z]{0,3}\s?[.:]\s*(?:\(1\)\s*)?", "", following)
            unit["heading"] = tidy_heading(choose_heading(listed, heading_from_text(unit), following) or unit["heading"])
            unglue_note_mark(unit)
        if unit["kind"] == "document":
            parts = [(unit["lines"], None)]
        else:
            levels = [SUB, ITEM, CLAUSE, None] if unit["key"] is not None else [ITEM, SUB, CLAUSE, None]
            parts = pieces(unit["lines"], levels)
        for n, (members, span) in enumerate(parts):
            text = flow(members)
            if not text.strip():
                continue
            pages = [line.page for line in members if line.page is not None]
            notes = list(dict.fromkeys(note for line in members for note in line.notes))
            solid = text.replace(" ", "").replace("\n", "")
            chunks.append({
                "unit": unit["kind"], "number": unit["number"], "heading": unit["heading"], "path": unit["path"],
                "span": span, "part": n + 1, "parts": len(parts),
                "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None,
                "table_like": bool(solid) and sum(char.isdigit() for char in solid) > 0.25 * len(solid),
                "text": text, "notes": notes, "stub": None,
                "_lead": opening_words(unit["lines"]) if n and unit["key"] is not None else None,
            })

    chunks = tidy_small(chunks)
    outline = [f"{chunk['number']}. {chunk['heading']}" for chunk in chunks
               if chunk["unit"] not in ("rule",) or word == "rule" if chunk["number"] and chunk["heading"] and chunk["part"] == 1]
    chunks.insert(0, card(record, outline))

    shared = {field: record.get(field) for field in ("source", "work_id", "family_id", "authority_rank", "title", "instrument_type", "year",
                                                      "jurisdiction", "status", "status_applies_in", "status_exceptions", "practice_area",
                                                      "text_source", "tier", "quarantine_reasons", "last_amended",
                                                      # what the law replaced and what replaced it (kg.link_succession)
                                                      "replaces", "replaced_by")}
    # how far the law can be trusted (confidence.py), and the one line of evidence for its status
    shared["confidence"] = {k: v for k, v in (record.get("confidence") or {}).items() if k != "reasons"}
    shared["status_evidence"] = (record.get("status_basis") or [None])[0]
    rows = []
    for seq, chunk in enumerate(chunks):
        lead = chunk.pop("_lead", None)
        header = label_of(record, chunk)
        body = (lead + " [...]\n" if lead else "") + chunk["text"]
        # which collection a chunk is served from: current (and unconfirmed, the Sindh laws labelled "status not
        # confirmed") answer questions; historical is kept apart, shown only as what the law was before; quarantined
        # laws and stubs (an omitted section, a contents list) are never served
        serve = record.get("tier") if record.get("tier") in ("current", "unconfirmed", "historical") and not chunk.get("stub") else None
        rows.append({"chunk_id": f"{record['work_id']}:{seq:05d}", "doc_id": record["_key"], **shared, "seq": seq,
                     "method": report["method"], **chunk, "chars": len(chunk["text"]), "serve": serve,
                     "embed_text": header + "\n" + body})
    report["chunks"] = len(rows)
    report["largest_chunk"] = max(row["chars"] for row in rows)
    return rows, report


def label_of(record: dict, chunk: dict) -> str:
    """The line put in front of a chunk before it is embedded: which law, and where in it."""
    bits = [record["title"]] + chunk["path"]
    if chunk["number"]:
        name = f"{chunk['unit'].title()} {chunk['number']}" + (f": {chunk['heading']}" if chunk["heading"] else "")
        if chunk["parts"] > 1:
            name += f" [{chunk['span']}]" if chunk["span"] else f" [part {chunk['part']} of {chunk['parts']}]"
        bits.append(name)
    elif chunk["unit"] == "preamble":
        bits.append("Title and preamble")
    elif chunk["unit"] == "card":
        bits.append("Overview")
    elif chunk["unit"] == "notes":
        bits.append("Notes on amendments")
    elif chunk["parts"] > 1:
        bits.append(f"part {chunk['part']} of {chunk['parts']}")
    return " — ".join(bits)


def status_line(record: dict) -> str | None:
    """The law's status in one sentence, from the evidence the knowledge graph settled on (kg.py, confidence.py)."""
    status, tier = record.get("status"), record.get("tier")
    related = record.get("related") or []
    by = next((x for x in related if x["relation"] == "repealed_by"), None)
    amended = f" Last amended {record['last_amended']}." if record.get("last_amended") else ""
    replaced = ""
    if record.get("replaces"):
        replaced = " It replaced " + "; ".join(f"{p['title']}" for p in record["replaces"][:4]) + "."
    if status == "in_force" and tier == "current":
        line = f"Status: in force ({record.get('status_applies_in') or record.get('jurisdiction')})."
        for exception in record.get("status_exceptions") or []:
            line += f" Repealed for the {exception['territory']} by {exception['by']}."
        return line + amended + replaced
    if tier == "unconfirmed":
        return ("Status: not confirmed. The Sindh Code does not say which laws are current; nothing found repeals this one."
                + amended + replaced)
    if status == "repealed":
        return (f"Status: repealed{' by ' + by['title'] if by else ''}; this is what the law was before."
                + ("" if (record.get("confidence") or {}).get("status") == "verified" else " (The repeal is probable, not confirmed.)"))
    if status == "superseded":
        enacted = next((x for x in related if x["relation"] == "enacted_as"), None)
        return f"Status: replaced{' by ' + enacted['title'] if enacted else ''}; not in force."
    if status == "lapsed_presumed":
        return "Status: an Ordinance presumed to have lapsed; not in force."
    return "Status: not confirmed."


def card(record: dict, outline: list[str]) -> dict:
    """One generated chunk describing the law as a whole, so that "which law covers X" finds it."""
    facts = ", ".join(str(value) for value in (record["instrument_type"], record["year"], record["jurisdiction"]) if value)
    text = f"{record['title']} ({facts})."
    described = record.get("summary") or record.get("one_line")
    if described:
        text += "\n" + described
    status = status_line(record)
    if status:
        text += "\n" + status
    if record.get("subjects"):
        text += "\nSubjects: " + ", ".join(record["subjects"][:8]) + "."
    if outline:
        listing = "\nProvisions: " + "; ".join(outline)
        text += listing if len(text) + len(listing) <= CARD_MAX else listing[:CARD_MAX - len(text)].rsplit(";", 1)[0] + "; ..."
    return {"unit": "card", "number": None, "heading": None, "path": [], "span": None, "part": 1, "parts": 1,
            "page_start": None, "page_end": None, "table_like": False, "text": text, "notes": [], "stub": None}


# ---------------------------------------------------------------- 5. copies of federal laws

# A provincial code prints many federal laws. Where the province has amended one (the Code of Civil Procedure on the
# Punjab Code, with its Punjab amendments) its copy is that province's law and is served for it. Where it merely
# reproduces one, the copy can be decades old: the Punjab Code's Penal Code still reads "375. Rape.- [Repealed by the
# Offence of Zina ... Ordinance, 1979]", lacks section 336A and the 38 others inserted since, and keeps section 325
# (attempt to commit suicide, omitted in 2022). Such a copy is not served; the Pakistan Code's text answers for the
# province instead. The Constitution can only be amended by Parliament, so no provincial copy of it is ever served
# (the Punjab Code's still gives the Supreme Court the original jurisdiction of Article 184, omitted in 2025).
OWN_PROVINCE = {"punjab": r"Punjab", "sindh": r"Sindh?", "kp": r"(?:N\.?\s?W\.?\s?F\.?\s?P\.?|Khyber\s+Pakhtunkhwa|North[- ]West\s+Frontier)",
                "balochistan": r"Ba[lu]ochistan"}
DEAD_TEXT = re.compile(r"\b(?:omitted|repealed|deleted)\b|\brep\.|(?:\*\s*){3,}", re.I)
CONSTITUTION = re.compile(r"\bconstitution of the islamic republic of pakistan\b", re.I)


def in_force_provisions(rows: list[dict]) -> dict[tuple[str, str], bool]:
    """(unit, number) -> whether the law's text gives that provision in force (not an omitted or repealed stub)."""
    out = {}
    for row in rows:
        if row["number"] and row["part"] == 1 and row["unit"] in ("section", "article", "rule"):
            dead = row.get("stub") == "omitted" or (len(row["text"]) < 300 and DEAD_TEXT.search(row["text"]))
            out[(row["unit"], re.sub(r"[^0-9A-Z]", "", row["number"].upper()))] = not dead
    return out


def federal_text(record: dict, rows: list[dict]) -> dict:
    return {"work_id": record["work_id"], "title": record["title"], "last_amended": record.get("last_amended"),
            "provisions": in_force_provisions(rows)}


def stale_copy(record: dict, rows: list[dict], federal: dict) -> dict | None:
    """Why a provincial code's copy of a federal law is not served, or None if it is. A copy is stale when the
    Pakistan Code's text gives at least three provisions in force that the copy lacks or prints as repealed, at least
    twice as many as the other way round, while the copy shows no amendment by its own province and none later than
    the Pakistan Code's latest."""
    own = OWN_PROVINCE.get(record["source"])
    base = federal.get(record.get("family_id"))
    if not own or not base or record.get("jurisdiction") != "Federal" or record.get("tier") not in ("current", "unconfirmed"):
        return None
    if CONSTITUTION.search(record["title"]):
        return {"work_id": base["work_id"], "title": base["title"], "why": "the Constitution is amended by Parliament alone"}
    copy = in_force_provisions(rows)
    only_federal = sum(1 for key, alive in base["provisions"].items() if alive and not copy.get(key))
    only_copy = sum(1 for key, alive in copy.items() if alive and not base["provisions"].get(key))
    amendments = sum(len(re.findall(rf"\b{own}\s+(?:Act|Ordinance|Amendment)\b|\(\s*{own}\s+Amendment", row["text"] + " " + " ".join(row["notes"]), re.I))
                     for row in rows)
    later = (record.get("last_amended") or 0) > (base.get("last_amended") or 0)
    if amendments or later or only_federal < 3 or only_federal < 2 * only_copy:
        return None
    return {"work_id": base["work_id"], "title": base["title"],
            "why": f"the Pakistan Code's text gives {only_federal} provisions in force that this copy lacks or prints as repealed "
                   f"({only_copy} the other way round), and this copy shows no amendment by its province"}


def federal_texts() -> dict:
    """family -> the Pakistan Code's current text of each federal law, from its chunks already written."""
    by_work = {}
    for row in read_jsonl(DATA_DIR / "chunks.jsonl"):
        if row["tier"] == "current":
            by_work.setdefault(row["work_id"], []).append(row)
    out = {}
    for rows in by_work.values():
        record = {"work_id": rows[0]["work_id"], "title": rows[0]["title"], "last_amended": rows[0].get("last_amended")}
        text = federal_text(record, rows)
        family = rows[0]["family_id"]
        if family not in out or sum(text["provisions"].values()) > sum(out[family]["provisions"].values()):
            out[family] = text
    return out


# ---------------------------------------------------------------- run

def laws(source: str, only: re.Pattern | None):
    """(metadata row, raw text) for each law of a source that goes in the index."""
    data_dir, key = source_layout(source)
    wanted = {}
    with (data_dir / "metadata_final.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["index_default"] and (only is None or only.search(row["title"])):
                row["_key"] = row[key]
                wanted[row[key]] = row
    with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row[key] in wanted:
                yield wanted[row[key]], row["raw_text"]


def write_sample(record: dict, rows: list[dict], report: dict) -> Path:
    SAMPLE_DIR.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", record["title"].lower()).strip("-")[:60]
    path = SAMPLE_DIR / f"{record['source']}--{name}.txt"
    with path.open("w", encoding="utf-8") as out:
        out.write(f"{record['title']}\n{json.dumps(report, ensure_ascii=False)}\n")
        for row in rows:
            pages = f" | pages {row['page_start']}-{row['page_end']}" if row["page_start"] else ""
            out.write(f"\n{'=' * 100}\n#{row['seq']} | {row['unit']} {row['number'] or ''} | {row['chars']} chars{pages}"
                      f"{' | table' if row['table_like'] else ''}\n{row['embed_text']}\n")
            for note in row["notes"]:
                out.write(f"    note: {note}\n")
    return path


def percentile(values: list[int], share: float) -> int:
    return sorted(values)[min(len(values) - 1, int(len(values) * share))] if values else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, help="one source only (default: all)")
    parser.add_argument("--only", help="only laws whose title matches this regular expression (writes nothing but samples)")
    parser.add_argument("--sample", action="store_true", help="write each law's chunks as readable text to data/chunk_samples/")
    args = parser.parse_args()
    only = re.compile(args.only, re.I) if args.only else None

    started = time.time()
    summary = {"by_source": {}}
    sizes, methods, units, problems = [], Counter(), Counter(), Counter()
    read_by_model = 0
    run = [args.source] if args.source else SOURCES
    # the Pakistan Code's text of each federal law, to judge the provincial codes' copies by: from this run when the
    # Pakistan Code is cut first in it, else from its chunks already written
    federal = {} if run[0] == "pakistancode" else federal_texts()
    stale = []
    for source in run:
        data_dir, key = source_layout(source)
        # written beside the old files and swapped in when complete: a power cut, or chunk_model.py
        # reading the report meanwhile, never meets a half-written file
        outputs = None if only else ((data_dir / "chunks.jsonl.tmp").open("w", encoding="utf-8"),
                                     (data_dir / "chunk_report.jsonl.tmp").open("w", encoding="utf-8"))
        count = chunk_count = 0
        pointed = {row["work_id"]: row for row in read_jsonl(data_dir / "chunk_model.jsonl")} if (data_dir / "chunk_model.jsonl").exists() else {}
        for record, raw in laws(source, only):
            rows, report = cut(record, raw, pointed.get(record["work_id"]))
            if source == "pakistancode" and record.get("tier") == "current":
                text = federal_text(record, rows)
                if record["family_id"] not in federal or sum(text["provisions"].values()) > sum(federal[record["family_id"]]["provisions"].values()):
                    federal[record["family_id"]] = text
            judged = stale_copy(record, rows, federal)
            if judged:
                for row in rows:
                    row["serve"] = None
                    row["stale_copy_of"] = judged["work_id"]
                report["stale_copy_of"] = judged
                stale.append((source, record["title"], judged["title"], judged["why"]))
            read_by_model += report.get("read_by_model", False)
            count += 1
            chunk_count += len(rows)
            methods[report["method"]] += 1
            for problem in report["problems"]:
                problems[re.sub(r"\d+", "N", problem)] += 1
            for row in rows:
                units[row["unit"]] += 1
                if row["unit"] != "card":
                    sizes.append(row["chars"])
            if outputs:
                for row in rows:
                    outputs[0].write(json.dumps(row, ensure_ascii=False) + "\n")
                outputs[1].write(json.dumps({key: record["_key"], "work_id": record["work_id"], **report}, ensure_ascii=False) + "\n")
            if only or args.sample:
                where = write_sample(record, rows, report) if args.sample else ""
                print(f"  {source:12s} {record['title'][:55]:55s} {report['method']:10s} {len(rows):5d} chunks  "
                      f"contents {report['contents_entries']:4d}  provisions {report['provisions']:4d}  {'; '.join(report['problems'])}  {where}")
        if outputs:
            for handle in outputs:
                handle.close()
            for name in ("chunks.jsonl", "chunk_report.jsonl"):
                os.replace(data_dir / (name + ".tmp"), data_dir / name)
        summary["by_source"][source] = {"laws": count, "chunks": chunk_count}
        print(f"{source:13s} {count:5d} laws  {chunk_count:7d} chunks")

    summary.update({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "laws": sum(s["laws"] for s in summary["by_source"].values()), "chunks": sum(s["chunks"] for s in summary["by_source"].values()),
        "laws_by_method": dict(methods.most_common()), "laws_structured_with_the_model": read_by_model,
        "chunks_by_unit": dict(units.most_common()),
        "chunk_chars": {"median": percentile(sizes, .5), "p10": percentile(sizes, .1), "p90": percentile(sizes, .9),
                        "p99": percentile(sizes, .99), "largest": max(sizes, default=0),
                        "under_200": sum(s < 200 for s in sizes), "over_2600": sum(s > SPLIT_ABOVE + 400 for s in sizes)},
        "laws_with_problems": dict(problems.most_common()),
        "stale_copies_not_served": [{"source": s, "title": t, "instead": i, "why": w} for s, t, i, w in stale],
    })
    if not only and not args.source:
        (DATA_DIR / "chunk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{summary['laws']} laws -> {summary['chunks']} chunks  ({time.time() - started:.0f}s)")
    print("  how laws were cut:", summary["laws_by_method"], "| structured with the model's help:", read_by_model)
    print("  chunks by unit:", summary["chunks_by_unit"])
    print("  chunk size (characters):", summary["chunk_chars"])
    print("  laws the rules struggled with:", summary["laws_with_problems"])
    print(f"  provincial copies of federal laws not served, the Pakistan Code's text being later ({len(stale)}):")
    for source, title, _, why in stale:
        print(f"    {source:12s} {title[:60]:60s} {why}")


if __name__ == "__main__":
    main()
