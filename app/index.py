"""
app/index.py

The served part of the corpus, held in memory for searching:

    chunks      every chunk with `serve` set (chunk_statutes.py): current and unconfirmed laws answer
                questions, historical laws are kept apart and shown only as what the law was before
    vectors     their Kanon 2 vectors, searched in the Qdrant vector database (app/vectordb.py) when it is running,
                else held here in half precision; a chunk whose text changed since it was embedded has no vector
                until eval/kanon_eval.py --embed embeds it again
    keywords    a BM25 index over the answer chunks, cached in data/app/ next to the chunk files it was built from
    sections    (law, provision number) -> chunks, for questions that cite a provision ("s. 489-F PPC")

Loading takes about a minute the first time (the caches are then reused while the chunk files are unchanged).
"""

import hashlib
import json
import math
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from app.config import ANSWER_COLLECTION, APP_DATA, EMBED_STORE, HISTORY_COLLECTION, ROOT

sys.path.insert(0, str(ROOT))
from ocr_scanned import SOURCES, source_layout  # noqa: E402

KEEP = ("chunk_id", "work_id", "family_id", "source", "title", "instrument_type", "year", "jurisdiction", "status",
        "status_applies_in", "status_exceptions", "status_evidence", "tier", "serve", "authority_rank", "unit", "number",
        "heading", "path", "span", "part", "parts", "seq", "text", "replaces", "replaced_by", "last_amended", "stub")
K1, B = 1.2, 0.75
TOKEN = re.compile(r"\w+")
STOP = set("""a an and are as at be by for from has have he her his i if in into is it its me my of on or our she so that the their them
then there these they this to was we were what when where which who will with you your can do does how may shall should would could not no
ka ki ke hai hain kya mein main se ko aur par ne to bhi ho kar karna liye wala wali ye yeh woh wo jo agar kis kaise kab""".split())
LOADED_AS = "2026-10-20 omitted provisions kept for citations"
SCORE_BLOCK = 20000   # rows scored at a time (half-precision rows are widened to float32 block by block)


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


SOURCE_HOME = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
# a law on the federal sites that names a place applies only there: the Islamabad Rent Restriction Ordinance, the Karachi
# Rent Restriction Act, the Sind Revenue Jurisdiction Act
PLACE_IN_TITLE = [(re.compile(r"\b(?:islamabad|capital territory|federal capital)\b", re.I), "Islamabad Capital Territory"),
                  (re.compile(r"\b(?:karachi|sindh?|hyderabad|sukkur|larkana)\b", re.I), "Sindh"),
                  (re.compile(r"\b(?:lahore|punjab|rawalpindi|multan|faisalabad|gujranwala)\b", re.I), "Punjab"),
                  (re.compile(r"\b(?:peshawar|khyber|pakhtunkhwa|n\.?\s?-?w\.?\s?f\.?\s?p|north[- ]west frontier)\b", re.I), "Khyber Pakhtunkhwa"),
                  (re.compile(r"\b(?:quetta|bal[ou]chistan|gwadar|kalat)\b", re.I), "Balochistan")]


def home_of(chunk: dict) -> str:
    """Where a chunk's law applies: the province whose code it is on, a place its title names, else "federal"."""
    home = SOURCE_HOME.get(chunk["source"])
    if home:
        return home
    for pattern, place in PLACE_IN_TITLE:
        if pattern.search(chunk["title"]):
            return place
    return "federal"


def provision_key(number) -> str:
    """A provision number as written either way: "489-F", "489 F" and "489F" are one section."""
    return re.sub(r"[^0-9A-Z]", "", str(number).upper())


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]    # as eval/kanon_eval.py names a chunk's text


def label(chunk: dict) -> str:
    """Where in the law a chunk sits: "Section 7: Talaq", "Schedule ... (part 2 of 3)"."""
    bits = list(chunk.get("path") or [])
    if chunk.get("number"):
        bits.append(f"{(chunk.get('unit') or 'section').title()} {chunk['number']}" + (f": {chunk['heading']}" if chunk.get("heading") else ""))
    elif chunk.get("unit") == "card":
        bits.append("Overview")
    elif chunk.get("unit") == "preamble":
        bits.append("Title and preamble")
    if (chunk.get("parts") or 1) > 1:
        bits.append(f"part {chunk['part']} of {chunk['parts']}")
    return " — ".join(bits)


class Corpus:
    def __init__(self, load_vectors: bool = True):
        started = time.time()
        APP_DATA.mkdir(parents=True, exist_ok=True)
        self.chunks = self._load_chunks()
        self.by_work = defaultdict(list)
        for cid, chunk in self.chunks.items():
            self.by_work[chunk["work_id"]].append(cid)
        for ids in self.by_work.values():
            ids.sort(key=lambda cid: self.chunks[cid]["seq"])
        self.fold_family = self._west_pakistan_copies()
        self.vector_db = None
        self.answer_ids, self.answer_vectors = [], None
        self.history_ids, self.history_vectors = [], None
        if load_vectors:
            from app import vectordb
            if vectordb.available() and vectordb.client().collection_exists("answers"):
                self.vector_db = vectordb.client()
                where = f"the Qdrant vector database ({self.vector_db.count('answers').count:,} answer and " \
                        f"{self.vector_db.count('history').count:,} history vectors)"
            else:
                self.answer_ids, self.answer_vectors = self._load_vectors(ANSWER_COLLECTION)
                self.history_ids, self.history_vectors = self._load_vectors(HISTORY_COLLECTION)
                where = f"memory ({len(self.answer_ids):,} answer and {len(self.history_ids):,} history vectors; Qdrant is not running)"
        self._build_keywords()
        self._build_sections()
        print(f"corpus ready: {len(self.chunks):,} chunks; vectors in {where if load_vectors else 'nothing (not loaded)'} "
              f"({time.time() - started:.0f}s)", flush=True)

    def _west_pakistan_copies(self) -> dict:
        """work -> the family its copies fold into, where it differs from its own: the Pakistan Code prints the laws of
        the old province of West Pakistan as they apply in Islamabad, and the provinces print their own versions under
        their own names ("The Punjab Control of Goondas Ordinance, 1959" for "The West Pakistan Control of Goondas
        Ordinance, 1959"), so the two are one law to fold -- the province's text for the province, else the Pakistan
        Code's. Same-named laws of two provinces are not joined: those are different laws."""
        def core(title: str) -> str:
            title = re.sub(r"\b(?:the|west\s+pakistan|punjab|sindh?|khyber\s+pakhtunkhwa|n\.?\s?w\.?\s?f\.?\s?p\.?|bal[ou]chistan)\b",
                           " ", title.lower())
            return re.sub(r"[^a-z0-9]+", " ", title).strip()
        first = {}
        for ids in self.by_work.values():
            chunk = self.chunks[ids[0]]
            first[chunk["work_id"]] = chunk
        west = {}
        for work, chunk in first.items():
            if chunk["source"] == "pakistancode" and chunk.get("jurisdiction") == "West Pakistan" and chunk["serve"] in ANSWER_COLLECTION:
                west.setdefault(core(chunk["title"]), chunk["family_id"])
        joined = {}
        for work, chunk in first.items():
            family = west.get(core(chunk["title"])) if chunk["source"] in SOURCE_HOME else None
            if family and family != chunk["family_id"]:
                joined[work] = family
        return joined

    # ---- loading
    def _load_chunks(self) -> dict:
        files = [source_layout(s)[0] / "chunks.jsonl" for s in SOURCES]
        stamp = [LOADED_AS] + [f.stat().st_mtime_ns for f in files]      # (what is kept changes with the code too)
        cache = APP_DATA / "chunks.pkl"
        if cache.exists():
            with cache.open("rb") as f:
                saved = pickle.load(f)
            if saved["stamp"] == stamp:
                return saved["chunks"]
        chunks = {}
        for path in files:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    # an omitted provision of a law that is served ("184. [* * * * * * *]") is kept too, for questions
                    # that cite it -- never embedded or keyword-searched: told nothing, the answer reasoned Article 184(3)
                    # still stood from a 2023 Act that mentions it
                    omitted = chunk.get("stub") == "omitted" and chunk.get("tier") in ANSWER_COLLECTION and not chunk.get("stale_copy_of")
                    if chunk.get("serve") or omitted:
                        kept = {k: chunk.get(k) for k in KEEP}
                        kept["serve"] = chunk.get("serve") or chunk["tier"]
                        kept["hash"] = digest(chunk["embed_text"])
                        kept["label"] = label(chunk)
                        chunks[chunk["chunk_id"]] = kept
        with cache.open("wb") as f:
            pickle.dump({"stamp": stamp, "chunks": chunks}, f)
        return chunks

    def _load_vectors(self, serves: tuple) -> tuple[list[str], np.ndarray]:
        ids, blocks = [], []
        for path in sorted(EMBED_STORE.glob("part_*.ids.json")):
            pairs = json.loads(path.read_text(encoding="utf-8"))
            keep = [n for n, (cid, h) in enumerate(pairs)
                    if cid in self.chunks and self.chunks[cid]["serve"] in serves and self.chunks[cid]["hash"] == h]
            if keep:
                ids += [pairs[n][0] for n in keep]
                blocks.append(np.load(str(path).replace(".ids.json", ".npy"))[keep])
        if not blocks:
            return [], np.zeros((0, 1792), dtype=np.float16)
        vectors = np.concatenate(blocks).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        return ids, vectors.astype(np.float16)

    def _build_keywords(self):
        ids = [cid for cid, c in self.chunks.items() if c["serve"] in ANSWER_COLLECTION and not c.get("stub")]
        cache = APP_DATA / "bm25.npz"
        meta = APP_DATA / "bm25.json"
        signature = digest("".join(ids) + "".join(self.chunks[cid]["hash"] for cid in ids))
        if cache.exists() and meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if saved["signature"] == signature:
                self.keyword_ids, self.vocab = saved["ids"], saved["vocab"]
                self.keyword_matrix = sparse.load_npz(cache).tocsc()
                return
        vocab, rows, cols, tfs, lengths = {}, [], [], [], []
        for n, cid in enumerate(ids):
            chunk = self.chunks[cid]
            words = tokens(f"{chunk['title']} {chunk['label']} {chunk['text']}")
            lengths.append(len(words))
            for term, tf in Counter(words).items():
                rows.append(n)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(tf)
        rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.float32)
        df = np.bincount(cols, minlength=len(vocab))
        idf = np.log((len(ids) - df + 0.5) / (df + 0.5) + 1).astype(np.float32)
        norm = K1 * (1 - B + B * lengths[rows] / lengths.mean())
        weights = idf[cols] * tfs * (K1 + 1) / (tfs + norm)
        matrix = sparse.csr_matrix((weights, (rows, cols)), shape=(len(ids), len(vocab)), dtype=np.float32)
        sparse.save_npz(cache, matrix)
        meta.write_text(json.dumps({"signature": signature, "ids": ids, "vocab": vocab}), encoding="utf-8")
        self.keyword_ids, self.vocab, self.keyword_matrix = ids, vocab, matrix.tocsc()

    def _build_sections(self):
        """(law title, provision number) -> chunk ids, for the laws a question names by a short name or citation."""
        self.sections = defaultdict(list)
        for cid, chunk in self.chunks.items():
            if chunk["serve"] in ANSWER_COLLECTION and chunk.get("number"):
                self.sections[(chunk["work_id"], provision_key(chunk["number"]))].append(cid)
        self.works_by_title = defaultdict(set)
        for cid, chunk in self.chunks.items():
            if chunk["serve"] in ANSWER_COLLECTION:
                self.works_by_title[chunk["title"].lower()].add(chunk["work_id"])

    # ---- searching
    def dense(self, query: np.ndarray, collection: str = "answers", k: int = 100, allowed=None, place: str | None = None) -> list[tuple[str, float]]:
        """The k chunks nearest the query. `place` filters inside the vector database; `allowed` (the same rule as a
        function) filters the in-memory search."""
        if self.vector_db is not None:
            from app import vectordb
            return [(cid, s) for cid, s in vectordb.search(self.vector_db, query, collection, k, place) if cid in self.chunks]
        ids, vectors = (self.answer_ids, self.answer_vectors) if collection == "answers" else (self.history_ids, self.history_vectors)
        if not ids:
            return []
        query = (query / (np.linalg.norm(query) + 1e-9)).astype(np.float32)
        scores = np.empty(len(ids), dtype=np.float32)
        for start in range(0, len(ids), SCORE_BLOCK):
            scores[start:start + SCORE_BLOCK] = vectors[start:start + SCORE_BLOCK].astype(np.float32) @ query
        if allowed is not None:
            scores[[n for n, cid in enumerate(ids) if not allowed(self.chunks[cid])]] = -1
        best = np.argpartition(-scores, min(k, len(ids) - 1))[:k]
        return [(ids[i], float(scores[i])) for i in best[np.argsort(-scores[best])] if scores[i] > -1]

    def keywords(self, text: str, k: int = 100, allowed=None) -> list[tuple[str, float]]:
        terms = [self.vocab[t] for t in set(tokens(text)) if t in self.vocab]
        if not terms:
            return []
        scores = np.asarray(self.keyword_matrix[:, terms].sum(axis=1)).ravel()
        order = np.argsort(-scores)[:k * 3]
        out = [(self.keyword_ids[i], float(scores[i])) for i in order if scores[i] > 0]
        if allowed is not None:
            out = [(cid, s) for cid, s in out if allowed(self.chunks[cid])]
        return out[:k]


# names students use for the laws they cite most; each maps to words its title contains
SHORT_NAMES = {
    "ppc": "pakistan penal code", "penal code": "pakistan penal code", "tazirat": "pakistan penal code",
    "crpc": "code of criminal procedure", "cr.p.c": "code of criminal procedure", "criminal procedure": "code of criminal procedure",
    "cpc": "code of civil procedure", "c.p.c": "code of civil procedure", "civil procedure": "code of civil procedure",
    "qso": "qanun-e-shahadat", "qanun-e-shahadat": "qanun-e-shahadat", "qanun e shahadat": "qanun-e-shahadat",
    "mflo": "muslim family laws ordinance", "constitution": "constitution of the islamic republic of pakistan",
    "aain": "constitution of the islamic republic of pakistan", "contract act": "contract act", "tpa": "transfer of property act",
    "transfer of property": "transfer of property act", "specific relief": "specific relief act", "limitation act": "limitation act",
    "family courts act": "family courts act", "guardians and wards": "guardians and wards act", "ata": "anti-terrorism act",
    "anti-terrorism": "anti-terrorism act", "cnsa": "control of narcotic substances act", "peca": "prevention of electronic crimes act",
    "nab ordinance": "national accountability ordinance", "income tax ordinance": "income tax ordinance", "sales tax act": "sales tax act",
    "companies act": "companies act", "succession act": "succession act", "registration act": "registration act",
    "stamp act": "stamp act", "evidence": "qanun-e-shahadat",
}
# the letter of "489-F" / "22A" is a capital that is not the start of a word: "section 498 of", "section 7 ki" are 498 and 7
SUFFIX = r"(?:\s?-?\s?(?-i:[A-Z]{1,2})(?![A-Za-z]))?"
CITATION = re.compile(rf"\b(section|sec\.?|s\.|dafa|daf'a|article|art\.?|rule)\s*(\d{{1,4}}{SUFFIX})(?![\dA-Za-z])|"
                      rf"\b(\d{{1,4}}(?:-(?-i:[A-Z]{{1,2}}))?)\s*(?=\s*(?:of\s+(?:the\s+)?)?(?:PPC|CrPC|CPC|QSO|MFLO)\b)", re.I)
KIND = {"section": "section", "sec": "section", "s": "section", "dafa": "section", "daf'a": "section",
        "article": "article", "art": "article", "rule": "rule"}
ORDER = re.compile(r"\border\s+([IVXL]+[A-Z]?)\b", re.I)          # "Order XXXIX rule 2 CPC"


def cited_provisions(corpus: Corpus, question: str) -> list[str]:
    """Chunks for the provisions a question cites outright ("s. 489-F PPC", "MFLO ki section 7", "Article 199",
    "Order XXXIX rule 2 CPC"): of the law named, the provision of that number and kind -- section 12 of the CPC, not
    rule 12 of one of its Orders -- under the Order named, if one is."""
    low = question.lower()
    names = [title for short, title in SHORT_NAMES.items() if re.search(rf"(?<![a-z]){re.escape(short)}(?![a-z])", low)]
    if "article" in low and not names:
        names = ["constitution of the islamic republic of pakistan"]
    cites = [(KIND.get(m.group(1).lower().rstrip(".")) if m.group(1) else None, provision_key(m.group(2) or m.group(3)))
             for m in CITATION.finditer(question)]
    orders = [m.group(1).upper() for m in ORDER.finditer(question)]
    if not names or not cites:
        return []
    works = {w for title, ws in corpus.works_by_title.items() for name in names if name in title for w in ws}
    rank = {}
    for work in works:
        for kind, number in cites:
            for cid in corpus.sections.get((work, number), []):
                chunk = corpus.chunks[cid]
                in_order = any(re.match(rf"Order {o}\b", p) for p in chunk.get("path") or [] for o in orders)
                kind_fits = chunk["unit"] == (kind or ("rule" if orders else "section")) or (kind == "section" and chunk["unit"] == "article")
                key = (bool(orders) and not in_order, not kind_fits)
                rank[cid] = min(rank.get(cid, key), key)
    # the provision cited first; of its copies the federal text (the provincial codes carry copies), then by authority
    return sorted(rank, key=lambda cid: (*rank[cid], corpus.chunks[cid]["source"] != "pakistancode",
                                         corpus.chunks[cid]["authority_rank"] or 9, corpus.chunks[cid]["part"] or 1))[:6]
