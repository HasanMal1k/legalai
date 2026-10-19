"""
app/search.py

The search behind an answer:

    1. provisions the question cites outright ("s. 489-F PPC") come first
    2. Kanon 2 vectors give the candidates; BM25 keywords add their ten best that Kanon missed (fusing the two
       evenly made results worse on the test questions: Kanon alone finds 83% in its top ten, fused 67%)
    3. copies of the same provision from several statute books are folded into one (the province's own, or the
       federal text, first); a law repealed for the chosen province is left out there
    4. the Kanon 2 Reranker orders the best 50; the top sections go to the answer
    5. what the law was before: historical laws the answering laws replaced (their `replaces`), and historical
       chunks very close to the question -- never mixed into the answer, shown beside it

The province filter: federal law applies everywhere it has not been repealed for that territory; a province's
own law only in that province.

A question of several parts ("what offences ... what can the shareholders do ... can he get bail?") is searched
issue by issue (answer.understand lists them): each issue has its own candidates and reranking, and each gives
its best sections to the answer -- one search for the whole question finds none of them well.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from isaacus import Isaacus

from app.config import EMBED_MODEL, RERANK_MODEL, secret
from app.index import PLACE_IN_TITLE, Corpus, cited_provisions, home_of, label, provision_key

PROVINCES = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan", "ict": "Islamabad Capital Territory"}
SOURCE_PROVINCE = {"punjab": "Punjab", "sindh": "Sindh", "kp": "Khyber Pakhtunkhwa", "balochistan": "Balochistan"}
RERANK_DEPTH = 50
KEYWORD_EXTRA = 10       # keyword hits added to the reranker's candidates
ANSWER_SOURCES = 8
HISTORY_CLOSE = 0.62     # cosine similarity a historical chunk needs to be shown unprompted
MIN_RELEVANCE = 0.12     # reranker score below which a section is not passed to the answer
ISSUE_DEPTH = 16         # per issue of a many-part question: candidates reranked (a focused issue needs few),
ISSUE_KEYWORDS = 4       # keyword hits added to them,
ISSUE_SOURCES = 3        # and sections it may give the answer,
ISSUE_CLOSE = 0.5        # those after its best scoring at least half as high
MAX_SOURCES = 28
MAX_GAPS = 8            # what a first answer said it could not cover, searched again
GAP_RELEVANCE = 0.05     # reranker score a section found for a gap needs (a gap the statutes do not cover finds noise)
WHOLE_UP_TO = 5000       # characters: a section cut into parts is given to the answer whole up to this size

_api = None


def api() -> Isaacus:
    global _api
    if _api is None:
        _api = Isaacus(api_key=secret("ISAACUS_API_KEY"), max_retries=4)
    return _api


def embed_query(text: str) -> np.ndarray:
    return embed_queries([text])[0]


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    response = api().embeddings.create(model=EMBED_MODEL, texts=texts, task="retrieval/query", overflow_strategy="drop_end")
    return [np.asarray(e.embedding, dtype=np.float32) for e in sorted(response.embeddings, key=lambda e: e.index)]


def place_named(text: str) -> str | None:
    """The province a question is about when it names one place only ("a company in Lahore"): used when the student
    has not chosen one. Two places named (a house in Lahore, a buyer in Karachi) decide nothing."""
    keys = {name: key for key, name in PROVINCES.items()}
    found = {keys[place] for pattern, place in PLACE_IN_TITLE if pattern.search(text)}
    return found.pop() if len(found) == 1 else None


def provision_of(chunk: dict) -> tuple:
    """One provision, whichever of its parts: (law, unit, number, where in the law)."""
    if not chunk.get("number"):
        return (chunk["chunk_id"],)
    return chunk["work_id"], chunk["unit"], provision_key(chunk["number"]), tuple(chunk.get("path") or [])


def whole_provision(corpus: Corpus, chunk: dict) -> dict:
    """A section cut into parts is given to the answer whole (up to WHOLE_UP_TO characters), else its opening and the
    part that matched: part 2 of a section alone (its definitions) leaves out what the section makes an offence."""
    if (chunk.get("parts") or 1) <= 1 or not chunk.get("number"):
        return chunk
    parts = sorted((corpus.chunks[cid] for cid in corpus.sections.get((chunk["work_id"], provision_key(chunk["number"])), [])
                    if provision_of(corpus.chunks[cid]) == provision_of(chunk) and corpus.chunks[cid]["parts"] == chunk["parts"]),
                   key=lambda c: c["part"])
    if len(parts) < 2:
        return chunk
    whole = sum(len(p["text"]) for p in parts) <= WHOLE_UP_TO
    keep = parts if whole else [p for p in parts if p["part"] in (1, chunk["part"])]
    pieces, last = [], 0
    for part in keep:
        if part["part"] != last + 1:
            pieces.append("[...]")
        pieces.append(part["text"].strip())
        last = part["part"]
    if last < chunk["parts"]:
        pieces.append("[...]")
    kept = " and ".join(str(p["part"]) for p in keep)
    shown = label(dict(chunk, parts=1)) + ("" if whole else f" — part{'s' if len(keep) > 1 else ''} {kept} of {chunk['parts']}")
    return dict(chunk, text="\n".join(pieces), label=shown)


def distinct(reranked: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """The reranked chunks, one per provision (its best-scoring part)."""
    seen, out = set(), []
    for chunk, score in reranked:
        if provision_of(chunk) not in seen:
            seen.add(provision_of(chunk))
            out.append((chunk, score))
    return out


def rerank(query: str, chunks: list[dict]) -> list[tuple[dict, float]]:
    if not chunks:
        return []
    texts = [f"{c['title']} — {c['label']}\n{c['text']}" for c in chunks]
    response = api().rerankings.create(model=RERANK_MODEL, query=query, texts=texts)
    return [(chunks[r.index], r.score) for r in response.results]


def allowed_in(province: str | None):
    """Which chunks may answer for a province (None: all of Pakistan)."""
    if not province:
        return None
    place = PROVINCES[province]

    def allowed(chunk: dict) -> bool:
        if any(x.get("territory") == place for x in chunk.get("status_exceptions") or []):
            return False                      # repealed there
        home = home_of(chunk)                 # a province's code, or a law naming its place, speaks for that place only
        return home == "federal" or home == place
    return allowed


def fuse(*rankings: list[tuple[str, float]], k: int = 60) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def fold_copies(corpus: Corpus, ids: list[str], province: str | None) -> list[str]:
    """One chunk per provision: the same section printed on several codes (the Penal Code on the Pakistan Code and
    on the Punjab Code) appears once -- the chosen province's copy, else the federal text."""
    place = PROVINCES.get(province) if province else None
    best = {}
    for order, cid in enumerate(ids):
        chunk = corpus.chunks[cid]
        key = (chunk["family_id"], chunk["unit"], str(chunk["number"] or chunk["seq"]), chunk["part"])
        rank = (SOURCE_PROVINCE.get(chunk["source"]) != place if place else chunk["source"] != "pakistancode",
                chunk["authority_rank"] or 9, order)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, cid, order)
    kept = {cid for _, cid, _ in best.values()}
    return [cid for cid in ids if cid in kept]


def search(corpus: Corpus, english: str, original: str, province: str | None = None, issues: list[str] | None = None) -> dict:
    if issues:
        return search_issues(corpus, english, original, province, issues)
    timing = {}
    started = time.time()
    allowed = allowed_in(province)
    cited = [cid for cid in cited_provisions(corpus, original) + cited_provisions(corpus, english)
             if allowed is None or allowed(corpus.chunks[cid])]
    query = embed_query(english)
    timing["embed"] = time.time() - started
    dense = corpus.dense(query, "answers", 100, allowed, PROVINCES.get(province))
    keywords = corpus.keywords(english, 100, allowed)
    timing["retrieve"] = time.time() - started
    lead = [cid for cid, _ in dense[:RERANK_DEPTH]]
    extra = [cid for cid, _ in keywords if cid not in set(lead)][:KEYWORD_EXTRA]
    candidates = fold_copies(corpus, list(dict.fromkeys(cited + lead + extra + [cid for cid, _ in dense[RERANK_DEPTH:]])), province)
    reranked = rerank(english, [corpus.chunks[cid] for cid in candidates[:RERANK_DEPTH + KEYWORD_EXTRA + len(cited)]])
    timing["rerank"] = time.time() - started
    # a provision the question names outright stays in, wherever the reranker put it
    chosen = [c for c in reranked if c[0]["chunk_id"] in cited][:3]
    # the rest only while the reranker finds them relevant at all (always the best three, for the answer to weigh)
    rest = [c for c in reranked if c not in chosen]
    chosen += [c for n, c in enumerate(rest) if n < 3 or c[1] >= MIN_RELEVANCE][:ANSWER_SOURCES - len(chosen)]
    picked = {c["chunk_id"] for c, _ in chosen}
    ranked = [c["chunk_id"] for c, _ in chosen] + [c["chunk_id"] for c, _ in reranked if c["chunk_id"] not in picked]
    return {"sources": given(corpus, chosen), "history": earlier_law(corpus, query, chosen),
            "ranked": ranked + [cid for cid in candidates[RERANK_DEPTH:]],     # the whole order, for eval/app_eval.py
            "timing": {k: round(v, 2) for k, v in timing.items()}}


def given(corpus: Corpus, chosen: list[tuple[dict, float]]) -> list[dict]:
    """What the answer is given: each chosen provision once, whole where it was cut into parts."""
    return [dict(whole_provision(corpus, chunk), score=round(score, 3)) for chunk, score in distinct(chosen)]


def issue_results(corpus: Corpus, issues: list[str], vectors: list[np.ndarray], province: str | None) -> list[list[tuple[dict, float]]]:
    """Each issue searched on its own: its nearest sections and a few keyword hits, reranked against the issue."""
    allowed, place = allowed_in(province), PROVINCES.get(province)

    def one(issue: str, vector: np.ndarray) -> list[tuple[dict, float]]:
        # a provision the issue names ("pre-arrest bail under section 498 of the Code of Criminal Procedure": words the
        # section itself never uses) is fetched outright and comes first
        named = [cid for cid in cited_provisions(corpus, issue) if allowed is None or allowed(corpus.chunks[cid])][:2]
        lead = [cid for cid, _ in corpus.dense(vector, "answers", ISSUE_DEPTH, allowed, place)]
        extra = [cid for cid, _ in corpus.keywords(issue, 40, allowed) if cid not in set(lead)][:ISSUE_KEYWORDS]
        candidates = fold_copies(corpus, list(dict.fromkeys(named + lead + extra)), province)
        results = rerank(issue, [corpus.chunks[cid] for cid in candidates])
        first = {provision_of(corpus.chunks[cid]) for cid in named}
        return distinct([r for r in results if provision_of(r[0]) in first] + [r for r in results if provision_of(r[0]) not in first])

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, issues, vectors))


def fill_gaps(corpus: Corpus, gaps: list[str], province: str | None, have: list[dict]) -> list[dict]:
    """Sections for what a first answer said its sources did not cover: each gap searched as an issue, its best
    section (and a second close to it) added when the reranker finds it relevant at all and it is not given yet."""
    gaps = gaps[:MAX_GAPS]
    found = issue_results(corpus, gaps, embed_queries(gaps), province) if gaps else []
    seen, added = {provision_of(chunk) for chunk in have}, []
    for results in found:
        for rank, (chunk, score) in enumerate(results[:2]):
            if score >= GAP_RELEVANCE and (rank == 0 or score >= ISSUE_CLOSE * results[0][1]) and provision_of(chunk) not in seen:
                seen.add(provision_of(chunk))
                added.append((chunk, score))
    return given(corpus, added)


def search_issues(corpus: Corpus, english: str, original: str, province: str | None, issues: list[str]) -> dict:
    """A question of several parts: each issue searched and reranked on its own, then each gives its best sections in
    turn (the best of every issue first, then the second best, ...) up to MAX_SOURCES."""
    timing = {}
    started = time.time()
    allowed = allowed_in(province)
    cited = [cid for cid in cited_provisions(corpus, original) + cited_provisions(corpus, english)
             if allowed is None or allowed(corpus.chunks[cid])]
    vectors = embed_queries(issues + [english])
    timing["embed"] = time.time() - started
    found = issue_results(corpus, issues, vectors[:len(issues)], province)
    timing["rerank"] = time.time() - started
    chosen = [(corpus.chunks[cid], 1.0) for cid in dict.fromkeys(cited)][:3]
    seen = {provision_of(chunk) for chunk, _ in chosen}
    for rank in range(ISSUE_SOURCES):
        for results in found:
            # an issue's best section always (the answer says if it does not help); its next ones while they score
            # near it -- a focused issue scores low against any one section, so the bar is the issue's own best
            if rank < len(results) and (rank == 0 or results[rank][1] >= ISSUE_CLOSE * results[0][1]) and provision_of(results[rank][0]) not in seen:
                seen.add(provision_of(results[rank][0]))
                chosen.append(results[rank])
    chosen = chosen[:MAX_SOURCES]
    ranked = list(dict.fromkeys([c["chunk_id"] for c, _ in chosen] + [c["chunk_id"] for results in found for c, _ in results]))
    return {"sources": given(corpus, chosen), "history": earlier_law(corpus, vectors[-1], chosen), "ranked": ranked,
            "issues": [[c["chunk_id"] for c, _ in results[:ISSUE_SOURCES]] for results in found],
            "timing": {k: round(v, 2) for k, v in timing.items()}}


def earlier_law(corpus: Corpus, query: np.ndarray, chosen: list[tuple[dict, float]]) -> list[dict]:
    """What the law was before: the historical laws the answering laws replaced, and historical chunks very close to
    the question. One entry per historical law, with its best-matching chunk."""
    found = {}
    replaced = {p["work_id"]: chunk for chunk, _ in chosen for p in (chunk.get("replaces") or [])}
    for cid, score in corpus.dense(query, "history", 40):
        chunk = corpus.chunks[cid]
        work = chunk["work_id"]
        if work in found:
            continue
        if work in replaced or score >= HISTORY_CLOSE:
            successor = chunk.get("replaced_by") or {}
            found[work] = {"chunk": chunk, "score": round(score, 3), "replaced_by": successor.get("title"),
                           "how": successor.get("how"), "why": "replaced by a law in this answer" if work in replaced else "closely matches the question"}
        if len(found) >= 3:
            break
    return list(found.values())
