"""
eval/kanon_eval.py

Embeds the corpus with Isaacus' Kanon 2 Embedder and runs the evaluation questions against it.

Needs the API key in the environment (never in a file under version control):
    setx ISAACUS_API_KEY "iuak_..."        (then open a new terminal)

    python eval/kanon_eval.py --estimate                 # tokens and cost, nothing sent
    python eval/kanon_eval.py --embed --price 0.35 --yes # embed every chunk (resumable)
    python eval/kanon_eval.py --search                   # embed the questions, write the runs
    python eval/score.py eval/runs/kanon2_raw.jsonl eval/runs/kanon2_translated.jsonl eval/runs/hybrid_translated.jsonl eval/runs/hybrid_reranked.jsonl

Only served chunks are embedded (`serve` set by chunk_statutes.py), in two collections:
  answers   current laws, and the Sindh laws served as "status not confirmed" -- what questions are answered from
  history   repealed, superseded and lapsed laws -- never mixed into answers; searched apart, to show what the
            law was before when a question matches one closely, or through a current law's `replaces`
Quarantined laws and stubs are not embedded at all.

Chunk vectors are kept in data/embeddings/kanon-2-embedder/ as float16 shards, each with its chunks' ids and a
hash of the text embedded. --embed sends only chunks whose (id, text) is not stored yet, so it resumes after an
interruption and, after a re-cut or a text repair, pays only for the chunks that changed; vectors of chunks that
no longer exist are ignored. The runs written (answers collection) are:
  kanon2_raw          the questions as written
  kanon2_translated   Urdu and Roman Urdu questions translated to English first (eval/translate.py)
  hybrid_translated   kanon2_translated fused with keyword search (reciprocal rank fusion)
  hybrid_reranked     the top 50 of hybrid_translated re-ordered by the Kanon 2 Reranker (about 2 M tokens for
                      the 134 questions)
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ocr_scanned import SOURCES, source_layout  # noqa: E402

MODEL = "kanon-2-embedder"
RERANKER = "kanon-2-reranker"
RERANK_DEPTH = 50  # candidates the reranker re-orders per question
STORE = HERE.parent / "data" / "embeddings" / MODEL
SHARD = 2048       # chunks per saved file
BATCH = 64         # texts per request
WORKERS = 4        # requests in flight
CHARS_PER_TOKEN = 3.6


def client():
    from isaacus import Isaacus  # imported here so --estimate works without the package or key
    from app.config import secret  # the environment, or the user's registry environment (setx, no new terminal)
    return Isaacus(api_key=secret("ISAACUS_API_KEY"), max_retries=6)


COLLECTIONS = {"answers": ("current", "unconfirmed"), "history": ("historical",)}


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def served_chunks() -> dict[str, tuple[str, str, str]]:
    """chunk_id -> (collection, text hash, text) for every chunk that is served."""
    collection_of = {serve: name for name, serves in COLLECTIONS.items() for serve in serves}
    out = {}
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                collection = collection_of.get(chunk.get("serve"))
                if collection:
                    out[chunk["chunk_id"]] = (collection, digest(chunk["embed_text"]), chunk["embed_text"])
    return out


def stored() -> set[tuple[str, str]]:
    """(chunk_id, text hash) of every vector already on disk."""
    have = set()
    for path in STORE.glob("*.ids.json"):
        have.update(tuple(pair) for pair in json.loads(path.read_text(encoding="utf-8")))
    return have


def embed(api, texts: list[str], task: str) -> np.ndarray:
    for attempt in range(8):
        try:
            response = api.embeddings.create(model=MODEL, texts=texts, task=task, overflow_strategy="drop_end")
            ordered = sorted(response.embeddings, key=lambda e: e.index)
            return np.asarray([e.embedding for e in ordered], dtype=np.float32), response.usage.input_tokens
        except Exception as e:      # the client already retries rate limits; this covers longer outages
            if attempt == 7:
                raise
            wait = 10 * 2 ** attempt
            print(f"  request failed ({type(e).__name__}: {str(e)[:120]}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def rerank(api, query: str, ids: list[str], text_of: dict[str, str]) -> list[str]:
    """`ids` re-ordered by the reranker's relevance to `query`."""
    for attempt in range(8):
        try:
            response = api.rerankings.create(model=RERANKER, query=query, texts=[text_of[cid] for cid in ids])
            return [ids[r.index] for r in response.results]
        except Exception as e:
            if attempt == 7:
                raise
            wait = 10 * 2 ** attempt
            print(f"  rerank failed ({type(e).__name__}: {str(e)[:120]}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def pending() -> list[tuple[str, str, str]]:
    """(chunk_id, hash, text) of the served chunks not embedded yet, answers first."""
    have = stored() if STORE.exists() else set()
    served = served_chunks()
    todo = [(cid, h, text) for cid, (collection, h, text) in served.items() if (cid, h) not in have]
    return sorted(todo, key=lambda row: served[row[0]][0] != "answers")


def estimate():
    served = served_chunks()
    todo = pending()
    by_collection = {name: sum(1 for c, _, _ in served.values() if c == name) for name in COLLECTIONS}
    tokens = sum(len(text) for _, _, text in todo) / CHARS_PER_TOKEN
    return by_collection, len(todo), tokens


def embed_corpus(price: float | None):
    api = client()
    STORE.mkdir(parents=True, exist_ok=True)
    todo = pending()
    print(f"{len(todo):,} chunks to embed")
    spent_tokens, started = 0, time.time()
    taken = {int(p.stem.split("_")[-1].split(".")[0]) for p in STORE.glob("part_*.npy")}
    number = max(taken, default=-1) + 1
    for first in range(0, len(todo), SHARD):
        part = todo[first:first + SHARD]
        batches = [part[i:i + BATCH] for i in range(0, len(part), BATCH)]
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(lambda batch: embed(api, [text for _, _, text in batch], "retrieval/document"), batches))
        vectors = np.concatenate([vectors for vectors, _ in results]).astype(np.float16)
        spent_tokens += sum(tokens for _, tokens in results)
        name = STORE / f"part_{number:05d}"
        np.save(name.with_suffix(".npy"), vectors)
        # the ids file is written last: a shard without one was interrupted and is not counted as stored
        name.with_suffix(".ids.json").write_text(json.dumps([[cid, h] for cid, h, _ in part]), encoding="utf-8")
        number += 1
        cost = f" | about ${spent_tokens / 1e6 * price:,.2f} so far" if price else ""
        print(f"  shard {number - 1:5d}: {len(part)} chunks | {first + len(part):,} of {len(todo):,} | {spent_tokens / 1e6:.1f} M tokens{cost} | "
              f"{time.time() - started:.0f}s", flush=True)
    print("corpus embedded")


def load_corpus(collection: str = "answers") -> tuple[list[str], np.ndarray]:
    """The vectors of one collection's current chunks; vectors of chunks re-cut or no longer served are left out."""
    served = served_chunks()
    ids, blocks = [], []
    for path in sorted(STORE.glob("part_*.ids.json")):
        pairs = json.loads(path.read_text(encoding="utf-8"))
        keep = [n for n, (cid, h) in enumerate(pairs) if cid in served and served[cid][0] == collection and served[cid][1] == h]
        if keep:
            ids += [pairs[n][0] for n in keep]
            blocks.append(np.load(str(path).replace(".ids.json", ".npy"))[keep])
    return ids, np.concatenate(blocks)


def top(vectors: np.ndarray, query: np.ndarray, k: int = 100) -> np.ndarray:
    scores = np.empty(len(vectors), dtype=np.float32)
    for start in range(0, len(vectors), 50000):         # in slices, to keep memory flat
        scores[start:start + 50000] = vectors[start:start + 50000].astype(np.float32) @ query
    best = np.argpartition(-scores, k)[:k]
    return best[np.argsort(-scores[best])]


def fuse(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal rank fusion of several ranked lists."""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:100]


def search():
    api = client()
    ids, vectors = load_corpus()
    print(f"loaded {len(ids):,} chunk vectors")
    questions = [json.loads(line) for line in (HERE / "eval_set.jsonl").open(encoding="utf-8")]
    english = {r["id"]: r["en"] for r in map(json.loads, (HERE / "translations.jsonl").open(encoding="utf-8"))}
    bm25 = {r["id"]: r["ranked"] for r in map(json.loads, (HERE / "runs" / "bm25_translated.jsonl").open(encoding="utf-8"))}
    variants = {"raw": [q["q"] for q in questions], "translated": [english.get(q["id"], q["q"]) for q in questions]}
    runs = {}
    for variant, texts in variants.items():
        query_vectors = np.concatenate([embed(api, texts[i:i + BATCH], "retrieval/query")[0] for i in range(0, len(texts), BATCH)])
        runs[variant] = {q["id"]: [ids[i] for i in top(vectors, query_vectors[n])] for n, q in enumerate(questions)}
        write(f"kanon2_{variant}", runs[variant])
    hybrid = {qid: fuse(ranked, bm25[qid]) for qid, ranked in runs["translated"].items()}
    write("hybrid_translated", hybrid)
    text_of = {cid: text for cid, (_, _, text) in served_chunks().items()}
    reranked = {}
    for q in questions:
        candidates = [cid for cid in hybrid[q["id"]] if cid in text_of][:RERANK_DEPTH]
        reranked[q["id"]] = rerank(api, variants["translated"][questions.index(q)], candidates, text_of) + hybrid[q["id"]][RERANK_DEPTH:]
    write("hybrid_reranked", reranked)


def write(name: str, run: dict):
    path = HERE / "runs" / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for qid, ranked in run.items():
            out.write(json.dumps({"id": qid, "ranked": ranked}) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--price", type=float, help="dollars per million tokens, to print the running cost")
    parser.add_argument("--yes", action="store_true", help="confirm that the embedding run may spend money")
    args = parser.parse_args()
    if args.estimate or (args.embed and not args.yes):
        collections, count, tokens = estimate()
        print(f"served chunks: {collections}; {count:,} not embedded yet, about {tokens / 1e6:.0f} M tokens"
              + (f", about ${tokens / 1e6 * args.price:,.0f} at ${args.price}/M" if args.price else ""))
        if args.embed:
            print("add --yes to start")
        return
    if args.embed:
        embed_corpus(args.price)
    if args.search:
        search()


if __name__ == "__main__":
    main()
