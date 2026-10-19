"""
app/vectordb.py

The vector database: Qdrant (tools\\qdrant\\qdrant.exe, data in data\\qdrant, http://127.0.0.1:6333/dashboard).
Two collections, as the chunks are served:

    answers   current laws and the Sindh laws served as "status not confirmed"
    history   repealed, superseded and lapsed laws, shown only as what the law was before

Each point is one chunk's Kanon 2 vector (1,792 numbers, cosine) with what the search filters on: the statute book it
comes from (`home`: "federal" or a province), the territories it is repealed for (`exceptions`), its law (`work_id`,
`family_id`) and tier. Vectors are kept on disk with an 8-bit copy in memory for speed; results are re-scored with the
full vectors.

    python -m app.vectordb        load or update the collections from data\\embeddings (the Kanon vectors on disk)

Updating is incremental: points whose chunk text is unchanged are left alone, changed or new ones are written, and
points of chunks no longer served are deleted.
"""

import json
import sys
import time
import uuid
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import ANSWER_COLLECTION, EMBED_STORE, HISTORY_COLLECTION
from app.index import Corpus, home_of

URL = "http://127.0.0.1:6333"
COLLECTIONS = {"answers": ANSWER_COLLECTION, "history": HISTORY_COLLECTION}
SIZE = 1792
BATCH = 256


def client() -> QdrantClient:
    return QdrantClient(url=URL, timeout=120)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload(chunk: dict) -> dict:
    return {"chunk_id": chunk["chunk_id"], "hash": chunk["hash"], "work_id": chunk["work_id"], "family_id": chunk["family_id"],
            "home": home_of(chunk), "tier": chunk["tier"], "title": chunk["title"],
            "exceptions": [x["territory"] for x in chunk.get("status_exceptions") or []], "year": chunk.get("year")}


def ensure(db: QdrantClient, name: str) -> None:
    if db.collection_exists(name):
        return
    db.create_collection(
        name, vectors_config=models.VectorParams(size=SIZE, distance=models.Distance.COSINE, on_disk=True),
        quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
        on_disk_payload=True)
    for field in ("home", "exceptions", "tier", "work_id", "family_id"):
        db.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def stored_points(db: QdrantClient, name: str) -> dict[str, tuple[str, str]]:
    """chunk_id -> (text hash, home) of every point in a collection."""
    have, offset = {}, None
    while True:
        points, offset = db.scroll(name, limit=5000, offset=offset, with_payload=["chunk_id", "hash", "home"], with_vectors=False)
        have.update({p.payload["chunk_id"]: (p.payload["hash"], p.payload.get("home")) for p in points})
        if offset is None:
            return have


def sync(corpus: Corpus) -> dict:
    db = client()
    report = {}
    for name, serves in COLLECTIONS.items():
        ensure(db, name)
        have = stored_points(db, name)
        wanted = {cid: c for cid, c in corpus.chunks.items() if c["serve"] in serves}
        stale = [cid for cid, (h, _) in have.items() if cid not in wanted or wanted[cid]["hash"] != h]
        if stale:
            db.delete(name, points_selector=models.PointIdsList(points=[point_id(cid) for cid in stale]))
        # where a law applies can change without its text changing (a better rule for it): rewrite only that field
        moved = defaultdict(list)
        for cid, (h, home) in have.items():
            if cid in wanted and wanted[cid]["hash"] == h and home != home_of(wanted[cid]):
                moved[home_of(wanted[cid])].append(point_id(cid))
        for home, ids in moved.items():
            for start in range(0, len(ids), 1000):
                db.set_payload(name, payload={"home": home}, points=ids[start:start + 1000])
        written, missing = 0, 0
        todo = {cid for cid, c in wanted.items() if (have.get(cid) or (None,))[0] != c["hash"]}
        for path in sorted(EMBED_STORE.glob("part_*.ids.json")):
            pairs = json.loads(path.read_text(encoding="utf-8"))
            rows = [n for n, (cid, h) in enumerate(pairs) if cid in todo and wanted[cid]["hash"] == h]
            if not rows:
                continue
            vectors = np.load(str(path).replace(".ids.json", ".npy"))
            for start in range(0, len(rows), BATCH):
                batch = rows[start:start + BATCH]
                db.upsert(name, wait=True, points=[
                    models.PointStruct(id=point_id(pairs[n][0]), vector=vectors[n].astype(np.float32).tolist(), payload=payload(wanted[pairs[n][0]]))
                    for n in batch])
                written += len(batch)
                todo -= {pairs[n][0] for n in batch}
        missing = len(todo)          # served chunks not embedded yet (eval/kanon_eval.py --embed)
        report[name] = {"points": db.count(name).count, "written": written, "deleted": len(stale),
                        "place updated": sum(len(ids) for ids in moved.values()), "not embedded yet": missing}
    return report


def province_filter(place: str | None) -> models.Filter | None:
    """Federal law applies everywhere it is not repealed for that territory; a province's code only in that province."""
    if not place:
        return None
    return models.Filter(must=[models.FieldCondition(key="home", match=models.MatchAny(any=[place, "federal"]))],
                         must_not=[models.FieldCondition(key="exceptions", match=models.MatchValue(value=place))])


def search(db: QdrantClient, query: np.ndarray, collection: str, k: int = 100, place: str | None = None) -> list[tuple[str, float]]:
    response = db.query_points(collection, query=query.astype(np.float32).tolist(), limit=k, query_filter=province_filter(place),
                               with_payload=["chunk_id"],
                               search_params=models.SearchParams(quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)))
    return [(p.payload["chunk_id"], float(p.score)) for p in response.points]


def available() -> bool:
    try:
        client().get_collections()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.time()
    corpus = Corpus(load_vectors=False)
    print(json.dumps(sync(corpus), indent=1))
    print(f"done in {time.time() - started:.0f}s")
