"""
app/server.py

The web app: a question box answered from the statutes, every statement cited, with feedback buttons.

    powershell -ExecutionPolicy Bypass -File run_app.ps1      (then open http://127.0.0.1:8000)

    POST /api/ask       {"question": "...", "province": null | "punjab" | "sindh" | "kp" | "balochistan" | "ict"}
    POST /api/feedback  {"question_id": "...", "rating": 1 | -1, "kind": "wrong" | ..., "comment": "..."}
    GET  /api/law/{work_id}   a served law's text, provision by provision

Needs ISAACUS_API_KEY (search) and DEEPSEEK_API_KEY (answers), and the corpus embedded (eval/kanon_eval.py --embed).
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import answer, search, store
from app.config import LLM_MODEL
from app.index import Corpus

STATIC = Path(__file__).resolve().parent / "static"
corpus: Corpus | None = None


@asynccontextmanager
async def lifespan(_):
    global corpus
    corpus = Corpus()
    yield


app = FastAPI(title="Pakistan Law Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    province: str | None = None


class Feedback(BaseModel):
    question_id: str
    rating: int = Field(ge=-1, le=1)
    kind: str | None = None
    comment: str | None = Field(default=None, max_length=4000)


def status_text(chunk: dict) -> str:
    if chunk["tier"] == "unconfirmed":
        return "Status not confirmed"
    if chunk["tier"] == "historical":
        return {"repealed": "Repealed", "superseded": "Replaced", "lapsed_presumed": "Lapsed"}.get(chunk["status"], "No longer in force")
    return "In force"


def public(chunk: dict, n: int | None = None, cited: bool = False) -> dict:
    return {"n": n, "chunk_id": chunk["chunk_id"], "work_id": chunk["work_id"], "title": chunk["title"], "label": chunk["label"],
            "year": chunk["year"], "jurisdiction": chunk.get("status_applies_in") or chunk["jurisdiction"], "status": status_text(chunk),
            "tier": chunk["tier"], "evidence": chunk.get("status_evidence"), "exceptions": [x["territory"] for x in chunk.get("status_exceptions") or []],
            "last_amended": chunk.get("last_amended"), "text": chunk["text"], "cited": cited}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.post("/api/ask")
def ask(body: Ask):
    if body.province and body.province not in search.PROVINCES:
        raise HTTPException(400, "unknown province")
    started = time.time()
    plan = answer.understand(body.question)
    language, english, issues = plan["language"], plan["english"], plan["issues"]
    # no province chosen, but the question names one place ("a company in Lahore"): answer for it, and say so
    inferred = None if body.province else search.place_named(body.question) or search.place_named(english)
    province = body.province or inferred
    found = search.search(corpus, english, body.question, province, issues)
    text = answer.write_answer(body.question, language, found["sources"], search.PROVINCES.get(province), english, issues)
    gaps, filled = [], []
    if issues and text:
        # what the first answer could not cover is searched again; found something, the answer is written again with it
        gaps = answer.find_gaps(english, text)
        filled = search.fill_gaps(corpus, gaps, province, found["sources"]) if gaps else []
        if filled:
            found["sources"] += filled
            text = answer.write_answer(body.question, language, found["sources"], search.PROVINCES.get(province), english, issues + gaps)
    cited = answer.cited_numbers(text, len(found["sources"]))
    seconds = round(time.time() - started, 2)
    qid = store.save_question(body.question, language, english, province, text, found["sources"], found["history"], seconds, LLM_MODEL)
    return {
        "id": qid, "language": language, "english": english, "seconds": seconds, "timing": found["timing"],
        "issues": issues, "gaps": gaps, "gap_sources": len(filled),
        "province": province, "province_inferred": search.PROVINCES.get(inferred) if inferred else None,
        "answer": text or "No section of the laws available matches this question.",
        "sources": [public(chunk, n, n in cited) for n, chunk in enumerate(found["sources"], 1)],
        "history": [dict(public(h["chunk"]), replaced_by=h["replaced_by"], how=h["how"], why=h["why"]) for h in found["history"]],
    }


@app.post("/api/feedback")
def feedback(body: Feedback):
    store.save_feedback(body.question_id, body.rating, body.kind, body.comment)
    return {"ok": True}


@app.get("/api/law/{work_id}")
def law(work_id: str):
    ids = corpus.by_work.get(work_id)
    if not ids:
        raise HTTPException(404, "not a served law")
    first = corpus.chunks[ids[0]]
    return {"title": first["title"], "year": first["year"], "status": status_text(first), "evidence": first.get("status_evidence"),
            "replaces": first.get("replaces") or [], "replaced_by": first.get("replaced_by"),
            "provisions": [{"chunk_id": cid, "label": corpus.chunks[cid]["label"], "text": corpus.chunks[cid]["text"]} for cid in ids]}


@app.get("/api/health")
def health():
    if corpus.vector_db is not None:
        vectors = {"store": "qdrant", "answers": corpus.vector_db.count("answers").count, "history": corpus.vector_db.count("history").count}
    else:
        vectors = {"store": "memory", "answers": len(corpus.answer_ids), "history": len(corpus.history_ids)}
    return {"chunks": len(corpus.chunks), "vectors": vectors}
