"""
app/store.py

Every question, its answer and the sources shown, and every piece of feedback, in data/app/app.db (SQLite),
for reviewing wrong answers and improving the app (as the terms of service will say).
"""

import json
import sqlite3
import time
import uuid

from app.config import APP_DATA

DB = APP_DATA / "app.db"


def connect() -> sqlite3.Connection:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""create table if not exists questions (id text primary key, asked_at text, question text, language text,
                  english text, province text, answer text, sources text, history text, seconds real, model text)""")
    db.execute("""create table if not exists feedback (id integer primary key autoincrement, question_id text, given_at text,
                  rating integer, kind text, comment text)""")
    return db


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_question(question, language, english, province, answer, sources, history, seconds, model) -> str:
    qid = uuid.uuid4().hex[:12]
    with connect() as db:
        db.execute("insert into questions values (?,?,?,?,?,?,?,?,?,?,?)",
                   (qid, now(), question, language, english, province, answer,
                    json.dumps([s["chunk_id"] for s in sources]), json.dumps([h["chunk"]["chunk_id"] for h in history]),
                    seconds, model))
    return qid


def save_feedback(question_id: str, rating: int, kind: str | None, comment: str | None) -> None:
    with connect() as db:
        db.execute("insert into feedback (question_id, given_at, rating, kind, comment) values (?,?,?,?,?)",
                   (question_id, now(), rating, kind, (comment or "")[:4000]))
