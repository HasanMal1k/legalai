"""
eval/app_eval.py

The app's own search (app/search.py: cited provisions, Kanon 2 vectors + BM25, copies folded, Kanon 2 Reranker) on
the evaluation questions, with the English translations eval/translate.py made (so no language-model calls).
Writes eval/runs/app_pipeline.jsonl for eval/score.py. Costs about 2 M reranker tokens (under a dollar).

    python eval/app_eval.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app.index import Corpus          # noqa: E402
from app.search import search         # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = Corpus()
    questions = [json.loads(line) for line in (HERE / "eval_set.jsonl").open(encoding="utf-8")]
    english = {r["id"]: r["en"] for r in map(json.loads, (HERE / "translations.jsonl").open(encoding="utf-8"))}
    path = HERE / "runs" / "app_pipeline.jsonl"
    with path.open("w", encoding="utf-8") as out:
        for n, q in enumerate(questions, 1):
            found = search(corpus, english.get(q["id"], q["q"]), q["q"])
            out.write(json.dumps({"id": q["id"], "ranked": found["ranked"][:100]}) + "\n")
            if n % 20 == 0:
                print(f"  {n} questions", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
