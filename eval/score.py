"""
eval/score.py

Scores a retrieval run against eval/eval_set.jsonl.

A run is a .jsonl file with one row per question: {"id": ..., "ranked": [chunk_id, ...]} (best
first, at least 50 deep). Reported per language and per kind of question, citations apart:

  hit@k     share of questions with a correct chunk in the top k (k = 1, 5, 10, 50)
  MRR@10    mean of 1/rank of the first correct chunk (0 if not in the top 10)
  law@10    share with any chunk of the right law in the top 10 (the right Act, wrong section)

    python eval/score.py eval/runs/bm25_raw.jsonl [more runs ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = (1, 5, 10, 50)


def law_of(chunk_id: str) -> str:
    return chunk_id.rsplit(":", 1)[0]


def score(run_path: Path) -> dict:
    questions = {row["id"]: row for row in map(json.loads, (HERE / "eval_set.jsonl").open(encoding="utf-8"))}
    runs = {row["id"]: row["ranked"] for row in map(json.loads, run_path.open(encoding="utf-8"))}
    groups = defaultdict(list)
    for qid, question in questions.items():
        ranked = runs.get(qid, [])
        relevant = set(question["relevant"])
        laws = {law_of(c) for c in relevant}
        first = next((n for n, c in enumerate(ranked, 1) if c in relevant), None)
        result = {f"hit@{k}": bool(first and first <= k) for k in KS}
        result["MRR@10"] = 1 / first if first and first <= 10 else 0.0
        result["law@10"] = any(law_of(c) in laws for c in ranked[:10])
        bucket = "citation" if question["kind"] == "citation" else question["lang"]
        groups[bucket].append(result)
        if question["kind"] != "citation":
            groups["all (no citations)"].append(result)
            groups[f"kind: {question['kind']}"].append(result)
    return {name: {metric: sum(r[metric] for r in rows) / len(rows) for metric in rows[0]} | {"n": len(rows)}
            for name, rows in groups.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for path in map(Path, sys.argv[1:]):
        table = score(path)
        print(f"\n{path.name}")
        print(f"   {'':22s} {'n':>4s} " + " ".join(f"{m:>7s}" for m in [f'hit@{k}' for k in KS] + ["MRR@10", "law@10"]))
        order = ["all (no citations)", "en", "ru", "ur", "citation"] + sorted(k for k in table if k.startswith("kind:"))
        for name in order:
            if name in table:
                row = table[name]
                print(f"   {name:22s} {row['n']:4d} " + " ".join(f"{row[m]:7.0%}" if m != "MRR@10" else f"{row[m]:7.2f}"
                                                         for m in [f'hit@{k}' for k in KS] + ["MRR@10", "law@10"]))


if __name__ == "__main__":
    main()
