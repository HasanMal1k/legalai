"""
eval/bm25.py

Plain keyword search (BM25) over the chunks questions are answered from (serve = current or
unconfirmed, as kanon_eval.py's "answers" collection), on the evaluation questions: the baseline that
shows how much a question gives away in its words. Writes eval/runs/bm25_<variant>.jsonl.

    python eval/bm25.py                 # the questions as written
    python eval/bm25.py --translated    # Urdu and Roman Urdu questions replaced by their English
                                        # translation (eval/translations.jsonl, from translate.py)
"""

import argparse
import heapq
import json
import math
import re
import sys
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ocr_scanned import SOURCES, source_layout  # noqa: E402

K1, B = 1.2, 0.75
TOKEN = re.compile(r"\w+")
STOP = set("""a an and are as at be by for from has have he her his i if in into is it its me my of on or our she so that the their them
then there these they this to was we were what when where which who will with you your can do does how may shall should would could not no
ka ki ke hai hain kya mein main se ko aur par ne to bhi ho kar karna liye wala wali ye yeh woh wo jo agar kis kaise kab""".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translated", action="store_true")
    args = parser.parse_args()
    questions = [json.loads(line) for line in (HERE / "eval_set.jsonl").open(encoding="utf-8")]
    if args.translated:
        english = {row["id"]: row["en"] for row in map(json.loads, (HERE / "translations.jsonl").open(encoding="utf-8"))}
        for question in questions:
            question["q"] = english.get(question["id"], question["q"])
    query_terms = {q["id"]: tokens(q["q"]) for q in questions}
    vocab = {t for terms in query_terms.values() for t in terms}

    started = time.time()
    ids, lengths = [], array("i")
    postings = defaultdict(lambda: (array("i"), array("i")))
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if chunk.get("serve") not in ("current", "unconfirmed"):
                    continue
                words = tokens(chunk["embed_text"])
                n = len(ids)
                ids.append(chunk["chunk_id"])
                lengths.append(len(words))
                for term, tf in Counter(w for w in words if w in vocab).items():
                    docs, tfs = postings[term]
                    docs.append(n)
                    tfs.append(tf)
    total = len(ids)
    average = sum(lengths) / total
    print(f"indexed {total:,} chunks in {time.time() - started:.0f}s; {len(vocab)} query terms")

    out_path = HERE / "runs" / f"bm25_{'translated' if args.translated else 'raw'}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for question in questions:
            scores = defaultdict(float)
            for term in set(query_terms[question["id"]]):
                if term not in postings:
                    continue
                docs, tfs = postings[term]
                idf = math.log((total - len(docs) + 0.5) / (len(docs) + 0.5) + 1)
                for doc, tf in zip(docs, tfs):
                    scores[doc] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lengths[doc] / average))
            best = heapq.nlargest(100, scores.items(), key=lambda item: item[1])
            out.write(json.dumps({"id": question["id"], "ranked": [ids[doc] for doc, _ in best]}) + "\n")
    print(f"wrote {out_path} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
