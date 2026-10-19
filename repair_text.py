"""
repair_text.py

Some PDFs have a text layer that extracts as garbage: doubled characters
("TThhee  AAcctt"), or an old embedded OCR layer that isn't readable
English. Those documents went into statutes.jsonl as "native" text. This
re-reads them from the page images with the GPU OCR (ocr_scanned.py) and
replaces the text when the OCR result is clearly better.

    python repair_text.py --dry-run    # list what would be repaired
    python repair_text.py              # repair

For each repaired document: statutes.jsonl gets the OCR text
("text_source": "ocr", "repaired_from": "native"), and its rows are
removed from metadata.jsonl and metadata_v2.jsonl so generate_metadata.py
and enrich_metadata.py describe it again from the corrected text. The
previous text is kept in data/<source>/repaired_text_backup.jsonl.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ocr_scanned import SOURCES, init_gpu_worker, looks_english, ocr_document, source_layout

DOUBLED = 0.12  # share of letters equal to the letter before; ordinary English is about 0.03
THIN = 300      # characters per page; a printed page of law holds 1,500 or more
GARBLED = 0.10  # share of words seen in fewer than RARE documents of the corpus; clean text is under 0.05
RARE = 5
WORD = re.compile(r"[A-Za-z]{3,}")


def corpus_word_counts() -> dict[str, int]:
    """In how many documents each word appears: real words recur, a scan's misreadings do not."""
    counts = {}
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                for word in {w.lower() for w in WORD.findall(json.loads(line)["raw_text"])}:
                    counts[word] = counts.get(word, 0) + 1
    return counts


def garble(text: str, counts: dict[str, int]) -> float:
    words = [w.lower() for w in WORD.findall(text)]
    return sum(counts.get(w, 0) < RARE for w in words) / max(1, len(words))


def doubled_share(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text[:20000])
    if len(letters) < 200:
        return 0.0
    return sum(a == b for a, b in zip(letters, letters[1:])) / (len(letters) - 1)


def is_bad(text: str) -> str | None:
    if doubled_share(text) >= DOUBLED:
        return "doubled characters"
    if not looks_english(text):
        return "not readable English"
    return None


def pdf_name(row: dict) -> str | None:
    return row.get("pdf_file") or (row.get("pdf_url") or "").split("/")[-1] or None


def candidates(source: str, counts: dict[str, int]) -> list[dict]:
    data_dir, key = source_layout(source)
    indexed = set()
    final = data_dir / "metadata_final.jsonl"
    if final.exists():
        indexed = {row[key] for row in map(json.loads, final.open(encoding="utf-8")) if row["index_default"]}
    found = []
    with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("text_source", "native") != "native":
                continue  # OCR and translated text were already checked when they were made
            reason = is_bad(row["raw_text"])
            pages = row.get("num_pages") or 0
            if not reason and pages >= 3 and len(row["raw_text"]) < THIN * pages:
                # a text layer on the first page only: the rest of the PDF is page images
                reason = "pages without text"
            share = garble(row["raw_text"], counts)
            if not reason and share > GARBLED and row[key] in indexed:
                # an old OCR layer that reads as English but is full of misreadings ("Prrlimdrt) rccoived")
                reason = "garbled text layer"
            name = pdf_name(row)
            if reason and name and (data_dir / "raw_pdfs" / name).exists():
                found.append({"id": row[key], "name": row.get("name"), "reason": reason, "chars": len(row["raw_text"]),
                              "garble": share, "pdf": str(data_dir / "raw_pdfs" / name)})
    return found


def drop_rows(path: Path, key: str, ids: set[str]) -> int:
    """Remove the rows for `ids` from a .jsonl file; returns how many went."""
    if not path.exists():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    dropped = 0
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if json.loads(line)[key] in ids:
                dropped += 1
            else:
                dst.write(line)
    os.replace(tmp, path)
    return dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="OCR processes sharing the GPU")
    args = parser.parse_args()

    counts = corpus_word_counts()
    todo = {source: candidates(source, counts) for source in SOURCES}
    for source, docs in todo.items():
        if docs:
            reasons = {}
            for doc in docs:
                reasons[doc["reason"]] = reasons.get(doc["reason"], 0) + 1
            print(f"{source}: {len(docs)} documents with a bad text layer {reasons}")
    total = sum(len(docs) for docs in todo.values())
    print(f"{total} documents to re-read")
    if args.dry_run or not total:
        return

    # OCR everything first (GPU), then rewrite each source's files once.
    new_text, rejected = {}, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_gpu_worker) as pool:
        futures = {pool.submit(ocr_document, doc["pdf"]): (source, doc) for source, docs in todo.items() for doc in docs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Re-reading", unit="doc"):
            source, doc = futures[future]
            try:
                text, pages = future.result()
            except Exception as e:
                print(f"  ! {doc['name']}: {type(e).__name__}: {e}", flush=True)
                continue
            # only accept the OCR text if it is actually good (and, for missing pages, much longer;
            # for a garbled layer, clearly less garbled and not much shorter)
            if doc["reason"] == "garbled text layer":
                better = garble(text, counts) <= doc["garble"] - 0.04 and len(text) >= 0.7 * doc["chars"]
            else:
                better = doc["reason"] != "pages without text" or len(text) >= 2 * doc["chars"]
            if len(text) >= 100 and not is_bad(text) and better:
                new_text[(source, doc["id"])] = (text, pages)
            else:
                rejected += 1

    print(f"OCR text accepted for {len(new_text)}; kept the old text for {rejected} (the OCR was no better)")
    for source, docs in todo.items():
        fixed = {doc["id"] for doc in docs if (source, doc["id"]) in new_text}
        if not fixed:
            continue
        data_dir, key = source_layout(source)
        path = data_dir / "statutes.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst, \
                (data_dir / "repaired_text_backup.jsonl").open("a", encoding="utf-8") as backup:
            for line in src:
                row = json.loads(line)
                if row[key] in fixed:
                    backup.write(json.dumps({key: row[key], "raw_text": row["raw_text"], "replaced_at": now},
                                            ensure_ascii=False) + "\n")
                    text, pages = new_text[(source, row[key])]
                    row.update(raw_text=text, num_pages=pages, text_source="ocr", repaired_from="native", repaired_at=now)
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                dst.write(line)
        os.replace(tmp, path)
        stale = sum(drop_rows(data_dir / name, key, fixed) for name in ("metadata.jsonl", "metadata_v2.jsonl"))
        print(f"{source}: repaired {len(fixed)} of {len(docs)}; {len(docs) - len(fixed)} left as they were "
              f"(OCR was no better); {stale} metadata rows cleared for regeneration")

    print("\nNext: python generate_metadata.py --source all; python enrich_metadata.py --source all; "
          "python finalize_metadata.py")


if __name__ == "__main__":
    main()
