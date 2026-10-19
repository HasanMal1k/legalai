"""
repair_pages.py

Mends garbled text, page by page, in the laws that are served (tier current or historical) and in
those held back only because of their text -- on the local GPU:

  stage ocr    every page re-read with docTR (ocr_scanned.py), and its garble measured
  stage vlm    pages still garbled are transcribed by the two local vision models (Gemma, then Qwen,
               through translate_urdu.ask): one model over every page, then the other, since both do
               not fit on the GPU at once
  stage apply  per page, the best reading wins -- the text layer the PDF came with, docTR's, or the
               models' transcription, the last only where the two models agree closely and both read
               clean. A document's text is replaced only when it comes out clearly less garbled; the
               old text goes to data/<source>/repaired_text_backup.jsonl.

    python repair_pages.py --stage ocr
    python repair_pages.py --stage vlm --minutes 100      (time-boxed, resumable; current-tier laws first)
    python repair_pages.py --stage apply                  then: finalize_metadata.py, chunk_statutes.py

Page readings are cached in data/<source>/page_repair.jsonl, one row per (document, page, reader), so
every stage can be stopped and resumed. Metadata rows are kept: the model's description was made from
the old text and still describes the same law.

"Garbled" is finalize_metadata's measure: the share of a page's ordinary lower-case words that appear
in fewer than 5 documents of the whole corpus. A page counts as clean at 2% or less.
"""

import argparse
import difflib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from ocr_scanned import SOURCES, source_layout

WORD = re.compile(r"(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])")
CLEAN = 0.02          # a page this clean is left alone
AGREE = 0.90          # the two models' readings must match this closely (share of words in sequence)
PROMPT = ("This is a scanned page of a Pakistani law or government notification, in English. Transcribe ALL the text on the page "
          "exactly as printed, line by line, top to bottom, keeping numbering, brackets and punctuation. Correct nothing, add nothing, "
          "summarise nothing. Where a word cannot be read, write [illegible]. Output only the transcription.")


def corpus_counts() -> Counter:
    seen = Counter()
    for source in SOURCES:
        data_dir, _ = source_layout(source)
        with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                seen.update(set(WORD.findall(json.loads(line)["raw_text"])))
    return seen


def garble(text: str, seen: Counter) -> float:
    words = WORD.findall(text)
    return sum(seen[w] < 5 for w in words) / max(1, len(words))


def targets() -> list[dict]:
    """Laws to mend: served or held only for their text, not clean, with their PDF on disk. Current first."""
    out = []
    for source in SOURCES:
        data_dir, key = source_layout(source)
        rows = {}
        with (data_dir / "statutes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows[row[key]] = row
        for record in map(json.loads, (data_dir / "metadata_final.jsonl").open(encoding="utf-8")):
            text_only = record["tier"] == "quarantine" and record["quarantine_reasons"] \
                and all(x.startswith("text:") for x in record["quarantine_reasons"])
            served = record["tier"] in ("current", "historical")
            if not (served or text_only) or record["confidence"]["text"] == "high" or record["text_source"] == "machine-translation":
                continue
            row = rows.get(record[key])
            pdf = data_dir / "raw_pdfs" / (row.get("pdf_file") or "") if row else None
            if pdf and pdf.is_file():
                out.append({"source": source, "key": key, "id": record[key], "title": record["title"], "tier": record["tier"],
                            "text_source": record["text_source"], "pdf": str(pdf), "garble": record["garbled_words"],
                            "chars": len(row["raw_text"])})
    order = {"current": 0, "quarantine": 1, "historical": 2}
    return sorted(out, key=lambda t: (order[t["tier"]], -t["garble"]))


def cache_path(source: str) -> Path:
    return source_layout(source)[0] / "page_repair.jsonl"


def load_cache() -> dict:
    done = {}
    for source in SOURCES:
        path = cache_path(source)
        if path.exists():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                done[(source, row["id"], row["page"], row["reader"])] = row["text"]
    return done


def save(source: str, doc_id: str, page: int, reader: str, text: str) -> None:
    with cache_path(source).open("a", encoding="utf-8") as out:
        out.write(json.dumps({"id": doc_id, "page": page, "reader": reader, "text": text}, ensure_ascii=False) + "\n")


def stage_ocr(todo: list[dict], cache: dict, minutes: float) -> None:
    """docTR on every page; the PDF's own text layer is kept per page as the 'layer' reading."""
    import pypdfium2 as pdfium
    import torch
    import ocr_scanned

    model = ocr_scanned.load_model()
    started, pages_done = time.time(), 0
    for n, doc in enumerate(todo, 1):
        pdf = pdfium.PdfDocument(doc["pdf"])
        try:
            for page in range(len(pdf)):
                if (doc["source"], doc["id"], page, "doctr") in cache:
                    continue
                if (doc["source"], doc["id"], page, "layer") not in cache:
                    layer = pdf[page].get_textpage().get_text_range() if doc["text_source"] == "native" else ""
                    save(doc["source"], doc["id"], page, "layer", layer)
                image = ocr_scanned.render_pages(pdf, page, page + 1)
                with torch.inference_mode(), torch.autocast("cuda"):
                    result = model(image)
                text = ocr_scanned.tidy(ocr_scanned.page_text(result.pages[0]))
                save(doc["source"], doc["id"], page, "doctr", text)
                cache[(doc["source"], doc["id"], page, "doctr")] = text
                pages_done += 1
        finally:
            pdf.close()
        if n % 20 == 0:
            rate = pages_done / max(1, time.time() - started)
            print(f"  ocr: {n}/{len(todo)} documents, {pages_done} pages ({rate:.1f} pages/s)", flush=True)
        if (time.time() - started) / 60 > minutes:
            print(f"  ocr: stopped after {minutes} minutes at document {n} of {len(todo)}", flush=True)
            return
    print(f"  ocr: done, {pages_done} pages read", flush=True)


def page_images(doc: dict, pages: list[int]) -> dict[int, bytes]:
    import pypdfium2 as pdfium
    from PIL import Image
    import ocr_scanned

    pdf = pdfium.PdfDocument(doc["pdf"])
    out = {}
    try:
        for page in pages:
            buffer = io.BytesIO()
            Image.fromarray(ocr_scanned.render_pages(pdf, page, page + 1)[0]).save(buffer, format="PNG")
            out[page] = buffer.getvalue()
    finally:
        pdf.close()
    return out


def best_old(doc: dict, page: int, cache: dict, seen: Counter) -> tuple[str, float]:
    readings = [cache.get((doc["source"], doc["id"], page, r)) or "" for r in ("layer", "doctr")]
    readings = [t for t in readings if t.strip()]
    if not readings:
        return "", 1.0
    longest = max(len(t) for t in readings)
    usable = [t for t in readings if len(t) >= 0.6 * longest]
    text = min(usable, key=lambda t: garble(t, seen))
    return text, garble(text, seen)


SAMPLE = 3        # pages a law's check starts with: its most garbled ones
CONFIRMS = 0.97   # a model's reading this close to docTR's (words in sequence) confirms the page as it is


def flagged_pages(doc: dict, cache: dict, seen: Counter) -> list[int]:
    """The law's pages that do not read clean, most garbled first."""
    pages = sorted({p for (s, i, p, r) in cache if s == doc["source"] and i == doc["id"] and r == "doctr"})
    scored = [(best_old(doc, p, cache, seen)[1], p) for p in pages]
    return [p for share, p in sorted(scored, reverse=True) if share > CLEAN]


def page_verdict(doc: dict, page: int, cache: dict, seen: Counter) -> str:
    """confirmed (a model reads the page as docTR or the PDF's own layer does), corrected (both models read it clean and
    agree, differently), open (no clean agreement), unread."""
    key = (doc["source"], doc["id"], page)
    a, b = cache.get((*key, "vlm_a")), cache.get((*key, "vlm_b"))
    if a is None:
        return "unread"
    old = best_old(doc, page, cache, seen)[0]
    if old and agreement(old, a) >= CONFIRMS:
        return "confirmed"
    if b and garble(a, seen) <= CLEAN and garble(b, seen) <= CLEAN and agreement(a, b) >= AGREE and "[illegible]" not in a \
            and len(a) >= 0.6 * max(1, len(old)):
        return "corrected"
    return "open"


def stage_vlm(todo: list[dict], cache: dict, seen: Counter, minutes: float, model_names: list[str]) -> None:
    """Tier by tier (current first): the first model reads a sample of each law's worst pages; where it confirms docTR
    on all of them, the law's text stands as checked. Where it does not, it reads the law's other flagged pages, and
    the second model reads the pages the first read clean but differently, to confirm the correction."""
    import translate_urdu as t

    started = time.time()

    def ask(doc, pages, model, reader):
        if not pages:
            return True
        images = page_images(doc, pages)
        for page in pages:
            text = t.ask(model, PROMPT, images[page])
            save(doc["source"], doc["id"], page, reader, text)
            cache[(doc["source"], doc["id"], page, reader)] = text
            if (time.time() - started) / 60 > minutes:
                print(f"  vlm: stopped after {minutes} minutes", flush=True)
                return False
        return True

    for tier in ("current", "quarantine", "historical"):
        docs = [d for d in todo if d["tier"] == tier]
        print(f"  vlm: {tier} laws ({len(docs)}), sampling with {model_names[0]}", flush=True)
        for doc in docs:
            sample = [p for p in flagged_pages(doc, cache, seen)[:SAMPLE] if (doc["source"], doc["id"], p, "vlm_a") not in cache]
            if not ask(doc, sample, model_names[0], "vlm_a"):
                return
        print(f"  vlm: {tier} laws, the rest of the pages where the sample disagreed", flush=True)
        for doc in docs:
            flagged = flagged_pages(doc, cache, seen)
            if all(page_verdict(doc, p, cache, seen) == "confirmed" for p in flagged[:SAMPLE]):
                continue
            rest = [p for p in flagged if (doc["source"], doc["id"], p, "vlm_a") not in cache]
            if not ask(doc, rest, model_names[0], "vlm_a"):
                return
        print(f"  vlm: {tier} laws, second readings with {model_names[1]}", flush=True)
        for doc in docs:
            second = [p for p in flagged_pages(doc, cache, seen) if page_verdict(doc, p, cache, seen) == "open"
                      and garble(cache.get((doc["source"], doc["id"], p, "vlm_a")) or "x", seen) <= CLEAN
                      and (doc["source"], doc["id"], p, "vlm_b") not in cache]
            if not ask(doc, second, model_names[1], "vlm_b"):
                return
    print("  vlm: all tiers done", flush=True)


def agreement(a: str, b: str) -> float:
    wa, wb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio() if wa and wb else 0.0


def stage_apply(todo: list[dict], cache: dict, seen: Counter) -> None:
    """Per law: the text is replaced only when the page readings make it clearly less garbled (docTR's or the PDF's own
    layer, and the models' agreed corrections); and the law's `text_check` records what the readings showed:
    confirmed (every flagged page read the same by a second reader, or corrected by two models), confirmed_by_sample
    (the worst pages were, the rest unread), or open."""
    report = Counter()
    changed, checks = {}, {}
    for doc in todo:
        pages = sorted({p for (s, i, p, r) in cache if s == doc["source"] and i == doc["id"] and r == "doctr"})
        if not pages:
            continue
        out, used = [], Counter()
        flagged = set(flagged_pages(doc, cache, seen))
        verdicts = Counter()
        for page in pages:
            text, share = best_old(doc, page, cache, seen)
            used["old" if text == (cache.get((doc["source"], doc["id"], page, "layer")) or None) else "doctr"] += 1
            if page in flagged:
                verdict = page_verdict(doc, page, cache, seen)
                verdicts[verdict] += 1
                if verdict == "corrected":
                    text = cache[(doc["source"], doc["id"], page, "vlm_a")]
                    used["models"] += 1
            out.append(text)
        sampled = flagged_pages(doc, cache, seen)[:SAMPLE]
        if not flagged or verdicts["open"] == verdicts["unread"] == 0:
            check = "confirmed"
        elif verdicts["open"] == 0 and sampled and all(page_verdict(doc, p, cache, seen) in ("confirmed", "corrected") for p in sampled):
            check = "confirmed_by_sample"
        else:
            check = "open"
        checks.setdefault(doc["source"], {})[doc["id"]] = {"verdict": check, "pages": dict(verdicts)}
        report[f"check: {check}"] += 1
        new = "\n".join(out).strip()
        new_share = garble(new, seen)
        if new_share <= doc["garble"] - 0.005 and len(new) >= 0.7 * max(1, doc.get("chars", len(new))):
            changed.setdefault(doc["source"], {})[doc["id"]] = (new, used)
            report["text replaced"] += 1
    print("apply:", dict(report))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for source in set(changed) | set(checks):
        docs, checked = changed.get(source, {}), checks.get(source, {})
        data_dir, key = source_layout(source)
        path = data_dir / "statutes.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst, \
                (data_dir / "repaired_text_backup.jsonl").open("a", encoding="utf-8") as backup:
            for line in src:
                row = json.loads(line)
                if row[key] in docs:
                    backup.write(json.dumps({key: row[key], "raw_text": row["raw_text"], "text_source": row.get("text_source"),
                                             "replaced_at": now}, ensure_ascii=False) + "\n")
                    text, used = docs[row[key]]
                    row.update(raw_text=text, text_source="ocr", repaired_from=row.get("text_source", "native"), repaired_at=now,
                               repair_readers=dict(used))
                if row[key] in checked:
                    row["text_check"] = {**checked[row[key]], "at": now}
                if row[key] in docs or row[key] in checked:
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                dst.write(line)
        os.replace(tmp, path)
        print(f"{source}: text replaced for {len(docs)} documents; checks recorded for {len(checked)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["list", "ocr", "vlm", "apply"], required=True)
    parser.add_argument("--minutes", type=float, default=100)
    parser.add_argument("--models", nargs=2, default=["gemma4:12b-it-qat", "qwen3.5:9b"])
    args = parser.parse_args()
    todo = targets()
    print(f"{len(todo)} laws to mend: {dict(Counter(t['tier'] for t in todo))}", flush=True)
    if args.stage == "list":
        return
    cache = load_cache()
    if args.stage == "ocr":
        stage_ocr(todo, cache, args.minutes)
        return
    seen = corpus_counts()
    if args.stage == "vlm":
        stage_vlm(todo, cache, seen, args.minutes, args.models)
    else:
        stage_apply(todo, cache, seen)


if __name__ == "__main__":
    main()
