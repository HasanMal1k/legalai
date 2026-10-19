"""
translate_urdu.py

Stage 4: the scanned documents that ocr_scanned.py rejected as "not
English" are Urdu-language laws. Read them with local vision models and
save an English translation as the document's text.

Classic OCR engines can't read the Nastaliq script these are printed in,
so each page is transcribed independently by two vision models (they
misread different words), and a third pass reconciles the two readings
and writes the English. Records are marked "text_source":
"machine-translation" and keep the Urdu transcription in "original_text":
this is a reading aid, not an authoritative translation.

Needs Ollama running locally (https://ollama.com) with both models pulled:
    ollama pull gemma4:12b-it-qat
    ollama pull qwen3.5:9b

Run (every stage is resumable; each page's work is cached on disk):
    python translate_urdu.py --stage transcribe-a
    python translate_urdu.py --stage transcribe-b
    python translate_urdu.py --stage translate
    python translate_urdu.py --stage save
    python translate_urdu.py                 # all four in order

The stages are separate passes because the two models don't fit in GPU
memory together.
"""

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import pypdfium2 as pdfium
import requests
from PIL import Image
from tqdm import tqdm

from ocr_scanned import SOURCES, read_jsonl, render_pages, source_layout, tidy

OLLAMA = "http://localhost:11434/api/chat"
MODEL_A = "gemma4:12b-it-qat"   # also the translator
MODEL_B = "qwen3.5:9b"
MIN_URDU_SHARE = 0.3            # below this a "not English" document isn't Urdu either

TRANSCRIBE = (
    "This is a scanned page of an Urdu-language law. Transcribe ALL the text on the page exactly as written, "
    "in Urdu script, line by line, top to bottom. Keep numbers and any English text as they appear. "
    "Do not translate, summarise, explain or add anything. Output only the transcription.")

TRANSLATE = """Below are two independent machine transcriptions (A and B) of the SAME page of an Urdu-language law of Pakistan. Each contains some misread words, usually in different places.

Work out what the page actually says: where A and B differ, choose the reading that is a real Urdu word and makes sense in a legal text; where both are doubtful, write [unclear].

Then output a clear, faithful English translation of the page in normal legal drafting style, keeping the numbering and structure. Do not add, omit, summarise or explain anything. Output only the English translation.

TRANSCRIPTION A:
{a}

TRANSCRIPTION B:
{b}
"""


TRANSLATE_ONE = """Translate this page of an Urdu-language law of Pakistan into clear, faithful English in normal legal drafting style, keeping the numbering and structure. It is a machine transcription and may contain misread words; where a word makes no sense write [unclear] once and move on. Every part of your answer must be in English. Do not add, omit, summarise or explain anything. Output only the English translation.

{a}
"""

FAILED_PAGE = "[page could not be translated]"


def page_ok(english: str) -> bool:
    """Reject the ways a translation goes wrong: nothing, a loop, or Urdu echoed back."""
    if not english.strip():
        return False
    lines = [line.strip() for line in english.split("\n") if line.strip()]
    most_repeated = max(lines.count(line) for line in set(lines))
    if most_repeated >= 8 and most_repeated / len(lines) > 0.3:
        return False
    if len(re.findall(r"\[(?:unclear|illegible)\]", english, re.I)) > 0.2 * len(english.split()):
        return False
    return len(re.findall(r"[؀-ۿ]", english)) / len(english) < 0.2


def ask(model: str, prompt: str, image_png: bytes | None = None) -> str:
    message = {"role": "user", "content": prompt}
    if image_png:
        message["images"] = [base64.b64encode(image_png).decode()]
    options = {"temperature": 0, "num_ctx": 8192, "num_predict": 3000}
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA, json={
                "model": model, "stream": False, "think": False, "messages": [message], "options": options,
            }, timeout=1800)
            if resp.status_code == 500 and "repeat limit" in resp.text:
                # The model looped on a repetitive page (tables, lists) and
                # Ollama cut it off. Try again discouraging repetition; if it
                # loops again the page is given up as unreadable.
                if "repeat_penalty" in options:
                    return ""
                options = {**options, "repeat_penalty": 1.3, "temperature": 0.2}
                continue
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as e:
            last_error = e
            time.sleep(10 * (attempt + 1))
    if last_error is None:
        return ""
    raise last_error


def urdu_documents() -> list[dict]:
    """Documents whose latest failure is 'not English' (and that have no record yet)."""
    docs = []
    for source in SOURCES:
        data_dir, key = source_layout(source)
        latest = {}
        for failure in read_jsonl(data_dir / "fetch_failures.jsonl"):
            latest[failure.get(key) or failure.get("id") or failure.get("detail_url")] = failure
        done = {row[key] for row in read_jsonl(data_dir / "statutes.jsonl")}
        catalog = {row[key]: row for row in read_jsonl(data_dir / "catalog.jsonl")}
        located = {row[key]: row for row in read_jsonl(data_dir / "scanned_files.jsonl")}
        for doc_id, failure in latest.items():
            if failure["reason"].startswith("not English") and doc_id not in done:
                docs.append({
                    "source": source, "key": key, "law": catalog[doc_id],
                    "pdf_url": located.get(doc_id, {}).get("pdf_url"),
                    "pdf_file": failure["pdf_file"],
                    "pdf_path": data_dir / "raw_pdfs" / failure["pdf_file"],
                    # per-page working files: p001.a.txt, p001.b.txt, p001.en.txt
                    "work_dir": data_dir / "urdu_work" / Path(failure["pdf_file"]).stem,
                })
    return docs


def page_file(doc: dict, index: int, kind: str) -> Path:
    return doc["work_dir"] / f"p{index + 1:03d}.{kind}.txt"


def urdu_share(text: str) -> float:
    letters = re.findall(r"[^\W\d_]", text)
    return sum("؀" <= c <= "ۿ" for c in letters) / max(1, len(letters))


def is_urdu(doc: dict, pages: int) -> bool:
    """Judged from model A's transcription of the first pages."""
    sample = "".join(page_file(doc, i, "a").read_text(encoding="utf-8")
                     for i in range(min(3, pages)) if page_file(doc, i, "a").exists())
    return not sample or urdu_share(sample) >= MIN_URDU_SHARE


def transcribe(docs: list[dict], model: str, kind: str) -> None:
    jobs = []
    for doc in docs:
        pdf = pdfium.PdfDocument(doc["pdf_path"])
        pages = len(pdf)
        pdf.close()
        if kind == "b" and not is_urdu(doc, pages):
            continue
        jobs += [(doc, i) for i in range(pages) if not page_file(doc, i, kind).exists()
                 # model B only reads pages model A has already read
                 and (kind == "a" or page_file(doc, i, "a").exists())]
    print(f"{model}: {len(jobs)} pages to transcribe", flush=True)
    for doc, index in tqdm(jobs, desc=f"Transcribing ({model})", unit="page"):
        # model A decides early whether the document is Urdu at all
        if kind == "a" and index >= 3 and not is_urdu(doc, 3):
            continue
        pdf = pdfium.PdfDocument(doc["pdf_path"])
        image = Image.fromarray(render_pages(pdf, index, index + 1)[0])
        pdf.close()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        text = ask(model, TRANSCRIBE, buffer.getvalue())
        doc["work_dir"].mkdir(parents=True, exist_ok=True)
        page_file(doc, index, kind).write_text(text, encoding="utf-8")


def translate(docs: list[dict]) -> None:
    jobs = []
    for doc in docs:
        for a_path in sorted(doc["work_dir"].glob("p*.a.txt")):
            b_path = a_path.with_name(a_path.name.replace(".a.txt", ".b.txt"))
            en_path = a_path.with_name(a_path.name.replace(".a.txt", ".en.txt"))
            if b_path.exists() and not en_path.exists():
                jobs.append((a_path, b_path, en_path))
    print(f"{MODEL_A}: {len(jobs)} pages to translate", flush=True)
    for a_path, b_path, en_path in tqdm(jobs, desc="Translating", unit="page"):
        a, b = a_path.read_text(encoding="utf-8"), b_path.read_text(encoding="utf-8")
        if max(len(a), len(b)) <= 20:
            # a blank page: nothing on it but, at most, a page number
            english = ""
        else:
            # Reconciling both readings is best, but on repetitive pages
            # (tables, lists) it can loop or echo the Urdu back, so fall back
            # to one reading at a time, then to the other model as translator.
            attempts = [(MODEL_A, TRANSLATE.format(a=a or b, b=b or a)),
                        (MODEL_A, TRANSLATE_ONE.format(a=a or b)),
                        (MODEL_B, TRANSLATE_ONE.format(a=b or a)),
                        (MODEL_B, TRANSLATE.format(a=a or b, b=b or a))]
            english = FAILED_PAGE
            for model, prompt in attempts:
                candidate = ask(model, prompt)
                if page_ok(candidate):
                    english = candidate
                    break
        en_path.write_text(english, encoding="utf-8")


def save(docs: list[dict]) -> None:
    for doc in docs:
        data_dir, key = source_layout(doc["source"])
        pdf = pdfium.PdfDocument(doc["pdf_path"])
        pages = len(pdf)
        pdf.close()
        english = [page_file(doc, i, "en") for i in range(pages)]
        name = doc["law"]["name"][:60]
        if not is_urdu(doc, pages):
            print(f"  skipped (not Urdu): {name}")
            continue
        if not all(p.exists() for p in english):
            print(f"  not finished ({sum(p.exists() for p in english)}/{pages} pages translated): {name}")
            continue
        text = tidy("\n\n".join(p.read_text(encoding="utf-8") for p in english))
        untranslated = [i + 1 for i, p in enumerate(english) if p.read_text(encoding="utf-8").startswith("[page could not")]
        urdu = tidy("\n\n".join(page_file(doc, i, "a").read_text(encoding="utf-8") for i in range(pages)))
        if len(text) < 100:
            print(f"  skipped (no text came out): {name}")
            continue
        with (data_dir / "statutes.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({
                **doc["law"],
                "pdf_url": doc["pdf_url"],
                "pdf_file": doc["pdf_file"],
                "raw_text": text,
                "num_pages": pages,
                "text_source": "machine-translation",
                "original_language": "ur",
                "original_text": urdu,
                "translation_note": f"Urdu scan transcribed by {MODEL_A} and {MODEL_B}, reconciled and translated "
                                    f"to English by {MODEL_A}. Not an official translation; may contain errors.",
                "untranslated_pages": untranslated,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ensure_ascii=False) + "\n")
        print(f"  saved: {name} ({pages} pages, {len(text)} chars of English"
              + (f"; pages {untranslated} could not be translated" if untranslated else "") + ")")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["transcribe-a", "transcribe-b", "translate", "save", "all"], default="all")
    args = parser.parse_args()

    docs = urdu_documents()
    print(f"{len(docs)} documents flagged as not English")
    if args.stage in ("transcribe-a", "all"):
        transcribe(docs, MODEL_A, "a")
    if args.stage in ("transcribe-b", "all"):
        transcribe(docs, MODEL_B, "b")
    if args.stage in ("translate", "all"):
        translate(docs)
    if args.stage in ("save", "all"):
        save(docs)


if __name__ == "__main__":
    main()
