"""
ocr_scanned.py

Stage 3: OCR the PDFs that fetch_statutes.py / fetch_sources.py skipped
as "likely scanned, needs OCR", and add them to the same statutes.jsonl
with "text_source": "ocr" so they can be told apart from native text.

OCR runs on the GPU with docTR (PyTorch): a few worker processes share
the card, each rendering PDF pages (pypdfium2) and recognising them.

Install:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install python-doctr

Run:
    python ocr_scanned.py --source sindh
    python ocr_scanned.py --source pakistancode --stage map   # only stage 1
    python ocr_scanned.py --source punjab --limit 500         # bounded batch

Two stages, both resumable:
  map  The failure log says which *document* was scanned, not which file
       that is, and for most sources the PDF URL is only on the document's
       detail page. So revisit that page (one request per document, at the
       usual polite pace -- no PDF is downloaded again) and note the file
       in data/<source>/scanned_files.jsonl.
  ocr  Render pages, recognise them on the GPU, write the record.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import requests
from tqdm import tqdm

import fetch_sources
from fetch_statutes import resolve_pdf_url

DATA_DIR = Path(__file__).parent / "data"
SOURCES = ["pakistancode", *fetch_sources.SOURCES]

OCR_DPI = 200
MAX_PAGE_PIXELS = 3000  # cap the long edge so an oversized scan can't exhaust memory
MIN_CHARS = 100         # same threshold the fetchers use for "no usable text"
PAGES_PER_CHUNK = 8     # pages rendered and sent to the GPU together


def source_layout(source: str) -> tuple[Path, str]:
    """(data directory, name of the field that identifies a document)."""
    if source == "pakistancode":
        return DATA_DIR, "detail_url"
    return DATA_DIR / source, "id"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def done_ids(path: Path, key: str) -> set[str]:
    return {row[key] for row in read_jsonl(path) if key in row}


def scanned_documents(source: str) -> list[dict]:
    """Catalog rows whose latest failure is 'scanned' and that still have no record."""
    data_dir, key = source_layout(source)
    latest = {}
    for failure in read_jsonl(data_dir / "fetch_failures.jsonl"):
        doc_id = failure.get(key) or failure.get("id") or failure.get("detail_url")
        latest[doc_id] = failure["reason"]
    done = done_ids(data_dir / "statutes.jsonl", key)
    wanted = {doc_id for doc_id, reason in latest.items() if "likely scanned" in reason and doc_id not in done}
    return [row for row in read_jsonl(data_dir / "catalog.jsonl") if row[key] in wanted]


def locate_file(law: dict, source: str, session: requests.Session) -> tuple[str | None, str | None, bool]:
    """(pdf_url, local file name, whether a request was made)."""
    if source == "pakistancode":
        pdf_url = resolve_pdf_url(law["detail_url"], session)
        return pdf_url, (pdf_url.split("/")[-1] if pdf_url else None), True
    if law.get("pdf_url"):
        return law["pdf_url"], fetch_sources.local_name(law["pdf_url"]), False
    pdf_url, _ = fetch_sources.read_detail_page(law, source, session)
    return pdf_url, (fetch_sources.local_name(pdf_url) if pdf_url else None), True


def map_files(source: str, limit: int | None) -> None:
    data_dir, key = source_layout(source)
    map_path = data_dir / "scanned_files.jsonl"
    mapped = done_ids(map_path, key)
    todo = [law for law in scanned_documents(source) if law[key] not in mapped]
    if limit:
        todo = todo[:limit]
    print(f"{source}: {len(todo)} scanned documents to locate ({len(mapped)} already located)", flush=True)
    session = requests.Session()
    missing = 0
    with map_path.open("a", encoding="utf-8") as out:
        for law in tqdm(todo, desc=f"Locating {source}"):
            try:
                pdf_url, pdf_file, requested = locate_file(law, source, session)
            except Exception as e:
                print(f"  ! {law['name']}: {type(e).__name__}: {e}", flush=True)
                time.sleep(5)
                continue
            on_disk = bool(pdf_file) and (data_dir / "raw_pdfs" / pdf_file).exists()
            missing += not on_disk
            out.write(json.dumps({key: law[key], "pdf_url": pdf_url, "pdf_file": pdf_file,
                                  "on_disk": on_disk}, ensure_ascii=False) + "\n")
            out.flush()
            if requested:
                time.sleep(1.5)  # polite pacing -- government servers
    if missing:
        print(f"  ! {missing} documents whose file is not in raw_pdfs/ (see on_disk=false in {map_path.name})")


def render_pages(pdf, start: int, end: int) -> list[np.ndarray]:
    """Render pages [start, end) of an open PDF to RGB arrays."""
    images = []
    for index in range(start, end):
        page = pdf[index]
        width, height = page.get_size()
        scale = min(OCR_DPI / 72, MAX_PAGE_PIXELS / max(width, height))
        images.append(np.ascontiguousarray(page.render(scale=scale).to_numpy()[:, :, :3]))
    return images


def load_model():
    # docTR builds and tears down a thread pool on every call; on Windows
    # that costs more than the OCR itself. Must be set before importing it.
    os.environ["DOCTR_MULTIPROCESSING_DISABLE"] = "TRUE"
    import torch
    from doctr.models import ocr_predictor

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch -- install the CUDA build of torch (see the top of this file).")
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True,
                          det_bs=PAGES_PER_CHUNK, reco_bs=512)
    return model.cuda()


_model = None  # one per GPU worker process


def init_gpu_worker() -> None:
    global _model
    _model = load_model()


def ocr_document(pdf_path: str) -> tuple[str, int]:
    """Text and page count of one PDF. Runs in a GPU worker process."""
    import torch

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        texts = []
        for start in range(0, len(pdf), PAGES_PER_CHUNK):
            images = render_pages(pdf, start, min(start + PAGES_PER_CHUNK, len(pdf)))
            # half precision: same output on these pages, less time and memory
            with torch.inference_mode(), torch.autocast("cuda"):
                result = _model(images)
            texts.extend(page_text(page) for page in result.pages)
        return tidy("\n".join(texts)), len(pdf)
    except Exception:
        torch.cuda.empty_cache()
        raise
    finally:
        pdf.close()


def page_text(page) -> str:
    """Rebuild reading-order lines from docTR's word boxes.

    docTR's own line builder splits widely spaced (justified) lines into
    fragments, so group words by vertical position instead: these pages
    are single-column, and that keeps each printed line together.
    """
    words = []
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (x0, y0), (x1, y1) = word.geometry[0], word.geometry[-1]
                words.append((min(y0, y1), max(y0, y1), min(x0, x1), word.value))
    if not words:
        return ""
    height = float(np.median([bottom - top for top, bottom, _, _ in words]))
    words.sort(key=lambda w: (w[0] + w[1]) / 2)
    lines = []  # each: [centre, bottom, [(x, text), ...]]
    for top, bottom, x, value in words:
        centre = (top + bottom) / 2
        if lines and abs(centre - lines[-1][0]) < 0.6 * height:
            line = lines[-1]
            line[2].append((x, value))
            line[0] += (centre - line[0]) / len(line[2])  # running mean follows a slight skew
            line[1] = max(line[1], bottom)
        else:
            lines.append([centre, bottom, [(x, value)]])
    out = []
    previous_bottom = None
    for centre, bottom, items in lines:
        if previous_bottom is not None and (centre - height / 2) - previous_bottom > 1.2 * height:
            out.append("")  # visible gap on the page -> paragraph break
        out.append(" ".join(value for _, value in sorted(items)))
        previous_bottom = bottom
    return "\n".join(out)


# Words that make up a large share of any English legal text.
COMMON_WORDS = frozenset(
    "the of and to in a be shall or by for any as is this such under which may with not on act section "
    "government that an at from it are have been his person provided rules order said court no sub "
    "authority board made members member other date province provincial pakistan law".split())
MIN_COMMON_SHARE = 0.15  # real English pages score 0.25+; Urdu scans read as English score under 0.10


def looks_english(text: str) -> bool:
    """False when the OCR output is gibberish -- in practice an Urdu page, which this English model can't read."""
    words = re.findall(r"[A-Za-z]+", text.lower())

    def common_share(sample: list[str]) -> float:
        return sum(word in COMMON_WORDS for word in sample) / max(1, len(sample))

    # Judge the opening too: an Act that is mostly a schedule of place names
    # has few common words overall but starts with ordinary legal English.
    return bool(words) and max(common_share(words), common_share(words[:400])) >= MIN_COMMON_SHARE


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_ocr(source: str, limit: int | None, workers: int) -> None:
    data_dir, key = source_layout(source)
    located = {row[key]: row for row in read_jsonl(data_dir / "scanned_files.jsonl")}
    scanned = scanned_documents(source)
    todo = [law for law in scanned if located.get(law[key], {}).get("on_disk")]
    if limit:
        todo = todo[:limit]
    print(f"{source}: {len(todo)} documents to OCR ({len(scanned) - len(todo)} scanned documents "
          f"not located yet or missing on disk)", flush=True)
    if not todo:
        return

    # The GPU sits mostly idle behind one process's CPU-side work (cropping
    # and resizing every word), so several workers share it. Each loads its
    # own copy of the model; this process is the only one that writes.
    workers = min(workers, len(todo))
    added = empty = 0
    with (data_dir / "statutes.jsonl").open("a", encoding="utf-8") as out_f, \
            (data_dir / "fetch_failures.jsonl").open("a", encoding="utf-8") as fail_f, \
            ProcessPoolExecutor(max_workers=workers, initializer=init_gpu_worker) as pool, \
            tqdm(total=len(todo), desc=f"OCR {source} ({workers} GPU workers)", unit="doc") as bar:

        def finish(law: dict, text: str, num_pages: int | None, error: str | None) -> None:
            nonlocal added, empty
            where = located[law[key]]
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if error or len(text) < MIN_CHARS or not looks_english(text):
                empty += 1
                # the reason deliberately doesn't say "scanned", so reruns don't retry it
                if error:
                    reason = f"OCR failed -- {error}"
                elif len(text) >= MIN_CHARS:
                    reason = "not English -- OCR output is unreadable (Urdu-language scan; needs an Urdu OCR model)"
                elif text:
                    # e.g. one-page placeholders that only say the law was repealed
                    reason = f"not a document -- OCR found only {len(text)} characters: {text[:80]!r}"
                else:
                    reason = "OCR found no text -- blank or illegible pages"
                fail_f.write(json.dumps({key: law[key], "name": law["name"], "reason": reason,
                                         "pdf_file": where["pdf_file"], "at": now}, ensure_ascii=False) + "\n")
                fail_f.flush()
                return
            added += 1
            out_f.write(json.dumps({
                **law,
                "pdf_url": where["pdf_url"],
                "pdf_file": where["pdf_file"],
                "raw_text": text,
                "num_pages": num_pages,
                "text_source": "ocr",
                "fetched_at": now,
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        futures = {pool.submit(ocr_document, str(data_dir / "raw_pdfs" / located[law[key]]["pdf_file"])): law
                   for law in todo}
        pages = 0
        started = time.time()
        for future in as_completed(futures):
            law = futures[future]
            try:
                text, num_pages = future.result()
                pages += num_pages
                finish(law, text, num_pages, None)
            except Exception as e:
                finish(law, "", None, f"{type(e).__name__}: {e}")
            bar.update(1)
            bar.set_postfix_str(f"{pages} pages, {pages / max(1e-9, time.time() - started):.1f}/s")

    print(f"\nOK -> {data_dir / 'statutes.jsonl'}  ({added} documents added, {empty} with no usable OCR text)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--stage", choices=["map", "ocr", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2,
                        help="OCR processes sharing the GPU; each takes about 4 GB of GPU memory "
                             "(2 fit a 10 GB card, 3 overflow it and run slower)")
    args = parser.parse_args()

    if args.stage in ("map", "all"):
        map_files(args.source, args.limit)
    if args.stage in ("ocr", "all"):
        run_ocr(args.source, args.limit, args.workers)


if __name__ == "__main__":
    main()
