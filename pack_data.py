"""
pack_data.py

The chunk files (data/<source>/chunks.jsonl, one per source) are the text the app searches and answers from.
Several are over GitHub's 100 MB limit, so the repository carries them gzipped (about a tenth of the size) and
they are unpacked after cloning:

    python pack_data.py unpack      chunks.jsonl.gz -> chunks.jsonl, for every source that has no chunks.jsonl yet
    python pack_data.py pack        chunks.jsonl -> chunks.jsonl.gz, after chunk_statutes.py, before committing

Packing is deterministic (no timestamp or file name in the gzip header), so a source whose chunks did not change
gives the same .gz bytes and nothing new to commit. Files are copied as bytes: line endings are kept as they are.
"""

import argparse
import gzip
import shutil
import time

from ocr_scanned import SOURCES, source_layout


def pack(force: bool) -> None:
    for source in SOURCES:
        plain = source_layout(source)[0] / "chunks.jsonl"
        packed = plain.with_name("chunks.jsonl.gz")
        if not plain.exists():
            print(f"{source:12s} no chunks.jsonl, skipped")
            continue
        if packed.exists() and packed.stat().st_mtime >= plain.stat().st_mtime and not force:
            print(f"{source:12s} up to date")
            continue
        started = time.time()
        with plain.open("rb") as f, open(packed, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            shutil.copyfileobj(f, out, 1 << 20)
        print(f"{source:12s} {plain.stat().st_size / 2**20:6.0f} MB -> {packed.stat().st_size / 2**20:5.0f} MB  ({time.time() - started:.0f}s)")


def unpack(force: bool) -> None:
    for source in SOURCES:
        plain = source_layout(source)[0] / "chunks.jsonl"
        packed = plain.with_name("chunks.jsonl.gz")
        if not packed.exists():
            print(f"{source:12s} no chunks.jsonl.gz, skipped")
            continue
        if plain.exists() and not force:
            print(f"{source:12s} chunks.jsonl already there (--force to overwrite)")
            continue
        with gzip.open(packed, "rb") as f, plain.open("wb") as out:
            shutil.copyfileobj(f, out, 1 << 20)
        print(f"{source:12s} {plain.stat().st_size / 2**20:6.0f} MB unpacked")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["pack", "unpack"])
    parser.add_argument("--force", action="store_true", help="redo every source, even those that look up to date")
    args = parser.parse_args()
    (pack if args.action == "pack" else unpack)(args.force)


if __name__ == "__main__":
    main()
