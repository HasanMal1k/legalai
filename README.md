# Pakistan case corpus

The downloader collects public judgments from the official Sindh High Court archive.
Pakistan Code contains legislation and should be kept as a separate RAG collection.

## Run (PowerShell)

```powershell
uv add pypdf
uv run main.py --limit 1000
```

Raise the total target to collect more:

```powershell
uv run main.py --limit 5000
```

The same command resumes completed downloads, verifies existing PDFs, and skips duplicate
case references and identical PDF contents. Do not run two writers against the same output directory.

## Options

- `--limit 1000`: total unique cases, including completed runs.
- `--output pakistan_cases`: corpus directory.
- `--groups family terrorism criminal civil labour_service tax_customs commercial constitutional`:
  select subject groups; all eight are enabled by default.
- `--document-type judgment`: default. Use `order` to collect orders in a separate output directory.
- `--delay 1`: minimum seconds between requests. Retry-After is respected.
- `--discover-only`: cache search results without downloading PDFs.
- `--seed 42`: reproducible sampling order.
- `--max-mb 40`: maximum size of an individual PDF.

Search categories and family topics are read from the live court form. Family searches also
include topic-filtered constitutional petitions and civil revisions. Categories reflect the
archive's labels, not an independent legal classification. In particular, an anti-terrorism
court appeal does not necessarily concern conduct legally held to constitute terrorism.
The downloader chooses the least-represented available group next; it cannot guarantee
equal numbers or every legal subject. Search results are cached; remove the cache directory
when you intentionally want a fresh archive search.

## Outputs

- `pdfs/*.pdf`: original judgment PDFs.
- `texts/*.json`: extracted text with original page numbers.
- `manifest.jsonl`: one record per unique case with source URL, citation where available,
  parties, case reference, raw decision date, judges, topic/category, hash and file paths.
- `summary.json`: verified case count, subject distribution, OCR flags.
- `errors.jsonl`: failed searches/downloads for investigation.

Empty or very short pages are flagged for OCR. They are not silently treated as usable text.
For RAG, split extracted text into chunks while retaining case ID, citation, court, decision date,
source URL and page number on every chunk. Keep scanned PDFs for a separate OCR pass.
Split evaluation data by case rather than by chunk to avoid overlap with evaluation cases.

This initial source covers Sindh High Court, not all Pakistan courts. Add separately verified
adapters for other courts to improve geographic and judicial coverage. Count distinct cases,
not PDF files, pages or chunks. Source PDFs may contain several connected cases; the manifest
counts the archive's primary case reference. Missing metadata is preserved as empty strings.
The downloader checks robots.txt and explicit AI-input restrictions; review source reuse
conditions before redistribution.
