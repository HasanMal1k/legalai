# Pakistan Law Assistant

Question answering over Pakistani legislation. Ask in English, Urdu or Roman Urdu and get an answer in which every
statement is cited to a section of a statute, drawn only from laws there is evidence are still in force.

The repository holds the whole pipeline behind it:

- scrapers for six official government portals;
- OCR and text repair on a local GPU;
- metadata extracted by a local LLM and checked by rules;
- a knowledge graph of repeals and amendments that decides which laws are in force;
- section-level chunking, vectors, and hybrid search with a reranker;
- a retrieval evaluation set;
- a small web app.

On the 134-question evaluation set, the right section is in the top 10 results for 90% of questions and ranked
first for 60% (keyword search alone: 18% and 9%).

> Legal information, not legal advice. Machine translations of Urdu laws are reading aids, not authoritative texts.

## How it works

```
6 government portals ─► crawl ─► fetch PDFs ─► text (OCR / Urdu translation / repair where needed)
   ─► metadata: 2 passes of a local LLM (qwen3.5:9b on Ollama, JSON-schema constrained), merged and cross-checked
   ─► deduplication ─► knowledge graph of repeals/amendments + rules ─► confidence tier per law
   ─► one chunk per section ─► Kanon 2 vectors ─► Qdrant ("answers" and "history" collections)

question ─► language detected, translated / split into issues (DeepSeek)
   ─► cited sections + vector search (province filter) + BM25 extras ─► copies folded ─► Kanon 2 Reranker
   ─► answer written by DeepSeek from numbered sources only, every statement cited [n]
```

Only laws with evidence of their status are served. Each law gets a tier:

| Tier | Laws | Use |
|---|---|---|
| current | 3,227 | answers |
| unconfirmed | 707 | answers, labelled "status not confirmed" (Sindh laws) |
| historical | 1,358 | shown only as "earlier law" |
| quarantine | 3,910 | never served |

## What is in the repository

| Path | What |
|---|---|
| `app/` | The web app: FastAPI server, search, answer writing, Qdrant sync, static page |
| `crawl_*.py`, `fetch_*.py` | Catalogue and download from the official portals (one request at a time, with delays) |
| `ocr_scanned.py`, `translate_urdu.py`, `repair_text.py`, `repair_pages.py` | OCR (docTR), Urdu transcription and translation with local vision models, text repair |
| `generate_metadata.py`, `enrich_metadata.py` | Metadata passes 1 and 2 with a local LLM through Ollama |
| `finalize_metadata.py`, `dedupe.py`, `kg.py`, `confidence.py` | Merge and check the metadata, deduplicate, build the knowledge graph, assign tiers |
| `crawl_status.py`, `register.py` | The government's own status lists, and a register of laws in force |
| `chunk_statutes.py`, `chunk_model.py` | Section-level chunking (rules; the local model only points at section starts in poor scans) |
| `eval/` | 134 evaluation questions, the embedding and search runs, scoring |
| `test_*.py`, `audit_*.py` | Tests for the repeal rules, knowledge graph and register; whole-corpus audits |
| `data/<source>/metadata_final.jsonl` | One record per document: metadata, status with evidence, deduplication, tier |
| `data/<source>/chunks.jsonl.gz` | The chunks (gzipped to fit GitHub; unpack with `pack_data.py`) |
| `data/embeddings/kanon-2-embedder/` | Kanon 2 vectors of every served chunk (float16 shards, each with chunk ids and text hashes) |
| `data/kg/`, `data/register/` | Knowledge graph and in-force register |

Not in the repository:

- the source PDFs (about 37 GB);
- the full extracted text (`statutes.jsonl`, rebuilt by the scrapers);
- the Qdrant database (rebuilt from the vectors, see below);
- the app's caches and question log.

`<source>` is `drs`, `punjab`, `sindh`, `kp` or `balochistan`. The Pakistan Code's files are in `data/` itself.

## Run the app (Windows, PowerShell)

1. Install (Python 3.12):

   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

2. Unpack the chunks:

   ```powershell
   .venv\Scripts\python pack_data.py unpack
   ```

3. Set the API keys. Isaacus is used for query embeddings and the reranker, DeepSeek for answers. Keys are read
   from the environment and never stored in files.

   ```powershell
   setx ISAACUS_API_KEY "..."
   setx DEEPSEEK_API_KEY "..."
   ```

4. Optional: set up Qdrant. Put the Qdrant 1.19 Windows binary at `tools\qdrant\qdrant.exe` (`config.yaml` is
   already there). Start the app once (it starts Qdrant), then load the vectors in a second terminal and restart
   the app:

   ```powershell
   .venv\Scripts\python -m app.vectordb
   ```

   Without Qdrant, the app searches the same vectors in memory.

5. Start the app and open http://127.0.0.1:8000:

   ```powershell
   powershell -ExecutionPolicy Bypass -File run_app.ps1
   ```

   The first start builds the BM25 index and chunk cache in `data\app\`, which takes about a minute.

## Rebuild the data

Every stage is resumable. The model stages need [Ollama](https://ollama.com) with `qwen3.5:9b`
(and `gemma4:12b-it-qat` for Urdu) and a GPU. OCR needs `torch` and `python-doctr`, listed at the end of
`requirements.txt`.

```powershell
python crawl_catalog.py; python crawl_sources.py --source all          # catalogues
python fetch_statutes.py; python fetch_sources.py --source all         # PDFs -> text
python ocr_scanned.py --source <name>; python translate_urdu.py        # scans, Urdu laws
python generate_metadata.py --source all; python enrich_metadata.py    # local LLM, two passes
python finalize_metadata.py        # merge, dedupe.py, kg.py, confidence.py -> metadata_final.jsonl
python test_repeal_check.py; python test_kg.py; python audit_kg.py     # must pass / be read
python chunk_statutes.py           # (chunk_model.py for laws it reports, then chunk_statutes.py again)
python eval/resolve.py             # re-map the evaluation answers to the new chunk ids
python eval/kanon_eval.py --embed --yes    # embeds only chunks whose text changed
python -m app.vectordb             # sync Qdrant
python pack_data.py pack           # before committing the chunks
```

Status register: `python crawl_status.py drs pakistancode drs-details`, then `python register.py` and
`python test_register.py`.

Retrieval evaluation: `python eval/app_eval.py`, then `python eval/score.py eval/runs/app_pipeline.jsonl`.

## Sources

The scrapers use these government portals only:

- the Pakistan Code (pakistancode.gov.pk);
- the Ministry of Law's Document Retrieval System (drs.molaw.gov.pk);
- the Punjab, Sindh, Khyber Pakhtunkhwa and Balochistan Codes.

They make one request at a time per site, with a pause between requests. Sites whose terms forbid scraping
(e.g. pakistanlawsite.com) are not used.

## Case-law downloader (not used by the app)

`main.py` downloads public judgments from the Sindh High Court archive into `pakistan_cases/` (not committed):
`python main.py --limit 1000`. Options include `--groups`, `--document-type order`, `--delay` and
`--discover-only`; see `python main.py --help`. It is resumable and checks robots.txt. Review the reuse conditions
before redistributing judgments.
