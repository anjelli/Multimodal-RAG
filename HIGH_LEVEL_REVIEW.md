# High-level repository review

## What this repository is

This is a **multimodal Retrieval-Augmented Generation (RAG)** prototype focused on PDF documents. It ingests PDFs, extracts text/tables/images, summarizes/indexes them into Chroma, retrieves relevant chunks for a query, and optionally asks an LLM to produce an answer from retrieved context.

At a high level, the architecture is split into:

- **Ingestion** (`src/ingestion/*`): PDF parsing and normalization into text/table/image buckets.
- **Retrieval** (`src/retriever/*`): embedding generation + Chroma persistence + query and ranking.
- **LLM output** (`src/llm_output/*`): prompt assembly and model invocation wrappers.
- **Integration scripts / CLIs** (`src/main.py`, `src/integration/run_integration.py`, helpers): end-to-end runs and inspection.

---

## What it is doing end-to-end

1. **Reads a PDF** from CLI `--source` or defaults to the first `data/*.pdf` file.
2. **Extracts structured elements** using `unstructured.partition.pdf.partition_pdf` (titles/text/list items/tables/images).
3. **Normalizes outputs** into `processed_data` categories and writes a summary JSON.
4. **Builds short summaries** for each content item (text/table/image), keeping raw/metadata in a docstore.
5. **Embeds summaries** using OpenAI when available, otherwise sentence-transformers, otherwise a hash fallback.
6. **Stores vectors in Chroma** and raw payloads in a local shelve docstore (`processed_data/docstore.db`).
7. **Retrieves top-k results** with vector similarity and returns summary + metadata + original stored content.
8. **Optionally formats context for LLM calls** and asks a chat model for a natural-language answer.

---

## Core components and responsibilities

### 1) Configuration and runtime defaults

- Centralized env-driven config and directory creation are in `src/config.py`.
- Key env vars include `OPENAI_API_KEY` and directory overrides (`MMRAG_*_DIR`).

### 2) Ingestion pipeline

- `IngestionPipeline.load_data()` uses Unstructured PDF partitioning with high-resolution strategy and image/table extraction output to an extracted-data directory.
- `process_data()` categorizes elements by type name and tries to serialize table structures to CSV.
- `get_processed_data()` writes a `*_summary.json` snapshot under `processed_data/`.
- There is also a fallback helper module (`pdfplumber_extract.py`) for text/tables/images via pdfplumber/PyMuPDF, but the current `load_data()` path does not auto-switch to it.

### 3) Retriever pipeline

- `EmbeddingClient` supports three backends in priority order:
  1. OpenAI embedding API,
  2. sentence-transformers,
  3. deterministic hash embedding fallback.
- `get_chroma_collection()` tries multiple Chroma client constructor styles for compatibility.
- `RetrieverPipeline.add_documents()` stores raw content in shelve by generated UUID and adds summaries (+ optional embeddings) to Chroma with metadata.
- `RetrieverPipeline.retrieve()` queries Chroma by query embedding and reconstructs result payloads using shelve-backed content lookup.

### 4) LLM output layer

- `LLMOutputGenerator.img_prompt_func()` creates chat messages with bounded textual context and lightweight image references.
- `ModelClient` tries LangChain `ChatOpenAI`, then OpenAI client, includes rough token counting and truncation to avoid context-window overflow.

### 5) Entry points and utilities

- `src/main.py` is the main CLI for ingestion and querying.
- `src/integration/run_integration.py` runs an end-to-end smoke flow (ingest → embed → index → basic queryable check → sample LLM call).
- `src/run_retriever_demo.py` indexes from a saved summary JSON.
- `src/run_queries_and_inspect.py` runs canned queries and prints/inspects summary content.

---

## Current maturity: what looks good

- Clear separation of concerns between ingestion/retrieval/model-output layers.
- Multiple defensive fallbacks for embeddings and Chroma client construction.
- Practical persistence design: vectors in Chroma + originals in local docstore.
- Helpful operational ergonomics: env vars, CLI flags, image-skip mode, and integration script.

---

## Notable gaps / risks to be aware of

1. **No tests currently present** (unit/integration assertions beyond runtime script behavior).
2. **Dependency versions are very mixed/old** (e.g., `langchain==0.0.1`, `unstructured==0.4.0`) and may conflict with modern code paths.
3. **Ingestion fallback not automatically used** in `load_data()` if `unstructured` path fails.
4. **Cross-platform helper script is Windows-specific** (`run_ingest_with_poppler.py` hardcodes a local Windows path).
5. **Repository includes tracked `__pycache__` artifacts**, which generally should be gitignored.

---

## Suggested next steps (priority order)

1. Add a **minimal test suite** for chunking, embedding backend selection, retriever add/query, and prompt builder formatting.
2. Normalize and upgrade dependency set (or pin to known-compatible stack) and document Python version.
3. Add robust fallback behavior in `IngestionPipeline.load_data()` to call the pdfplumber/PyMuPDF path automatically when Unstructured fails.
4. Add a compact architecture diagram / runbook in README for faster onboarding.
5. Clean repository hygiene (`.gitignore`, remove committed cache/log artifacts where appropriate).

