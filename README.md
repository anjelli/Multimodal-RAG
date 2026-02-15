# Multimodal RAG Project

A PDF-focused **multimodal Retrieval-Augmented Generation (RAG)** prototype.

It ingests PDFs, extracts text/tables/images, indexes summaries in Chroma, stores original payloads in a local docstore, retrieves relevant context, and optionally sends context to an LLM for final answers.

## Python and environment

- **Recommended Python**: `3.10` or `3.11`
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Required environment variables:
  - `OPENAI_API_KEY` (required only when you want OpenAI embeddings/LLM calls)
- Optional directory overrides:
  - `MMRAG_DATA_DIR`, `MMRAG_EXTRACTED_DIR`, `MMRAG_PROCESSED_DIR`, `MMRAG_CHROMA_DIR`
- `--openai-key` can be provided to `src.main` and is applied for both embeddings and LLM answer generation.

## Architecture (compact)

```mermaid
flowchart LR
    A[PDF files in data/] --> B[IngestionPipeline]
    B --> C[Text / Table / Image elements]
    C --> D[Summaries + metadata]
    D --> E[EmbeddingClient\n(OpenAI | sbert | hash)]
    E --> F[Chroma collection]
    C --> G[Shelve docstore\nprocessed_data/docstore.db]
    H[User question] --> I[RetrieverPipeline.query]
    I --> F
    I --> G
    I --> J[Retrieved summaries + full content]
    J --> K[LLMOutputGenerator + ModelClient]
    K --> L[Final answer + sources]
```

## Runbook

### 1) Ingest a PDF into Chroma

```bash
python -m src.main \
  --source data/<your_report>.pdf \
  --persist-dir chroma_db \
  --extracted-dir extracted_data
```

If Poppler/image tooling is unavailable, skip image extraction:

```bash
MMRAG_SKIP_IMAGES=1 python -m src.main --source data/<your_report>.pdf
# or
python -m src.main --source data/<your_report>.pdf --no-images
```

### 2) Query indexed content (+ generated answer)

```bash
python -m src.main --question "What are the net-zero goals?" --top-k 4 --openai-key "$OPENAI_API_KEY"
```

### 3) End-to-end smoke run

```bash
python -m src.integration.run_integration
```

## Project structure

- `src/ingestion/`: PDF parsing and normalization.
- `src/retriever/`: embeddings, Chroma client, retrieval logic.
- `src/llm_output/`: prompt construction and model adapter.
- `src/integration/run_integration.py`: end-to-end integration runner.
- `src/main.py`: CLI entrypoint for ingestion and querying.

## Notes

- Ingestion now automatically falls back to `pdfplumber`/`PyMuPDF` extraction when `unstructured` fails.
- Retrieved items include both vector-match summaries and full stored content payloads.
- Local runtime artifacts (`__pycache__`, logs, generated db folders) are gitignored.
