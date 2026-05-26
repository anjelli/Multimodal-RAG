# Enterprise Multimodal RAG — ESG & Bid-Document Intelligence

A production-oriented Multimodal Retrieval-Augmented Generation (RAG) pipeline for querying complex financial and sustainability (ESG) reports containing text, tables, charts, and images. Designed for **local, private deployment** with full citation traceability and hallucination guardrails.

---

## What's Inside

- Multimodal PDF ingestion: native text, OCR fallback, tables (pdfplumber / Camelot), and embedded images/charts
- ESG-aware semantic and metadata chunking
- Text embeddings (SentenceTransformers) + CLIP image embeddings
- Persistent ChromaDB vector store + SQLite docstore
- Hybrid retrieval: dense vectors + BM25 + reciprocal-rank fusion
- Multi-vector parent-child retrieval for page-traceable citations
- Cross-encoder reranking for precision after broad recall
- Query rewriting and conversation memory
- Citation-grounded local answer generation via **Ollama** (`qwen2.5:7b-instruct`)
- Hallucination guardrails with grounded fallback answers
- Automated evaluation: retrieval accuracy, faithfulness proxy, context precision/recall, answer relevancy, hallucination rate, and latency
- Logging, caching, error handling, and config-driven architecture

---

## Architecture

```text
                ┌────────────────────┐
                │   PDF Documents    │
                └─────────┬──────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
     Text / Table Extraction    Visual Extraction
     (PyMuPDF · pdfplumber ·    (CLIP embeddings ·
      Camelot · OCR fallback)    OCR · image summary)
            │                           │
            └──────────┬────────────────┘
                       │
              ESG-aware chunking
              (parent-child + metadata)
                       │
         ┌─────────────┴──────────────┐
         │                            │
   Text embeddings            CLIP image embeddings
   (SentenceTransformers)
         │                            │
         └─────────────┬──────────────┘
                       │
               ChromaDB + BM25
                       │
           Reciprocal-Rank Fusion
                       │
          Cross-Encoder Reranking
                       │
        Citation-Grounded Prompt
                       │
              Ollama (local LLM)
                       │
        Answer + Citations + Audit Log
```

---

## Tech Stack

| Layer | Components |
|---|---|
| PDF parsing | PyMuPDF, pdfplumber, Camelot, pytesseract |
| Embeddings | SentenceTransformers, CLIP (torchvision) |
| Vector store | ChromaDB (persistent) + SQLite docstore |
| Sparse retrieval | rank-bm25 |
| Reranking | Cross-encoder (sentence-transformers) |
| LLM | Ollama — `qwen2.5:7b-instruct` (fully local) |
| Chunking | LangChain text splitters + custom ESG chunker |
| Caching | diskcache |
| Evaluation | Custom evaluator (retrieval + faithfulness + latency) |

---

## Why Fully Local?

This pipeline runs entirely on-device — no API keys, no cloud calls. This makes it suitable for:

- Confidential bid documents and ESG filings
- Air-gapped or private-network enterprise environments
- Avoiding third-party data retention

---

## Prerequisites

### Windows External Tools

| Tool | Download |
|---|---|
| Ollama | https://ollama.com/download |
| Tesseract OCR | https://github.com/UB-Mannheim/tesseract/wiki |
| Poppler | https://github.com/oschwartz10612/poppler-windows/releases |
| Ghostscript (optional, for Camelot) | https://www.ghostscript.com/releases/gsdnld.html |

> **Python version:** Use Python 3.10–3.12. Python 3.13+ may have compatibility issues with PyTorch, Chroma, and Camelot.

### Pull the LLM

```bash
ollama pull qwen2.5:7b-instruct
```

---

## Installation

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
python -m venv rag_env
source rag_env/bin/activate        # Windows: rag_env\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
jupyter notebook
# Open: multimodal_RAG_fixed.ipynb
```

Place your PDFs in the configured `LOCAL_PDF_DIR` (auto-discovered or set explicitly in the config cell), then run cells top-to-bottom.

---

## Evaluation Results

### Quick-Check Evaluation (3 generic ESG queries)

| Metric | Score |
|---|---|
| Hit Rate | 1.00 |
| Precision@K | 1.00 |
| Recall@K | 0.83 |
| MRR | 1.00 |
| Context Precision | 1.00 |
| Context Recall | 0.83 |
| Answer Relevancy | 0.70 |
| Faithfulness | 0.81 |
| Hallucination Rate | 0.17 |
| Avg Latency | ~51 sec |

> A full 11-case ground-truth evaluation set (covering text, table, image/chart, and cross-modal queries) is included in the notebook and produces the modal-wise breakdowns below.

### Modal-Wise Performance (11-case ground-truth set)

| Modality | Recall | Faithfulness | Notes |
|---|---|---|---|
| Text | 0.89 | 0.86 | Strong semantic reasoning; occasional hallucinations |
| Table | 1.00 | 0.55 | Perfect retrieval; weaker numerical reasoning |
| Image / Chart | 0.61 | 0.85 | Good grounding; lower visual recall |
| Cross-Modal | 0.90 | 0.69 | Good retrieval; multimodal fusion still challenging |

---

## Key Findings

**Strengths**
- Near-perfect retrieval (Hit Rate 1.0, MRR 1.0)
- Fully local and private — no external API calls
- Page-traceable citations via parent-child docstore
- ESG-term recall boosted by hybrid dense + BM25 retrieval
- Conservative generation: answers "I don't know" when evidence is absent

**Limitations**
- Table numerical reasoning degrades faithfulness
- Cross-modal evidence fusion is imperfect
- High latency (~50 sec/query) on CPU; GPU improves this significantly
- Chart recall (0.61) is the weakest modality

---

## Example Queries

```python
"What emissions metrics are disclosed?"
"What renewable energy commitments or progress are reported?"
"What sustainability actions are companies prioritizing?"
"Compare executive sustainability investment trends across years."
"What do the charts indicate about extreme weather concerns?"
```

---

## Project Structure

```text
├── data/
│   ├── pdfs/
│   ├── extracted_images/
│   └── processed_tables/
├── embeddings/
├── vectorstore/
├── evaluation/
├── outputs/
├── multimodal_RAG_fixed.ipynb
├── requirements.txt
└── README.md
```

---

## Production Deployment Considerations

The notebook maps cleanly to microservices:

- **Ingestion service** — async workers for PDF parsing, OCR, table/chart extraction, image summarization
- **Storage** — object store for PDFs, relational DB for metadata/docstore, managed vector DB (Qdrant, Weaviate, or Chroma server)
- **Retrieval API** — query rewrite, hybrid retrieval, metadata filters, reranking, source lineage
- **Answer API** — strict citation prompt, hallucination guard, answer audit trail, red-team filters
- **Security** — tenant isolation, RBAC, encryption at rest, PII/contract-term redaction, prompt injection filtering
- **Observability** — latency by stage, retrieval hit rate, faithfulness drift, CPU/GPU telemetry
- **Evaluation CI** — fixed ground-truth benchmark on every prompt, model, chunking, or retriever change

---

## Future Improvements

- Table-aware reasoning models (e.g. TAPAS, TableLlama)
- Layout-aware parsing with LayoutLM / DocTR for dense forms
- Chart-to-structured-data conversion (capture underlying data tables)
- Stronger local vision-language model for chart understanding
- Qdrant/Weaviate hybrid sparse+dense retrieval for server-side scalability
- ESG ontology extraction (commitments, KPIs, baselines, targets, owners)
- Multi-vendor bid response comparison and compliance matrix generation
- Metadata filters for issuer, year, region, ESG standard, and supplier category
