# Multimodal RAG 

A high-performance Multimodal Retrieval-Augmented Generation (RAG) pipeline for querying complex financial and sustainability reports containing:

* Text
* Tables
* Charts
* Images
* Mixed-layout PDF documents

Built using multimodal document parsing, hybrid retrieval, vector databases, and LLM-based answer synthesis.

---

# Features

* Multimodal PDF parsing
* Text, table, and image extraction
* OCR-enabled chart understanding
* Hybrid semantic retrieval
* Cross-modal context fusion
* Vector embeddings for heterogeneous data
* Context-aware answer generation
* Quantitative evaluation pipeline
* Hallucination and faithfulness analysis

---

# Architecture

```text
                ┌────────────────────┐
                │   PDF Documents    │
                └─────────┬──────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
     Text Extraction             Visual Extraction
            │                           │
     ┌──────┴──────┐          ┌─────────┴─────────┐
     │             │          │                   │
 Text Chunks    Tables     Charts/Images       OCR
     │             │          │                   │
     └──────┬──────┘          └─────────┬─────────┘
            │                           │
            └──────────┬────────────────┘
                       │
              Embedding Generation
                       │
               Vector Database
                       │
                 Hybrid Retrieval
                       │
              Cross-Modal Context
                       │
                    LLM
                       │
               Final Grounded Answer
```

---

# Tech Stack

## Core Frameworks

* Python
* LangChain
* ChromaDB
* Unstructured
* SentenceTransformers

## LLMs

* Groq LLM API
* Ollama
* Open-source embedding models

## Multimodal Components

* OCR-based chart extraction
* Table parsing
* Image captioning
* Semantic chunking

---

# Retrieval Pipeline

The system processes multiple modalities independently and fuses them during retrieval.

## Modalities Supported

| Modality | Processing                           |
| -------- | ------------------------------------ |
| Text     | Semantic chunking + embeddings       |
| Tables   | Structured serialization + embedding |
| Charts   | OCR + caption extraction             |
| Images   | Visual-text semantic encoding        |

---

# Evaluation Metrics

The pipeline was evaluated using:

| Metric             | Description                     |
| ------------------ | ------------------------------- |
| Hit Rate           | Retrieval success               |
| Precision@K        | Relevant retrieved chunks       |
| Recall@K           | Retrieval coverage              |
| MRR                | Ranking quality                 |
| Context Precision  | Context relevance               |
| Context Recall     | Relevant context coverage       |
| Answer Relevancy   | Semantic answer quality         |
| Faithfulness       | Grounding to retrieved evidence |
| Hallucination Rate | Unsupported generations         |
| Latency            | End-to-end response time        |

---

# Results

## Overall Performance

| Metric             | Score    |
| ------------------ | -------- |
| Hit Rate           | 1.00     |
| Precision@K        | 0.95     |
| Recall@K           | 0.85     |
| MRR                | 1.00     |
| Answer Relevancy   | 0.73     |
| Faithfulness       | 0.74     |
| Hallucination Rate | 0.045    |
| Avg Latency        | 33.3 sec |

---

# Modal-Wise Performance

## Text Retrieval

| Metric             | Score |
| ------------------ | ----- |
| Recall             | 0.89  |
| Faithfulness       | 0.86  |
| Hallucination Rate | 0.17  |

Strong semantic reasoning performance with occasional hallucinations.

---

## Table Retrieval

| Metric           | Score |
| ---------------- | ----- |
| Recall           | 1.00  |
| Faithfulness     | 0.55  |
| Answer Relevancy | 0.61  |

Perfect retrieval but weaker numerical reasoning and table interpretation.

---

## Image / Chart Retrieval

| Metric       | Score |
| ------------ | ----- |
| Recall       | 0.61  |
| Faithfulness | 0.85  |

Good grounding quality with weaker retrieval coverage for visual elements.

---

## Cross-Modal Queries

| Metric       | Score |
| ------------ | ----- |
| Recall       | 0.90  |
| Faithfulness | 0.69  |

Cross-modal retrieval performs well, while multimodal reasoning remains challenging.

---

# Key Findings

## Strengths

* Near-perfect retrieval quality
* Excellent ranking performance
* Strong multimodal indexing
* Effective chart grounding
* Low hallucination rate

## Limitations

* Table reasoning remains difficult
* Cross-modal evidence fusion is imperfect
* High latency for real-time systems
* Numerical synthesis can degrade faithfulness

---

# Example Queries

```python
"What sustainability actions are companies prioritizing?"
```

```python
"What do the charts indicate about extreme weather concerns?"
```

```python
"Compare executive sustainability investment trends across years."
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
```

## Create Environment

```bash
python -m venv rag_env
source rag_env/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## Start Notebook

```bash
jupyter notebook
```

Open:

```text
multimodal_RAG.ipynb
```

---

# Project Structure

```text
├── data/
│   ├── pdfs/
│   ├── extracted_images/
│   └── processed_tables/
│
├── embeddings/
├── vectorstore/
├── evaluation/
├── outputs/
│
├── multimodal_RAG.ipynb
├── requirements.txt
└── README.md
```

---

# Possible Future Improvements

* Table-aware reasoning models
* Chart-to-structured-data conversion
* Faster retrieval pipelines
* Modality-specific reranking
* Answer verification layers
* Agentic multimodal reasoning

---

# Research Insights

The evaluation demonstrates that retrieval is no longer the primary bottleneck in multimodal RAG systems.

The main challenge lies in:

* faithful multimodal reasoning,
* numerical grounding,
* and structured evidence synthesis.

This project highlights the gap between:

* retrieving the correct evidence,
* and generating fully grounded answers from heterogeneous modalities.

---



