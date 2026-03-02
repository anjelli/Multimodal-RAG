import logging
import os
import uuid
from pathlib import Path

from src.config import Config
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunking import chunk_text_items
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.pipeline import RetrieverPipeline
from src.llm_output.pipeline import LLMOutputGenerator
from src.llm_output.adapter import ModelClient


def _sentence_boundary_summary(text: str, max_chars: int = 400) -> str:
    """Return a summary that respects sentence boundaries up to max_chars.
    Falls back to word-boundary truncation, then hard truncation."""
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    # Try to end on a sentence boundary (. ! ?)
    boundary = max(s.rfind(". ", 0, max_chars), s.rfind("! ", 0, max_chars), s.rfind("? ", 0, max_chars))
    if boundary > max_chars // 2:
        return s[: boundary + 1]
    # Fall back to word boundary
    word_boundary = s.rfind(" ", 0, max_chars)
    if word_boundary > max_chars // 2:
        return s[:word_boundary] + "..."
    return s[:max_chars] + "..."


# Keep original name for backward compatibility
def simple_text_summary(text: str, max_chars: int = 400) -> str:
    return _sentence_boundary_summary(text, max_chars)


def _normalize_text_items(data, source_pdf: str):
    text_items = []
    for key in ("NarrativeText", "ListItem", "Text"):
        for item in data.get(key, []):
            if isinstance(item, dict):
                text_items.append(
                    {
                        "text": item.get("text", ""),
                        "metadata": {
                            "source": source_pdf,
                            "source_document": Path(source_pdf).name,
                            "content_type": key.lower(),
                            "type": key.lower(),
                            "extraction_method": "unstructured",
                            **(item.get("metadata") or {}),
                        },
                    }
                )
            else:
                text_items.append(
                    {
                        "text": str(item),
                        "metadata": {
                            "source": source_pdf,
                            "source_document": Path(source_pdf).name,
                            "content_type": key.lower(),
                            "type": key.lower(),
                            "extraction_method": "unstructured",
                        },
                    }
                )
    return text_items


def run_once(source_pdf: str = None):
    logging.basicConfig(level=logging.INFO)
    Config.ensure_dirs()

    if source_pdf is None:
        p = Config.DATA_DIR
        pdfs = list(p.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF available in data dir for integration test")
        source_pdf = str(pdfs[0])

    logging.info("Running ingestion for %s", source_pdf)
    skip_images = os.environ.get("MMRAG_SKIP_IMAGES", "0") in ("1", "true", "True")
    ing = IngestionPipeline(source_pdf, extracted_dir=str(Config.EXTRACTED_DIR), extract_images=not skip_images)
    ing.load_data()
    ing.process_data()
    data = ing.get_processed_data()

    # Prepare summaries and full content
    text_items = _normalize_text_items(data, source_pdf)
    chunked_texts = chunk_text_items(text_items, chunk_size=1000, overlap=100)

    # Issue 1: use full chunk text for embedding (not 400-char truncations)
    # Issue 5: keep sentence-boundary summaries for display/storage only
    text_summaries = [_sentence_boundary_summary(t["text"], max_chars=400) for t in chunked_texts]

    table_items = data.get("Table", [])
    table_summaries = []
    table_contents = []
    for t in table_items:
        if isinstance(t, dict) and t.get("csv_path"):
            shape = t.get("shape")
            # Issue 5/13: include shape and column info in table summary for better retrieval
            summary = f"Table {Path(t['csv_path']).name}"
            if shape:
                summary += f" ({shape[0]} rows x {shape[1]} cols)"
            table_summaries.append(summary)
            table_contents.append({
                **t,
                "type": "table",
                "content_type": "table",
                "source": source_pdf,
                "source_document": Path(source_pdf).name,
                "extraction_method": "pdfplumber",
            })
        else:
            raw = t.get("raw") if isinstance(t, dict) else str(t)
            table_summaries.append(_sentence_boundary_summary(str(raw), 300))
            table_contents.append({
                "raw": raw,
                "type": "table",
                "content_type": "table",
                "source": source_pdf,
                "source_document": Path(source_pdf).name,
                "extraction_method": "unstructured",
            })

    images = data.get("Image", [])
    image_summaries = []
    image_contents = []
    for im in images:
        if isinstance(im, dict) and im.get("path"):
            img_path = Path(im["path"])
            # Issue 6: use file name + extension as a more informative summary
            summary = f"Image {img_path.name} (type: {img_path.suffix.lstrip('.') or 'unknown'})"
            image_summaries.append(summary)
            image_contents.append({
                **im,
                "type": "image",
                "content_type": "image",
                "source": source_pdf,
                "source_document": Path(source_pdf).name,
                "extraction_method": "pdfplumber",
            })
        else:
            image_summaries.append(_sentence_boundary_summary(str(im), 200))
            image_contents.append({
                "raw": str(im),
                "type": "image",
                "content_type": "image",
                "source": source_pdf,
                "source_document": Path(source_pdf).name,
                "extraction_method": "unstructured",
            })

    # Issue 1: embed full text content (not truncated summaries) to preserve semantic meaning
    text_embed_docs = [t["text"] for t in chunked_texts]
    # For tables: embed the summary (no full text available without reading CSV)
    # For images: embed the summary (no pixel-level features without vision model)
    all_embed_docs = text_embed_docs + table_summaries + image_summaries
    all_summaries = text_summaries + table_summaries + image_summaries

    if not all_embed_docs:
        raise SystemExit("No documents found to index during integration run")

    # create embeddings
    embedder = EmbeddingClient()
    logging.info("Embedding %d documents using backend %s", len(all_embed_docs), embedder.backend)
    embeddings = embedder.embed_texts(all_embed_docs)

    client, collection = get_chroma_collection(
        collection_name="mmrag_demo",
        persist_dir=str(Config.CHROMA_PERSIST_DIR),
    )
    retriever = RetrieverPipeline(embedding_model=embedder, vectorstore=collection)
    text_contents = [
        {
            "text": item["text"],
            "metadata": item["metadata"],
            "type": item["metadata"].get("type", "text"),
            "content_type": item["metadata"].get("content_type", "text"),
            "source": item["metadata"].get("source", source_pdf),
            "source_document": item["metadata"].get("source_document", Path(source_pdf).name),
            "extraction_method": item["metadata"].get("extraction_method", "unstructured"),
        }
        for item in chunked_texts
    ]
    # Issue 1: pass full-content embeddings but store summaries as the indexed documents
    retriever.add_documents(
        all_summaries,
        text_contents + table_contents + image_contents,
        embeddings=embeddings,
    )

    try:
        client.persist()
    except Exception:
        pass

    # smoke check: ensure collection has at least one id
    res = collection.get(include=["ids"]) if hasattr(collection, "get") else None
    # best-effort test: if collection.count is available
    count = None
    try:
        count = collection.count()
    except Exception:
        try:
            count = len(collection.get(include=["ids"]) ["ids"])
        except Exception:
            pass

    logging.info("Chroma collection count (best-effort): %s", count)

    if not count or count == 0:
        raise SystemExit("Integration failed: no documents persisted in Chroma collection")

    logging.info("Integration run completed successfully. Persisted %d docs", count)

    # example question + LLM invocation
    model = ModelClient()
    sample_q = "Give a short summary of the indexed documents and list sources."
    data_dict = {"context": {"texts": [t["text"] for t in chunked_texts], "images": image_summaries}, "question": sample_q}
    messages = LLMOutputGenerator.img_prompt_func(data_dict)
    answer = model.invoke(messages)
    logging.info("LLM answer: %s", answer)
    return True


if __name__ == "__main__":
    run_once()
