import logging
import os
from pathlib import Path

from src.config import Config
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunking import chunk_text_items
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.pipeline import RetrieverPipeline
from src.llm_output.pipeline import LLMOutputGenerator
from src.llm_output.adapter import ModelClient


def simple_text_summary(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    s = str(text).strip()
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


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
                            "type": key.lower(),
                            **(item.get("metadata") or {}),
                        },
                    }
                )
            else:
                text_items.append(
                    {"text": str(item), "metadata": {"source": source_pdf, "type": key.lower()}}
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

    # Prepare simple summaries
    text_items = _normalize_text_items(data, source_pdf)
    chunked_texts = chunk_text_items(text_items, chunk_size=1000, overlap=100)
    text_summaries = [simple_text_summary(t["text"], max_chars=400) for t in chunked_texts]

    table_items = data.get("Table", [])
    table_summaries = []
    table_contents = []
    for t in table_items:
        if isinstance(t, dict) and t.get("csv_path"):
            table_summaries.append(f"Table {Path(t['csv_path']).name} shape={t.get('shape')}")
            table_contents.append({**t, "type": "table", "source": source_pdf})
        else:
            table_summaries.append(simple_text_summary(t.get("raw") if isinstance(t, dict) else str(t), 300))
            table_contents.append({"raw": t.get("raw") if isinstance(t, dict) else str(t), "type": "table", "source": source_pdf})

    images = data.get("Image", [])
    image_summaries = []
    image_contents = []
    for im in images:
        if isinstance(im, dict) and im.get("path"):
            image_summaries.append(f"Image file {Path(im['path']).name}")
            image_contents.append({**im, "type": "image", "source": source_pdf})
        else:
            image_summaries.append(simple_text_summary(str(im), 200))
            image_contents.append({"raw": str(im), "type": "image", "source": source_pdf})

    # create embeddings
    embedder = EmbeddingClient()
    docs = []

    # combine all content types as documents
    for s, orig in zip(text_summaries, chunked_texts):
        docs.append(s)

    for s, orig in zip(table_summaries, table_contents):
        docs.append(s)

    for s, orig in zip(image_summaries, image_contents):
        docs.append(s)

    if not docs:
        raise SystemExit("No documents found to index during integration run")

    logging.info("Embedding %d documents using backend %s", len(docs), embedder.backend)
    embeddings = embedder.embed_texts(docs)

    client, collection = get_chroma_collection()
    retriever = RetrieverPipeline(embedding_model=embedder, vectorstore=collection)
    text_contents = [
        {"text": item["text"], "metadata": item["metadata"], "type": item["metadata"].get("type", "text")}
        for item in chunked_texts
    ]
    retriever.add_documents(
        docs,
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
