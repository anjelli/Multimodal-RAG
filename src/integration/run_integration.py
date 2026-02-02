import logging
import os
import uuid
from pathlib import Path

from src.config import Config
from src.ingestion.pipeline import IngestionPipeline
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.llm_output.pipeline import LLMOutputGenerator
from src.llm_output.adapter import ModelClient


def simple_text_summary(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    s = str(text).strip()
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


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
    texts = data.get("NarrativeText", []) + data.get("ListItem", [])
    text_summaries = [simple_text_summary(t, max_chars=400) for t in texts]

    table_items = data.get("Table", [])
    table_summaries = []
    for t in table_items:
        if isinstance(t, dict) and t.get("csv_path"):
            table_summaries.append(f"Table {Path(t['csv_path']).name} shape={t.get('shape')}")
        else:
            table_summaries.append(simple_text_summary(t.get("raw") if isinstance(t, dict) else str(t), 300))

    images = data.get("Image", [])
    image_summaries = []
    image_refs = []
    for im in images:
        if isinstance(im, dict) and im.get("path"):
            image_refs.append(im["path"])
            image_summaries.append(f"Image file {Path(im['path']).name}")
        else:
            image_summaries.append(simple_text_summary(str(im), 200))

    # create embeddings
    embedder = EmbeddingClient()
    docs = []
    metadatas = []
    ids = []

    # combine all content types as documents
    for s, orig in zip(text_summaries, texts):
        ids.append(str(uuid.uuid4()))
        docs.append(s)
        metadatas.append({"source": source_pdf, "type": "text"})

    for s, orig in zip(table_summaries, table_items):
        ids.append(str(uuid.uuid4()))
        docs.append(s)
        metadatas.append({"source": source_pdf, "type": "table", **(orig if isinstance(orig, dict) else {})})

    for s, orig in zip(image_summaries, image_refs if image_refs else images):
        ids.append(str(uuid.uuid4()))
        docs.append(s)
        metadatas.append({"source": source_pdf, "type": "image", "ref": orig})

    if not docs:
        raise SystemExit("No documents found to index during integration run")

    logging.info("Embedding %d documents using backend %s", len(docs), embedder.backend)
    embeddings = embedder.embed_texts(docs)

    client, collection = get_chroma_collection()

    # add into chroma collection
    # chroma collection.add expects ids, documents, metadatas, embeddings
    try:
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    except Exception:
        # try with minimal fields
        collection.add(ids=ids, documents=docs, metadatas=metadatas)

    # persist client
    try:
        client.persist()
    except Exception:
        pass

    # smoke check: ensure collection has at least one id
    res = collection.get(ids=ids[:10], include=["ids"]) if hasattr(collection, "get") else None
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
    data_dict = {"context": {"texts": texts, "images": image_summaries}, "question": sample_q}
    messages = LLMOutputGenerator.img_prompt_func(data_dict)
    answer = model.invoke(messages)
    logging.info("LLM answer: %s", answer)
    return True


if __name__ == "__main__":
    run_once()
