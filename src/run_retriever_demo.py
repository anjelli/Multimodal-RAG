import json
from pathlib import Path
import logging

from src.config import Config
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.pipeline import RetrieverPipeline
import pandas as pd


def load_summary(summary_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document_lists(summary):
    text_cats = ["Title", "NarrativeText", "Text", "ListItem"]
    texts = []
    text_summaries = []
    for c in text_cats:
        for t in summary.get(c, []):
            s = str(t).strip()
            if not s:
                continue
            texts.append(s)
            text_summaries.append(s[:400])

    tables = []
    table_summaries = []
    for t in summary.get("Table", []):
        if isinstance(t, dict) and t.get("csv_path"):
            try:
                df = pd.read_csv(t["csv_path"])
                preview = df.head(5).to_csv(index=False)
                tables.append({"csv_path": t["csv_path"], "shape": t.get("shape")})
                table_summaries.append(preview[:800])
            except Exception:
                tables.append(t)
                table_summaries.append(str(t)[:400])
        else:
            tables.append(t)
            table_summaries.append(str(t)[:400])

    images = []
    image_summaries = []
    for im in summary.get("Image", []):
        if isinstance(im, dict) and im.get("path"):
            images.append({"path": im.get("path")})
            image_summaries.append(f"Image at {im.get('path')}")
        else:
            images.append(im)
            image_summaries.append(str(im)[:200])

    return text_summaries, texts, table_summaries, tables, image_summaries, images


def main():
    logging.basicConfig(level=logging.INFO)
    processed = Path("processed_data")
    # pick the first summary json
    summaries = list(processed.glob("*_summary.json"))
    if not summaries:
        raise SystemExit("No summary JSON found in processed_data/")
    summary_path = summaries[0]
    logging.info("Using summary: %s", summary_path)

    summary = load_summary(summary_path)
    text_summaries, texts, table_summaries, tables, image_summaries, images = build_document_lists(summary)

    emb = EmbeddingClient()
    try:
        client, collection = get_chroma_collection(
            collection_name="mmrag_demo",
            persist_dir=str(Config.CHROMA_PERSIST_DIR),
        )
        vectorstore = collection
    except Exception as e:
        raise SystemExit("Failed to create Chroma collection: %s" % e)

    pipeline = RetrieverPipeline(embedding_model=emb, vectorstore=vectorstore)

    def chunked_add(summaries, contents, chunk_size=1000):
        if not summaries:
            return
        n = len(summaries)
        logging.info("Adding %s documents in chunks of %s", n, chunk_size)
        for i in range(0, n, chunk_size):
            s_chunk = summaries[i : i + chunk_size]
            c_chunk = contents[i : i + chunk_size]
            pipeline.add_documents(s_chunk, c_chunk)

    logging.info("Adding text documents: %s", len(texts))
    chunked_add(text_summaries, texts, chunk_size=1000)
    logging.info("Adding table documents: %s", len(tables))
    chunked_add(table_summaries, tables, chunk_size=200)
    logging.info("Adding image documents: %s", len(images))
    chunked_add(image_summaries, images, chunk_size=200)

    # report collection count
    try:
        cnt = collection.count()
    except Exception:
        # fallback: try to get ids
        try:
            cnt = len(collection.get(ids=True)["ids"])
        except Exception:
            cnt = "unknown"
    logging.info("Chroma collection document count: %s", cnt)


if __name__ == "__main__":
    main()
