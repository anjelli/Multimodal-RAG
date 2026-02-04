import argparse
import logging
import os
from pathlib import Path

from src.config import Config
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.pipeline import RetrieverPipeline


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal RAG ingestion and retrieval")
    p.add_argument("--source", help="Path to source PDF or directory", default=None)
    p.add_argument("--persist-dir", help="Chroma persist directory", default=str(Config.CHROMA_PERSIST_DIR))
    p.add_argument("--extracted-dir", help="extracted images dir", default=str(Config.EXTRACTED_DIR))
    p.add_argument("--openai-key", help="OpenAI API key (ENV fallback)", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--no-images", help="Skip extracting images (don't require poppler)", action="store_true")
    p.add_argument("--question", help="Query the vectorstore instead of ingesting", default=None)
    p.add_argument("--top-k", help="Number of results to return for queries", type=int, default=4)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    Config.ensure_dirs()

    if args.question:
        logging.info("Running query against vectorstore")
        embedder = EmbeddingClient()
        client, collection = get_chroma_collection(
            collection_name="mmrag_demo",
            persist_dir=args.persist_dir,
        )
        retriever = RetrieverPipeline(embedding_model=embedder, vectorstore=collection)
        results = retriever.retrieve(args.question, k=args.top_k)
        for idx, result in enumerate(results, start=1):
            print(f"[{idx}] score={result.get('distance')}")
            print(f"summary: {result.get('summary')}")
            print(f"metadata: {result.get('metadata')}")
            print(f"content: {result.get('content')}\n")
        return

    source = args.source
    if source is None:
        # pick first pdf in data dir
        p = Config.DATA_DIR
        pdfs = list(p.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF found in data directory; pass --source or add PDFs to data/")
        source = str(pdfs[0])

    logging.info("Source PDF: %s", source)

    # Respect env var MMRAG_SKIP_IMAGES to forcibly disable image extraction
    skip_images_env = os.environ.get("MMRAG_SKIP_IMAGES")
    if skip_images_env is not None and str(skip_images_env).lower() in ("1", "true", "yes"):
        use_images = False
        logging.info("MMRAG_SKIP_IMAGES is set; skipping image extraction")
    else:
        use_images = not args.no_images

    # Ingestion
    from src.ingestion.pipeline import IngestionPipeline

    ingestion = IngestionPipeline(source, extracted_dir=str(args.extracted_dir), extract_images=use_images)
    ingestion.load_data()
    ingestion.process_data()
    data = ingestion.get_processed_data()

    text_cats = ["Title", "NarrativeText", "Text", "ListItem"]
    text_summaries = []
    text_contents = []
    for cat in text_cats:
        for item in data.get(cat, []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            text = str(text).strip()
            if not text:
                continue
            text_summaries.append(text[:400])
            text_contents.append(text)

    table_summaries = []
    table_contents = []
    for table in data.get("Table", []):
        if isinstance(table, dict) and table.get("csv_path"):
            summary = f"Table {Path(table['csv_path']).name} shape={table.get('shape')}"
            table_summaries.append(summary)
            table_contents.append(table)
        else:
            table_text = str(table.get("raw") if isinstance(table, dict) else table)
            table_summaries.append(table_text[:400])
            table_contents.append(table)

    image_summaries = []
    image_contents = []
    for image in data.get("Image", []):
        if isinstance(image, dict) and image.get("path"):
            summary = f"Image at {image.get('path')}"
            image_summaries.append(summary)
            image_contents.append(image)
        else:
            image_text = str(image)
            image_summaries.append(image_text[:200])
            image_contents.append(image)

    if not (text_summaries or table_summaries or image_summaries):
        raise SystemExit("No documents found to index after ingestion")

    embedder = EmbeddingClient()
    client, collection = get_chroma_collection(
        collection_name="mmrag_demo",
        persist_dir=args.persist_dir,
    )
    retriever = RetrieverPipeline(embedding_model=embedder, vectorstore=collection)
    if text_summaries:
        retriever.add_documents(text_summaries, text_contents)
    if table_summaries:
        retriever.add_documents(table_summaries, table_contents)
    if image_summaries:
        retriever.add_documents(image_summaries, image_contents)

    print("Ingestion complete. Processed keys:", list(data.keys()))


if __name__ == "__main__":
    main()
