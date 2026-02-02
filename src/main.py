import argparse
import logging
import os
from pathlib import Path

from src.config import Config
from src.ingestion.pipeline import IngestionPipeline
from src.retriever.pipeline import RetrieverPipeline
from src.image_processing.pipeline import ImageProcessor
from src.llm_output.pipeline import LLMOutputGenerator


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal RAG ingestion and retrieval")
    p.add_argument("--source", help="Path to source PDF or directory", default=None)
    p.add_argument("--persist-dir", help="Chroma persist directory", default=str(Config.CHROMA_PERSIST_DIR))
    p.add_argument("--extracted-dir", help="extracted images dir", default=str(Config.EXTRACTED_DIR))
    p.add_argument("--openai-key", help="OpenAI API key (ENV fallback)", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--no-images", help="Skip extracting images (don't require poppler)", action="store_true")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    Config.ensure_dirs()

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
    ingestion = IngestionPipeline(source, extracted_dir=str(args.extracted_dir), extract_images=use_images)
    ingestion.load_data()
    ingestion.process_data()
    data = ingestion.get_processed_data()

    # Note: embedding/vectorstore setup is left to user configuration.
    print("Ingestion complete. Processed keys:", list(data.keys()))


if __name__ == "__main__":
    main()