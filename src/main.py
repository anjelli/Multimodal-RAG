import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import logging
from pathlib import Path

from src.config import Config
from src.retriever.embeddings import EmbeddingClient
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.pipeline import RetrieverPipeline


def setup_logging(level=logging.WARNING):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal RAG ingestion and retrieval")
    p.add_argument("--source", default=None)
    p.add_argument("--persist-dir", default=str(Config.CHROMA_PERSIST_DIR))
    p.add_argument("--extracted-dir", default=str(Config.EXTRACTED_DIR))
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--question", default=None)
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


# ---------------------- CONTEXT BUILDER ----------------------

def build_context(results, max_chars=3000, max_chunks=5):
    seen = set()
    combined = []
    total = 0

    for r in results:
        text = r.get("content") or r.get("summary")
        if isinstance(text, dict):
            continue
        if not text:
            continue

        text = str(text).strip()

        # drop low-information fragments
        if len(text) < 30:
            continue

        # deduplicate
        if text in seen:
            continue
        seen.add(text)

        if total + len(text) > max_chars:
            break

        combined.append(text)
        total += len(text)

        if len(combined) >= max_chunks:
            break

    return "\n\n".join(combined)


# ---------------------- MAIN ----------------------

def main():
    setup_logging()
    args = parse_args()
    Config.ensure_dirs()

    # ---------------------- QUERY MODE ----------------------

    if args.question:
        print("\nRetrieving...\n")

        embedder = EmbeddingClient()
        _, collection = get_chroma_collection(
            collection_name="mmrag_demo",
            persist_dir=args.persist_dir,
        )

        retriever = RetrieverPipeline(
            embedding_model=embedder,
            vectorstore=collection
        )

        results = retriever.retrieve(args.question, k=args.top_k)

        print("Top Retrieved Chunks\n")
        for idx, result in enumerate(results, start=1):
            print(f"[{idx}] score={round(result.get('distance', 0), 4)}")
            print(result.get("summary"))
            print()

        context = build_context(results)

        from src.llm_output.adapter import ModelClient

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise corporate analyst.\n"
                    "Answer ONLY using the provided context.\n"
                    "If insufficient information exists, say: "
                    "'The answer is not explicitly stated in the provided context.'\n"
                    "Do not repeat phrases. Do not invent details.\n"
                    "Provide a concise structured answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {args.question}\n\n"
                    "Provide a clear, structured answer."
                ),
            },
        ]

        print("\nGenerating answer...\n")

        answer = ModelClient().invoke(messages)

        print("\nFinal Answer\n")
        print(answer.strip())
        return

    # ---------------------- INGESTION MODE ----------------------

    source = args.source
    if source is None:
        pdfs = list(Config.DATA_DIR.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF found in data directory")
        source = str(pdfs[0])

    print(f"\nIngesting: {source}\n")

    from src.ingestion.pipeline import IngestionPipeline

    ingestion = IngestionPipeline(
        source,
        extracted_dir=str(args.extracted_dir),
        extract_images=not args.no_images,
    )

    ingestion.load_data()
    ingestion.process_data()
    data = ingestion.get_processed_data()

    text_summaries = []
    text_contents = []

    text_cats = ["Title", "NarrativeText", "Text", "ListItem"]

    for cat in text_cats:
        for item in data.get(cat, []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            text = str(text).strip()
            if not text:
                continue
            text_summaries.append(text[:400])
            text_contents.append(text)

    if not text_summaries:
        raise SystemExit("No documents found to index")

    embedder = EmbeddingClient()

    _, collection = get_chroma_collection(
        collection_name="mmrag_demo",
        persist_dir=args.persist_dir,
    )

    retriever = RetrieverPipeline(
        embedding_model=embedder,
        vectorstore=collection,
    )

    retriever.add_documents(text_summaries, text_contents)

    print("Ingestion complete.")
    print("Indexed chunks:", len(text_summaries))


if __name__ == "__main__":
    main()
