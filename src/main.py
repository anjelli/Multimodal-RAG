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

import re

def build_context(results, max_chars=6000, max_chunks=8, query: str = ""):
    seen = set()
    combined = []
    total = 0

    keyword_set = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2} if query else set()

    def keyword_score(text: str) -> int:
        if not keyword_set:
            return 1
        tokens = {w for w in re.split(r"\W+", text.lower()) if len(w) > 2}
        return len(keyword_set & tokens)

    # filter out chunks with no keyword overlap when query is provided
    filtered = []
    for r in results:
        text = r.get("content") or r.get("summary")
        if isinstance(text, dict):
            continue
        if not text:
            continue
        text = str(text).strip()
        if len(text) < 30:
            continue
        if keyword_set and keyword_score(text) == 0:
            continue
        filtered.append(text)

    for text in filtered:
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

def clean_response(text: str) -> str:
    """Remove section headings, deduplicate sentences, and validate completeness."""
    # Remove lines that look like section headings (short, title-case or all-caps, no period)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Skip lines that are purely heading-like: short, end without punctuation
        if len(stripped) < 80 and not stripped.endswith((".", "?", "!")):
            words = stripped.split()
            if words and all(w[0].isupper() or not w[0].isalpha() for w in words if w):
                continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    # Deduplicate sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen_sents: set = set()
    unique_sents = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key and key not in seen_sents:
            seen_sents.add(key)
            unique_sents.append(s)
    text = " ".join(unique_sents).strip()

    # If the result is empty or suspiciously short, return a fallback message
    if len(text) < 10:
        return "The answer is not explicitly stated in the provided context."

    return text

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
            print(f"[{idx}] score={{round(result.get('distance', 0), 4)}}")
            print(result.get("summary"))
            print()  

        context = build_context(results, query=args.question)

        from src.llm_output.adapter import ModelClient

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise corporate analyst answering questions from official documents.\n"
                    "Rules:\n"
                    "1. Extract ONLY factual information explicitly stated in the context.\n"
                    "2. Do NOT include section headings, report titles, or structural labels in your answer.\n"
                    "3. Do NOT repeat phrases or sentences.\n"
                    "4. Do NOT invent numbers, names, or details not present in the context.\n"
                    "5. Write in complete, grammatically correct sentences.\n"
                    "6. If the context does not contain enough information to answer, respond with: "
                    "'The answer is not explicitly stated in the provided context.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{{context}}\n\n"
                    f"Question: {{args.question}}\n\n"
                    "Provide a clear, structured answer."
                ),
            },
        ]

        print("\nGenerating answer...\n")

        answer = ModelClient().invoke(messages)

        print("\nFinal Answer\n")
        print(clean_response(answer.strip()))
        return

    # ---------------------- INGESTION MODE ----------------------

    source = args.source
    if source is None:
        pdfs = list(Config.DATA_DIR.glob("*.pdf"))
        if not pdfs:
            raise SystemExit("No PDF found in data directory")
        source = str(pdfs[0])

    print(f"\nIngesting: {{source}}\n")

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