import logging
from src.retriever.chroma_client import get_chroma_collection
from src.retriever.embeddings import EmbeddingClient
import shelve
import json
from pathlib import Path


def query_collection(collection, emb, queries, top_k=3):
    results = []
    # Try text-based query first
    try:
        resp = collection.query(query_texts=queries, n_results=top_k, include=["metadatas", "documents", "distances"])
        # resp may have 'ids', 'metadatas', 'documents'
        results = resp
    except Exception:
        try:
            resp = collection.query(queries=queries, n_results=top_k, include=["metadatas", "documents", "distances"])
            results = resp
        except Exception:
            # fallback to embedding-based query
            emb_queries = emb.embed_texts(queries)
            resp = collection.query(query_embeddings=emb_queries, n_results=top_k, include=["metadatas", "documents", "distances"])
            results = resp
    return results


def print_query_results(results, queries):
    metadatas = results.get("metadatas") or results.get("result_metadatas") or []
    docs = results.get("documents") or results.get("result_documents") or []
    for qi, q in enumerate(queries):
        print(f"\nQuery: {q}\n")
        meta_list = metadatas[qi] if qi < len(metadatas) else []
        doc_list = docs[qi] if qi < len(docs) else []
        for rank in range(len(meta_list)):
            meta = meta_list[rank] or {}
            _id = meta.get("doc_id") or meta.get("id") or None
            doc = doc_list[rank] if rank < len(doc_list) else None
            print(f"- Rank {rank+1}: id={_id} meta={meta}")
            if doc:
                snippet = str(doc)[:400]
                print(f"  snippet: {snippet}")


def inspect_summary(summary_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    titles = summary.get("Title", [])
    tables = summary.get("Table", [])
    images = summary.get("Image", [])

    print(f"\nSummary: {summary_path}\n")
    print(f"Titles (showing up to 10):\n")
    for t in titles[:10]:
        print(f"- {t}")

    print(f"\nTables found: {len(tables)}")
    for t in tables[:5]:
        if isinstance(t, dict) and t.get("csv_path"):
            print(f"- CSV: {t.get('csv_path')} shape={t.get('shape')}")
            try:
                import pandas as pd

                df = pd.read_csv(t.get("csv_path"))
                print(df.head(3).to_csv(index=False))
            except Exception as e:
                print(f"  (failed to read csv: {e})")
        else:
            print(f"- {t}")

    print(f"\nImages found: {len(images)} (showing up to 5):")
    for im in images[:5]:
        print(f"- {im}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processed = Path("processed_data")
    summary_files = list(processed.glob("*_summary.json"))
    if not summary_files:
        raise SystemExit("No summary JSON found in processed_data/")
    summary_path = summary_files[0]

    # connect to chroma
    client, collection = get_chroma_collection(collection_name="mmrag_demo")

    emb = EmbeddingClient()

    queries = [
        "ITC net zero targets",
        "water stewardship initiatives",
        "employee engagement index",
        "sustainable supply chain"
    ]

    print("Running queries against Chroma collection...")
    results = query_collection(collection, emb, queries, top_k=3)
    print_query_results(results, queries)

    print("\nInspecting summary JSON...\n")
    inspect_summary(summary_path)

    # show a couple of docstore entries for the top id of first query
    metadatas = results.get("metadatas") or []
    if metadatas and metadatas[0]:
        top_meta = metadatas[0][0]
        top_id = top_meta.get("doc_id") or top_meta.get("id")
        if top_id:
            print(f"\nLooking up stored content for id {top_id} in shelve docstore")
            try:
                with shelve.open(str(processed / "docstore.db")) as ds:
                    content = ds.get(top_id)
                    print("Stored content type:", type(content))
                    if isinstance(content, str):
                        print(content[:500])
                    else:
                        print(str(content)[:500])
            except Exception as e:
                print("Failed to open docstore:", e)
