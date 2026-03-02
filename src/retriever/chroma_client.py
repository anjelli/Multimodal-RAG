from typing import Optional

def get_chroma_collection(collection_name: str, persist_dir: str):
    if not collection_name:
        raise ValueError("collection_name must be provided")
    if not persist_dir:
        raise ValueError("persist_dir must be provided")

    try:
        from chromadb import PersistentClient
    except Exception as e:
        raise RuntimeError(
            "chromadb with PersistentClient is required. Install/update chromadb."
        ) from e

    client = PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    return client, collection
