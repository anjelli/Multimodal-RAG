from typing import Optional
from src.config import Config


def get_chroma_collection(collection_name: str = "MMRAG"):
    try:
        import chromadb
    except Exception as e:
        raise RuntimeError(
            "chromadb is required for Chroma persistence. Install it with `pip install chromadb`."
        ) from e

    # Try older and newer client constructors to be compatible across chroma versions
    client = None
    try:
        from chromadb.config import Settings

        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(Config.CHROMA_PERSIST_DIR)))
    except Exception:
        try:
            # newer chroma versions accept simple kwargs
            client = chromadb.Client(persist_directory=str(Config.CHROMA_PERSIST_DIR))
        except Exception:
            try:
                # last resort: default constructor
                client = chromadb.Client()
            except Exception as e:
                raise RuntimeError("Failed to construct a Chroma client: %s" % e)

    # create or get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        try:
            collection = client.create_collection(name=collection_name)
        except Exception:
            # fallback: some versions expect different args
            collection = client.create_collection(collection_name)

    return client, collection
