from typing import List, Dict, Any, Optional
import uuid
import logging
import shelve
from src.config import Config


class RetrieverPipeline:
    def __init__(self, embedding_model, vectorstore=None, docstore_path: Optional[str] = None, id_key="doc_id"):
        """If `vectorstore` is None, caller should supply a chroma collection via chroma_client.get_chroma_collection().
        `docstore_path` if provided will be used for a simple persistent shelve-backed docstore."""
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.id_key = id_key
        # initialize persistent docstore
        self.docstore_path = docstore_path or str(Config.PROCESSED_DIR / "docstore.db")

    def _open_docstore(self):
        return shelve.open(self.docstore_path)

    def add_documents(self, summaries: List[str], contents: List[Any], embeddings: Optional[List[List[float]]] = None):
        from langchain_core.documents import Document

        if len(summaries) != len(contents):
            raise ValueError("summaries and contents must be same length")

        doc_ids = [str(uuid.uuid4()) for _ in contents]
        metadatas = []
        for i, content in enumerate(contents):
            meta = {self.id_key: doc_ids[i]}
            if isinstance(content, dict):
                # include provenance fields
                for k in ("path", "csv_path", "shape", "type"):
                    if content.get(k) is not None:
                        meta[k] = content.get(k)
                extra_meta = content.get("metadata")
                if isinstance(extra_meta, dict):
                    meta.update(extra_meta)
            metadatas.append(meta)

        # If vectorstore is a chroma collection (has add(ids=...)), use its API
        try:
            add_fn = getattr(self.vectorstore, "add", None)
        except Exception:
            add_fn = None

        # persist contents in shelve-backed docstore
        try:
            with self._open_docstore() as ds:
                for _id, c in zip(doc_ids, contents):
                    ds[_id] = c
        except Exception:
            logging.exception("Failed to persist to shelve docstore %s", self.docstore_path)

        # add to vectorstore
        if add_fn:
            try:
                # chroma expects embeddings possibly; if not provided compute using embedding_model
                if embeddings is None and hasattr(self.embedding_model, "embed_texts"):
                    embeddings = self.embedding_model.embed_texts(summaries)

                # call collection.add
                kwargs = {"ids": doc_ids, "documents": summaries, "metadatas": metadatas}
                if embeddings is not None:
                    kwargs["embeddings"] = embeddings
                add_fn(**kwargs)
                # try persist on client if available (chroma client may be attached)
                client = getattr(self.vectorstore, "client", None)
                if client is not None:
                    try:
                        client.persist()
                    except Exception:
                        pass
            except TypeError:
                # fallback if add signature different
                try:
                    self.vectorstore.add(summaries)
                except Exception:
                    logging.exception("Failed to add documents to vectorstore")
        else:
            # fallback for older vectorstore APIs (langchain-style)
            docs = [Document(page_content=s, metadata=m) for s, m in zip(summaries, metadatas)]
            try:
                self.vectorstore.add_documents(docs)
                persist = getattr(self.vectorstore, "persist", None)
                if callable(persist):
                    persist()
            except Exception:
                logging.exception("Failed to add using vectorstore.add_documents")

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not query:
            return []
        if not hasattr(self.embedding_model, "embed_texts"):
            raise RuntimeError("Embedding model does not support embed_texts.")

        query_embedding = self.embedding_model.embed_texts([query])
        query_fn = getattr(self.vectorstore, "query", None)
        if not callable(query_fn):
            raise RuntimeError("Vectorstore does not support query().")

        result = query_fn(
            query_embeddings=query_embedding,
            n_results=k,
            include=["ids", "documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        contents = {}
        try:
            with self._open_docstore() as ds:
                for _id in ids:
                    if _id in ds:
                        contents[_id] = ds.get(_id)
        except Exception:
            logging.exception("Failed to read from docstore %s", self.docstore_path)

        results = []
        for _id, doc, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                {
                    "id": _id,
                    "summary": doc,
                    "metadata": meta,
                    "content": contents.get(_id),
                    "distance": dist,
                }
            )
        return results

    def create_multi_vector_retriever(self, text_summaries, texts, table_summaries, tables, image_summaries, images):
        # Ensure documents are added first
        if text_summaries:
            self.add_documents(text_summaries, texts)
        if table_summaries:
            self.add_documents(table_summaries, tables)
        if image_summaries:
            self.add_documents(image_summaries, images)

        # Build a simple retriever wrapper if using chroma
        try:
            from langchain.retrievers.multi_vector import MultiVectorRetriever
            retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=None,
                id_key=self.id_key,
            )
            return retriever
        except Exception:
            # fallback: return the raw vectorstore object and let caller use its query APIs
            return self.vectorstore
