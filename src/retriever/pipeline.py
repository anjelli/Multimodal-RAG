from typing import List, Dict, Any, Optional
import uuid
import logging
import shelve
import re
from src.config import Config


class RetrieverPipeline:
    def __init__(self, embedding_model, vectorstore=None, docstore_path: Optional[str] = None, id_key="doc_id"):
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.id_key = id_key
        self.docstore_path = docstore_path or str(Config.PROCESSED_DIR / "docstore.db")

    def _open_docstore(self):
        return shelve.open(self.docstore_path)

    def add_documents(
        self,
        summaries: List[str],
        contents: List[Any],
        embeddings: Optional[List[List[float]]] = None,
    ):
        if len(summaries) != len(contents):
            raise ValueError("summaries and contents must be same length")

        doc_ids = [str(uuid.uuid4()) for _ in contents]

        metadatas = []
        for i, content in enumerate(contents):
            meta = {self.id_key: doc_ids[i]}
            if isinstance(content, dict):
                for k in ("path", "csv_path", "shape", "type"):
                    if content.get(k) is not None:
                        meta[k] = content.get(k)
                extra_meta = content.get("metadata")
                if isinstance(extra_meta, dict):
                    meta.update(extra_meta)
            metadatas.append(meta)

        # Persist original contents
        try:
            with self._open_docstore() as ds:
                for _id, c in zip(doc_ids, contents):
                    ds[_id] = c
        except Exception:
            logging.exception("Failed to persist to shelve docstore %s", self.docstore_path)

        # Add to Chroma
        try:
            if embeddings is None and hasattr(self.embedding_model, "embed_texts"):
                embeddings = self.embedding_model.embed_texts(summaries)

            kwargs = {
                "ids": doc_ids,
                "documents": summaries,
                "metadatas": metadatas,
            }

            if embeddings is not None:
                kwargs["embeddings"] = embeddings

            self.vectorstore.add(**kwargs)

            logging.info("Added %s documents to collection", len(summaries))
            try:
                count = self.vectorstore.count()
            except Exception:
                count = "unknown"
            logging.info("Collection count after ingestion: %s", count)

        except Exception:
            logging.exception("Failed to add documents to vectorstore")

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not query:
            return []

        if not hasattr(self.embedding_model, "embed_texts"):
            raise RuntimeError("Embedding model does not support embed_texts.")

        query_embedding = self.embedding_model.embed_texts([query])

        try:
            count = self.vectorstore.count()
        except Exception:
            count = "unknown"

        logging.info("Collection count before query: %s", count)

        # FIXED: removed "ids" from include (invalid in Chroma 0.6.x)
        result = self.vectorstore.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # ids are returned automatically by Chroma
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

        # Keyword re-ranking
        keyword_set = {
            w for w in re.split(r"\W+", str(query).lower()) if len(w) > 2
        }

        def keyword_overlap(text: str) -> int:
            if not text or not keyword_set:
                return 0
            tokens = [w for w in re.split(r"\W+", str(text).lower()) if len(w) > 2]
            return len(keyword_set.intersection(tokens))

        def rank_key(item: Dict[str, Any]):
            text = item.get("content") or item.get("summary") or ""
            overlap = keyword_overlap(text)
            dist = item.get("distance")
            dist_val = dist if dist is not None else 1.0
            return (-overlap, dist_val)

        results.sort(key=rank_key)
        return results

    def create_multi_vector_retriever(
        self,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        images,
    ):
        if text_summaries:
            self.add_documents(text_summaries, texts)
        if table_summaries:
            self.add_documents(table_summaries, tables)
        if image_summaries:
            self.add_documents(image_summaries, images)

        return self.vectorstore
