from typing import List, Dict, Any, Optional
import uuid
import logging
import shelve
import re
import hashlib
from src.config import Config


class RetrieverPipeline:
    def __init__(self, embedding_model, vectorstore=None, docstore_path: Optional[str] = None, id_key="doc_id"):
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.id_key = id_key
        self.docstore_path = docstore_path or str(Config.PROCESSED_DIR / "docstore.db")

    def _open_docstore(self):
        return shelve.open(self.docstore_path)

    @staticmethod
    def _content_hash(content: Any) -> str:
        """Compute SHA-256 hash of content for deduplication (Issue 3)."""
        text = content if isinstance(content, str) else str(content)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_documents(self, summaries: List[str], contents: List[Any], embeddings: Optional[List[List[float]]] = None):
        if len(summaries) != len(contents):
            raise ValueError("summaries and contents must be same length")

        # Issue 3: deduplicate via SHA-256 content hash stored in docstore
        existing_hashes: set = set()
        try:
            with self._open_docstore() as ds:
                existing_hashes = set(ds.get("_content_hashes", {}).keys())
        except Exception:
            logging.warning("Could not read existing content hashes from docstore; skipping dedup check")

        filtered_summaries, filtered_contents, filtered_indices = [], [], []
        for i, (summary, content) in enumerate(zip(summaries, contents)):
            h = self._content_hash(content)
            if h in existing_hashes:
                logging.debug("Skipping duplicate document (hash %s)", h[:12])
                continue
            filtered_summaries.append(summary)
            filtered_contents.append(content)
            filtered_indices.append(i)

        if not filtered_summaries:
            logging.info("All %d documents already indexed; nothing to add.", len(summaries))
            return

        if len(filtered_summaries) < len(summaries):
            logging.info(
                "Deduplication: skipping %d already-indexed documents; adding %d new.",
                len(summaries) - len(filtered_summaries),
                len(filtered_summaries),
            )
            # re-slice embeddings to match filtered set
            if embeddings is not None:
                embeddings = [embeddings[i] for i in filtered_indices]

        doc_ids = [str(uuid.uuid4()) for _ in filtered_contents]
        metadatas = []
        for i, content in enumerate(filtered_contents):
            meta = {self.id_key: doc_ids[i]}
            if isinstance(content, dict):
                # Issue 4: include provenance/metadata fields
                for k in ("path", "csv_path", "shape", "type", "source", "page", "extraction_method", "content_type"):

                    if content.get(k) is not None:
                        meta[k] = content.get(k)
                extra_meta = content.get("metadata")
                if isinstance(extra_meta, dict):
                    # carry through page_number, source_document, etc.
                    for k in ("page", "page_number", "source", "source_document", "extraction_method", "content_type"):
                        if k in extra_meta:
                            meta[k] = extra_meta[k]
                    meta.update(extra_meta)
            metadatas.append(meta)

        # If vectorstore is a chroma collection (has add(ids=...)), use its API
        try:
            add_fn = getattr(self.vectorstore, "add", None)
        except Exception:
            add_fn = None

        # persist contents and updated hashes in shelve-backed docstore
        try:
            with self._open_docstore() as ds:
                for _id, c in zip(doc_ids, filtered_contents):
                    ds[_id] = c
                # Issue 3: update the set of known content hashes
                known_hashes = ds.get("_content_hashes", {})
                for content in filtered_contents:
                    h = self._content_hash(content)
                    known_hashes[h] = True
                ds["_content_hashes"] = known_hashes
        except Exception:
            logging.exception("Failed to persist to shelve docstore %s", self.docstore_path)

        # add to vectorstore
        if add_fn:
            try:
                # chroma expects embeddings possibly; if not provided compute using embedding_model
                if embeddings is None and hasattr(self.embedding_model, "embed_texts"):
                    embeddings = self.embedding_model.embed_texts(filtered_summaries)

                # call collection.add
                kwargs = {"ids": doc_ids, "documents": filtered_summaries, "metadatas": metadatas}
                if embeddings is not None:
                    kwargs["embeddings"] = embeddings
                add_fn(**kwargs)
                logging.info("Added %s documents to collection", len(filtered_summaries))
                try:
                    count = self.vectorstore.count()
                except Exception:
                    count = "unknown"
                logging.info("Collection count after ingestion: %s", count)
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
                    self.vectorstore.add(filtered_summaries)
                    logging.info("Added %s documents to collection", len(filtered_summaries))
                    try:
                        count = self.vectorstore.count()
                    except Exception:
                        count = "unknown"
                    logging.info("Collection count after ingestion: %s", count)
                except Exception:
                    logging.exception("Failed to add documents to vectorstore")
        else:
            # fallback for older vectorstore APIs (langchain-style)
            from langchain_core.documents import Document

            docs = [Document(page_content=s, metadata=m) for s, m in zip(filtered_summaries, metadatas)]
            try:
                self.vectorstore.add_documents(docs)
                logging.info("Added %s documents to collection", len(filtered_summaries))
                try:
                    count = self.vectorstore.count()
                except Exception:
                    count = "unknown"
                logging.info("Collection count after ingestion: %s", count)
                persist = getattr(self.vectorstore, "persist", None)
                if callable(persist):
                    persist()
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
