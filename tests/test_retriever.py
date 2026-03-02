from src.retriever.pipeline import RetrieverPipeline


class DummyEmbedder:
    def embed_texts(self, texts):
        return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self):
        self.docs = []
        self.metas = []
        self.ids = []

    def add(self, ids, documents, metadatas, embeddings=None):
        self.ids.extend(ids)
        self.docs.extend(documents)
        self.metas.extend(metadatas)

    def count(self):
        return len(self.ids)

    def query(self, query_embeddings, n_results, include):
        k = min(n_results, len(self.ids))
        return {
            "ids": [self.ids[:k]],
            "documents": [self.docs[:k]],
            "metadatas": [self.metas[:k]],
            "distances": [[0.2 + i * 0.1 for i in range(k)]],
        }


def test_retriever_add_and_query(tmp_path):
    store = FakeVectorStore()
    retriever = RetrieverPipeline(
        embedding_model=DummyEmbedder(),
        vectorstore=store,
        docstore_path=str(tmp_path / "docstore.db"),
    )

    retriever.add_documents(
        summaries=["net zero commitment", "water usage reduction"],
        contents=[{"text": "doc1", "type": "text"}, {"text": "doc2", "type": "text"}],
    )

    results = retriever.retrieve("net zero targets", k=2)
    assert len(results) == 2
    assert results[0]["summary"]
    assert "doc_id" in results[0]["metadata"]
    assert results[0]["content"] is not None


def test_retriever_deduplication(tmp_path):
    """Issue 3: adding the same content twice should only index it once."""
    store = FakeVectorStore()
    retriever = RetrieverPipeline(
        embedding_model=DummyEmbedder(),
        vectorstore=store,
        docstore_path=str(tmp_path / "docstore.db"),
    )

    content = {"text": "unique document content", "type": "text"}
    retriever.add_documents(summaries=["summary1"], contents=[content])
    retriever.add_documents(summaries=["summary1"], contents=[content])

    # Vector store should only contain one document, not two
    assert store.count() == 1, f"Expected 1 doc after dedup, got {store.count()}"


def test_retriever_metadata_includes_provenance(tmp_path):
    """Issue 4: metadata should include source, content_type, extraction_method fields."""
    store = FakeVectorStore()
    retriever = RetrieverPipeline(
        embedding_model=DummyEmbedder(),
        vectorstore=store,
        docstore_path=str(tmp_path / "docstore.db"),
    )

    content = {
        "text": "sample text",
        "type": "text",
        "source": "test.pdf",
        "source_document": "test.pdf",
        "content_type": "narrativetext",
        "extraction_method": "unstructured",
    }
    retriever.add_documents(summaries=["sample summary"], contents=[content])

    meta = store.metas[0]
    assert meta.get("source") == "test.pdf"
    assert meta.get("content_type") == "narrativetext"
    assert meta.get("extraction_method") == "unstructured"
