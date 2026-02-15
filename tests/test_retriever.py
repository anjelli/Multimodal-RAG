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
