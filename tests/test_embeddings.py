import os
import types

from src.retriever.embeddings import EmbeddingClient


def test_embedding_backend_hash_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    client = EmbeddingClient()
    assert client.backend == "hash"
    emb = client.embed_texts(["abc"])
    assert len(emb) == 1
    assert len(emb[0]) > 0


def test_embedding_backend_sbert_selected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeModel:
        def encode(self, texts, show_progress_bar=False):
            return types.SimpleNamespace(tolist=lambda: [[0.1, 0.2] for _ in texts])

    fake_module = types.SimpleNamespace(SentenceTransformer=lambda name: FakeModel())
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)

    client = EmbeddingClient()
    assert client.backend == "sbert"
    emb = client.embed_texts(["one", "two"])
    assert emb == [[0.1, 0.2], [0.1, 0.2]]
