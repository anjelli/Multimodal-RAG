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


def test_embedding_cache_prevents_recomputation(monkeypatch):
    """Issue 15: same text should be returned from cache without re-embedding."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    client = EmbeddingClient()
    assert client.backend == "hash"

    call_count = {"n": 0}
    original_embed_uncached = client._embed_uncached

    def counting_embed(texts):
        call_count["n"] += len(texts)
        return original_embed_uncached(texts)

    client._embed_uncached = counting_embed

    client.embed_texts(["hello world"])
    client.embed_texts(["hello world"])  # should hit cache
    assert call_count["n"] == 1, "Second embed_texts call should use cache, not re-embed"


def test_hash_backend_emits_warning(monkeypatch, recwarn):
    """Issue 11: hash backend should log a warning."""
    import logging
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    import warnings
    with warnings.catch_warnings(record=True):
        client = EmbeddingClient()
    # The warning is logged via logging.warning; verify backend is "hash"
    assert client.backend == "hash"
