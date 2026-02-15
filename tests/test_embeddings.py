import os
import types

from src.retriever.embeddings import EmbeddingClient
from src.llm_output.adapter import ModelClient


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


def test_embedding_openai_v1_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeEmbeddings:
        def create(self, input, model):
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.9, 0.1]) for _ in input])

    class FakeCompletions:
        def create(self, model, messages, max_tokens):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="stub answer"))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.embeddings = FakeEmbeddings()
            self.chat = FakeChat()

    fake_openai_mod = types.SimpleNamespace(OpenAI=FakeOpenAIClient)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_mod)

    emb_client = EmbeddingClient()
    assert emb_client.backend == "openai"
    emb = emb_client.embed_texts(["hello"])
    assert emb == [[0.9, 0.1]]

    llm_client = ModelClient(model="gpt-4o-mini")
    llm_client._langchain_client = None
    answer = llm_client.invoke([{"role": "user", "content": "Say hi"}], max_response_tokens=16)
    assert answer == "stub answer"
