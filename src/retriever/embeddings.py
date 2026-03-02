import os
from typing import List
import time
import logging
import hashlib


class EmbeddingClient:
    """Provides a simple embedding interface. Prefers OpenAI if OPENAI_API_KEY set, else falls back to sentence-transformers."""

    def __init__(self, model: str = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MMRAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self._openai_mode = None
        # In-memory cache: sha256(text) → embedding vector (Issue 15)
        self._cache: dict = {}
        self._total_tokens_used: int = 0  # cumulative token count for cost tracking (Issue 14)
        if self.api_key:
            try:
                import openai

                if hasattr(openai, "OpenAI"):
                    # OpenAI SDK v1+
                    self._client = openai.OpenAI(api_key=self.api_key)
                    self._openai_mode = "v1"
                else:
                    # OpenAI SDK legacy
                    openai.api_key = self.api_key
                    self._client = openai
                    self._openai_mode = "legacy"
                self.backend = "openai"
            except Exception:
                self._client = None
                self.backend = "none"
        else:
            # fallback to sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(os.environ.get("MMRAG_ST_MODEL", "all-MiniLM-L6-v2"))
                self.backend = "sbert"
            except Exception:
                # final lightweight fallback: deterministic hash-based embeddings
                self.backend = "hash"
                self._hash_fn = hashlib.sha256
                self._dim = int(os.environ.get("MMRAG_HASH_EMBED_DIM", 512))
                # Issue 11: warn that hash backend is not semantic and unsuitable for production
                logging.warning(
                    "EmbeddingClient: no semantic backend available (no OPENAI_API_KEY and "
                    "sentence-transformers not installed). Falling back to deterministic hash "
                    "embeddings which are NOT semantic and will produce meaningless retrieval "
                    "results. Set OPENAI_API_KEY or install sentence-transformers."
                )

    @staticmethod
    def _text_cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Issue 15: check cache first; collect texts that need embedding
        results: List = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        for i, t in enumerate(texts):
            key = self._text_cache_key(t)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if not uncached_texts:
            return results  # type: ignore[return-value]

        new_embeddings = self._embed_uncached(uncached_texts)

        # populate cache and results
        for orig_i, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            key = self._text_cache_key(text)
            self._cache[key] = emb
            results[orig_i] = emb

        return results  # type: ignore[return-value]

    def _embed_uncached(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "openai":
            # batching for OpenAI
            embeds = []
            for i in range(0, len(texts), 16):
                chunk = texts[i : i + 16]
                # retry logic
                for attempt in range(4):
                    try:
                        if self._openai_mode == "v1":
                            resp = self._client.embeddings.create(input=chunk, model=self.model)
                            embeds.extend([e.embedding for e in resp.data])
                            # Issue 14: log token usage for cost tracking
                            usage = getattr(resp, "usage", None)
                            if usage is not None:
                                tokens = getattr(usage, "total_tokens", 0) or 0
                                self._total_tokens_used += tokens
                                logging.info(
                                    "OpenAI embeddings: %d tokens used in this batch (cumulative: %d)",
                                    tokens,
                                    self._total_tokens_used,
                                )
                        else:
                            resp = self._client.Embedding.create(input=chunk, model=self.model, request_timeout=30)
                            embeds.extend([e["embedding"] for e in resp["data"]])
                        break
                    except Exception as e:
                        wait = (2 ** attempt) * 0.5
                        logging.warning("Embedding request failed (attempt %s): %s; retrying in %ss", attempt + 1, e, wait)
                        time.sleep(wait)
                        if attempt == 3:
                            raise
            return embeds
        elif self.backend == "sbert":
            # SentenceTransformer returns numpy arrays; convert to lists
            try:
                arr = self._model.encode(texts, show_progress_bar=False)
                return arr.tolist()
            except Exception:
                # fallback: simple split embeddings (should not happen)
                return [list(map(float, str(t).encode("utf-8")[:16])) for t in texts]
        elif self.backend == "hash":
            # deterministic hash-based embeddings (not semantic, for demo only)
            import math

            out = []
            for t in texts:
                h = self._hash_fn(str(t).encode("utf-8")).digest()
                vec = []
                # expand digest to required dim by repeating
                while len(vec) < self._dim:
                    for b in h:
                        if len(vec) >= self._dim:
                            break
                        vec.append((b / 255.0) - 0.5)
                # normalize
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vec = [x / norm for x in vec]
                out.append(vec)
            return out
        else:
            raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers.")
