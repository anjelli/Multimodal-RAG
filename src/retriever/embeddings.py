import os
from typing import List
import time
import logging


class EmbeddingClient:
    """Provides a simple embedding interface. Prefers OpenAI if OPENAI_API_KEY set, else falls back to sentence-transformers."""

    def __init__(self, model: str = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MMRAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self._openai_mode = None
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
                import hashlib

                self.backend = "hash"
                self._hash_fn = hashlib.sha256
                self._dim = int(os.environ.get("MMRAG_HASH_EMBED_DIM", 512))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
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
