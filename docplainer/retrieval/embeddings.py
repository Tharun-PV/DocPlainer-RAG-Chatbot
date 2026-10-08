from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

from docplainer.config import Settings, get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """ONNX embeddings on CPU (no torch). Vectors are L2-normalised."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words embedder with no model download.

    Lexical only (no semantics), so it is for tests and offline smoke runs, never for a
    deployed demo.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = zlib.crc32(g.encode())
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


_cache: dict[str, Embedder] = {}


def get_embedder(settings: Settings | None = None) -> Embedder:
    cfg = settings or get_settings()
    key = f"{cfg.embedder_backend}:{cfg.embedding_model}"
    if key not in _cache:
        _cache[key] = (
            HashingEmbedder()
            if cfg.embedder_backend == "hashing"
            else FastEmbedder(cfg.embedding_model)
        )
    return _cache[key]
