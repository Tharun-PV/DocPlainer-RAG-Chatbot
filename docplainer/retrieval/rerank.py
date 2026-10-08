from __future__ import annotations

import math
from typing import Protocol

from docplainer.config import Settings, get_settings


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance logit per text (higher is more relevant)."""
        ...


class CrossEncoderReranker:
    """Cross-encoder via fastembed's ONNX runtime: CPU only, no torch."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


_cache: dict[str, Reranker] = {}


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    cfg = settings or get_settings()
    if not cfg.use_reranker or cfg.embedder_backend == "hashing":
        return None
    if cfg.reranker_model not in _cache:
        _cache[cfg.reranker_model] = CrossEncoderReranker(cfg.reranker_model)
    return _cache[cfg.reranker_model]
