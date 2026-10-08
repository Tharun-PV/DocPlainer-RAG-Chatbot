from __future__ import annotations

from dataclasses import dataclass, field

from docplainer.config import Settings, get_settings
from docplainer.retrieval.index import HybridIndex
from docplainer.retrieval.rerank import Reranker, sigmoid
from docplainer.schemas import Hit


@dataclass
class RetrievalResult:
    hits: list[Hit]
    confident: bool  # False means "the documents probably do not contain the answer"
    trace: dict = field(default_factory=dict)


class Retriever:
    """Hybrid search -> rerank -> relevance gate.

    Wide recall first (dense + BM25, fused with RRF), then a cross-encoder re-scores the best
    candidates against the query. The top relevance score gates whether the caller should
    answer from the documents or fall back (web search / "not in the documents").
    """

    def __init__(
        self, index: HybridIndex, reranker: Reranker | None, settings: Settings | None = None
    ):
        self.index = index
        self.reranker = reranker
        self.cfg = settings or get_settings()

    def retrieve(self, query: str, doc_ids: set[str] | None = None) -> RetrievalResult:
        cfg = self.cfg
        candidates = self.index.search(
            query, cfg.dense_k, cfg.sparse_k, cfg.rrf_k, doc_ids=doc_ids
        )
        pool = candidates[: cfg.rerank_candidates]
        if not pool:
            return RetrievalResult([], False, {"candidates": 0, "reranked": False})

        if self.reranker is not None:
            logits = self.reranker.score(query, [h.chunk.embed_text for h in pool])
            for hit, logit in zip(pool, logits, strict=True):
                hit.score = logit
                hit.relevance = sigmoid(logit)
            pool.sort(key=lambda h: h.score, reverse=True)
            threshold = cfg.min_relevance
        else:
            threshold = cfg.min_dense_cosine

        hits = pool[: cfg.final_k]
        top = max(h.relevance for h in hits)
        return RetrievalResult(
            hits=hits,
            confident=top >= threshold,
            trace={
                "candidates": len(candidates),
                "reranked": self.reranker is not None,
                "top_relevance": round(top, 4),
                "threshold": threshold,
            },
        )
