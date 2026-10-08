from __future__ import annotations

import numpy as np

from docplainer.retrieval.bm25 import BM25
from docplainer.retrieval.embeddings import Embedder
from docplainer.schemas import Chunk, Hit


class HybridIndex:
    """In-memory dense + BM25 index fused with Reciprocal Rank Fusion.

    Kept in memory on purpose: the demo indexes per visitor session, corpora are small, and
    nothing needs to survive a restart. Re-adding a document that is already indexed is a
    no-op, so repeated uploads do not create duplicates.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._bm25 = BM25()

    def __len__(self) -> int:
        return len(self.chunks)

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks}

    def add(self, chunks: list[Chunk]) -> int:
        """Index chunks, skipping documents that are already present. Returns chunks added."""
        have = self.doc_ids()
        new = [c for c in chunks if c.doc_id not in have]
        if not new:
            return 0
        vectors = self.embedder.embed_documents([c.embed_text for c in new])
        self._matrix = vectors if not len(self.chunks) else np.vstack([self._matrix, vectors])
        self.chunks += new
        self._bm25.fit([c.embed_text for c in self.chunks])
        return len(new)

    def remove_doc(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self._bm25.fit([c.embed_text for c in self.chunks])

    def search(
        self,
        query: str,
        dense_k: int = 30,
        sparse_k: int = 30,
        rrf_k: int = 60,
        doc_ids: set[str] | None = None,
    ) -> list[Hit]:
        if not self.chunks:
            return []
        allowed = (
            np.array([c.doc_id in doc_ids for c in self.chunks]) if doc_ids else None
        )
        sims = self._matrix @ self.embedder.embed_query(query)
        bm = np.asarray(self._bm25.scores(query), dtype=np.float32)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
            bm = np.where(allowed, bm, -np.inf)

        dense_order = [int(i) for i in np.argsort(-sims)[:dense_k] if np.isfinite(sims[i])]
        sparse_order = [
            int(i) for i in np.argsort(-bm)[:sparse_k] if np.isfinite(bm[i]) and bm[i] > 0
        ]
        dense_rank = {i: r for r, i in enumerate(dense_order, start=1)}
        sparse_rank = {i: r for r, i in enumerate(sparse_order, start=1)}

        fused: dict[int, float] = {}
        for ranks in (dense_rank, sparse_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r)
        ordered = sorted(fused, key=fused.get, reverse=True)  # type: ignore[arg-type]
        return [
            Hit(
                chunk=self.chunks[i],
                score=fused[i],
                relevance=float(sims[i]),
                dense_score=float(sims[i]),
                dense_rank=dense_rank.get(i),
                sparse_rank=sparse_rank.get(i),
            )
            for i in ordered
        ]
