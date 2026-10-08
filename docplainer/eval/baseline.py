"""The v1 pipeline, re-created for before/after comparison.

pypdf page text -> fixed 800/120 character splits -> dense-only top-4. No OCR, no table
handling, no keyword search, no reranking, no relevance gate. Same embedder as v2 so the
comparison isolates the pipeline, not the model.
"""

from __future__ import annotations

from pathlib import Path

from docplainer.eval.dataset import Question
from docplainer.retrieval.embeddings import Embedder
from docplainer.retrieval.index import HybridIndex
from docplainer.retrieval.retriever import RetrievalResult
from docplainer.schemas import Chunk, Hit


def _split(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            cut = cut if cut > start + size // 2 else text.rfind(" ", start, end)
            end = cut if cut > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [t for t in out if t]


class BaselineRetriever:
    def __init__(self, files: list[Path], embedder: Embedder, k: int = 4):
        from pypdf import PdfReader

        self.k = k
        self.index = HybridIndex(embedder)
        chunks: list[Chunk] = []
        for f in files:
            if f.suffix.lower() != ".pdf":
                continue
            for page_no, page in enumerate(PdfReader(str(f)).pages, start=1):
                for n, text in enumerate(_split(page.extract_text() or "")):
                    chunks.append(
                        Chunk(
                            id=f"{f.name}:{page_no}:{n}",
                            doc_id=f.name,
                            file_name=f.name,
                            page=page_no,
                            section="",
                            kind="text",
                            text=text,
                            embed_text=text,
                        )
                    )
        self.index.add(chunks)

    def retrieve(self, query: str) -> RetrievalResult:
        # dense only: the baseline never used keyword search
        sims = self.index._matrix @ self.index.embedder.embed_query(query)
        order = sims.argsort()[::-1][: self.k]
        hits = [
            Hit(self.index.chunks[i], float(sims[i]), float(sims[i]), float(sims[i]))
            for i in order
        ]
        return RetrievalResult(hits=hits, confident=True, trace={"baseline": True})


def question_query(q: Question) -> str:
    return q.question
