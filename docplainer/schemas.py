from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table"]
ChunkKind = Literal["text", "table", "web"]


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: int = 0  # heading level, 1-based
    caption: str = ""  # tables only


@dataclass
class ParsedPage:
    number: int  # 1-based page / slide number; 0 for unpaginated formats
    blocks: list[Block] = field(default_factory=list)
    ocr: bool = False
    ocr_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    doc_id: str
    file_name: str
    pages: list[ParsedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    doc_id: str
    file_name: str
    page: int  # 0 when the format has no pages
    section: str
    kind: ChunkKind
    text: str  # what the LLM sees and what is cited
    embed_text: str  # what gets embedded / BM25-indexed (adds context, table summary)
    ocr: bool = False
    url: str = ""  # web chunks only

    def label(self) -> str:
        if self.kind == "web":
            return self.url or "web search"
        where = f" p.{self.page}" if self.page else ""
        return f"{self.file_name}{where}"


@dataclass
class Hit:
    chunk: Chunk
    score: float  # final ordering score
    relevance: float  # normalised 0..1, used for the "no answer" threshold
    dense_rank: int | None = None
    sparse_rank: int | None = None
