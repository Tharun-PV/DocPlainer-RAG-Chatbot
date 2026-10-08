"""Structure-aware chunking.

* Headings start a new chunk and build a section path ("Report > Revenue > By region").
* Paragraphs are packed up to a size budget with a small sentence overlap.
* Tables are never cut mid-row; oversized ones split by rows with the header repeated.
* Every chunk's embed_text carries file + section context, and table chunks also carry a
  short generated description, so a question about "revenue by region" finds the table.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from docplainer.config import Settings, get_settings
from docplainer.ingestion import tables
from docplainer.schemas import Chunk, ParsedDocument

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
Summarizer = Callable[[str], str]


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for sentence in _SENTENCE.split(text):
        while len(sentence) > limit:  # no sentence boundary: hard split on spaces
            cut = sentence.rfind(" ", 0, limit) or limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if cur and len(cur) + len(sentence) + 1 > limit:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def _tail(text: str, size: int) -> str:
    """Last whole sentence(s) of text within size chars, used as overlap."""
    if size <= 0 or not text:
        return ""
    sentences = _SENTENCE.split(text)
    out = ""
    for s in reversed(sentences):
        if len(out) + len(s) + 1 > size:
            break
        out = f"{s} {out}".strip()
    return out


def _split_table(markdown: str, limit: int) -> list[str]:
    lines = markdown.splitlines()
    if len(markdown) <= limit or len(lines) <= 3:
        return [markdown]
    header, rows = lines[:2], lines[2:]
    parts, cur = [], []
    size = sum(len(h) for h in header)
    for row in rows:
        if cur and size + len(row) > limit:
            parts.append("\n".join(header + cur))
            cur, size = [], sum(len(h) for h in header)
        cur.append(row)
        size += len(row) + 1
    if cur:
        parts.append("\n".join(header + cur))
    return parts


class _Chunker:
    """Walks a parsed document once, tracking the heading path and a paragraph buffer."""

    def __init__(self, doc: ParsedDocument, cfg: Settings, summarizer: Summarizer | None):
        self.doc, self.cfg, self.summarizer = doc, cfg, summarizer
        self.chunks: list[Chunk] = []
        self.stack: list[tuple[int, str]] = []
        self.buf: list[str] = []
        self.size = 0
        self.carry = ""
        self.buf_sec = ""
        self.page = 0
        self.ocr = False

    def section(self) -> str:
        return " > ".join(t for _, t in self.stack)

    def context(self, sec: str) -> str:
        return f"{self.doc.file_name} | {sec}" if sec else self.doc.file_name

    def emit(self, kind: str, text: str, embed: str, sec: str) -> None:
        self.chunks.append(
            Chunk(
                id=f"{self.doc.doc_id}:{self.page}:{len(self.chunks) + 1}",
                doc_id=self.doc.doc_id,
                file_name=self.doc.file_name,
                page=self.page,
                section=sec,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                embed_text=embed,
                ocr=self.ocr,
            )
        )

    def flush(self) -> None:
        body = " ".join(self.buf).strip()
        if len(body) >= 20:
            text = body if not self.carry else f"{self.carry} {body}"
            embed = f"{self.context(self.buf_sec)}\n{text}"
            self.emit("text", text, embed, self.buf_sec)
            self.carry = _tail(body, self.cfg.chunk_overlap_chars)
        self.buf, self.size = [], 0

    def heading(self, level: int, text: str) -> None:
        self.flush()
        self.carry = ""
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    def paragraph(self, text: str) -> None:
        for piece in _split_long(text, self.cfg.chunk_max_chars):
            if self.buf and self.size + len(piece) > self.cfg.chunk_max_chars:
                self.flush()
            if not self.buf:
                self.buf_sec = self.section()
            self.buf.append(piece)
            self.size += len(piece) + 1

    def table(self, markdown: str, caption: str) -> None:
        self.flush()
        self.carry = ""
        grid = tables.parse_markdown_table(markdown)
        summary = (
            self.summarizer(markdown) if self.summarizer else tables.describe(grid, caption)
        )
        sec = self.section()
        for part in _split_table(markdown, self.cfg.table_max_chars):
            shown = f"{caption}\n{part}" if caption else part
            self.emit("table", shown, f"{self.context(sec)}\n{summary}\n{part}", sec)

    def run(self) -> list[Chunk]:
        for page in self.doc.pages:
            self.page, self.ocr = page.number, page.ocr
            for block in page.blocks:
                if block.kind == "heading":
                    self.heading(block.level, block.text)
                elif block.kind == "paragraph":
                    self.paragraph(block.text)
                else:
                    self.table(block.text, block.caption)
            self.flush()  # chunks never span pages, so every citation has an exact page
        return self.chunks


def chunk_document(
    doc: ParsedDocument,
    settings: Settings | None = None,
    summarizer: Summarizer | None = None,
) -> list[Chunk]:
    return _Chunker(doc, settings or get_settings(), summarizer).run()
