"""PDF parsing with per-page triage.

Each page takes the cheapest path that works:
  digital text   -> PyMuPDF text blocks, with headings inferred from font size
  ruled tables   -> PyMuPDF table finder, rendered as Markdown (table area removed from text)
  scanned / image-only page -> render and OCR (tables inferred from column gaps)
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable

import numpy as np
import pymupdf as fitz

from docplainer.config import Settings, get_settings
from docplainer.ingestion import tables
from docplainer.ingestion.ocr import (
    OcrEngine,
    get_default_engine,
    ocr_to_blocks,
    recover_missed_lines,
)
from docplainer.schemas import Block, ParsedDocument, ParsedPage

Progress = Callable[[int, int, str], None]
_CAPTION = re.compile(r"^(table|exhibit|figure)\s*[\dA-Za-z.\-]*[:.\s]", re.I)


def _inside(bbox: tuple[float, ...], regions: list[fitz.Rect]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in regions)


def _body_font_size(doc: fitz.Document, pages: int) -> float:
    """Median span size weighted by character count: the 'normal' text size."""
    sizes: list[float] = []
    for i in range(min(pages, 12)):
        for b in doc[i].get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for span in line["spans"]:
                    sizes.extend([round(span["size"], 1)] * len(span["text"].strip()))
    return statistics.median(sizes) if sizes else 11.0


def _text_blocks(page: fitz.Page, body: float, skip: list[fitz.Rect]) -> list[tuple[float, Block]]:
    out: list[tuple[float, Block]] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0 or _inside(b["bbox"], skip):
            continue
        lines = []
        size = 0.0
        bold = False
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                lines.append(text)
                size = max(size, max(s["size"] for s in line["spans"]))
                bold = bold or any(s["flags"] & 16 for s in line["spans"])
        if not lines:
            continue
        text = " ".join(lines)
        ratio = size / body if body else 1.0
        short = len(text) <= 120 and len(lines) <= 2
        if short and (ratio >= 1.15 or (bold and ratio >= 0.98 and not text.endswith("."))):
            level = 1 if ratio >= 1.5 else 2 if ratio >= 1.25 else 3
            out.append((b["bbox"][1], Block("heading", text, level=level)))
        else:
            out.append((b["bbox"][1], Block("paragraph", text)))
    return out


def _table_blocks(page: fitz.Page) -> tuple[list[tuple[float, Block]], list[fitz.Rect]]:
    found: list[tuple[float, Block]] = []
    regions: list[fitz.Rect] = []
    try:
        finder = page.find_tables()
    except Exception:  # table finder can fail on malformed pages; fall back to plain text
        return found, regions
    for t in finder.tables:
        grid = tables.clean_rows(t.extract())
        if not tables.is_meaningful(grid):
            continue
        rect = fitz.Rect(t.bbox)
        regions.append(rect)
        found.append((rect.y0, Block("table", tables.to_markdown(grid))))
    return found, regions


def _attach_captions(items: list[tuple[float, Block]]) -> list[Block]:
    items.sort(key=lambda it: it[0])
    blocks: list[Block] = []
    for _, block in items:
        if block.kind == "table" and blocks:
            prev = blocks[-1]
            if prev.kind == "paragraph" and len(prev.text) < 160 and _CAPTION.match(prev.text):
                block.caption = blocks.pop().text
        blocks.append(block)
    return blocks


def _render(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def parse_pdf(
    path: str,
    doc_id: str,
    file_name: str,
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    progress: Progress | None = None,
) -> ParsedDocument:
    cfg = settings or get_settings()
    parsed = ParsedDocument(doc_id=doc_id, file_name=file_name)
    doc = fitz.open(path)
    try:
        if doc.needs_pass:
            parsed.warnings.append("PDF is password protected and could not be read.")
            return parsed
        total = min(doc.page_count, cfg.max_pages)
        if doc.page_count > cfg.max_pages:
            parsed.warnings.append(
                f"Only the first {cfg.max_pages} of {doc.page_count} pages were processed."
            )
        body = _body_font_size(doc, total)
        for i in range(total):
            page = doc[i]
            if progress:
                progress(i + 1, total, f"{file_name}: page {i + 1}/{total}")
            tbl_items, regions = _table_blocks(page)
            txt_items = _text_blocks(page, body, regions)
            chars = sum(len(b.text) for _, b in txt_items)
            if chars < cfg.min_page_chars and not tbl_items:
                engine = ocr_engine or (get_default_engine() if cfg.ocr_enabled else None)
                parsed.pages.append(_ocr_page(page, i + 1, cfg, engine, file_name))
            else:
                parsed.pages.append(
                    ParsedPage(number=i + 1, blocks=_attach_captions(tbl_items + txt_items))
                )
    finally:
        doc.close()
    return parsed


def _ocr_page(
    page: fitz.Page, number: int, cfg: Settings, engine: OcrEngine | None, file_name: str
) -> ParsedPage:
    result = ParsedPage(number=number, ocr=True)
    if engine is None:
        result.ocr = False
        result.warnings.append(
            f"Page {number} of {file_name} has no text layer and OCR is unavailable."
        )
        return result
    image = _render(page, cfg.ocr_dpi)
    words = recover_missed_lines(image, engine.read(image), engine)
    if not words:
        result.warnings.append(f"Page {number} of {file_name}: OCR found no text.")
        return result
    result.ocr_confidence = float(np.mean([w.conf for w in words]))
    result.blocks = ocr_to_blocks(words)
    if result.ocr_confidence < 0.7:
        result.warnings.append(
            f"Page {number} of {file_name}: low OCR confidence ({result.ocr_confidence:.0%}); "
            "answers from this page may be unreliable."
        )
    return result
