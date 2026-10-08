"""OCR for scanned pages. RapidOCR (ONNX, CPU-only) is optional and loaded lazily."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from docplainer.ingestion import tables
from docplainer.schemas import Block


@dataclass
class OcrWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float


class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> list[OcrWord]: ...


class RapidOcrEngine:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # optional dependency

        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> list[OcrWord]:
        # Lower-than-default detection thresholds: the defaults silently dropped whole text
        # lines on JPEG-compressed scans in testing, and a missing line is worse than noise.
        result, _ = self._engine(image, box_thresh=0.3, unclip_ratio=2.0, text_score=0.3)
        words = []
        for box, text, conf in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            words.append(OcrWord(text, min(xs), min(ys), max(xs), max(ys), float(conf)))
        return words


def _ink_bands(image: np.ndarray) -> list[tuple[int, int]]:
    """Vertical bands (y0, y1) that contain dark pixels, i.e. probable text lines."""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    ink_rows = (gray < 140).sum(axis=1) > max(3, image.shape[1] * 0.002)
    bands, start = [], None
    for y, on in enumerate(ink_rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(ink_rows)))
    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    return merged


def recover_missed_lines(
    image: np.ndarray, words: list[OcrWord], engine: OcrEngine
) -> list[OcrWord]:
    """Re-read horizontal bands that contain ink but no OCR box.

    Text detectors occasionally skip an entire line; the page then silently loses a sentence.
    Cropping the uncovered band and reading it on its own recovers most of those lines.
    """
    if not words:
        return words
    heights = sorted(w.y1 - w.y0 for w in words)
    line_h = heights[len(heights) // 2]
    recovered: list[OcrWord] = []
    for y0, y1 in _ink_bands(image):
        if y1 - y0 < 0.4 * line_h:
            continue  # specks, rules and table borders
        spans = sorted((max(w.y0, y0), min(w.y1, y1)) for w in words if w.y1 > y0 and w.y0 < y1)
        covered_px, edge = 0.0, y0
        for a, b in spans:
            a = max(a, edge)
            if b > a:
                covered_px += b - a
                edge = b
        if covered_px >= 0.4 * (y1 - y0):
            continue  # tall bands are usually ruled tables whose rows are already read
        pad = max(6, int(0.35 * line_h))  # tight: wider crops pull in neighbouring lines
        top, bottom = max(0, y0 - pad), min(image.shape[0], y1 + pad)
        for w in engine.read(image[top:bottom]):
            # Re-centre at the typical line height so a crop never looks like a heading.
            cy = (w.y0 + w.y1) / 2 + top
            recovered.append(OcrWord(w.text, w.x0, cy - line_h / 2, w.x1, cy + line_h / 2, w.conf))
    return words + recovered


_engine: OcrEngine | None = None


def get_default_engine() -> OcrEngine | None:
    """Return the shared engine, or None if the ocr extra is not installed."""
    global _engine
    if _engine is None:
        try:
            _engine = RapidOcrEngine()
        except ImportError:
            return None
    return _engine


def words_to_lines(words: list[OcrWord]) -> list[list[OcrWord]]:
    """Group boxes into visual lines, top to bottom, left to right."""
    if not words:
        return []
    heights = sorted(w.y1 - w.y0 for w in words)
    line_h = heights[len(heights) // 2] or 10.0
    ordered = sorted(words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    lines: list[list[OcrWord]] = [[ordered[0]]]
    for w in ordered[1:]:
        last = lines[-1]
        cy_last = sum((x.y0 + x.y1) / 2 for x in last) / len(last)
        if abs((w.y0 + w.y1) / 2 - cy_last) < 0.5 * line_h:
            last.append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _join_line(line: list[OcrWord], char_w: float) -> tuple[str, int]:
    """Join a line's boxes; wide gaps become ' | ' so table rows keep their columns."""
    parts = [line[0].text]
    cells = 1
    for prev, cur in zip(line, line[1:], strict=False):
        if cur.x0 - prev.x1 > 1.2 * char_w:
            parts.append(" | ")
            cells += 1
        else:
            parts.append(" ")
        parts.append(cur.text)
    return "".join(parts), cells


def ocr_to_blocks(words: list[OcrWord]) -> list[Block]:
    """Turn OCR boxes into paragraph and table blocks (runs of columnar lines -> tables)."""
    lines = words_to_lines(words)
    if not lines:
        return []
    widths = [(w.x1 - w.x0) / max(len(w.text), 1) for w in words]
    char_w = sorted(widths)[len(widths) // 2]
    heights = sorted(w.y1 - w.y0 for w in words)
    line_h = heights[len(heights) // 2] or 10.0

    rendered = [_join_line(line, char_w) for line in lines]
    tops = [min(w.y0 for w in line) for line in lines]
    bottoms = [max(w.y1 for w in line) for line in lines]
    heights_by_line = [max(w.y1 - w.y0 for w in line) for line in lines]

    blocks: list[Block] = []
    para: list[str] = []
    run: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(Block("paragraph", " ".join(para)))
            para.clear()

    def flush_run() -> None:
        if not run:
            return
        grid = tables.clean_rows([r.split(" | ") for r in run])
        if len(run) >= 3 and tables.is_meaningful(grid):
            blocks.append(Block("table", tables.to_markdown(grid)))
        else:
            para.extend(run)
        run.clear()

    for i, (text, cells) in enumerate(rendered):
        if i and tops[i] - bottoms[i - 1] > 0.9 * line_h:
            flush_run()
            flush_para()
        ratio = heights_by_line[i] / line_h
        if cells == 1 and ratio >= 1.2 and len(text) <= 120 and not text.endswith("."):
            flush_run()
            flush_para()
            blocks.append(Block("heading", text, level=1 if ratio >= 1.6 else 2))
        elif cells >= 2:
            if not run:
                flush_para()
            run.append(text)
        else:
            flush_run()
            para.append(text)
    flush_run()
    flush_para()
    return blocks
