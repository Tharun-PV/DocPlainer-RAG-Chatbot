from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docplainer.config import Settings, get_settings
from docplainer.ingestion.chunking import Summarizer, chunk_document
from docplainer.ingestion.ocr import OcrEngine
from docplainer.ingestion.parsers import parse_file
from docplainer.ingestion.pdf import Progress
from docplainer.schemas import Chunk


@dataclass
class IngestResult:
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # file name -> reason
    doc_ids: dict[str, str] = field(default_factory=dict)  # file name -> doc id


def ingest_files(
    paths: list[str | Path],
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    progress: Progress | None = None,
    summarizer: Summarizer | None = None,
) -> IngestResult:
    """Parse and chunk files. One bad file never aborts the rest."""
    cfg = settings or get_settings()
    result = IngestResult()
    for raw in paths:
        path = Path(raw)
        try:
            doc = parse_file(path, cfg, ocr_engine, progress)
            chunks = chunk_document(doc, cfg, summarizer)
        except Exception as exc:  # surface per-file failures in the UI instead of crashing
            result.errors[path.name] = str(exc)
            continue
        result.warnings += doc.warnings + [w for p in doc.pages for w in p.warnings]
        if not chunks:
            result.errors[path.name] = "No extractable text found."
            continue
        result.chunks += chunks
        result.doc_ids[doc.file_name] = doc.doc_id
    return result
