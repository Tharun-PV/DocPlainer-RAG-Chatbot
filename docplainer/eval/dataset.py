from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from docplainer.schemas import Chunk


@dataclass
class Gold:
    file: str
    page: int | None = None
    contains: str | None = None


@dataclass
class Question:
    id: str
    type: str  # text | table | scan | unanswerable
    question: str
    gold: list[Gold] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def load_golden(path: str | Path) -> list[Question]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            gold = [Gold(**g) for g in raw.pop("gold", [])]
            questions.append(Question(gold=gold, **raw))
    return questions


def _norm(text: str) -> str:
    # Whitespace-insensitive so OCR spacing noise does not decide whether a chunk "contains"
    # the answer, and so the metric does not depend on how a chunker lays out whitespace.
    return re.sub(r"\s+", "", text).lower()


def is_relevant(chunk: Chunk, gold: list[Gold]) -> bool:
    for g in gold:
        if chunk.file_name != g.file:
            continue
        if g.page is not None and chunk.page != g.page:
            continue
        if g.contains and _norm(g.contains) not in _norm(chunk.text):
            continue
        return True
    return False
