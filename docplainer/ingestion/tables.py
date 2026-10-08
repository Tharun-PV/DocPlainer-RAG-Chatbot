"""Table helpers shared by every parser: cleaning, Markdown rendering, summaries."""

from __future__ import annotations

import re
from collections.abc import Sequence

_WS = re.compile(r"\s+")


def _clean(cell: object) -> str:
    text = "" if cell is None else str(cell)
    return _WS.sub(" ", text).replace("|", "/").strip()


def clean_rows(rows: Sequence[Sequence[object]]) -> list[list[str]]:
    """Normalise cells, drop empty rows/columns and pad ragged rows."""
    grid = [[_clean(c) for c in row] for row in rows]
    grid = [row for row in grid if any(row)]
    if not grid:
        return []
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [i for i in range(width) if any(r[i] for r in grid)]
    return [[r[i] for i in keep] for r in grid]


def is_meaningful(grid: list[list[str]]) -> bool:
    """Reject layout artefacts: need >=2 rows, >=2 columns and some filled cells."""
    if len(grid) < 2 or len(grid[0]) < 2:
        return False
    cells = [c for r in grid for c in r]
    return sum(1 for c in cells if c) / len(cells) >= 0.4


def to_markdown(grid: list[list[str]]) -> str:
    if not grid:
        return ""
    header, *body = grid
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_markdown_table(text: str) -> list[list[str]]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} and c for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def describe(grid: list[list[str]], caption: str = "") -> str:
    """Deterministic one-line summary used to help retrieval find a table by its shape."""
    header = ", ".join(c for c in grid[0] if c)
    first_col = [r[0] for r in grid[1:] if r[0]]
    rows_hint = f" Rows include: {', '.join(first_col[:8])}." if first_col else ""
    cap = f"{caption}. " if caption else ""
    return f"{cap}Table with columns: {header}. {len(grid) - 1} data rows.{rows_hint}"
