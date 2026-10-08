from __future__ import annotations

from docplainer.schemas import Chunk, Hit


def web_search(query: str, max_results: int = 5) -> list[Hit]:
    """DuckDuckGo snippets as Hits. Returns [] on any failure: the web is best-effort."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    hits = []
    for i, r in enumerate(results):
        body = (r.get("body") or "").strip()
        if not body:
            continue
        chunk = Chunk(
            id=f"web:{i}",
            doc_id="web",
            file_name=(r.get("title") or "web result").strip(),
            page=0,
            section="",
            kind="web",
            text=body,
            embed_text=body,
            url=r.get("href") or "",
        )
        hits.append(Hit(chunk=chunk, score=1.0, relevance=1.0))
    return hits
