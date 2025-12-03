from typing import List, Dict
try:
    from ddgs import DDGS  # type: ignore
except Exception:  # fallback if only old package exists
    from duckduckgo_search import DDGS  # type: ignore


def web_search_snippets(query: str, max_results: int = 5) -> List[Dict]:
    results: List[Dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r)
    return results
