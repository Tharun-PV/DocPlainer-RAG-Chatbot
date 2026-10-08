"""Fetch every model at image-build time so the first request never waits on a download."""

from docplainer.config import get_settings
from docplainer.retrieval import get_embedder, get_reranker

cfg = get_settings()
get_embedder(cfg).embed_query("warm up")
reranker = get_reranker(cfg)
if reranker:
    reranker.score("warm up", ["warm up"])
try:
    from docplainer.ingestion.ocr import get_default_engine

    get_default_engine()
except Exception as exc:  # OCR is optional
    print(f"OCR engine not available: {exc}")
print("models ready")
