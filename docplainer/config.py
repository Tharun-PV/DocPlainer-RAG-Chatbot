from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables, overridable with DOCPLAINER_* environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="DOCPLAINER_", env_file=".env", extra="ignore"
    )

    # --- models (all run locally on CPU) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    # "hashing" swaps in an offline embedder (tests, CI smoke runs, no downloads)
    embedder_backend: str = "fastembed"

    # --- ingestion ---
    max_upload_mb: int = 15
    max_pages: int = 40
    ocr_enabled: bool = True
    ocr_dpi: int = 170
    min_page_chars: int = 40
    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 150
    table_max_chars: int = 2200

    # --- retrieval ---
    dense_k: int = 30
    sparse_k: int = 30
    rerank_candidates: int = 20
    final_k: int = 6
    rrf_k: int = 60
    # Normalised relevance (0..1) below which the corpus is treated as "no answer".
    min_relevance: float = 0.25

    # --- generation / demo guard rails ---
    history_turns: int = 4
    demo_question_limit: int = 15
    demo_upload_limit: int = 3
    web_fallback: bool = True

    # API keys come from secrets/env; never committed.
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
