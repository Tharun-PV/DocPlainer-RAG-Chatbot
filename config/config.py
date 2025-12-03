import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    # LLM provider keys
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Default models (can be overridden in UI)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # Gemini models often require the "-latest" suffix in current APIs
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")

    # Embeddings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chroma
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")

    # Chunking (moved from UI)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Conversational retrieval toggle
    conversational_retrieval_enabled: bool = os.getenv(
        "CONV_RETRIEVAL", "true").lower() in ("1", "true", "yes")
    history_char_limit: int = int(os.getenv("HISTORY_CHAR_LIMIT", "800"))


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def set_env_if_provided(key: str, value: Optional[str]):
    if value:
        os.environ[key] = value
