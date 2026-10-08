from pathlib import Path

import pytest

from docplainer.config import Settings

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def cfg() -> Settings:
    return Settings(embedder_backend="hashing", use_reranker=False)


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE


def have_ocr() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True
