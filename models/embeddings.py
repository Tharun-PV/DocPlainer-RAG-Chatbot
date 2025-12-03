from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings


_embedder: Optional[HuggingFaceEmbeddings] = None


def get_hf_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder
