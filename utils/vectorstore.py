from __future__ import annotations

from typing import List, Tuple
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    def __init__(self, persist_dir: str, embedding: Embeddings):
        os.makedirs(persist_dir, exist_ok=True)
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="rag_collection",
        )

    def add(self, docs: List[Document]) -> None:
        self.store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> List[Document]:
        # Chroma supports filtering by metadata via 'filter' dict
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @staticmethod
    def purge_directory(persist_dir: str) -> bool:
        #Delete the Chroma persistence directory to remove previous session data.
        try:
            if os.path.isdir(persist_dir):
                # On Windows, files can be locked; try best-effort removal
                shutil.rmtree(persist_dir, ignore_errors=True)
            # Double-check and attempt secondary cleanup if remnants remain
            if os.path.isdir(persist_dir):
                for root, dirs, files in os.walk(persist_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(persist_dir)
                except Exception:
                    pass
            return not os.path.isdir(persist_dir)
        except Exception:
            return False
