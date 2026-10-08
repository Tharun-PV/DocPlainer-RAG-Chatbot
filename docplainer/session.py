"""Per-visitor state: the document index, usage limits and the answer service.

Kept free of Streamlit so it can be unit-tested and reused behind an API later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docplainer.config import Settings, get_settings
from docplainer.generation import AnswerService
from docplainer.generation.answer import ChatLLM
from docplainer.ingestion import IngestResult, ingest_files
from docplainer.ingestion.pdf import Progress
from docplainer.retrieval import HybridIndex, Retriever, get_embedder, get_reranker


class LimitReached(Exception):
    pass


@dataclass
class RagSession:
    settings: Settings = field(default_factory=get_settings)
    demo_mode: bool = True  # True: shared demo keys with caps. False: visitor's own keys.
    questions_used: int = 0
    uploads_used: int = 0
    docs: dict[str, str] = field(default_factory=dict)  # file name -> doc id

    def __post_init__(self) -> None:
        self.index = HybridIndex(get_embedder(self.settings))
        self.retriever = Retriever(self.index, get_reranker(self.settings), self.settings)

    # --- limits ------------------------------------------------------------------------
    @property
    def questions_left(self) -> int | None:
        if not self.demo_mode:
            return None
        return max(self.settings.demo_question_limit - self.questions_used, 0)

    def check_can_ask(self) -> None:
        if self.questions_left == 0:
            raise LimitReached(
                f"Demo limit of {self.settings.demo_question_limit} questions reached. "
                "Add your own free Groq or Gemini key in the sidebar to keep going."
            )

    # --- documents ---------------------------------------------------------------------
    def add_files(
        self, paths: list[Path], progress: Progress | None = None, count_upload: bool = True
    ) -> tuple[IngestResult, int]:
        """Ingest and index files. Returns (ingest result, chunks newly added)."""
        if (
            count_upload
            and self.demo_mode
            and self.uploads_used + len(paths) > self.settings.demo_upload_limit
        ):
            raise LimitReached(
                f"Demo limit of {self.settings.demo_upload_limit} uploads reached. "
                "Use your own API key for unlimited uploads, or try the sample documents."
            )
        result = ingest_files(list(paths), self.settings, progress=progress)
        added = self.index.add(result.chunks)
        self.docs.update(result.doc_ids)
        if count_upload:
            self.uploads_used += len(result.doc_ids)
        return result, added

    def remove(self, file_name: str) -> None:
        doc_id = self.docs.pop(file_name, None)
        if doc_id:
            self.index.remove_doc(doc_id)

    # --- answering ---------------------------------------------------------------------
    def service(self, llm: ChatLLM) -> AnswerService:
        return AnswerService(self.retriever, llm, self.settings)

    def selected_doc_ids(self, names: list[str] | None) -> set[str] | None:
        if not names or set(names) == set(self.docs):
            return None
        return {self.docs[n] for n in names if n in self.docs}
