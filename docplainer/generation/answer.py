from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from docplainer.config import Settings, get_settings
from docplainer.generation import prompts
from docplainer.generation.llm import LLMError, Messages
from docplainer.generation.web import web_search
from docplainer.retrieval import Retriever
from docplainer.schemas import Hit

Mode = Literal["documents", "web", "none", "chat"]
_CITE = re.compile(r"\[(\d+)\]")
_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|good (morning|evening)|bye)\b[\s!.?]*$", re.I
)


class ChatLLM(Protocol):
    def complete(self, messages: Messages, **kw) -> str: ...

    def stream(self, messages: Messages, **kw) -> Iterator[str]: ...


@dataclass
class Citation:
    n: int
    label: str  # "file.pdf p.3"
    snippet: str
    kind: str  # text | table | web
    url: str = ""
    ocr: bool = False


@dataclass
class Prepared:
    """Everything decided before the LLM speaks, so the UI can stream the answer itself."""

    mode: Mode
    hits: list[Hit] = field(default_factory=list)
    messages: Messages | None = None  # None: `reply` is the whole answer, no LLM call
    reply: str = ""
    query: str = ""
    trace: dict = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    mode: Mode
    citations: list[Citation]
    trace: dict = field(default_factory=dict)


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: ChatLLM,
        settings: Settings | None = None,
        search_web: Callable[[str], list[Hit]] = web_search,
    ):
        self.retriever, self.llm = retriever, llm
        self.cfg = settings or get_settings()
        self.search_web = search_web

    # --- steps -------------------------------------------------------------------------
    def standalone_query(self, question: str, history: list[dict[str, str]]) -> str:
        """Resolve follow-ups ("what about 2023?") into a self-contained search query."""
        if not history:
            return question
        convo = prompts.format_history(history, self.cfg.history_turns)
        try:
            out = self.llm.complete(
                [
                    {"role": "system", "content": prompts.REWRITE_PROMPT},
                    {"role": "user", "content": f"History:\n{convo}\n\nQuestion: {question}"},
                ],
                temperature=0.0,
                max_tokens=80,
            ).strip()
        except LLMError:
            return question
        return out.splitlines()[0].strip(' "') if out else question

    def prepare(
        self,
        question: str,
        history: list[dict[str, str]],
        allow_web: bool = True,
        style: str = "concise",
        doc_ids: set[str] | None = None,
    ) -> Prepared:
        if _SMALL_TALK.match(question):
            return Prepared("chat", reply="Hi! Ask me anything about your documents.")

        query = self.standalone_query(question, history)
        trace: dict = {"query": query}
        has_docs = len(self.retriever.index) > 0
        result = self.retriever.retrieve(query, doc_ids) if has_docs else None
        if result is not None:
            trace |= result.trace

        if result is not None and result.confident:
            mode: Mode = "documents"
            hits, intro = result.hits, ""
        else:
            hits = self.search_web(query) if (allow_web and self.cfg.web_fallback) else []
            if hits:
                mode, intro = "web", prompts.WEB_NOTE
            else:
                why = (
                    "I couldn't find that in your documents."
                    if has_docs
                    else "Upload a document first, and I'll answer from it."
                )
                return Prepared("none", reply=why, query=query, trace=trace)

        system = prompts.SYSTEM_PROMPT.format(style=prompts.STYLES.get(style, ""))
        convo = prompts.format_history(history, self.cfg.history_turns)
        user = (
            (f"{intro}\n\n" if intro else "")
            + (f"Conversation so far:\n{convo}\n\n" if convo else "")
            + f"Context passages:\n{prompts.build_context(hits)}\n\nQuestion: {question}"
        )
        return Prepared(
            mode,
            hits,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            query=query,
            trace=trace,
        )

    @staticmethod
    def finalize(prepared: Prepared, text: str) -> Answer:
        """Attach only the passages the model actually cited (all of them if it cited none)."""
        if prepared.messages is None:
            return Answer(prepared.reply, prepared.mode, [], prepared.trace)
        cited = sorted({int(n) for n in _CITE.findall(text) if 0 < int(n) <= len(prepared.hits)})
        numbers = cited or list(range(1, len(prepared.hits) + 1))
        citations = []
        for n in numbers:
            c = prepared.hits[n - 1].chunk
            citations.append(Citation(n, c.label(), c.text[:600], c.kind, url=c.url, ocr=c.ocr))
        return Answer(text, prepared.mode, citations, prepared.trace)

    # --- convenience -------------------------------------------------------------------
    def answer(self, question: str, history: list[dict[str, str]] | None = None, **kw) -> Answer:
        prepared = self.prepare(question, history or [], **kw)
        if prepared.messages is None:
            return self.finalize(prepared, prepared.reply)
        return self.finalize(prepared, self.llm.complete(prepared.messages))
