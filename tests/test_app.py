"""End-to-end UI smoke test using Streamlit's AppTest (no network, no model downloads)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeChain:
    def complete(self, messages, **kw):
        return "standalone question"

    def stream(self, messages, **kw):
        yield from ("Europe made ", "141.9 million EUR ", "[1].")


@pytest.fixture
def app(monkeypatch):
    from docplainer.config import get_settings

    monkeypatch.setenv("DOCPLAINER_EMBEDDER_BACKEND", "hashing")
    monkeypatch.setenv("DOCPLAINER_MIN_DENSE_COSINE", "0.05")
    monkeypatch.setenv("DOCPLAINER_WEB_FALLBACK", "false")
    get_settings.cache_clear()
    monkeypatch.setattr("docplainer.generation.build_chain", lambda keys, cfg=None: FakeChain())
    at = AppTest.from_file(APP, default_timeout=90).run()
    yield at
    get_settings.cache_clear()


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_sample_docs_then_cited_answer(app):
    assert not app.exception
    app = click(app, "Load sample documents")
    assert not app.exception
    assert set(app.session_state.rag.docs) == {
        "northwind_handbook.pdf",
        "northwind_annual_report.pdf",
        "northwind_scanned_memo.pdf",
    }
    app.chat_input[0].set_value("What was Europe's revenue in FY2024?").run()
    assert not app.exception
    reply = app.chat_message[-1]
    assert "141.9" in reply.markdown[-1].value or any("141.9" in m.value for m in reply.markdown)
    assert app.session_state.rag.questions_used == 1
    assert app.session_state.messages[-1]["citations"][0]["label"].startswith("northwind_")


def test_question_without_documents_asks_for_upload(app):
    app.chat_input[0].set_value("hello there, what is in my files?").run()
    assert not app.exception
    assert any("Upload a document" in m.value for m in app.chat_message[-1].markdown)


def test_demo_limit_blocks_further_questions(app):
    rag = app.session_state.rag
    rag.questions_used = rag.settings.demo_question_limit
    app.chat_input[0].set_value("anything").run()
    assert any("Demo limit" in w.value for w in app.warning)
