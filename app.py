"""DocPlainer: chat with your documents (Streamlit UI over the docplainer package)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from docplainer.config import get_settings
from docplainer.generation import LLMError, build_chain
from docplainer.ingestion import SUPPORTED_EXTS
from docplainer.session import LimitReached, RagSession

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"
EXAMPLES = [
    "How many days of paid annual leave do employees get?",
    "What was Europe's revenue in FY2024?",
    "What was the total cost of the Hamburg pilot?",
    "What is the capital city of Australia?",
]
MODE_BADGE = {
    "documents": ":green[From your documents]",
    "web": ":orange[From a web search, not your documents]",
    "none": ":red[Not found]",
    "chat": "",
}

st.set_page_config(page_title="DocPlainer", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def warm_models() -> bool:
    """Load the embedder and reranker once per process, shared by all visitor sessions."""
    from docplainer.retrieval import get_embedder, get_reranker

    cfg = get_settings()
    get_embedder(cfg)
    get_reranker(cfg)
    return True


def session() -> RagSession:
    if "rag" not in st.session_state:
        warm_models()
        st.session_state.rag = RagSession(get_settings())
        st.session_state.messages = []
    return st.session_state.rag


def llm_keys(rag: RagSession) -> dict[str, str | None]:
    cfg = rag.settings
    if rag.demo_mode:
        return {
            "groq": cfg.groq_api_key,
            "gemini": cfg.google_api_key,
            "nvidia": cfg.nvidia_api_key,
            "openrouter": cfg.openrouter_api_key,
        }
    return st.session_state.get("byok", {})


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            badge = " · scanned page (OCR)" if c["ocr"] else ""
            title = f"**[{c['n']}] {c['label']}**{badge}"
            if c["url"]:
                title = f"**[{c['n']}] [{c['label']}]({c['url']})**"
            st.markdown(title)
            if c["kind"] == "table":
                st.markdown(c["snippet"])  # tables render as real tables
            else:
                st.caption(c["snippet"])


def sidebar(rag: RagSession) -> dict:
    with st.sidebar:
        st.header("Documents")
        if st.button(
            "Load sample documents", use_container_width=True, disabled=not SAMPLE_DIR.exists()
        ):
            files = sorted(SAMPLE_DIR.glob("*.pdf"))
            with st.status("Indexing sample documents...", expanded=True) as status:
                result, _ = rag.add_files(
                    files, progress=lambda i, n, m: st.write(m), count_upload=False
                )
                status.update(
                    label=f"Indexed {len(result.doc_ids)} sample documents", state="complete"
                )

        uploads = st.file_uploader(
            "Upload PDF, DOCX, PPTX, TXT or MD",
            type=[e.lstrip(".") for e in SUPPORTED_EXTS],
            accept_multiple_files=True,
            help=f"Up to {rag.settings.max_upload_mb} MB and {rag.settings.max_pages} pages each. "
            "Scanned PDFs are read with OCR.",
        )
        if uploads and st.button("Index uploaded files", type="primary", use_container_width=True):
            new = [u for u in uploads if u.name not in rag.docs]
            if not new:
                st.info("These files are already indexed.")
            else:
                index_uploads(rag, new)

        if rag.docs:
            st.caption("Indexed documents")
            for name in list(rag.docs):
                left, right = st.columns([5, 1])
                left.caption(f"📄 {name}")
                if right.button("", icon=":material/close:", key=f"rm-{name}", help="Remove"):
                    rag.remove(name)
                    st.rerun()
        selected = (
            st.multiselect("Search in", list(rag.docs), default=list(rag.docs))
            if len(rag.docs) > 1
            else list(rag.docs)
        )

        st.divider()
        st.header("Answers")
        style = st.radio("Style", ["concise", "detailed"], horizontal=True)
        allow_web = st.toggle(
            "Search the web when documents don't cover it",
            value=rag.settings.web_fallback,
            help="Web answers are always labelled as such.",
        )

        st.divider()
        st.header("API key")
        own = st.toggle("Use my own free key", value=not rag.demo_mode)
        rag.demo_mode = not own
        if own:
            st.caption("Held in this browser session only. Never stored or logged.")
            st.session_state.byok = {
                "groq": st.text_input("Groq key", type="password", key="k_groq") or None,
                "gemini": st.text_input("Gemini key", type="password", key="k_gem") or None,
                "nvidia": st.text_input("NVIDIA key", type="password", key="k_nv") or None,
                "openrouter": st.text_input("OpenRouter key", type="password", key="k_or") or None,
            }
        elif rag.questions_left is not None:
            st.caption(f"Demo questions left: **{rag.questions_left}**")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return {"style": style, "allow_web": allow_web, "doc_ids": rag.selected_doc_ids(selected)}


def index_uploads(rag: RagSession, uploads: list) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for u in uploads:
            path = Path(tmp) / Path(u.name).name
            path.write_bytes(u.getbuffer())
            paths.append(path)
        bar = st.progress(0.0, text="Starting...")

        def progress(done: int, total: int, msg: str) -> None:
            bar.progress(done / total, text=msg)

        try:
            result, added = rag.add_files(paths, progress=progress)
        except LimitReached as exc:
            bar.empty()
            st.error(str(exc))
            return
        bar.empty()
    for name, reason in result.errors.items():
        st.error(f"{name}: {reason}")
    for warning in dict.fromkeys(result.warnings):
        st.warning(warning)
    if added:
        st.success(f"Indexed {added} passages from {len(result.doc_ids)} file(s).")


def main() -> None:
    rag = session()
    opts = sidebar(rag)

    st.title("📄 DocPlainer")
    st.caption("Ask questions about your documents. Answers cite the page they came from.")

    if not st.session_state.messages and not rag.docs:
        st.info("Load the sample documents or upload your own in the sidebar to get started.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                st.caption(MODE_BADGE.get(m["mode"], ""))
            st.markdown(m["content"])
            render_citations(m.get("citations", []))

    question = st.chat_input("Ask about your documents...")
    if not question and rag.docs and not st.session_state.messages:
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES, strict=False):
            if col.button(ex, key=f"ex-{ex}"):
                question = ex
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            rag.check_can_ask()
            chain = build_chain(llm_keys(rag), rag.settings)
        except LimitReached as exc:
            st.warning(str(exc))
            return
        except ValueError:
            st.warning("No API key available. Add a free Groq or Gemini key in the sidebar.")
            return

        service = rag.service(chain)
        try:
            with st.spinner("Searching your documents..."):
                prepared = service.prepare(
                    question, history, opts["allow_web"], opts["style"], opts["doc_ids"]
                )
            st.caption(MODE_BADGE.get(prepared.mode, ""))
            if prepared.messages is None:
                text = prepared.reply
                st.markdown(text)
            else:
                text = st.write_stream(chain.stream(prepared.messages))
        except LLMError as exc:
            st.error(f"All free model providers are busy or rejected the request. {exc}")
            return
        answer = service.finalize(prepared, text)
        if prepared.mode != "chat":
            rag.questions_used += 1
        citations = [c.__dict__ for c in answer.citations]
        render_citations(citations)
        with st.expander("How this answer was found"):
            st.json(answer.trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "mode": answer.mode, "citations": citations}
    )


main()
