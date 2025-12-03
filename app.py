import streamlit as st
import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from config.config import load_config, set_env_if_provided, AppConfig
from models.llm import get_llm
from models.embeddings import get_hf_embeddings
from utils.doc_loaders import load_documents, SUPPORTED_EXTS
from utils.text_split import split_documents
from utils.vectorstore import VectorStoreManager
from utils.websearch import web_search_snippets


def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"


def _build_retrieval_query(current_question: str, history: List[dict], char_limit: int = 800) -> tuple[str, bool]:

    pieces: List[str] = []
    total = 0
    # Walk history from latest to oldest, keep only user messages, stop when limit reached
    for msg in reversed(history):
        role = msg.get("role", "")
        if role != "user":
            continue
        content = msg.get("content", "") or ""
        if not content:
            continue
        line = f"user: {content}"
        if total + len(line) > char_limit:
            break
        pieces.append(line)
        total += len(line)
    used_history = len(pieces) > 0
    # Put oldest-first order for readability
    pieces = list(reversed(pieces))
    combined = "\n".join(
        pieces + [f"question: {current_question}"]) if pieces else current_question
    return combined, used_history


def _build_history_context(history: List[dict], char_limit: int = 800) -> tuple[str, bool]:

    pieces: List[str] = []
    total = 0
    for msg in reversed(history):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role not in ("user", "assistant") or not content:
            continue
        line = f"{role}: {content}"
        if total + len(line) > char_limit:
            break
        pieces.append(line)
        total += len(line)
    used = len(pieces) > 0
    pieces = list(reversed(pieces))
    return "\n".join(pieces), used


def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown(
        "Welcome! Follow these instructions to set up and use the chatbot.")

    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Upload documents** on the Chat page and build the vector store
    3. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "vector" not in st.session_state:
        st.session_state.vector = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "provider" not in st.session_state:
        st.session_state.provider = "Groq"
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "OpenAI": None, "Groq": None, "Gemini": None}
    # no file restriction; search across all docs


def _build_sidebar(cfg: AppConfig):
    st.subheader("Model & Keys")
    provider = st.selectbox(
        "LLM Provider", ["OpenAI", "Groq", "Gemini"], index=1)

    # Default model per provider
    default_model = (
        cfg.openai_model if provider == "OpenAI" else cfg.groq_model if provider == "Groq" else cfg.google_model
    )
    model_name = st.text_input("Model name", value=default_model)

    # API key handling (per-session, not global env)
    current_key = st.session_state.api_keys.get(provider)
    key_input = st.text_input(
        f"Enter {provider} API Key", value=current_key or "", type="password")
    if key_input and key_input != current_key:
        st.session_state.api_keys[provider] = key_input
        current_key = key_input
        st.success(f"{provider} API key stored for this session")

    st.session_state.provider = provider
    st.session_state.model_name = model_name

    # Removed chunking/embeddings/store controls from UI; using config.
    persist_dir = cfg.chroma_dir
    emb_model = cfg.embedding_model_name
    chunk_size = cfg.chunk_size
    chunk_overlap = cfg.chunk_overlap

    st.divider()
    st.subheader("Conversational Retrieval")
    conv_enabled = st.checkbox(
        "Use chat history in retrieval", value=cfg.conversational_retrieval_enabled)
    history_chars = cfg.history_char_limit

    return {
        "api_key": current_key,
        "model_name": model_name,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "emb_model": emb_model,
        "persist_dir": persist_dir,
        "history_chars": int(history_chars),
        "conv_enabled": conv_enabled,
    }


def _render_sources(sources: List[Tuple[str, str]]):
    if not sources:
        return
    # Deduplicate to avoid repeated lines for same file/page
    unique = []
    seen = set()
    for label, detail in sources:
        key = (label, detail)
        if key not in seen:
            seen.add(key)
            unique.append((label, detail))
    st.markdown("\n---\n")
    st.caption("Sources:")
    for label, detail in unique:
        st.caption(f"- {label}: {detail}")


def _ensure_vectorstore(emb_model: str, persist_dir: str):
    if st.session_state.embedder is None:
        st.session_state.embedder = get_hf_embeddings(emb_model)
    if st.session_state.vector is None:
        st.session_state.vector = VectorStoreManager(
            persist_dir, st.session_state.embedder)


def chat_page():
    st.title("🤖 RAG Chatbot")

    _init_state()
    cfg: AppConfig = st.session_state.config

    with st.sidebar:
        sidebar = _build_sidebar(cfg)

        st.divider()
        st.subheader("Upload Documents")
        uploaded = st.file_uploader(
            "Upload files (pdf, txt, md, pptx, docx)",
            type=[e.lstrip(".") for e in SUPPORTED_EXTS],
            accept_multiple_files=True,
        )
        if uploaded:
            temp_dir = Path("./uploaded_files")
            temp_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: List[Path] = []
            for uf in uploaded:
                file_path = temp_dir / uf.name
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_paths.append(file_path)

            if st.button("Process documents", use_container_width=True):
                with st.spinner("Loading, splitting, and indexing documents..."):
                    _ensure_vectorstore(
                        sidebar["emb_model"], sidebar["persist_dir"])
                    docs = load_documents(saved_paths)
                    # Filter out empty documents before splitting
                    docs = [d for d in docs if (d.page_content or "").strip()]
                    chunks = split_documents(
                        docs, sidebar["chunk_size"], sidebar["chunk_overlap"])
                    # Guard against empty chunks which cause Chroma upsert errors
                    chunks = [c for c in chunks if (
                        c.page_content or "").strip()]
                    if chunks:
                        st.session_state.vector.add(chunks)
                    else:
                        st.warning(
                            "No valid text content found to index. Please upload documents with extractable text.")
                st.success(
                    f"Indexed {len(chunks)} chunks from {len(saved_paths)} file(s)")
                # Track uploaded file names for filtering
                st.session_state.uploaded_files = [p.name for p in saved_paths]

        # Full purge: delete Chroma directory on disk
        if st.button("Purge Chroma Data (Disk)", use_container_width=True):
            # Release any in-memory store before deleting on disk
            st.session_state.vector = None
            ok = VectorStoreManager.purge_directory(sidebar["persist_dir"])
            if ok:
                st.success(
                    "Chroma data purged from disk. Upload and process files to start fresh.")
            else:
                st.warning(
                    "Could not fully purge Chroma directory. Close running sessions and try again.")

    # Build LLM
    api_key = sidebar["api_key"]
    provider = st.session_state.provider
    model_name = st.session_state.model_name
    chat_model = None
    if api_key and model_name:
        try:
            chat_model = get_llm(provider, model_name, api_key)
        except Exception as e:
            st.error(f"LLM init error: {e}")

        # Response style is controlled from sidebar; initialize default here
        if "response_style" not in st.session_state:
            st.session_state.response_style = "concise"

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    system_prompt = (
        "You are a Document Explainer, Analyzer, and Summarizer. Answer ONLY using the provided context. "
        "Do NOT add external citations or URLs in your answer text. "
        "The app will show sources below: if context comes from documents, cite file name and page; "
        "if web snippets were used, it will show 'web search'. If context is insufficient, say so succinctly."
    )

    user_input = st.chat_input(
        "Ask a question about your documents or the web...")
    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not chat_model:
                st.info("Provide a valid API key and model in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    context_blocks: List[str] = []
                    sources: List[Tuple[str, str]] = []

                    # Try vector retrieval if available
                    used_web = False
                    vector_hits = 0
                    retrieval_query, used_history = _build_retrieval_query(
                        user_input, st.session_state.messages, char_limit=sidebar["history_chars"])
                    if not sidebar["conv_enabled"]:
                        retrieval_query = user_input
                        used_history = False
                    if st.session_state.vector is not None:
                        # Retrieve top-k; only fallback to web if zero results
                        vec_scored = st.session_state.vector.similarity_search_with_scores(
                            retrieval_query, k=4)
                        vector_hits = len(vec_scored)
                        for doc, score in vec_scored:
                            meta = doc.metadata or {}
                            page = meta.get("page")
                            file_name = meta.get("file_name", "")
                            context_blocks.append(doc.page_content)
                            if file_name:
                                label = "vector db"
                                detail = f"{file_name}{' p.' + str(page) if page else ''}"
                                sources.append((label, detail))
                        if not vec_scored:
                            used_web = True
                    else:
                        used_web = True

                    web_hits = 0
                    if used_web:
                        search_hits = web_search_snippets(
                            retrieval_query, max_results=5)
                        web_hits = len(search_hits)
                        for hit in search_hits:
                            snippet = hit.get("body") or hit.get("title") or ""
                            url = hit.get("href") or ""
                            if snippet:
                                context_blocks.append(
                                    f"Snippet: {snippet}\nURL: {url}")
                        if search_hits:
                            sources.append(("web search", "duckduckgo"))

                    # (debug info shown after response)

                    context_text = "\n\n".join(context_blocks[:8])
                    history_context_text = ""
                    used_history_in_context = False
                    if sidebar["conv_enabled"]:
                        history_context_text, used_history_in_context = _build_history_context(
                            st.session_state.messages, char_limit=sidebar["history_chars"]
                        )
                    style_instruction = (
                        "Respond concisely in 2-4 sentences." if st.session_state.response_style == "concise" else
                        "Provide a detailed answer with clear structure, but avoid external URLs in the text."
                    )
                    composed_parts = [
                        "Use ONLY this context to answer. If insufficient, say so.",
                    ]
                    if history_context_text:
                        composed_parts.append(
                            f"Conversation (recent):\n{history_context_text}")
                    if context_text:
                        composed_parts.append(
                            f"Retrieved Context:\n{context_text}")
                    composed_parts.append(f"Question: {user_input}")
                    composed_parts.append(f"Style: {style_instruction}")
                    composed = "\n\n".join(composed_parts)

                    response = get_chat_response(
                        chat_model,
                        st.session_state.messages +
                        [{"role": "user", "content": composed}],
                        system_prompt,
                    )

                    # If model indicates insufficient context, perform web fallback and retry
                    response_text = response if isinstance(
                        response, str) else str(response)
                    did_web_retry = False
                    if "insufficient context" in response_text.lower() and not used_web:
                        web_hits_retry = web_search_snippets(
                            retrieval_query, max_results=5)
                        if web_hits_retry:
                            did_web_retry = True
                            sources.append(("web search", "duckduckgo"))
                            web_blocks = []
                            for hit in web_hits_retry:
                                snippet = hit.get(
                                    "body") or hit.get("title") or ""
                                url = hit.get("href") or ""
                                if snippet:
                                    web_blocks.append(
                                        f"Snippet: {snippet}\nURL: {url}")
                            context_text_web = "\n\n".join(web_blocks[:6])
                            composed_web_parts = [
                                "Use ONLY this context to answer. If insufficient, say so.",
                            ]
                            if history_context_text:
                                composed_web_parts.append(
                                    f"Conversation (recent):\n{history_context_text}")
                            if context_text:
                                composed_web_parts.append(
                                    f"Retrieved Context:\n{context_text}")
                            composed_web_parts.append(
                                f"Web Context:\n{context_text_web}")
                            composed_web_parts.append(
                                f"Question: {user_input}")
                            composed_web_parts.append(
                                f"Style: {style_instruction}")
                            composed_web = "\n\n".join(composed_web_parts)
                            response = get_chat_response(
                                chat_model,
                                st.session_state.messages +
                                [{"role": "user", "content": composed_web}],
                                system_prompt,
                            )
                            response_text = response if isinstance(
                                response, str) else str(response)

                    st.markdown(response_text)
                    # Prefer web sources when web fallback or retry happened; otherwise show vector sources
                    final_sources = (
                        [s for s in sources if s[0] == "web search"]
                        if (used_web or did_web_retry)
                        else [s for s in sources if s[0] == "vector db"]
                    )
                    _render_sources(final_sources)
                    with st.expander("Retrieval info", expanded=False):
                        st.write({
                            "vector_hits": vector_hits,
                            "used_web_fallback": used_web or did_web_retry,
                            "web_hits": web_hits if not did_web_retry else len(web_hits_retry),
                            "used_history_in_retrieval": used_history,
                            "used_history_in_prompt": used_history_in_context,
                            "history_char_limit": sidebar["history_chars"],
                        })

                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )

        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            # Response style controls (sidebar)
            if "response_style" not in st.session_state:
                st.session_state.response_style = "concise"
            s1, s2 = st.columns(2)
            with s1:
                if st.button("Concise", use_container_width=True):
                    st.session_state.response_style = "concise"
            with s2:
                if st.button("Detailed", use_container_width=True):
                    st.session_state.response_style = "detailed"
            st.caption(f"Style: {st.session_state.response_style}")

            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
