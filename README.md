# DocPlainer RAG Chatbot (Streamlit)

This is DocPlainer (Document Explainer) Retrieval‑Augmented Chatbot. Upload documents, chat naturally, and get answers with source citations. When your files don’t have what you need, it can fall back to a quick web search.

## Highlights

- Uploads: `PDF`, `DOCX`, `PPTX`, `TXT`, `MD`
- Chunking via `RecursiveCharacterTextSplitter`
- Embeddings: Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
- Vector DB: Chroma (persistent to disk)
- LLMs: OpenAI, Groq, Google Gemini (pick in the UI)
- Conversational retrieval: optionally use recent chat turns in retrieval and prompts
- Web fallback: DuckDuckGo snippets when no vector hits
- Source citations: file name + page/slide or “web search”
- Sidebar response style: Concise or Detailed
- Debug expander: shows vector hits, web fallback, and history usage
- Purge button: clear the on‑disk Chroma cache

## Quick Start (Windows `cmd`)

1. Create and activate a virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Add API keys (the app can also prompt inline)

- `OPENAI_API_KEY` (OpenAI)
- `GROQ_API_KEY` (Groq)
- `GOOGLE_API_KEY` (Gemini)

Create a `.env` in the project root if you prefer:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

4. Run the app

```
streamlit run app.py
```

## Using the App

- Sidebar → “Model & Keys”: pick provider and model; enter an API key if not found in your env.
- Sidebar → “Conversational Retrieval”: enable to use recent chat history for smarter follow‑ups.
- Sidebar → “Upload Documents”: add your files and click “Process documents” to index.
- Main panel: chat as usual. You’ll see sources listed under each answer.
- Optional: use the “Purge Chroma Data (Disk)” button to fully reset the vector store.
- Response style: toggle Concise/Detailed in the sidebar to adjust output length.

## Behavior & Tips

- RAG‑first: the bot searches your indexed chunks. If zero relevant chunks are found, it falls back to web snippets.
- Conversation‑aware: when enabled, the app includes a compact history excerpt in the prompt and uses user turns to improve retrieval queries.
- Citations: you’ll see either document references (file name + page/slide) or “web search” when web snippets contributed.
- PDFs: page numbers come from PyPDF; PPTX uses slide numbers.
- Empty or scanned PDFs: if text can’t be extracted, chunks won’t be indexed. Consider adding OCR (we can add this if needed).

## Project Structure

```
app.py
requirements.txt
config/
	config.py
models/
	embeddings.py
	llm.py
utils/
	doc_loaders.py
	text_split.py
	vectorstore.py
	websearch.py
```

## Local Data & Git Hygiene

- Chroma persistence: `./chroma_db` (configurable). Use the sidebar purge to reset.
- Uploaded files: stored under `./uploaded_files`.
- Secrets: store keys in `.env` (not committed).

We include a `.gitignore` to exclude local artifacts:

```
.env
chroma_db/
uploaded_files/
__pycache__/
.streamlit/
.vscode/
```

## Troubleshooting

- Missing API key: add it in the sidebar or `.env`.
- Model name errors: verify provider docs for correct names.
- “Insufficient context”: ensure documents are processed; enable conversational retrieval for follow‑ups; or let the app fall back to web search.
- Stale results: click “Purge Chroma Data (Disk)” and re‑process docs.
