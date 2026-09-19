# Document AI Assistant

A simple Streamlit app that lets you upload documents (or load them from
Google Drive), then ask questions that are answered **only** from the content
of those documents, using Groq as the LLM.

## How it works

1. **Load documents** — upload PDF / DOCX / TXT / MD files, or paste a public
   Google Drive file/folder link.
2. **Extract text** — a dedicated function per file type:
   - PDF → text per page (page numbers preserved)
   - DOCX → full document text (no native page numbers)
   - TXT / MD → full file text
3. **Chunk** — text is split into overlapping ~200-word chunks. Every chunk
   keeps its source filename and page number (if any).
4. **Embed** — each chunk is turned into a vector using
   `sentence-transformers` (`all-MiniLM-L6-v2`). Embeddings are computed
   **once** per document and stored in `st.session_state`, so re-asking
   questions never re-embeds anything.
5. **Search** — a FAISS index gives semantic search over the embeddings. A
   simple keyword scorer checks for overlap with important words from the
   question. Both scores are combined into one **hybrid search** ranking.
6. **Answer** — the top matching chunks are sent to Groq along with the
   question. The model is instructed to answer only from that context, and to
   say so if the answer isn't in the documents.
7. **Sources** — every answer is followed by the exact chunks used (filename,
   page number when available, and the retrieved text).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your Groq API key to Streamlit secrets. Create a file at
   `.streamlit/secrets.toml` (do **not** commit this file) with:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
   The app reads the key only from `st.secrets["GROQ_API_KEY"]` — it is never
   hardcoded in `app.py`.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using Google Drive

- Paste a **public** Google Drive file link or folder link in the sidebar and
  click "Load from Google Drive". The app uses `gdown` to download it, so the
  link must be shared as "Anyone with the link can view".
- Files loaded from Drive go through the exact same extraction, chunking,
  embedding and search pipeline as local uploads.

## Notes on performance

- The embedding model is loaded once per app process (`st.cache_resource`).
- Documents are hashed on upload; a document that was already processed is
  never re-extracted or re-embedded.
- Embeddings and the FAISS index live in `st.session_state`, so they persist
  across questions within the same session without recomputation.

## Files

- `app.py` — the full application
- `requirements.txt` — Python dependencies
- `README.md` — this file
