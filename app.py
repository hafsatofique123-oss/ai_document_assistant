"""
Simple RAG-style AI Assistant built with Streamlit.

Pipeline:
1. Load documents (local upload OR Google Drive link) -> PDF / DOCX / TXT / MD
2. Extract text (per page for PDF, whole document for DOCX/TXT/MD)
3. Split text into overlapping chunks (metadata: filename, page)
4. Embed chunks once with SentenceTransformers, cache in session_state
5. Build a FAISS index over the embeddings (rebuilt only when new chunks appear)
6. On a question: hybrid search = semantic (FAISS) + keyword score
7. Send top chunks + question to Groq, force "context-only" answers
8. Show the answer, then the sources used to build it

Everything expensive (model loading, embeddings, FAISS index) is cached or
stored in st.session_state so it is computed once per document / once per app
session, not on every question.
"""

import os
import re
import hashlib
import tempfile
from collections import Counter

import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CHUNK_SIZE_WORDS = 200      # words per chunk
CHUNK_OVERLAP_WORDS = 50    # words overlap between consecutive chunks
TOP_K = 5                   # how many chunks to retrieve per question
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Simple stopword list for the keyword search (kept tiny on purpose)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "about", "as", "by", "this", "that", "these", "those", "it", "its",
    "do", "does", "did", "what", "which", "who", "whom", "how", "why",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "our",
    "your", "their", "not", "no", "yes", "than", "then", "so", "such",
}


# --------------------------------------------------------------------------
# Cached, expensive-to-create resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Load the sentence-embedding model once per app process."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Create the Groq client using the key from Streamlit secrets."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Extraction functions (one per file type)
# Each returns a list of {"text": str, "page": int|None}
# --------------------------------------------------------------------------
def extract_pdf(path):
    """Extract text page by page so we can keep real page numbers."""
    reader = PdfReader(path)
    sections = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            sections.append({"text": text, "page": i + 1})
    return sections


def extract_docx(path):
    """DOCX has no reliable page boundaries, so we return one section."""
    doc = DocxDocument(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "page": None}] if text.strip() else []


def extract_txt_or_md(path):
    """Plain text / markdown: read the whole file as one section."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "page": None}] if text.strip() else []


def extract_document(path, filename):
    """Dispatch to the right extractor based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        sections = extract_pdf(path)
    elif ext == ".docx":
        sections = extract_docx(path)
    elif ext in (".txt", ".md"):
        sections = extract_txt_or_md(path)
    else:
        return []
    # attach filename to every section
    for s in sections:
        s["filename"] = filename
    return sections


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text, filename, page):
    """
    Split text into overlapping word-based chunks.
    Every chunk keeps the filename and page it came from.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append({
            "filename": filename,
            "page": page,
            "text": chunk_str,
        })
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS  # move forward, keep overlap
    return chunks


def build_chunks_for_document(path, filename):
    """Extract + chunk a single document, return list of chunk dicts."""
    sections = extract_document(path, filename)
    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_text(section["text"], filename, section["page"]))
    return all_chunks


# --------------------------------------------------------------------------
# Embeddings + FAISS index (built once, reused across questions)
# --------------------------------------------------------------------------
def embed_texts(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
    return embeddings


def rebuild_faiss_index():
    """Rebuild the FAISS index from everything currently in session_state."""
    embeddings = st.session_state.embeddings
    if embeddings is None or len(embeddings) == 0:
        st.session_state.faiss_index = None
        return
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    st.session_state.faiss_index = index


def add_documents_to_pipeline(file_infos):
    """
    file_infos: list of (path, filename, file_hash)
    Extracts, chunks, embeds and adds new documents to session_state.
    Documents already processed (same hash) are skipped -> no recomputation.
    """
    new_chunks = []
    for path, filename, file_hash in file_infos:
        if file_hash in st.session_state.processed_hashes:
            continue
        doc_chunks = build_chunks_for_document(path, filename)
        new_chunks.extend(doc_chunks)
        st.session_state.processed_hashes.add(file_hash)
        st.session_state.processed_filenames.add(filename)

    if not new_chunks:
        return 0

    new_embeddings = embed_texts([c["text"] for c in new_chunks])

    if st.session_state.embeddings is None:
        st.session_state.embeddings = new_embeddings
    else:
        st.session_state.embeddings = np.vstack(
            [st.session_state.embeddings, new_embeddings]
        )

    st.session_state.chunks.extend(new_chunks)
    rebuild_faiss_index()
    return len(new_chunks)


# --------------------------------------------------------------------------
# Hybrid search: semantic (FAISS) + keyword overlap
# --------------------------------------------------------------------------
def extract_keywords(question):
    words = re.findall(r"[a-zA-Z0-9']+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def keyword_score(chunk_text_value, keywords):
    if not keywords:
        return 0.0
    text_lower = chunk_text_value.lower()
    word_counts = Counter(re.findall(r"[a-zA-Z0-9']+", text_lower))
    matches = sum(word_counts.get(k, 0) for k in keywords)
    # normalize by number of keywords so score is roughly comparable across questions
    return matches / len(keywords)


def semantic_search(query, top_k):
    """Return list of (chunk_index, similarity_score)."""
    index = st.session_state.faiss_index
    if index is None:
        return []
    query_emb = embed_texts([query])
    scores, indices = index.search(query_emb, min(top_k, len(st.session_state.chunks)))
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score)))
    return results


def hybrid_search(query, top_k=TOP_K, semantic_weight=0.7, keyword_weight=0.3):
    """
    Combine semantic similarity and keyword overlap into one score,
    return the top_k chunks (with metadata) sorted by that combined score.
    """
    chunks = st.session_state.chunks
    if not chunks:
        return []

    # Semantic scores: search a wider pool so keyword-only matches still have a chance
    pool_size = min(len(chunks), max(top_k * 4, 20))
    semantic_results = semantic_search(query, pool_size)
    semantic_scores = {idx: score for idx, score in semantic_results}

    keywords = extract_keywords(query)
    keyword_scores = {}
    candidate_indices = set(semantic_scores.keys())
    # also scan all chunks for keyword matches so purely keyword-relevant
    # chunks that FAISS ranked low still get a chance to surface
    for i, c in enumerate(chunks):
        k_score = keyword_score(c["text"], keywords)
        if k_score > 0:
            keyword_scores[i] = k_score
            candidate_indices.add(i)

    if not candidate_indices:
        return []

    max_kw = max(keyword_scores.values()) if keyword_scores else 1.0

    scored = []
    for i in candidate_indices:
        sem = semantic_scores.get(i, 0.0)
        kw = keyword_scores.get(i, 0.0) / max_kw if max_kw > 0 else 0.0
        combined = semantic_weight * sem + keyword_weight * kw
        scored.append((combined, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    results = []
    for combined_score, i in top:
        chunk = chunks[i]
        results.append({
            "filename": chunk["filename"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": combined_score,
        })
    return results


# --------------------------------------------------------------------------
# Groq: answer strictly from retrieved context
# --------------------------------------------------------------------------
def build_context_block(retrieved_chunks):
    parts = []
    for i, c in enumerate(retrieved_chunks, start=1):
        page_str = f", page {c['page']}" if c["page"] else ""
        parts.append(f"[Source {i}: {c['filename']}{page_str}]\n{c['text']}")
    return "\n\n".join(parts)


def ask_groq(question, retrieved_chunks):
    client = get_groq_client()
    if client is None:
        return "⚠️ No GROQ_API_KEY found in Streamlit secrets. Please add it to run questions."

    context_block = build_context_block(retrieved_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "context provided below. Do not use outside knowledge. "
        "If the answer cannot be found in the context, say clearly that the "
        "information is not available in the provided documents. "
        "Keep answers concise and cite which source number you used when relevant."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Google Drive loading (public file / folder links via gdown)
# --------------------------------------------------------------------------
def load_from_google_drive(link):
    """
    Download a public Google Drive file or folder link into a temp folder,
    then return a list of (path, filename, file_hash) for supported files.
    """
    import gdown

    tmp_dir = tempfile.mkdtemp(prefix="gdrive_")
    downloaded_paths = []

    try:
        if "folder" in link:
            gdown.download_folder(url=link, output=tmp_dir, quiet=True, use_cookies=False)
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    downloaded_paths.append(os.path.join(root, f))
        else:
            out_path = gdown.download(url=link, output=os.path.join(tmp_dir, ""), quiet=True, fuzzy=True)
            if out_path:
                downloaded_paths.append(out_path)
    except Exception as e:
        st.error(f"Could not load from Google Drive: {e}")
        return []

    file_infos = []
    for path in downloaded_paths:
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        file_infos.append((path, filename, file_hash))
    return file_infos


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chunks": [],                 # list of chunk dicts (filename, page, text)
        "embeddings": None,           # np.array aligned with st.session_state.chunks
        "faiss_index": None,          # faiss index built from embeddings
        "processed_hashes": set(),    # avoid re-processing the same file content
        "processed_filenames": set(), # for display
        "chat_history": [],           # list of (question, answer, sources)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("📁 Documents")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        file_infos = []
        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            suffix = os.path.splitext(uf.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file_bytes)
            tmp.close()
            file_infos.append((tmp.name, uf.name, file_hash))

        if st.sidebar.button("Process uploaded files"):
            with st.spinner("Extracting, chunking and embedding..."):
                added = add_documents_to_pipeline(file_infos)
            st.sidebar.success(f"Added {added} new chunks.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Google Drive")
    drive_link = st.sidebar.text_input("Paste a Drive file or folder link")
    if st.sidebar.button("Load from Google Drive"):
        if drive_link:
            with st.spinner("Downloading from Google Drive..."):
                file_infos = load_from_google_drive(drive_link)
            if file_infos:
                with st.spinner("Extracting, chunking and embedding..."):
                    added = add_documents_to_pipeline(file_infos)
                st.sidebar.success(f"Added {added} new chunks from Drive.")
            else:
                st.sidebar.warning("No supported files found at that link.")
        else:
            st.sidebar.warning("Please paste a Drive link first.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Document info")
    st.sidebar.write(f"Files processed: {len(st.session_state.processed_filenames)}")
    st.sidebar.write(f"Total chunks: {len(st.session_state.chunks)}")
    if st.session_state.processed_filenames:
        for name in sorted(st.session_state.processed_filenames):
            st.sidebar.caption(f"• {name}")


def render_main():
    st.title("🤖 Document AI Assistant")
    st.write(
        "Upload documents or load them from Google Drive, then ask questions. "
        "Answers are generated only from the content of your documents."
    )

    if st.session_state.chunks:
        st.info(f"Ready to answer questions using {len(st.session_state.chunks)} chunks "
                f"from {len(st.session_state.processed_filenames)} document(s).")

    question = st.text_input("Ask a question about your documents")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        if not st.session_state.chunks:
            st.warning("Please upload or load at least one document first.")
        else:
            with st.spinner("Searching documents and asking Groq..."):
                retrieved = hybrid_search(question, top_k=TOP_K)
                answer = ask_groq(question, retrieved)
            st.session_state.chat_history.append((question, answer, retrieved))

    # show chat history, most recent first
    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"### ❓ {q}")
        st.markdown(a)
        with st.expander("📚 Sources used for this answer"):
            for i, s in enumerate(sources, start=1):
                page_str = f" (page {s['page']})" if s["page"] else ""
                st.markdown(f"**Source {i}: {s['filename']}{page_str}** — score: {s['score']:.3f}")
                st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
        st.markdown("---")


def main():
    st.set_page_config(page_title="Document AI Assistant", page_icon="🤖", layout="wide")
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
