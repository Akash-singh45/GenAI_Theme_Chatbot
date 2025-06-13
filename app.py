import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Gemini API key
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# ----------------------------
# Load Sentence Transformer
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="GenAI Theme Chatbot", layout="wide")
st.title("📄 GenAI Theme Chatbot")
st.write("Upload a document and ask intelligent questions. Powered by Gemini + FAISS.")

# Temporary directory to hold uploads
UPLOAD_DIR = tempfile.gettempdir()
INDEX_FILE = os.path.join(UPLOAD_DIR, "faiss_index.index")
META_FILE = os.path.join(UPLOAD_DIR, "faiss_metadata.pkl")

# ----------------------------
# Utilities
# ----------------------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
        for idx, para in enumerate(paragraphs):
            chunks.append({
                "doc_id": os.path.basename(path),
                "page": page_num + 1,
                "paragraph": idx + 1,
                "text": para
            })
    return chunks

def extract_text_from_image(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return [{
        "doc_id": os.path.basename(path),
        "page": 1,
        "paragraph": 1,
        "text": text.strip()
    }]

def save_to_faiss(chunks):
    texts = [c['text'] for c in chunks]
    embeddings = embedding_model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'wb') as f:
        pickle.dump(chunks, f)
    return len(chunks)

def load_faiss_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, []
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def ask_gemini(question, chunks):
    context = "\n\n".join([
        f"[Doc: {c['doc_id']} | Page {c['page']} | Para {c['paragraph']}]: {c['text']}"
        for c in chunks
    ])
    prompt = f"""
You are a legal research assistant AI. Use only the provided context to answer clearly and concisely.

### Question:
{question}

### Context:
{context}

### Format:
Answer: <main answer>
Citations:
- Doc ID, Page, Para
Theme Summary: <if multiple docs involved>
"""
    response = model.generate_content(prompt)
    return response.text

def search_faiss(query, top_k):
    index, metadata = load_faiss_index()
    if index is None:
        return [], "No FAISS index found. Please upload a document."
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    results = [metadata[i] for i in I[0]]
    return results, None

# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📤 Upload a PDF or image document")
uploaded_file = st.file_uploader("Upload file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Saved {uploaded_file.name}")
    ext = os.path.splitext(file_path)[-1].lower()

    with st.spinner("Extracting text and embedding..."):
        if ext == ".pdf":
            chunks = extract_text_from_pdf(file_path)
        else:
            chunks = extract_text_from_image(file_path)

        if chunks:
            count = save_to_faiss(chunks)
            st.success(f"✅ {count} paragraphs saved to FAISS index")
        else:
            st.error("❌ No text extracted from file")

# ----------------------------
# Query Section
# ----------------------------
st.subheader("💬 Ask a question about the document")
query = st.text_input("Enter your question")

if st.button("Get Answer") and query:
    with st.spinner("Searching and asking Gemini..."):
        chunks, error = search_faiss(query, top_k=3)
        if error:
            st.error(error)
        else:
            answer = ask_gemini(query, chunks)
            st.markdown("### 🤖 Gemini’s Answer")
            st.success(answer)

            st.markdown("### 📚 Cited Chunks")
            for c in chunks:
                st.markdown(f"**📄 {c['doc_id']}** — Page {c['page']}, Para {c['paragraph']}")
                st.write(c['text'])
                st.markdown("---")
