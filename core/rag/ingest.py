import os
import re
import json
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Resolve root path based on project layout
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = BASE_DIR / "knowledge_docs"
CACHE_DIR = BASE_DIR / "foggy_vector_db"

print("⚙️ Initializing RAG Knowledge Ingestion Engine...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"  ❌ Error parsing PDF {pdf_path}: {e}")
        return ""

def clean_text(raw_text):
    text = raw_text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2014', '—')
    text = re.sub(r'\(text, \d{4}\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or re.match(r'^\d+$', line):
            continue
        if len(line) < 15 and ("page" in line.lower() or "vol" in line.lower()):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)

def build_overlapping_chunks(text, chunk_size=250, overlap=35):
    """Word-based overlapping chunks optimized for BSF operational context."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        window = words[i: i + chunk_size]
        if len(window) > 15:
            chunks.append(" ".join(window))
        i += (chunk_size - overlap)
    return chunks

def run_ingestion():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    supported_extensions = ('.txt', '.md', '.pdf')
    
    if not DOCS_DIR.exists():
        print(f"❌ Error: {DOCS_DIR} directory does not exist.")
        return

    found_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(supported_extensions)]

    if not found_files:
        print(f"⚠️ No manuals found in '{DOCS_DIR}'. Place your PDFs there.")
        return

    print(f"📦 Processing {len(found_files)} source manuals from {DOCS_DIR}...")
    
    all_chunk_texts = []
    metadata_store = []

    for filename in found_files:
        filepath = DOCS_DIR / filename
        raw_content = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else filepath.read_text(encoding="utf-8")
        sanitized_text = clean_text(raw_content)
        file_chunks = build_overlapping_chunks(sanitized_text, chunk_size=250, overlap=35)
        
        for idx, chunk in enumerate(file_chunks):
            all_chunk_texts.append(chunk)
            metadata_store.append({
                "source": filename,
                "chunk_id": idx,
                "text": chunk
            })
            
        print(f"  └─ {filename} ({len(file_chunks)} chunks)")

    print("\n🧮 Generating MiniLM Dense Embeddings...")
    embeddings = embed_model.encode(all_chunk_texts, convert_to_numpy=True, show_progress_bar=True)

    # Save outputs to foggy_vector_db/
    chunks_json_path = CACHE_DIR / "chunks.json"
    vectors_path = CACHE_DIR / "embeddings.npy"

    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunk_texts, f, indent=2)

    np.save(vectors_path, embeddings)

    print(f"💾 Vector DB successfully built! Total nodes: {len(all_chunk_texts)}")
    print(f"   ├─ Text Chunks: {chunks_json_path}")
    print(f"   └─ Embeddings:  {vectors_path}")

if __name__ == "__main__":
    run_ingestion()