"""
ingest.py  (v2)

Builds the hybrid (dense + BM25) knowledge base from documents in
knowledge_docs/.

Fixes vs the uploaded version:
- chunks.json now stores full metadata (source filename + chunk_id) per
  chunk, not just bare text. The original code built this metadata
  (`metadata_store`) but never wrote it to disk — only the bare text list
  was saved — so the retriever had no way to tell the LLM which document a
  chunk came from. That meant any "citation" the LLM gave would have to be
  invented, which is exactly the failure mode fixed earlier in this project.
  Now that there's a real corpus, real per-chunk citations are possible.
- Embeddings are now L2-normalized at encode time (normalize_embeddings=True)
  so retriever.py's dot-product search is actually equivalent to cosine
  similarity. Without this, the dot product conflates embedding magnitude
  with semantic relevance — chunks with larger norms could rank higher for
  reasons unrelated to how well they answer the query.
"""

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

    # Each record carries its source filename through to disk now — this is
    # the fix. Previously this same structure was built and then discarded.
    chunk_records = []

    for filename in found_files:
        filepath = DOCS_DIR / filename
        raw_content = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else filepath.read_text(encoding="utf-8")
        sanitized_text = clean_text(raw_content)
        file_chunks = build_overlapping_chunks(sanitized_text, chunk_size=250, overlap=35)

        for idx, chunk in enumerate(file_chunks):
            chunk_records.append({
                "source": filename,
                "chunk_id": idx,
                "text": chunk
            })

        print(f"  └─ {filename} ({len(file_chunks)} chunks)")

    if not chunk_records:
        print("⚠️ No chunks were produced — check that source documents contain extractable text.")
        return

    print("\n🧮 Generating MiniLM Dense Embeddings...")
    texts = [r["text"] for r in chunk_records]
    # normalize_embeddings=True is required for retriever.py's dot-product
    # search to behave as true cosine similarity rather than a
    # magnitude-biased inner product.
    embeddings = embed_model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
    )

    # Save outputs to foggy_vector_db/
    chunks_json_path = CACHE_DIR / "chunks.json"
    vectors_path = CACHE_DIR / "embeddings.npy"

    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    np.save(vectors_path, embeddings)

    print(f"💾 Vector DB successfully built! Total nodes: {len(chunk_records)}")
    print(f"   ├─ Chunks (with source metadata): {chunks_json_path}")
    print(f"   └─ Embeddings:  {vectors_path}")


if __name__ == "__main__":
    run_ingestion()