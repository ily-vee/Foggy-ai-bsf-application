import os
import re
import json
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

print("⚙️ Initializing Background Knowledge Vectorizer...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCS_DIR = "knowledge_docs"
CACHE_DIR = "foggy_vector_db"

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


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


def build_overlapping_chunks(text, chunk_size=300, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        window = words[i: i + chunk_size]
        if len(window) > 10:
            chunks.append(" ".join(window))
        i += (chunk_size - overlap)
    return chunks


# Collect and process
all_chunks = []
supported_extensions = ('.txt', '.md', '.pdf')
found_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(supported_extensions)]

if found_files:
    print(f"📦 Processing {len(found_files)} source manuals...")
    for filename in found_files:
        filepath = os.path.join(DOCS_DIR, filename)
        raw_content = extract_text_from_pdf(filepath) if filename.endswith('.pdf') else open(filepath, "r",
                                                                                             encoding="utf-8").read()
        sanitized_text = clean_text(raw_content)
        file_chunks = build_overlapping_chunks(sanitized_text, chunk_size=300, overlap=30)
        all_chunks.extend(file_chunks)
        print(f"  └─ {filename} ({len(file_chunks)} chunks)")

    print("🧮 Calculating matrix arrays...")
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)

    # Save to disk
    with open(os.path.join(CACHE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    np.save(os.path.join(CACHE_DIR, "embeddings.npy"), chunk_embeddings)
    print("💾 Success! Database files written to 'foggy_vector_db/'.")
else:
    print("⚠️ No documents found inside 'knowledge_docs/'.")
