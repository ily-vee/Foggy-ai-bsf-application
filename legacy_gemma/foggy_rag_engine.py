import os
import re
import json
import sys
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader  # Added to extract PDF data automatically

print("🤖 Initializing Deep Cleaning & Multi-Doc RAG Core...")

# 1. Initialize local embedding weights
embed_model = SentenceTransformer("all-MiniLM-L6-v2") # loads a lightweight local embedding model. Its job is to
# translate human sentences into list arrays of numbers (vectors) that capture the mathematical meaning of the text.

DOCS_DIR = "knowledge_docs"
CACHE_DIR = "foggy_vector_db"  # Directory to save calculated matrix cache

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    print(f"📁 Created '{DOCS_DIR}/' folder. Please drop your files inside it.")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# --- TEXT EXTRACTION & SANITIZATION PIPELINE ---
# Extracts text page-by-page from PDFs using PdfReader (ignoring empty pages)
def extract_text_from_pdf(pdf_path):
    """ Reads text from an unencrypted PDF document page-by-page. """
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        print(f"  ❌ Error parsing PDF extraction on {pdf_path}: {e}")
        return ""

# Normalizes quotes/dashes, strips web links, and filters out page numbers or headers.
def clean_text(raw_text):
    """ Applies regex sanitization to strip document noise, layouts, and fragments. """
    text = raw_text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2014', '—')
    text = re.sub(r'\(text, \d{4}\)', '', text)
    text = re.sub(r'https?://\S+', '', text)

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+$', line):  # Ignore pure numbers/page counts
            continue
        if len(line) < 15 and ("page" in line.lower() or "vol" in line.lower()):
            continue
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)

# Cuts text into 300-word windows with a 30-word overlap (so no context is split).
def build_overlapping_chunks(text, chunk_size=300, overlap=30):
    """ Slices normalized text into semantic windows with a sliding overlap boundary. """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        window = words[i: i + chunk_size]
        if len(window) > 10:
            chunks.append(" ".join(window))
        i += (chunk_size - overlap)
    return chunks


# --- HYBRID CACHE & PROCESSING RECOVERY TRACK ---

all_chunks = []
chunk_embeddings = None

# Create cache file paths
chunks_cache_file = os.path.join(CACHE_DIR, "chunks.json")
vectors_cache_file = os.path.join(CACHE_DIR, "embeddings.npy")

# Add .pdf to your accepted document streams
supported_extensions = ('.txt', '.md', '.pdf')
found_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(supported_extensions)]

# Check if cache matches the current files to skip processing completely
cache_valid = os.path.exists(chunks_cache_file) and os.path.exists(vectors_cache_file)

if cache_valid and found_files:
    print("💾 Found existing vector database storage. Loading indices instantly...")
    with open(chunks_cache_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(vectors_cache_file)
    print(f"✅ Loaded {len(all_chunks)} historical nodes from persistent disk memory.")
else:
    if found_files:
        print(f"🔍 Processing fresh/updated files inside '{DOCS_DIR}/'...")
        for filename in found_files:
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                # Direct route fork based on file extension type
                if filename.endswith('.pdf'):
                    raw_content = extract_text_from_pdf(filepath)
                else:
                    with open(filepath, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                sanitized_text = clean_text(raw_content)
                file_chunks = build_overlapping_chunks(sanitized_text, chunk_size=300, overlap=30)

                # Tag metadata slightly for cleaner validation boundaries
                all_chunks.extend(file_chunks)
                print(f"  └─ {filename}: Cleaned and split into {len(file_chunks)} windows.")
            except Exception as e:
                print(f"  ❌ Error reading {filename}: {e}")
    else:
        print("⚠️ No documents detected inside 'knowledge_docs/'. Using fallback seed records.")
        all_chunks = [
            "Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C. High "
            "deviations cause stress.",
            "BSF crates must maintain a moisture profile between 60% and 70%. Excess moisture creates an anaerobic "
            "substrate.",
        ]

    print(f"📊 Total cleaned processing nodes in active vector memory: {len(all_chunks)}")
    print("🧮 Generating mathematical matrix arrays...")
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)

    # Save calculated indices so the next startup takes 0 seconds
    with open(chunks_cache_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    np.save(vectors_cache_file, chunk_embeddings)
    print("💾 Matrix calculation saved to disk memory successfully.")


# --- MATRICES LOOKUP & INFERENCE ---

def retrieve_relevant_context(user_query, top_k=3):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
system_blueprint = ("You are Foggy, a precision BSF assistant. Answer queries using the verified dynamic local data "
                    "provided.")

conversation = []
print("\n🌱 Foggy RAG Core Engine Active and Streaming. Ready for questions!\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    context_extracted = retrieve_relevant_context(user_input, top_k=3)
    current_system_prompt = f"{system_blueprint}\nDYNAMIC LOCAL DATA:\n{context_extracted}"

    active_payload_messages = [{"role": "system", "content": current_system_prompt}]
    for past_turn in conversation:
        active_payload_messages.append(past_turn)
    active_payload_messages.append({"role": "user", "content": user_input})

    print("\nFoggy: ", end="")
    sys.stdout.flush()

    payload = {"messages": active_payload_messages, "temperature": 0.2, "stream": True}

    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), stream=True)
        full_reply = ""

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').lstrip('data: ')
                if decoded_line.strip() == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(decoded_line)
                    delta_content = chunk_json['choices'][0]['delta'].get('content') or ''
                    sys.stdout.write(delta_content)
                    sys.stdout.flush()
                    full_reply += delta_content
                except json.JSONDecodeError:
                    continue
        print("\n")
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": full_reply})

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Ensure your background llama-server terminal is running on port 8080.")
        break
