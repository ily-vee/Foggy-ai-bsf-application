import os
import requests
import json
import sys
from sentence_transformers import SentenceTransformer
import numpy as np

import logging
import warnings

# 1. Block Hugging Face specific warning outputs and progress logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("🌱 Foggy BSF Assistant (Gemma 3 4B - Real-Time Streaming Mode)")
print("Type 'exit' to quit\n")

# 1. Initialize local embedding weights for quick user queries
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Quietly load the pre-calculated vector database maps
CACHE_DIR = "foggy_vector_db"
chunks_path = os.path.join(CACHE_DIR, "chunks.json")
vectors_path = os.path.join(CACHE_DIR, "embeddings.npy")

if os.path.exists(chunks_path) and os.path.exists(vectors_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(vectors_path)
else:
    # Safe fallback if background builder hasn't run yet
    all_chunks = ["Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C."]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)


def retrieve_relevant_context(user_query, top_k=3):
    """Calculates vector distance against the cached disk matrix instantaneously."""
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


system_blueprint = ("You are Foggy, a precision AI assistant for BSF farmers. Give practical advice using the verified "
                    "dynamic local data provided.")
conversation = []

API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Foggy shutting down...")
        break

    # 3. Dynamic RAG execution on the fly
    context_extracted = retrieve_relevant_context(user_input, top_k=3)
    current_system_prompt = f"{system_blueprint}\nDYNAMIC LOCAL DATA:\n{context_extracted}"

    # Re-build active stream history payload
    active_payload_messages = [{"role": "system", "content": current_system_prompt}]
    for past_turn in conversation:
        active_payload_messages.append(past_turn)
    active_payload_messages.append({"role": "user", "content": user_input})

    print("\nFoggy: ", end="")
    sys.stdout.flush()

    payload = {
        "messages": active_payload_messages,
        "temperature": 0.3,
        "stream": True,
        "max_tokens": 2048
    }

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
        print("\n❌ Error: Could not connect to the local inference server. Check port 8080.")
        break
