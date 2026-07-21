import os
import re
import json
import sys
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# Silence HuggingFace warnings completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("⚙️ Booting Unified Foggy Engine Core...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Load Vector Matrix Cache
CACHE_DIR = "foggy_vector_db"
chunks_path = os.path.join(CACHE_DIR, "chunks.json")
vectors_path = os.path.join(CACHE_DIR, "embeddings.npy")

if os.path.exists(chunks_path) and os.path.exists(vectors_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(vectors_path)
    print(f"✅ Vector Database Connected. Loaded {len(all_chunks)} manual nodes.")
else:
    all_chunks = ["Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C."]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    print("⚠️ Warning: Vector DB fallback initialized.")

API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}


def retrieve_relevant_context(user_query, top_k=3):
    """Fetches matching source text from local documents instantly."""
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


# ==========================================
# CORE UTILITY 1: STRICT EXTRACTION ENGINE
# ==========================================
def extract_metrics_to_json(user_query):
    """Queries the local server using 0.0 temperature for fixed data parameters."""
    context = retrieve_relevant_context(user_query, top_k=2)

    system_blueprint = (
        "You are an offline data extraction engine. You must output raw valid JSON code only. "
        "Do not include any text commentary, intro, or markdown fences like ```json. "
        "Extract metrics from the provided data using this strict template:\n"
        "{\n"
        '  "topic_identified": "string",\n'
        '  "extracted_metric": "string or number",\n'
        '  "unit_of_measurement": "string or null",\n'
        '  "confidence_level": "High/Medium/Low"\n'
        "}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": f"{system_blueprint}\n\nDATA:\n{context}"},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 512
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        raw_reply = response.json()['choices'][0]['message']['content'].strip()
        clean_json_str = re.sub(r'^```json\s*|```$', '', raw_reply, flags=re.MULTILINE).strip()
        return json.loads(clean_json_str)
    except Exception:
        return None


# ==========================================
# CORE UTILITY 2: CONVERSATIONAL STREAMER
# ==========================================
def stream_conversation(user_query, history_buffer):
    """Streams friendly conversational paragraphs back to the console or client."""
    context = retrieve_relevant_context(user_query, top_k=3)
    system_blueprint = ("You are Foggy, a precision AI assistant for BSF farmers. Give practical advice using the "
                        "verified dynamic local data provided.")

    active_messages = [{"role": "system", "content": f"{system_blueprint}\n\nDYNAMIC LOCAL DATA:\n{context}"}]
    for turn in history_buffer:
        active_messages.append(turn)
    active_messages.append({"role": "user", "content": user_query})

    payload = {
        "messages": active_messages,
        "temperature": 0.3,
        "stream": True,
        "max_tokens": 1024
    }

    full_reply = ""
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), stream=True)
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
        print()
        return full_reply
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not reach llama-server.")
        return ""


# ==========================================
# AUTOMATED RUNTIME ROUTER LOOP
# ==========================================
if __name__ == "__main__":
    print("\n🌱 Foggy Hybrid Router Active. Type 'exit' to quit.\n")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # 1. INTENT RECOGNITION STEP
        # Check if user specifically wants a structural parameter log
        if any(keyword in user_input.lower() for keyword in ["extract", "json", "log", "metric", "save parameter"]):
            print("\n[SYSTEM: Routing to Structured Data Extraction Engine...]")
            extracted_data = extract_metrics_to_json(user_input)

            if extracted_data:
                print("📋 Extracted Machine Dictionary Structure:")
                print(json.dumps(extracted_data, indent=4))

                # Here is where your future code would run a SQL save:
                # database.save(extracted_data["extracted_metric"])
            else:
                print("⚠️ System was unable to enforce JSON parameters for this query.")
            print("-" * 50 + "\n")

        else:
            # 2. CONVERSATIONAL STEP
            print("\nFoggy: ", end="")
            sys.stdout.flush()
            reply = stream_conversation(user_input, conversation_history)

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": reply})
            print("-" * 50 + "\n")