import os
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# 1. Initialize local embedding weights quietly
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_DIR = "foggy_vector_db"
chunks_path = os.path.join(CACHE_DIR, "chunks.json")
vectors_path = os.path.join(CACHE_DIR, "embeddings.npy")

if os.path.exists(chunks_path) and os.path.exists(vectors_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(vectors_path)
else:
    all_chunks = ["Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C."]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)


def retrieve_relevant_context(user_query, top_k=2):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


# 2. Main Extraction Engine
def extract_structured_data(user_query):
    context_extracted = retrieve_relevant_context(user_query, top_k=2)

    # Crucial System Blueprint: We strictly demand raw JSON structure and give an example template
    system_blueprint = (
        "You are a data extraction engine. You must output raw valid JSON code only. "
        "Do not include any chat commentary, introduction, or markdown code blocks (like ```json). "
        "Analyze the provided local data and fill out this specific structure:\n"
        "{\n"
        '  "topic_identified": "string",\n'
        '  "extracted_metric": "string or number",\n'
        '  "unit_of_measurement": "string or null",\n'
        '  "confidence_level": "High/Medium/Low"\n'
        "}"
    )

    current_system_prompt = f"{system_blueprint}\n\nDYNAMIC LOCAL DATA:\n{context_extracted}"

    payload = {
        "messages": [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,  # Dropping temperature to 0.0 forces strict precision over creativity
        "stream": False  # Disabling streaming since we need the whole JSON block complete before parsing
    }

    API_URL = "http://localhost:8080/v1/chat/completions"
    HEADERS = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        raw_reply = response.json()['choices'][0]['message']['content'].strip()

        # CLEANING LAYER: Strip away markdown code fences if the model included them
        clean_json_str = re.sub(r'^```json\s*|```$', '', raw_reply, flags=re.MULTILINE).strip()

        # Parse the cleaned string directly into a usable Python Dictionary
        parsed_data = json.loads(clean_json_str)
        return parsed_data

    except json.JSONDecodeError:
        print("⚠️ Failed to parse response as clean JSON. Raw Output was:")
        return raw_reply
    except Exception as e:
        return f"❌ Connection Error: {e}"


# 3. Test the extraction pipeline
if __name__ == "__main__":
    print("🔬 Testing Structured Data Extraction Pipeline...\n")

    test_query = "What is the optimal crude protein range for larvae found in the Kenyan waste study?"

    print(f"User Query: '{test_query}'")
    print("Extracting parameters...")

    result = extract_structured_data(test_query)

    print("\n🖥️ Python Dictionary Output received successfully:")
    print(json.dumps(result, indent=4))