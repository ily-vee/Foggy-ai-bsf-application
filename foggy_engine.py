import os
import re
import json
import sys
import torch
import torch.nn as nn
import numpy as np
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer

# Silence warnings completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("⚙️ Booting Unified Foggy Engine Core...")

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CACHE_DIR = "foggy_vector_db"
chunks_path = os.path.join(CACHE_DIR, "chunks.json")
vectors_path = os.path.join(CACHE_DIR, "embeddings.npy")

SIGLIP_LOCAL_PATH = "./siglip2_local"
CLASSIFIER_WEIGHTS = "bsf_classifier.pth"
API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

CLASS_LABELS = {
    0: "1_eggs",
    1: "2_early_larvae",
    2: "3_feeding_larvae",
    3: "4_pupae",
    4: "5_bsf_adult"
}

# ==========================================
# MODEL INITIALIZATIONS (INSTANT LOCAL LOAD)
# ==========================================
# 1. Text embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Vector DB Loading
if os.path.exists(chunks_path) and os.path.exists(vectors_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(vectors_path)
    print(f"✅ Vector Database Connected. Loaded {len(all_chunks)} manual nodes.")
else:
    all_chunks = ["Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C."]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    print("⚠️ Warning: Vector DB fallback initialized.")


# 3. Custom PyTorch MLP Architecture
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super(BSFClassifierHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# 4. Load Vision Models
print("👁️ Loading Local SigLIP 2 Vision backend...")
vision_processor = AutoProcessor.from_pretrained(SIGLIP_LOCAL_PATH, local_files_only=True)
vision_model = AutoModel.from_pretrained(SIGLIP_LOCAL_PATH, local_files_only=True)
vision_model.eval()

print("🧠 Loading Trained MLP Weights...")
classifier = BSFClassifierHead()
if os.path.exists(CLASSIFIER_WEIGHTS):
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS))
    classifier.eval()
    print("✅ Classifier Head Activated successfully.")
else:
    print("⚠️ Warning: 'bsf_classifier.pth' not found. Classification will fail.")


# ==========================================
# CORE PIPELINE FUNCTIONS
# ==========================================
def run_vision_inference(image_path):
    """Processes image and returns prediction string with confidence levels."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            inputs = vision_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                siglip_outputs = vision_model.vision_model(**inputs)
                embedding = siglip_outputs.pooler_output
                classifier_outputs = classifier(embedding)
                probabilities = torch.nn.functional.softmax(classifier_outputs, dim=1).squeeze().numpy()

            predicted_idx = np.argmax(probabilities)
            label = CLASS_LABELS[predicted_idx]
            confidence = probabilities[predicted_idx] * 100
            return label, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0


def retrieve_relevant_context(user_query, top_k=3):
    """Fetches matching text blocks from local documents instantly."""
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


def stream_conversation(user_query, history_buffer, image_info=None):
    """Streams friendly conversational paragraphs back using unified RAG."""
    doc_context = retrieve_relevant_context(user_query, top_k=3)

    system_blueprint = (
        "You are Foggy, a precision AI assistant for BSF farmers. Give practical advice "
        "using the verified dynamic local data provided. Be concise and precise."
    )

    # Build contextual background from both RAG and Vision Classifier
    context_payload = f"DYNAMIC LOCAL KNOWLEDGE:\n{doc_context}"
    if image_info:
        context_payload += f"\n\nLIVE IMAGE CLASSIFICATION: The farmer uploaded an image which has been classified as {image_info['label']} (Confidence: {image_info['confidence']:.2f}%)."

    active_messages = [{"role": "system", "content": f"{system_blueprint}\n\n{context_payload}"}]
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
# CONVERSATIONAL LOOP
# ==========================================
if __name__ == "__main__":
    print("\n🌱 Foggy Unified Multi-Modal Core Active. Type 'exit' to quit.\n")
    conversation_history = []

    while True:
        user_input = input("You (use '<image_path> <your prompt>' to include an image): ")
        if user_input.lower() == "exit":
            break

        # Check for image file names inside user input (e.g., testimage1.jpeg)
        image_match = re.search(r'\b\w+\.(?:jpeg|jpg|png)\b', user_input, re.IGNORECASE)
        image_info = None

        if image_match:
            img_path = image_match.group(0)
            if os.path.exists(img_path):
                print(f"[SYSTEM: Analyzing local image '{img_path}' via SigLIP 2 Classifier...]")
                label, confidence = run_vision_inference(img_path)
                image_info = {"label": label, "confidence": confidence}
                print(f"🎯 Vision Result: {label} ({confidence:.2f}% confidence)")
                # Remove filename from the prompt to avoid confusing the text engine
                user_input = user_input.replace(img_path, "").strip()
            else:
                print(f"⚠️ Warning: Image file '{img_path}' was detected in prompt but not found on disk.")

        print("\nFoggy: ", end="")
        sys.stdout.flush()

        reply = stream_conversation(user_input, conversation_history, image_info)

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})
        print("-" * 50 + "\n")
