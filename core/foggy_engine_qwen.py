"""
foggy_engine_qwen.py
Unified Foggy AI Engine combining SigLIP 2 feature extraction, Mahalanobis OOD detection,
Vector RAG retrieval, and conversation history streaming.
"""

import os
import re
import json
import sys
import pickle
import torch
import numpy as np
from threading import Thread
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer

from preprocess import preprocess_and_embed, PhotoValidationError
from train_kfold import BSFClassifierHead

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CACHE_DIR = "foggy_vector_db"
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
VECTORS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")

CLASSIFIER_PATH = "classifier_head.pt"
OOD_PATH = "bsf_ood_detector.pkl"
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE_DISPLAY_MAP = {
    "1_eggs": {
        "title": "Egg Stage (Egg Clutches)",
        "desc": "Keep incubation area moist (60–70%) and sheltered from direct sunlight. Ensure hatchlings have immediate access to soft feed substrate."
    },
    "2_early_larvae": {
        "title": "Early Larvae (Neonates / Young Instars)",
        "desc": "Requires easily digestible, finely ground substrate with 65–70% moisture. Maintain temperatures around 27°C–30°C."
    },
    "3_feeding_larvae": {
        "title": "Active Feeding Larvae (3rd - 5th Instar)",
        "desc": "Peak waste processing phase! Feed kitchen waste, fruit peels, or brewer's waste. Keep substrate temperature below 38°C."
    },
    "4_pupae": {
        "title": "Prepupae / Pupae Stage",
        "desc": "Feeding has stopped; larvae seek dry ground to pupate. Provide clean, dry, dark crawl-off areas (e.g., dry sawdust)."
    },
    "5_bsf_adult": {
        "title": "Adult Fly Stage (Love Cages / Breeding)",
        "desc": "Adults require bright sunlight/LEDs (350–450 nm) and clean water misting. They do not eat waste—focus on mating humidity and egg traps."
    }
}

# ==========================================
# 1. VECTOR RAG RETRIEVAL ENGINE
# ==========================================
print("📚 Connecting Text Embedding Model for RAG...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(VECTORS_PATH)
    print(f"✅ Vector Database Loaded ({len(all_chunks)} knowledge nodes).")
else:
    all_chunks = [
        "Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C.",
        "Feeding larvae require high moisture (60-70%) and nutrient-rich organic waste.",
        "Adult flies require clean water misting and direct sunlight or 6000K LED lighting for optimal mating.",
        "Prepupae migrate away from high moisture toward dry, dark areas to pupate."
    ]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    print("⚠️ Warning: Vector DB files not found on disk. Initialized fallback memory.")

def retrieve_relevant_context(user_query: str, top_k: int = 3) -> str:
    """Computes cosine similarity against local vector embeddings and retrieves context."""
    if not user_query.strip():
        return ""
    query_vec = embed_model.encode([user_query], convert_to_numpy=True)
    
    # Normalize for cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-10)
    db_norm = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-10)
    
    scores = np.dot(db_norm, query_norm.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


# ==========================================
# 2. MULTIMODAL FOGGY ENGINE CORE
# ==========================================
class FoggyBSFEngine:
    def __init__(self, max_history_turns: int = 10):
        self.device = DEVICE
        self.max_history_turns = max_history_turns
        self.history = []

        # Load PyTorch Classifier Head
        print("🧠 Loading BSF Classifier Head & OOD Detector...")
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=self.device)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = checkpoint["idx_to_class"]

        self.classifier = BSFClassifierHead(input_dim=768, num_classes=len(self.class_to_idx)).to(self.device)
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier.eval()

        # Load Mahalanobis OOD Parameters
        with open(OOD_PATH, "rb") as f:
            self.ood_data = pickle.load(f)

        # Load Qwen LLM for Streaming Chat
        print(f"👁️ Initializing Qwen VL LLM Core on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        print("⚡ Unified Foggy Engine fully initialized!")

    def _compute_min_mahalanobis_distance(self, embedding_np: np.ndarray) -> float:
        inv_cov = self.ood_data["inv_cov"]
        class_means = self.ood_data["class_means"]
        distances = []
        for mean in class_means.values():
            delta = embedding_np - mean
            dist = float(np.sqrt(np.dot(np.dot(delta, inv_cov), delta)))
            distances.append(dist)
        return min(distances)

    def analyze_image(self, image_path: str) -> dict:
        """Processes image through SigLIP 2, performs OOD check, and predicts life stage."""
        try:
            raw_emb = preprocess_and_embed(image_path)
        except PhotoValidationError as e:
            return {"status": "error", "message": f"Validation failed: {str(e)}"}

        norm_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
        emb_np = norm_emb.squeeze().numpy()

        # OOD Mahalanobis Check
        min_distance = self._compute_min_mahalanobis_distance(emb_np)
        threshold = self.ood_data["threshold"]

        if min_distance > threshold:
            return {
                "status": "rejected",
                "is_ood": True,
                "mahalanobis_distance": round(min_distance, 4),
                "threshold": round(threshold, 4),
                "message": "Out-of-Distribution image detected. The photo does not appear to be a BSF life stage."
            }

        # Neural Classification
        with torch.no_grad():
            logits = self.classifier(norm_emb.to(self.device))
            probabilities = torch.softmax(logits, dim=-1).cpu().squeeze().numpy()

        pred_idx = int(np.argmax(probabilities))
        pred_label = self.idx_to_class[pred_idx]
        confidence = float(probabilities[pred_idx])

        display_info = STAGE_DISPLAY_MAP.get(pred_label, {
            "title": pred_label,
            "desc": "Standard Black Soldier Fly management principles apply."
        })

        return {
            "status": "success",
            "is_ood": False,
            "predicted_stage": pred_label,
            "display_title": display_info["title"],
            "stage_guide": display_info["desc"],
            "confidence": round(confidence * 100, 2),
            "mahalanobis_distance": round(min_distance, 4)
        }

    def stream_integrated_response(self, user_query: str, image_info: dict = None) -> str:
        """Injects vector RAG knowledge, vision analysis, and session history into Qwen generation stream."""
        rag_context = retrieve_relevant_context(user_query, top_k=3)

        system_content = (
            "You are Foggy, an expert AI assistant specializing in Black Soldier Fly (BSF) farming.\n"
            "MANDATE: Do not claim you cannot process photos. An integrated vision core pre-analyzes uploaded photos for you.\n\n"
            "OUT-OF-SCOPE PROTOCOL:\n"
            "If asked non-BSF questions:\n"
            "1. State you are Foggy, a BSF specialist.\n"
            "2. Identify the topic.\n"
            "3. Provide a clear, complete answer.\n\n"
            "IN-SCOPE BSF PROTOCOL:\n"
            "Deliver practical, high-value guidance in clear farmer-friendly language."
        )

        context_payload = f"VERIFIED BSF KNOWLEDGE BASE:\n{rag_context}"

        if image_info:
            if image_info.get("is_ood"):
                context_payload += (
                    f"\n\nVISUAL ANALYSIS RESULT: OUT-OF-DISTRIBUTION REJECTION\n"
                    f"- Distance: {image_info['mahalanobis_distance']} (Threshold: {image_info['threshold']})\n"
                    f"- Action: Inform the user that the image is not recognized as a BSF stage."
                )
            elif image_info.get("status") == "success":
                context_payload += (
                    f"\n\nVISUAL ANALYSIS RESULT (VERIFIED CLASSIFICATION):\n"
                    f"- Identified Stage: {image_info['display_title']}\n"
                    f"- Model Confidence: {image_info['confidence']}%\n"
                    f"- Stage Guidelines: {image_info['stage_guide']}\n"
                    f"INSTRUCTION: Treat this identified stage as ground truth and direct your answers to it."
                )

        messages = [{"role": "system", "content": f"{system_content}\n\n{context_payload}"}]

        # Inject session context (up to max_history_turns)
        for turn in self.history[-self.max_history_turns:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_query if user_query else "Analyze this BSF image."})

        # Tokenize and stream
        tokenizer = self.processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], return_tensors="pt").to(self.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        for new_text in streamer:
            sys.stdout.write(new_text)
            sys.stdout.flush()
            full_response += new_text

        print()

        # Update persistent history
        self.history.append({"role": "user", "content": user_query if user_query else "Uploaded image"})
        self.history.append({"role": "assistant", "content": full_response.strip()})

        return full_response.strip()


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================
if __name__ == "__main__":
    engine = FoggyBSFEngine()
    print("\n🌱 Foggy Multi-Modal Core Active. Type 'exit' to quit or 'clear' to reset history.\n")

    while True:
        try:
            raw_input = input("You (format: '<image_path> <prompt>' or just prompt): ")

            if raw_input.strip().lower() == "exit":
                break

            if raw_input.strip().lower() == "clear":
                engine.history = []
                print("🧹 Session context cleared.")
                continue

            if not raw_input.strip():
                continue

            # Extract image filename pattern (e.g., testimage1.jpeg, photo.png)
            image_match = re.search(r'\b[\w\-\\./]+\.(?:jpeg|jpg|png)\b', raw_input, re.IGNORECASE)
            image_info = None
            user_text_query = raw_input

            if image_match:
                img_path = image_match.group(0)
                if os.path.exists(img_path):
                    print(f"🔍 [SYSTEM: Running SigLIP 2 + OOD check on '{img_path}'...]")
                    image_info = engine.analyze_image(img_path)
                    user_text_query = raw_input.replace(img_path, "").strip()

                    if image_info.get("status") == "success":
                        print(f"🎯 Classifier Output: {image_info['display_title']} ({image_info['confidence']}% confidence)")
                    elif image_info.get("is_ood"):
                        print(f"⚠️ OOD Defense: Image rejected (Distance: {image_info['mahalanobis_distance']})")
                else:
                    print(f"⚠️ Warning: Image '{img_path}' was not found on disk.")

            print("\nFoggy: ", end="")
            sys.stdout.flush()

            engine.stream_integrated_response(user_query=user_text_query, image_info=image_info)
            print("\n" + "━" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting Foggy Engine...")
            break