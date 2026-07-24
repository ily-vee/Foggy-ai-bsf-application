import os
import re
import json
import sys
import torch
import torch.nn as nn
import numpy as np
from threading import Thread
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("⚙️ Booting Unified Foggy Engine Core (Fixed Vision Pipeline)...")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CACHE_DIR = "foggy_vector_db"
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
VECTORS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")

CLASSIFIER_PATH = "bsf_classifier.pth"
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CLASS_NAMES = [
    "1_eggs",
    "2_early_larvae",
    "3_feeding_larvae",
    "4_pupae",
    "5_bsf_adult"
]

STAGE_DISPLAY_MAP = {
    "1_eggs": {
        "title": "Egg Stage (Egg Clutches)",
        "desc": "Keep incubation area moist (60-70%) and sheltered from direct sunlight. Ensure hatchlings have immediate access to soft feed substrate."
    },
    "2_early_larvae": {
        "title": "Early Larvae (Neonates / Young Instars)",
        "desc": "Requires easily digestible, finely ground substrate with 65-70% moisture. Maintain temperatures around 27°C–30°C."
    },
    "3_feeding_larvae": {
        "title": "Feeding Larvae (Active Processing Phase)",
        "desc": "Peak voracity phase. Feed organic waste daily (up to 10% body weight equivalent). Ensure proper airflow and drain excess leachate."
    },
    "4_pupae": {
        "title": "Prepupae / Pupae Stage",
        "desc": "Larvae have stopped feeding and turned dark brown/black. Provide a clean, dry, and dark crawl-off migration area for pupation."
    },
    "5_bsf_adult": {
        "title": "Adult Fly Stage (Love Cages / Breeding)",
        "desc": "Adults require bright sunlight or full-spectrum lighting and clean water misting. They do not eat waste—focus on mating humidity and egg traps."
    }
}

CONFIDENCE_THRESHOLD = 50.0

# ==========================================
# 1. VECTOR RAG RETRIEVAL SETUP
# ==========================================
print("📚 Loading Text Embedding Model for RAG...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_embeddings = np.load(VECTORS_PATH)
    print(f"✅ Vector Database Connected. Loaded {len(all_chunks)} knowledge nodes.")
else:
    all_chunks = [
        "Optimal temperature range for Black Soldier Fly (BSF) larvae development is 27°C to 30°C.",
        "Feeding larvae require high moisture (60-70%) and nutrient-rich organic waste.",
        "Adult flies require clean water misting and direct sunlight or 6000K LED lighting for optimal mating.",
        "Prepupae migrate away from high moisture toward dry, dark areas to pupate."
    ]
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    print("⚠️ Warning: Vector DB not found, using fallback memory.")

def retrieve_relevant_context(user_query, top_k=3):
    query_vector = embed_model.encode([user_query], convert_to_numpy=True)
    scores = np.dot(chunk_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join([all_chunks[idx] for idx in top_indices])


# ==========================================
# 2. PYTORCH MLP CLASSIFIER ARCHITECTURE
# ==========================================
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim=1280, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. UNIFIED MULTIMODAL FOGGY ENGINE
# ==========================================
class FoggyEngineQwen:
    def __init__(self, max_history_turns=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"👁️ Loading Qwen 2.5-VL Model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_ID,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=self.device
        )

        if hasattr(self.model, "visual"):
            self.vision_encoder = self.model.visual
        elif hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            self.vision_encoder = self.model.model.visual
        else:
            raise AttributeError("Unable to locate visual encoder inside Qwen model.")

        self.vision_encoder.eval()

        print("🧠 Loading Trained Classifier Head...")
        self.classifier = BSFClassifierHead(input_dim=1280, num_classes=len(CLASS_NAMES)).to(self.device)
        if os.path.exists(CLASSIFIER_PATH):
            self.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=self.device))
            self.classifier.eval()
            print("✅ PyTorch classifier head successfully loaded.")
        else:
            print(f"⚠️ Warning: '{CLASSIFIER_PATH}' not found!")

        self.history = []
        self.max_history_turns = max_history_turns

    def run_vision_inference(self, image_path):
        """Passes image through Qwen visual backbone + MLP classifier head cleanly."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Pass through visual encoder backbone
                output = self.vision_encoder(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
                
                if hasattr(output, "last_hidden_state"):
                    patch_embeds = output.last_hidden_state
                elif isinstance(output, (tuple, list)):
                    patch_embeds = output[0]
                else:
                    patch_embeds = output

                # Global Mean Pooling across patch embeddings safely
                if patch_embeds.ndim == 3:
                    feature_vec = patch_embeds.mean(dim=1)  # Batch x Features
                elif patch_embeds.ndim == 2:
                    feature_vec = patch_embeds.mean(dim=0, keepdim=True)
                else:
                    feature_vec = patch_embeds

                # Enforce float32 for PyTorch BatchNorm precision
                feature_vec = feature_vec.to(dtype=torch.float32)

                logits = self.classifier(feature_vec)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(torch.argmax(logits, dim=1).item())

            predicted_raw = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx] * 100)

            display_info = STAGE_DISPLAY_MAP.get(predicted_raw, {
                "title": predicted_raw,
                "desc": "Standard Black Soldier Fly management principles apply."
            })

            return {
                "raw_label": predicted_raw,
                "display_title": display_info["title"],
                "stage_guide": display_info["desc"],
                "confidence": confidence,
                "is_low_confidence": confidence < CONFIDENCE_THRESHOLD
            }
        except Exception as e:
            print(f"❌ Vision Error: {e}")
            return None

    def stream_integrated_response(self, user_query, image_info=None):
        rag_context = retrieve_relevant_context(user_query, top_k=3)

        system_content = (
            "You are Foggy, a precision AI assistant specialized in Black Soldier Fly (BSF) farming.\n"
            "STRICT MANDATE: Never state 'I don't have the ability to analyze images'. "
            "You possess an integrated PyTorch computer vision engine that pre-analyzes uploaded images for you.\n\n"
            "OUT-OF-SCOPE PROTOCOL:\n"
            "If the user asks a non-BSF question:\n"
            "1. Identify yourself as Foggy, a BSF assistant.\n"
            "2. Identify the target knowledge field/domain of the question.\n"
            "3. Answer the user's question fully.\n\n"
            "IN-SCOPE BSF PROTOCOL:\n"
            "Answer directly and practically in clear farmer-friendly terms."
        )

        context_payload = f"VERIFIED BSF KNOWLEDGE BASE:\n{rag_context}"

        if image_info:
            if image_info.get("is_low_confidence"):
                context_payload += (
                    f"\n\nVISUAL ANALYSIS RESULT: Low confidence prediction ({image_info['confidence']:.2f}%). "
                    "Politely inform the farmer that the image is unclear or out of focus."
                )
            else:
                context_payload += (
                    f"\n\nVISUAL ANALYSIS RESULT (VERIFIED CLASSIFICATION):\n"
                    f"- Identified BSF Stage: {image_info['display_title']}\n"
                    f"- Model Confidence: {image_info['confidence']:.2f}%\n"
                    f"- Recommended Actions: {image_info['stage_guide']}\n"
                    "INSTRUCTION: Treat the identified stage as confirmed fact and answer all parts of the farmer's prompt specifically for this stage."
                )

        messages = [
            {"role": "system", "content": f"{system_content}\n\n{context_payload}"}
        ]
        
        for turn in self.history[-self.max_history_turns:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_query})

        tokenizer = self.processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], return_tensors="pt").to(self.device)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
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

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": full_response.strip()})

        return full_response.strip()


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================
if __name__ == "__main__":
    engine = FoggyEngineQwen()
    print("\n🌱 Foggy Multi-Modal Core Active. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You (format: '<image_path> <prompt>'): ")
            
            if user_input.strip().lower() == "exit":
                break

            if user_input.strip().lower() == "clear":
                engine.history = []
                print("🧹 Conversation history cleared.")
                continue

            if not user_input.strip():
                continue

            image_match = re.search(r'\b\w+\.(?:jpeg|jpg|png)\b', user_input, re.IGNORECASE)
            image_info = None

            if image_match:
                img_path = image_match.group(0)
                if os.path.exists(img_path):
                    print(f"[SYSTEM: Extracting features from '{img_path}' via Vision Backbone...]")
                    image_info = engine.run_vision_inference(img_path)
                    if image_info:
                        print(f"🎯 Classifier Output: {image_info['display_title']} ({image_info['confidence']:.2f}% confidence)")
                    user_input = user_input.replace(img_path, "").strip()
                else:
                    print(f"⚠️ Warning: Image file '{img_path}' not found on disk.")

            print("\nFoggy: ", end="")
            sys.stdout.flush()

            engine.stream_integrated_response(user_query=user_input, image_info=image_info)
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting Foggy Core...")
            break