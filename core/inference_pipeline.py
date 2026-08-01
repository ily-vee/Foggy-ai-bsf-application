"""
inference_pipeline.py

Unified Interactive Engine for Foggy AI.
Supports streaming tokens, single-line image+text input, optional text-only queries,
and persistent session memory.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from threading import Thread
from transformers import (
    AutoProcessor, 
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextIteratorStreamer
)
from peft import LoraConfig, get_peft_model, PeftModel

# Configuration Paths
SIGLIP_BASE = "google/siglip2-base-patch16-224"
QWEN_BASE = "Qwen/Qwen2.5-3B-Instruct"

SIGLIP2_DIR = "models/siglip2_bsf_lora"
QWEN_LORA_DIR = "models/qwen_bsf_qlora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

RAG_KNOWLEDGE_BASE = {
    "egg": "Optimal temp: 27-30°C. Moisture: >60% RH. Eggs must sit in dry crevices above feed. Hatching time: ~4 days.",
    "larva": "Optimum substrate temp: 27-30°C (max 35°C). Moisture target: 65-70%. Feed conversion peak at 5th instar. Split overcrowded trays.",
    "prepupa": "Non-feeding stage. Ramps needed at 30-45 degree incline. Pupation medium depth: 15-20cm, 60% moisture.",
    "pupa": "Non-feeding, motionless. Emergence: 7-14 days at 27-30°C. Medium moisture: ~60%.",
    "adult": "Non-feeding on solids. Requires liquid water / sugar solution. Temp: 27-30°C, RH: 70%. Requires strong light for flight mating."
}

class Siglip2BSFClassifier(torch.nn.Module):
    def __init__(self, base_vision_model, num_classes, hidden_dim=768):
        super().__init__()
        self.vision_model = base_vision_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits, pooled

class FoggyEngine:
    def __init__(self):
        print("🔹 Loading Vision Model & OOD Data...")
        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_BASE)
        full_model = AutoModel.from_pretrained(SIGLIP_BASE)
        base_vision_model = full_model.vision_model

        with open(f"{SIGLIP2_DIR}/class_mapping.json", "r") as f:
            self.class_mapping = json.load(f)
        num_classes = len(self.class_mapping)

        peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
        )
        lora_vision_model = get_peft_model(base_vision_model, peft_config)
        
        self.vision_classifier = Siglip2BSFClassifier(
            lora_vision_model, num_classes=num_classes, hidden_dim=base_vision_model.config.hidden_size
        )
        state_dict = torch.load(f"{SIGLIP2_DIR}/siglip2_bsf_model.pt", map_location=device, weights_only=True)
        self.vision_classifier.load_state_dict(state_dict)
        self.vision_classifier.to(device)
        self.vision_classifier.eval()

        self.ood_data = torch.load(f"{SIGLIP2_DIR}/ood_embeddings.pt", map_location="cpu", weights_only=False)

        print("🔹 Loading Language Model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
        base_qwen = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if os.path.exists(QWEN_LORA_DIR):
            self.qwen_model = PeftModel.from_pretrained(base_qwen, QWEN_LORA_DIR)
        else:
            self.qwen_model = base_qwen

    def check_ood(self, embedding_vec, threshold=0.55):
        norm_emb = embedding_vec / np.linalg.norm(embedding_vec)
        max_sim = -1.0
        
        for stage, centroid in self.ood_data["centroids"].items():
            norm_centroid = centroid / np.linalg.norm(centroid)
            sim = np.dot(norm_emb, norm_centroid)
            if sim > max_sim:
                max_sim = sim
                
        is_ood = max_sim < threshold
        return is_ood, float(max_sim)

    def analyze_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = self.siglip_processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits, pooled_emb = self.vision_classifier(inputs["pixel_values"])
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item() * 100
            
        detected_stage = self.class_mapping[str(pred_idx)]
        emb_np = pooled_emb.cpu().squeeze(0).numpy()

        is_ood, sim_score = self.check_ood(emb_np)
        return is_ood, sim_score, detected_stage, confidence

    def generate_stream(self, image_path: str, user_query: str):
        # Case A: An image is attached
        if image_path:
            is_ood, sim_score, detected_stage, confidence = self.analyze_image(image_path)
            
            if is_ood:
                print(f"\n⚠️ **Out-of-Distribution Warning**: Image similarity ({sim_score*100:.1f}%) is below threshold.")
                print("Please provide a clear image of BSF eggs, larvae, prepupae, pupae, or adults.\n")
                return

            print(f"\n🔍 [Detected Stage: {detected_stage.capitalize()} ({confidence:.1f}% Confidence)]")
            rag_context = RAG_KNOWLEDGE_BASE.get(detected_stage, "No specific guidelines available.")

            prompt = (
                f"[Vision Pipeline Analysis]\n"
                f"Detected Stage: {detected_stage.capitalize()}\n"
                f"Stage Confidence: {confidence:.1f}%\n"
                f"Status: In-Distribution\n\n"
                f"[Retrieved Reference Manual Context]\n{rag_context}\n\n"
                f"User Question: {user_query}"
            )

        # Case B: Text-only query (No image attached or clear command used)
        else:
            prompt = f"User Question: {user_query}"

        messages = [
            {"role": "system", "content": "You are Foggy, an expert AI assistant for Black Soldier Fly precision farming. Provide clear, direct, and actionable step-by-step guidance."},
            {"role": "user", "content": prompt}
        ]

        text_input = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.qwen_tokenizer([text_input], return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(self.qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=384,
            temperature=0.3,
            top_p=0.9
        )

        thread = Thread(target=self.qwen_model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\n[Foggy AI]: ", end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)
        print("\n")

def parse_input_line(input_str: str, current_active_image: str):
    """
    Parses a single input line into (image_path, query).
    Handles reset keywords, paths containing spaces/quotes, and image extensions.
    """
    clean_input = input_str.strip()
    
    # 1. Handle image memory clearing commands
    reset_keywords = ("clear", "reset", "clear image", "clear img", "no image", "no img")
    if clean_input.lower() in reset_keywords or clean_input.lower().startswith(("clear ", "reset ")):
        # If user typed 'clear' or 'reset' with extra query text afterwards
        query_after_clear = ""
        parts = clean_input.split(maxsplit=1)
        if len(parts) > 1 and not parts[1].strip().lower().endswith(IMAGE_EXTENSIONS):
            query_after_clear = parts[1].strip()
            
        return None, query_after_clear

    tokens = clean_input.split()
    if not tokens:
        return current_active_image, ""

    first_token = tokens[0].strip("'\"")

    # 2. Check if input starts with a valid image path
    if first_token.lower().endswith(IMAGE_EXTENSIONS) or os.path.exists(first_token):
        potential_img = first_token
        query_parts = tokens[1:]
    else:
        # Text-only query; retain previously active image if available
        potential_img = current_active_image
        query_parts = tokens

    query = " ".join(query_parts).strip()
    return potential_img, query

def main():
    engine = FoggyEngine()
    print("\n" + "="*60)
    print("        🐛 FOGGY AI INTERACTIVE TERMINAL ENGINE 🐛")
    print(" Options:")
    print("  • Both:    dataset/test.jpg What is the target moisture?")
    print("  • Text:    What temperature is needed for BSF eggs?")
    print("  • Reset:   clear image (removes active image memory)")
    print("  • Exit:    Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    current_image = None

    while True:
        try:
            status_indicator = f"📷 [{current_image}]" if current_image else "💬 [Text Mode]"
            user_input = input(f"{status_indicator} > ").strip()

            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            image_path, query = parse_input_line(user_input, current_image)

            # 1. Check if user cleared image memory
            if current_image and image_path is None:
                current_image = None
                print("🧹 Image memory cleared. Switched to Text Mode.\n")
                if not query:
                    continue  # Stop execution here - don't stream anything!

            # 2. Validate image path if provided
            if image_path and not os.path.exists(image_path):
                print(f"❌ File not found: '{image_path}'. Running in text-only mode.\n")
                image_path = None

            current_image = image_path

            # 3. Default prompt if an image is attached without a explicit question
            if not query and image_path:
                query = "Identify the life stage shown in this image and provide general guidance."

            # 4. Guard against empty queries in text mode
            if not query and not image_path:
                continue

            # 5. Stream generation
            engine.generate_stream(image_path, query)

        except (KeyboardInterrupt, EOFError):
            break

    print("\n👋 Exiting Foggy AI. Goodbye!")


if __name__ == "__main__":
    main()