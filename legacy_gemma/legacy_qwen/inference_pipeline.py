"""
core/inference_pipeline.py  (PATCHED)

Unified Interactive Engine for Foggy AI.
Features:
- Dual-mode input (image + query or text-only)
- Streamed generation via TextIteratorStreamer
- Out-Of-Distribution (OOD) similarity thresholding
- In-memory RAG protocol augmentation
- Bounded conversation history (real session memory)
- Post-hoc citation sanitization (safety net against hallucinated sources)
- Session memory management and clean teardown
"""

import os
import re
import sys
import json
import logging
from typing import Optional, Tuple, List, Dict
from threading import Thread

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from peft import LoraConfig, get_peft_model, PeftModel

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FoggyEngine")

# --- System Configuration ---
SIGLIP_BASE = "google/siglip2-base-patch16-224"
QWEN_BASE = "Qwen/Qwen2.5-3B-Instruct"

SIGLIP2_DIR = "models/siglip2_bsf_lora"
QWEN_LORA_DIR = "models/qwen_bsf_qlora"

OOD_THRESHOLD = 0.55
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How many prior (user, assistant) turns to keep in context.
# Keep this small — Qwen2.5-3B has a limited effective context and BSF
# answers don't need deep history, just enough to stay self-consistent.
MAX_HISTORY_TURNS = 5

# Safety-net regex for citation-style hallucinations. This does NOT fix the
# root cause (likely citation-formatted text in the QLoRA SFT set) but strips
# fabricated DOIs/footnotes before they reach the user.
CITATION_PATTERNS = [
    r"\[cite:\s*[^\]]*\]",       # [cite: 10.1007/...]
    r"\[\^cite:\d+\]",           # [^cite:1]
    r"\[cite-link\]",           # [cite-link]
    r"\[\^\d+\]",                # [^1]
]
CITATION_RE = re.compile("|".join(CITATION_PATTERNS))

RAG_KNOWLEDGE_BASE = {
    "egg": "Optimal temp: 27-30°C. Moisture: >60% RH. Eggs must sit in dry crevices above feed. Hatching time: ~4 days.",
    "larva": "Optimum substrate temp: 27-30°C (max 35°C). Moisture target: 65-70%. Feed conversion peak at 5th instar. Split overcrowded trays. Feeding stage duration: 13-18 days.",
    "prepupa": "Non-feeding stage. Ramps needed at 30-45 degree incline. Pupation medium depth: 15-20cm, 60% moisture. Wandering prepupal duration: 7-10 days.",
    "pupa": "Non-feeding, motionless. Emergence: 7-14 days at 27-30°C. Medium moisture: ~60%.",
    "adult": "Non-feeding on solids. Requires liquid water / sugar solution. Temp: 27-30°C, RH: 70%. Requires strong light for flight mating. Lifespan: 5-8 days minimum without feeding, up to 16-40+ days with water/sugar solution."
}

# Keyword map for text-only RAG retrieval. Without this, any query typed in
# pure text mode (no image) gets ZERO reference context, and the model falls
# back entirely on fine-tuned parametric recall — which is why numbers like
# larval feeding duration drift between "13-18", "16-19", "12-15+" across
# turns even though the training data is internally consistent at "13-18".
STAGE_KEYWORDS = {
    "egg": ["egg", "eggs", "hatch", "hatching", "oviposition", "lay eggs"],
    "larva": ["larva", "larvae", "larval", "feeding stage", "instar", "wandering", "feed rate"],
    "prepupa": ["prepupa", "prepupae", "pre-pupa", "pre-pupae", "self-harvest", "exit ramp", "migrate"],
    "pupa": ["pupa", "pupae", "pupal", "pupation", "cocoon", "emerge", "emergence"],
    "adult": ["adult", "adults", "fly", "flies", "mating", "mate", "cage", "sugar solution", "lifespan"],
}


def detect_stage_from_text(query: str) -> Optional[str]:
    """Best-effort keyword match to pick a relevant KB entry for text-only
    queries. Not a real retriever — just enough to stop the model from
    answering numeric questions with zero grounding. Replace with a proper
    embedding-based retriever if query variety grows."""
    q_lower = query.lower()
    best_stage, best_hits = None, 0
    for stage, keywords in STAGE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q_lower)
        if hits > best_hits:
            best_stage, best_hits = stage, hits
    return best_stage if best_hits > 0 else None

SYSTEM_PROMPT = (
    "You are Foggy, an expert AI assistant for Black Soldier Fly precision farming. "
    "Provide clear, direct, and actionable guidance.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "1. Base every specific number (temperature, moisture %, duration, quantity) ONLY on the "
    "values given in the [Reference Context] or earlier in this conversation. Never invent, "
    "adjust, or 'improve' a number that isn't present there.\n"
    "2. If the user's question asks for a detail not covered by the reference context, say so "
    "plainly (e.g. 'The reference data doesn't specify this — as general practice...') rather "
    "than fabricating a precise figure.\n"
    "3. Never include citations, DOIs, footnote markers, or references to external papers "
    "(e.g. '[cite: ...]', '[^1]'). You have no access to external sources — only the reference "
    "context provided. Presenting a fabricated citation is worse than no citation.\n"
    "4. Use Celsius only, matching the reference context. Do not switch to Fahrenheit.\n"
    "5. Stay consistent with what you or the user said earlier in this conversation."
)


class Siglip2BSFClassifier(torch.nn.Module):
    """Vision Head wrapper for SigLIP 2 feature embeddings."""
    def __init__(self, base_vision_model: torch.nn.Module, num_classes: int, hidden_dim: int = 768):
        super().__init__()
        self.vision_model = base_vision_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits, pooled


class FoggyEngine:
    """Core multi-modal inference engine for Foggy AI."""
    def __init__(self):
        logger.info(f"Initializing Foggy Engine on device: {DEVICE}")

        # Bounded conversation history: list of {"role": ..., "content": ...}
        # excluding the system prompt (which is re-added fresh each call).
        self.chat_history: List[Dict[str, str]] = []

        # 1. Load Vision Components
        self._load_vision_pipeline()

        # 2. Load LLM Components
        self._load_language_pipeline()

    def _load_vision_pipeline(self):
        logger.info("Loading SigLIP 2 Vision Pipeline & OOD Checkpoints...")

        mapping_path = os.path.join(SIGLIP2_DIR, "class_mapping.json")
        weights_path = os.path.join(SIGLIP2_DIR, "siglip2_bsf_model.pt")
        ood_path = os.path.join(SIGLIP2_DIR, "ood_embeddings.pt")

        if not os.path.exists(weights_path) or not os.path.exists(ood_path):
            raise FileNotFoundError(
                f"Missing vision artifacts in '{SIGLIP2_DIR}'. "
                "Ensure siglip2_bsf_model.pt, class_mapping.json, and ood_embeddings.pt exist."
            )

        with open(mapping_path, "r") as f:
            self.class_mapping = json.load(f)

        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_BASE)
        full_model = AutoModel.from_pretrained(SIGLIP_BASE)

        peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
        )
        lora_vision_model = get_peft_model(full_model.vision_model, peft_config)

        self.vision_classifier = Siglip2BSFClassifier(
            lora_vision_model,
            num_classes=len(self.class_mapping),
            hidden_dim=full_model.config.vision_config.hidden_size
        )

        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        self.vision_classifier.load_state_dict(state_dict)
        self.vision_classifier.to(DEVICE)
        self.vision_classifier.eval()

        self.ood_data = torch.load(ood_path, map_location="cpu", weights_only=False)

    def _load_language_pipeline(self):
        logger.info("Loading Qwen2.5 Language Model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_qwen = AutoModelForCausalLM.from_pretrained(
            QWEN_BASE,
            device_map="auto",
            torch_dtype=torch_dtype
        )

        if os.path.exists(QWEN_LORA_DIR):
            logger.info(f"Applying QLoRA adapters from '{QWEN_LORA_DIR}'...")
            self.qwen_model = PeftModel.from_pretrained(base_qwen, QWEN_LORA_DIR)
        else:
            logger.warning(f"No LoRA adapter found at '{QWEN_LORA_DIR}'. Running base Qwen2.5.")
            self.qwen_model = base_qwen

        self.qwen_model.eval()

        # Build a hard ban list of tokens whose decoded text contains CJK or
        # fullwidth-form characters. This is an English-only farming
        # assistant; there is no legitimate reason for these tokens to ever
        # appear, and banning them at the logits level is more reliable than
        # hoping generation settings alone prevent leakage.
        logger.info("Scanning vocabulary for non-Latin tokens to suppress...")
        cjk_re = re.compile(r'[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]')
        self.banned_token_ids = [
            [tid] for tok, tid in self.qwen_tokenizer.get_vocab().items()
            if cjk_re.search(self.qwen_tokenizer.convert_tokens_to_string([tok]))
        ]
        logger.info(f"Suppressing {len(self.banned_token_ids)} non-Latin tokens.")

    def check_ood(self, embedding_vec: np.ndarray) -> Tuple[bool, float]:
        norm_emb = embedding_vec / (np.linalg.norm(embedding_vec) + 1e-8)
        max_sim = -1.0

        for centroid in self.ood_data["centroids"].values():
            norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim = float(np.dot(norm_emb, norm_centroid))
            if sim > max_sim:
                max_sim = sim

        is_ood = max_sim < OOD_THRESHOLD
        return is_ood, max_sim

    def analyze_image(self, image_path: str) -> Tuple[bool, float, str, float]:
        try:
            img = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to open image: {e}")
            raise ValueError(f"Invalid or corrupted image file: '{image_path}'")

        inputs = self.siglip_processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            logits, pooled_emb = self.vision_classifier(inputs["pixel_values"])
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item() * 100

        detected_stage = self.class_mapping[str(pred_idx)]
        emb_np = pooled_emb.cpu().squeeze(0).numpy()

        is_ood, sim_score = self.check_ood(emb_np)
        return is_ood, sim_score, detected_stage, confidence

    def _sanitize(self, text: str) -> str:
        """Strip hallucinated citation/footnote markup as a safety net."""
        cleaned = CITATION_RE.sub("", text)
        # Collapse any double spaces left behind by removed markup.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned

    def _trim_history(self):
        # Keep only the last MAX_HISTORY_TURNS (user, assistant) pairs.
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    def reset_history(self):
        """Call this when the user explicitly clears/resets the session."""
        self.chat_history = []

    def generate_stream(self, image_path: Optional[str], user_query: str):
        if image_path:
            try:
                is_ood, sim_score, detected_stage, confidence = self.analyze_image(image_path)
            except ValueError as err:
                print(f"\n❌ Error: {err}\n")
                return

            if is_ood:
                print(f"\n⚠️  **Out-of-Distribution Warning**: Similarity score ({sim_score*100:.1f}%) is below safety threshold.")
                print("    Please provide a clear image of BSF eggs, larvae, prepupae, pupae, or adults.\n")
                return

            print(f"\n🔍 [Detected Stage: {detected_stage.capitalize()} ({confidence:.1f}% Confidence)]")
            rag_context = RAG_KNOWLEDGE_BASE.get(detected_stage, "No specific guidelines available.")

            prompt = (
                f"[Vision Analysis]\nDetected Stage: {detected_stage.capitalize()} ({confidence:.1f}% confidence)\n\n"
                f"[Reference Context]\n{rag_context}\n\n"
                f"User Question: {user_query}"
            )
        else:
            # Text-only mode previously got NO reference context at all,
            # leaving the model to answer numeric questions from parametric
            # recall alone — the cause of day-range drift across turns.
            # Do a best-effort keyword match to inject grounding context here too.
            detected_stage = detect_stage_from_text(user_query)
            if detected_stage:
                rag_context = RAG_KNOWLEDGE_BASE[detected_stage]
                prompt = (
                    f"[Reference Context — {detected_stage.capitalize()} stage]\n{rag_context}\n\n"
                    f"User Question: {user_query}"
                )
            else:
                prompt = f"User Question: {user_query}"

        # Build messages: system + bounded history + current turn.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": prompt})

        text_input = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.qwen_tokenizer([text_input], return_tensors="pt").to(DEVICE)

        streamer = TextIteratorStreamer(self.qwen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            bad_words_ids=self.banned_token_ids,
        )

        thread = Thread(target=self.qwen_model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\n[Foggy AI]: ", end="", flush=True)
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            print(new_text, end="", flush=True)
        print("\n")

        # Ensure the generation thread has fully finished before returning —
        # without this join, the next input() prompt can race the tail end
        # of generation on slower systems.
        thread.join()

        # Post-hoc grounding safety net: flag (don't silently hide) if the
        # model fabricated citation-style markup, so you can track how often
        # this is still happening after prompt/training fixes.
        if CITATION_RE.search(full_response):
            logger.warning("Detected hallucinated citation markup in response — check QLoRA SFT data for citation-formatted training examples.")

        sanitized = self._sanitize(full_response)

        # Store the bounded turn history using the sanitized response, so
        # future turns don't inherit fabricated citations either.
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": sanitized})
        self._trim_history()


def parse_input_line(input_str: str, current_active_image: Optional[str]) -> Tuple[Optional[str], str]:
    """Parses single-line interactive terminal input."""
    clean_input = input_str.strip()
    reset_keywords = ("clear", "reset", "clear image", "clear img", "no image", "no img")

    if clean_input.lower() in reset_keywords or clean_input.lower().startswith(("clear ", "reset ")):
        parts = clean_input.split(maxsplit=1)
        query_after = parts[1].strip() if len(parts) > 1 and not parts[1].strip().lower().endswith(IMAGE_EXTENSIONS) else ""
        return None, query_after

    tokens = clean_input.split()
    if not tokens:
        return current_active_image, ""

    first_token = tokens[0].strip("'\"")

    if first_token.lower().endswith(IMAGE_EXTENSIONS) or os.path.exists(first_token):
        return first_token, " ".join(tokens[1:]).strip()

    return current_active_image, clean_input


def main():
    engine = FoggyEngine()

    print("\n" + "="*60)
    print("        🐛 FOGGY AI INTERACTIVE TERMINAL ENGINE 🐛")
    print("  • Combined: dataset/test.jpg What is the target moisture?")
    print("  • Text:     What temperature is needed for BSF eggs?")
    print("  • Reset:    'clear' or 'reset' (switches to text mode + clears history)")
    print("  • Exit:     'exit' or 'quit'")
    print("="*60 + "\n")

    current_image: Optional[str] = None

    while True:
        try:
            status_indicator = f"📷 [{current_image}]" if current_image else "💬 [Text Mode]"
            user_input = input(f"{status_indicator} > ").strip()

            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            image_path, query = parse_input_line(user_input, current_image)

            # Detect Image Clearing
            if current_image and image_path is None:
                current_image = None
                print("🧹 Image memory cleared. Switched to Text Mode.\n")
                if not query:
                    continue

            # Explicit reset also clears conversation history, not just the image.
            if user_input.lower() in ("clear", "reset"):
                engine.reset_history()

            # Validate Image Path
            if image_path and not os.path.exists(image_path):
                print(f"❌ File not found: '{image_path}'. Defaulting to Text Mode.\n")
                image_path = None

            current_image = image_path

            # Default Prompt for image-only queries
            if not query and image_path:
                query = "Identify the life stage shown in this image and provide general guidance."

            if not query and not image_path:
                continue

            engine.generate_stream(image_path, query)

        except (KeyboardInterrupt, EOFError):
            break

    print("\n👋 Exiting Foggy AI Engine. Goodbye!")


if __name__ == "__main__":
    main()