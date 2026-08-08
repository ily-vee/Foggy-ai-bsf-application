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
import time
import queue
import logging
from typing import Optional, Tuple, List, Dict
from threading import Thread

import torch
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from peft import LoraConfig, get_peft_model, PeftModel

try:
    from retriever import HybridRetriever
except ImportError:
    HybridRetriever = None

try:
    from bsf_calculators import (
        calculate_feed_quantity,
        estimate_stage_transition,
        calculate_moisture_dilution,
        FEED_RATE_MIN_G_PER_LARVA_DAY,
        FEED_RATE_MAX_G_PER_LARVA_DAY,
    )
except ImportError:
    calculate_feed_quantity = None
    estimate_stage_transition = None
    calculate_moisture_dilution = None

try:
    from audio_transcriber import transcribe_audio, load_whisper_model
except ImportError:
    transcribe_audio = None
    load_whisper_model = None

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
VECTOR_DB_DIR = "foggy_vector_db"

OOD_THRESHOLD = 0.55
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
AUDIO_EXTENSIONS = ('.ogg', '.wav', '.mp3', '.m4a', '.flac', '.opus')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How long to wait for the next token before treating generation as hung.
# 512 max_new_tokens on a 3B model should never realistically take this
# long per-token; this exists purely as a circuit breaker against a wedged
# generate() call blocking a request indefinitely.
GENERATION_TIMEOUT_SECONDS = 120

# How many prior (user, assistant) turns to keep in context.
# Keep this small — Qwen2.5-3B has a limited effective context and BSF
# answers don't need deep history, just enough to stay self-consistent.
MAX_HISTORY_TURNS = 3

# Safety-net regex for citation-style hallucinations. This does NOT fix the
# root cause (likely citation-formatted text in the QLoRA SFT set) but strips
# fabricated DOIs/footnotes before they reach the user.
CITATION_PATTERNS = [
    r"\[cite:\s*[^\]]*\]",       # [cite: 10.1007/...]
    r"\[\^cite:\d+\]",           # [^cite:1]
    r"\[cite-link\]",           # [cite-link]
    r"\[\^\d+\]",                # [^1]
    r"\(Source:\s*[^)]*\)",     # (Source: manual.pdf)
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


# --- Numeric intent detection for the compute-first tool layer ---
# Deliberately simple regex matching, not NLU. It will miss many phrasings —
# that's an acceptable trade-off here, since a missed match just means the
# model answers in prose as before (no regression), whereas a wrong match
# risks feeding it an irrelevant computed number. When this starts missing
# too often in practice, the more robust next step is a small structured-
# extraction pass (even a constrained LLM call) rather than more regex.
FEED_INTENT_RE = re.compile(r"how much (feed|food)|feed (quantity|amount|rate)|how much to feed", re.I)
STAGE_TIMING_INTENT_RE = re.compile(r"when (will|does|do)|how (long|many days?)|ready to|pupate|emerge", re.I)
MOISTURE_INTENT_RE = re.compile(r"moisture|water to add|dilut", re.I)

NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(larvae|larva|kg|kilograms?|kilos?)?", re.I)
MOISTURE_TRIPLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kg.*?(\d+(?:\.\d+)?)\s*%.*?(\d+(?:\.\d+)?)\s*%", re.I)

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
    "3. Never include citations, DOIs, footnote markers, or source/document references of any "
    "kind (e.g. '[cite: ...]', '[^1]', '(Source: ...)') in your answer. Just state the information "
    "directly, in your own words, as if you simply know it.\n"
    "4. Use Celsius only, matching the reference context. Do not switch to Fahrenheit.\n"
    "5. Stay consistent with what you or the user said earlier in this conversation.\n"
    "6. You may receive a [Visual Diagnostics] block with automated, heuristic image "
    "measurements (dominant color, possible mold/fungal color coverage, sharpness, exposure). "
    "These are approximate cues from basic image analysis, NOT a certified health diagnosis. "
    "When asked about appearance or health, reference these specific measurements directly rather "
    "than deflecting — but say plainly that it's an automated visual cue and a human should confirm "
    "in person, especially if the mold/fungal coverage figure is notably high or the image is flagged "
    "blurry/poorly exposed.\n"
    "7. If a [Computed Values] block is present, those numbers were calculated exactly in code — "
    "restate them as given rather than recalculating, rounding differently, or approximating them "
    "yourself. Never invent your own arithmetic for feed quantities, timing estimates, or dilution "
    "amounts when this block is present."
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

        # Cache of (path -> analysis) so a follow-up question about an
        # already-classified image ("do they look healthy?") doesn't re-run
        # the full SigLIP2 forward pass + OOD check + CV heuristics from
        # scratch on every turn. Keyed by path, invalidated by file mtime so
        # a genuinely new photo saved to the same path isn't served stale
        # results. Also tracks first_seen so stage-transition timing
        # estimates have something to measure "days in stage" against.
        self.image_cache: Dict[str, Dict] = {}

        # 1. Load Vision Components
        self._load_vision_pipeline()

        # 2. Load LLM Components
        self._load_language_pipeline()

        # 3. Load real hybrid (dense + BM25) retriever if a vector DB has
        # been built. Falls back to the static 5-line RAG_KNOWLEDGE_BASE
        # dict if ingest.py hasn't been run yet, so the engine still starts.
        self.retriever = None
        if HybridRetriever is not None:
            try:
                self.retriever = HybridRetriever(vector_db_dir=VECTOR_DB_DIR)
                logger.info(f"Loaded HybridRetriever from '{VECTOR_DB_DIR}'.")
            except FileNotFoundError as e:
                logger.warning(f"{e} Falling back to the static RAG_KNOWLEDGE_BASE dict.")
        else:
            logger.warning("retriever.py not importable. Falling back to the static RAG_KNOWLEDGE_BASE dict.")

        # Load audio transcription (Whisper) once at startup, same pattern
        # as the retriever: degrade gracefully rather than crash if it's
        # unavailable, since audio support is additive to the core pipeline.
        self.audio_enabled = transcribe_audio is not None
        if self.audio_enabled:
            try:
                load_whisper_model(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16" if torch.cuda.is_available() else "int8",
                )
                logger.info("Whisper audio transcription ready.")
            except Exception as e:
                logger.warning(f"Could not load Whisper model ({e}); audio input will be unavailable.")
                self.audio_enabled = False
        else:
            logger.warning("audio_transcriber.py not importable. Voice notes will be unavailable.")

    def get_reference_context(self, query: str, detected_stage: Optional[str] = None) -> str:
        """Single entry point for grounding context, used by both the
        image and text-only paths. Prefers real retrieval; falls back to
        the static per-stage dict only if the retriever is unavailable or
        found nothing relevant."""
        if self.retriever:
            context = self.retriever.retrieve(query, detected_stage=detected_stage, top_k=3)
            if context:
                return context
            # Retriever ran but nothing cleared the relevance floor — try
            # the static fallback before giving up entirely.
        if detected_stage and detected_stage in RAG_KNOWLEDGE_BASE:
            return RAG_KNOWLEDGE_BASE[detected_stage]
        if not detected_stage:
            keyword_stage = detect_stage_from_text(query)
            if keyword_stage:
                return RAG_KNOWLEDGE_BASE[keyword_stage]
        return ""

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

    def compute_visual_diagnostics(self, img: Image.Image) -> str:
        """Lightweight, non-learned heuristic signals computed directly from
        the image pixels. This is the ONLY way the (text-only) LLM can speak
        to appearance/health questions at all — without this, it has nothing
        but a stage label and confidence score, which is why it correctly
        refused to comment on "do they look healthy" before this patch.
        These are rough automated cues, not a certified diagnosis, and the
        system prompt is written to make the model say so."""
        arr = np.array(img.convert("RGB")).astype(np.float32)

        # Brightness (perceptual luma)
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        brightness = float(luma.mean())

        # Sharpness via edge-response variance (cheap Laplacian-style proxy)
        gray = img.convert("L")
        edges = np.array(gray.filter(ImageFilter.FIND_EDGES)).astype(np.float32)
        sharpness = float(edges.var())

        # HSV-based color cues
        hsv = np.array(img.convert("HSV")).astype(np.float32)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Whitish (low sat, high value) or greenish (hue band + some sat):
        # rough proxy for mold/fungal growth, NOT a trained classifier.
        whitish_mask = (sat < 40) & (val > 180)
        greenish_mask = (hue > 60) & (hue < 100) & (sat > 60)
        mold_like_pct = float((whitish_mask | greenish_mask).mean() * 100)

        r, g, b = arr.reshape(-1, 3).mean(axis=0)
        if r < 60 and g < 60 and b < 60:
            color_desc = "dark brown/black"
        elif r > 180 and g > 180 and b > 160:
            color_desc = "pale/whitish (can indicate desiccation or mold if unexpected for this stage, or early-instar larvae if expected)"
        elif r > 150 and g > 140 and b < 120:
            color_desc = "cream/off-white"
        else:
            color_desc = f"mixed/medium tone (avg RGB ~{int(r)},{int(g)},{int(b)})"

        sharp_note = "in focus" if sharpness > 50 else "blurry — treat color/mold cues below with lower confidence"
        expo_note = "well-lit" if 60 < brightness < 200 else "under/over-exposed — treat color cues below with lower confidence"

        return (
            f"Dominant coloration: {color_desc}. "
            f"Estimated whitish/greenish (possible mold or fungal growth) surface coverage: {mold_like_pct:.1f}% of frame. "
            f"Image sharpness: {sharp_note}. "
            f"Image exposure: {expo_note}."
        )

    def analyze_image(self, image_path: str) -> Tuple[bool, float, str, float, str]:
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
        diagnostics_text = self.compute_visual_diagnostics(img)
        return is_ood, sim_score, detected_stage, confidence, diagnostics_text

    def get_image_analysis(self, image_path: str) -> Tuple[bool, float, str, float, str, float]:
        """Cache-aware wrapper around analyze_image(). Returns the same
        4-tuple as before plus days_since_first_seen, used by the
        stage-transition calculator below."""
        try:
            mtime = os.path.getmtime(image_path)
        except OSError:
            mtime = None

        cached = self.image_cache.get(image_path)
        if cached is not None and cached["mtime"] == mtime:
            logger.info(f"Image cache HIT for '{image_path}' — skipping vision model + diagnostics recompute.")
            days_since = (time.time() - cached["first_seen"]) / 86400.0
            return (cached["is_ood"], cached["sim_score"], cached["detected_stage"],
                    cached["confidence"], cached["diagnostics_text"], days_since)

        logger.info(f"Image cache MISS for '{image_path}' — running full analysis.")
        is_ood, sim_score, detected_stage, confidence, diagnostics_text = self.analyze_image(image_path)
        first_seen = time.time()
        self.image_cache[image_path] = {
            "mtime": mtime, "is_ood": is_ood, "sim_score": sim_score,
            "detected_stage": detected_stage, "confidence": confidence,
            "diagnostics_text": diagnostics_text, "first_seen": first_seen,
        }
        return is_ood, sim_score, detected_stage, confidence, diagnostics_text, 0.0

    def try_compute(self, query: str, detected_stage: Optional[str] = None,
                     days_in_stage: Optional[float] = None) -> Optional[str]:
        """Compute-first numeric grounding: detects a narrow set of
        calculation intents via regex, runs real arithmetic, and returns a
        formatted string for the [Computed Values] prompt block. Returns
        None if no known calculation applies OR the required calculator
        module isn't importable — callers must handle that gracefully
        rather than assuming a computed block always exists."""
        if calculate_feed_quantity is None:
            return None

        q = query

        if FEED_INTENT_RE.search(q):
            num_match = NUM_UNIT_RE.search(q)
            if num_match:
                value, unit = num_match.groups()
                try:
                    if unit and unit.lower().startswith(("kg", "kilo")):
                        result = calculate_feed_quantity(biomass_kg=float(value))
                        note = " (larvae count estimated from biomass using an approximate average weight — treat as a rough figure, not a precise count)"
                    else:
                        result = calculate_feed_quantity(larvae_count=float(value))
                        note = ""
                    return (
                        f"Feed quantity for {result['larvae_count_used']} larvae{note}: "
                        f"{result['feed_min_g_per_day']}-{result['feed_max_g_per_day']} g/day "
                        f"({result['feed_min_kg_per_day']}-{result['feed_max_kg_per_day']} kg/day), "
                        f"based on a documented rate of {FEED_RATE_MIN_G_PER_LARVA_DAY}-{FEED_RATE_MAX_G_PER_LARVA_DAY} g per larva per day."
                    )
                except (ValueError, ZeroDivisionError):
                    pass

        if detected_stage and days_in_stage is not None and STAGE_TIMING_INTENT_RE.search(q):
            try:
                result = estimate_stage_transition(detected_stage, days_in_stage)
                lo, hi = result["estimated_remaining_days_range"]
                lo_d, hi_d = result["typical_duration_range_days"]
                return (
                    f"This {detected_stage} has been tracked for {result['days_in_stage']} day(s) since first "
                    f"photographed. Typical {detected_stage} duration is {lo_d}-{hi_d} days, so the estimated "
                    f"remaining time is approximately {lo}-{hi} day(s)."
                )
            except ValueError:
                pass

        if MOISTURE_INTENT_RE.search(q):
            triple = MOISTURE_TRIPLE_RE.search(q)
            if triple:
                try:
                    dry_mass, cur_pct, target_pct = map(float, triple.groups())
                    result = calculate_moisture_dilution(dry_mass, cur_pct, target_pct)
                    return (
                        f"To bring {dry_mass}kg of substrate from {cur_pct}% to {target_pct}% moisture, "
                        f"add approximately {result['water_to_add_kg']} kg (~{result['water_to_add_liters']} L) of water."
                    )
                except ValueError:
                    pass

        return None

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

    def handle_message(self, text: Optional[str] = None, image_path: Optional[str] = None,
                        audio_path: Optional[str] = None) -> str:
        """Single entry point for downstream integration, regardless of
        channel. Callers pass whatever the incoming message actually
        contains — text, an image, a voice note, or image+caption — and get
        back a plain text reply. This is the full contract: everything
        else (vision classification, retrieval, computed values, LLM
        generation, transcription) happens inside this call.

        Precedence when multiple inputs are given:
          - audio_path, if present, is transcribed and used as the query
            text (a voice note is the user's actual question).
          - text, if also present alongside audio, is ignored — a message
            has one real "question," and audio takes precedence since
            transcribing it is the whole point of sending a voice note.
          - image_path can accompany either text or audio, same as the CLI.
        """
        heard_prefix = ""

        if audio_path:
            if not self.audio_enabled:
                return "Voice notes aren't supported right now — could you type your question instead?"

            result = transcribe_audio(audio_path)
            if result["empty"]:
                return "I couldn't make out anything in that voice note — could you try again somewhere quieter, or type your question?"

            query = result["text"]
            if result["low_confidence"]:
                # Surface what was actually heard rather than silently
                # acting on a possibly-garbled transcription.
                heard_prefix = f'(I heard: "{query}") '
        else:
            query = (text or "").strip()

        if not query:
            return "I didn't receive a question — could you send some text, a photo, or a voice note?"

        response = self.generate_response(image_path, query)
        return f"{heard_prefix}{response}" if heard_prefix else response

    def generate_response(self, image_path: Optional[str], user_query: str) -> Optional[str]:
        """Runs the full pipeline (vision + retrieval + compute + LLM) and
        returns the final sanitized response text. No printing here — this
        is the reusable core both the CLI and the WhatsApp webhook call.
        Returns None for the terminal-only 'nothing to say' cases (OOD
        warning, corrupted image) which the CLI wrapper prints directly;
        callers like a webhook should treat None as 'send the OOD/error
        message that was logged' — see generate_stream for that text.
        """
        if image_path:
            try:
                is_ood, sim_score, detected_stage, confidence, diagnostics_text, days_in_stage = self.get_image_analysis(image_path)
            except ValueError as err:
                logger.error(f"Image error: {err}")
                return f"❌ {err}"

            if is_ood:
                return (
                    f"⚠️ I couldn't confidently match this image to an egg, larva, prepupa, pupa, or adult "
                    f"(similarity {sim_score*100:.1f}%, below threshold). Could you send a clearer, closer photo?"
                )

            stage_hook = getattr(self, "_on_stage_detected", None)
            if stage_hook:
                stage_hook(detected_stage, confidence)

            rag_context = self.get_reference_context(user_query, detected_stage=detected_stage)
            context_block = rag_context if rag_context else "No closely matching reference material found for this specific question."
            computed = self.try_compute(user_query, detected_stage=detected_stage, days_in_stage=days_in_stage)
            computed_block = f"\n\n[Computed Values]\n{computed}" if computed else ""

            prompt = (
                f"[Vision Analysis]\nDetected Stage: {detected_stage.capitalize()} ({confidence:.1f}% confidence)\n\n"
                f"[Visual Diagnostics — heuristic, not a certified assessment]\n{diagnostics_text}\n\n"
                f"[Reference Context]\n{context_block}"
                f"{computed_block}\n\n"
                f"User Question: {user_query}"
            )
        else:
            rag_context = self.get_reference_context(user_query)
            computed = self.try_compute(user_query)
            computed_block = f"\n\n[Computed Values]\n{computed}" if computed else ""

            if rag_context or computed_block:
                context_line = f"[Reference Context]\n{rag_context}\n\n" if rag_context else ""
                prompt = f"{context_line}User Question: {user_query}{computed_block}"
            else:
                prompt = f"User Question: {user_query}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": prompt})

        text_input = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.qwen_tokenizer([text_input], return_tensors="pt").to(DEVICE)

        streamer = TextIteratorStreamer(
            self.qwen_tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            bad_words_ids=self.banned_token_ids,
        )

        # Previously: Thread(target=self.qwen_model.generate, kwargs=...).
        # If .generate() raised inside that thread (OOM, bad input shape,
        # etc.), the exception vanished silently — the caller just got an
        # empty response with no explanation, and the streamer's `for`
        # loop below could block forever waiting on a queue that would
        # never receive its final "done" signal. Wrapping the target
        # captures the exception; the streamer's timeout (set above)
        # bounds how long the loop will wait before giving up either way.
        generation_error: Dict[str, Optional[BaseException]] = {"exc": None}

        def _run_generate():
            try:
                self.qwen_model.generate(**generation_kwargs)
            except Exception as e:
                generation_error["exc"] = e
                logger.error(f"Generation thread failed: {e}", exc_info=True)

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()

        full_response = ""
        try:
            for new_text in streamer:
                full_response += new_text
                yield_hook = getattr(self, "_on_token", None)
                if yield_hook:
                    yield_hook(new_text)
        except queue.Empty:
            logger.error(f"Generation timed out after {GENERATION_TIMEOUT_SECONDS}s waiting for tokens.")

        thread.join(timeout=5)

        if generation_error["exc"] is not None:
            return f"Sorry, something went wrong generating a response ({type(generation_error['exc']).__name__}). Please try again."

        if not full_response.strip():
            logger.error("Generation produced no output (likely a timeout or silent failure).")
            return "Sorry, I wasn't able to generate a response that time. Please try again."

        if CITATION_RE.search(full_response):
            logger.warning("Detected hallucinated citation markup in response — check QLoRA SFT data for citation-formatted training examples.")

        sanitized = self._sanitize(full_response)

        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": sanitized})
        self._trim_history()

        return sanitized

    def generate_stream(self, image_path: Optional[str], user_query: str):
        """Terminal/CLI wrapper: prints tokens live as they arrive, then
        returns the full text too. Kept so main() below doesn't need to
        change at all."""
        token_count = {"n": 0}

        def on_token(tok):
            token_count["n"] += 1
            print(tok, end="", flush=True)

        self._on_token = on_token
        self._on_stage_detected = lambda stage, conf: print(f"\n🔍 [Detected Stage: {stage.capitalize()} ({conf:.1f}% Confidence)]")
        print("\n[Foggy AI]: ", end="", flush=True)
        response = self.generate_response(image_path, user_query)
        self._on_token = None
        self._on_stage_detected = None

        # OOD/error responses return before any token streaming happens, so
        # they'd otherwise never actually get printed — print them directly
        # in that case instead of relying on the (never-fired) token hook.
        if token_count["n"] == 0 and response:
            print(response, end="")

        print("\n")
        return response


def parse_input_line(input_str: str, current_active_image: Optional[str]) -> Tuple[Optional[str], str]:
    """Parses single-line interactive terminal input."""
    clean_input = input_str.strip()
    reset_keywords = ("clear", "reset", "clear image", "clear img", "no image", "no img")

    # Exact match on a full reset phrase ("clear image", "no img", etc.) —
    # this is a complete command with no leftover query. Previously this
    # fell through to the split-based extraction below, which incorrectly
    # treated the second word ("image" in "clear image") as a leftover query
    # and sent it to the LLM as if the user had typed "image".
    if clean_input.lower() in reset_keywords:
        return None, ""

    # Prefix form ("clear what temp for eggs?") — here the intent really is
    # "reset, then answer this new question", so extract the trailing text.
    if clean_input.lower().startswith(("clear ", "reset ")):
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
    print("  • Voice:    dataset/voice_note.ogg  (transcribes, then answers)")
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

            # Voice-note test path: type a single audio file path with no
            # other text, same convention as typing an image path alone.
            # This exists so the audio pipeline can be validated directly
            # from the CLI, independent of whatever webhook/channel
            # integration eventually calls handle_message() in production.
            if (len(user_input.split()) == 1
                    and user_input.lower().endswith(AUDIO_EXTENSIONS)
                    and os.path.exists(user_input)):
                if not engine.audio_enabled:
                    print("⚠️  Audio transcription isn't available (Whisper failed to load, or audio_transcriber.py is missing).\n")
                    continue

                print("🎤 Transcribing voice note...")
                result = transcribe_audio(user_input)
                if result["empty"]:
                    print("❌ Couldn't make out anything in that audio file.\n")
                    continue

                confidence_note = " (low confidence — verify this looks right)" if result["low_confidence"] else ""
                print(f"🎤 Transcribed: \"{result['text']}\"{confidence_note}")

                engine.generate_stream(current_image, result["text"])
                continue

            image_path, query = parse_input_line(user_input, current_image)

            # Detect Image Clearing
            if current_image and image_path is None:
                current_image = None
                # Reset history any time the image is actually cleared, not
                # just on the literal words "clear"/"reset" — this was the
                # second half of the bug: "clear image" cleared the image
                # but left old [Visual Diagnostics] context sitting in
                # chat_history, so the next reply kept referencing the
                # previous image as if it were still live.
                engine.reset_history()
                print("🧹 Image memory cleared. Conversation history reset. Switched to Text Mode.\n")
                if not query:
                    continue

            # Explicit reset also clears conversation history even when no
            # image was active (pure text-mode reset).
            if user_input.lower() in ("clear", "reset") and not current_image:
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