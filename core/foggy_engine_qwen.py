import os
import re
import sys
import time
import json
import torch
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from threading import Thread

# ----------------------------------------------------
# Dynamic Path Resolution
# ----------------------------------------------------
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent

for path in [PROJECT_ROOT, CORE_DIR / "finetune"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_siglip import SiglipBSFClassifier
from core.rag.retriever import HybridRetriever

from transformers import (
    SiglipImageProcessor,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Model & Adapter Paths
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

LORA_VISION_DIR = Path("models/siglip_bsf_lora")
OOD_PATH = Path("models/bsf_ood_detector.pkl")
LORA_QWEN_DIR = Path("models/qwen_bsf_qlora")

VECTOR_DB_DIR = Path("foggy_vector_db")
CLASSES = ["1_eggs", "2_early_larvae", "3_feeding_larvae", "4_pupae", "5_adult_bsf"]


class FoggyBrainEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Booting Unified Foggy Brain Core on {self.device}...")

        # 1. Load Local Hybrid RAG Retriever (BM25 + Dense Vectors)
        self.retriever = HybridRetriever(vector_db_dir=VECTOR_DB_DIR)
        print(f"✅ Hybrid RAG Engine Connected ({len(self.retriever.chunks)} total knowledge nodes).")

        # 2. Load Fine-Tuned SigLIP Vision Classifier
        local_siglip = Path("siglip2_local")
        if local_siglip.exists():
            self.siglip_processor = SiglipImageProcessor.from_pretrained(str(local_siglip))
        else:
            self.siglip_processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_ID)

        self.vision_model = SiglipBSFClassifier(num_classes=5)
        adapter_path = LORA_VISION_DIR / "vision_adapter"
        head_path = LORA_VISION_DIR / "classifier_head.pt"

        if adapter_path.exists():
            self.vision_model.backbone.load_adapter(str(adapter_path), adapter_name="default")
        if head_path.exists():
            self.vision_model.classifier.load_state_dict(torch.load(head_path, map_location=self.device))

        self.vision_model.to(self.device)
        self.vision_model.eval()

        # 3. Load Recalibrated OOD Detector
        if OOD_PATH.exists():
            self.ood_detector = joblib.load(OOD_PATH)
            print("✅ Recalibrated OOD Detector loaded successfully.")
        else:
            self.ood_detector = None
            print("⚠️ Warning: OOD detector pickle not found.")

        # 4. Load Qwen2.5-VL with QLoRA Adapter
        print("🧠 Loading 4-bit Qwen2.5-VL Multi-Modal LLM Core...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        if LORA_QWEN_DIR.exists():
            self.qwen_model = PeftModel.from_pretrained(base_qwen, str(LORA_QWEN_DIR))
            print("✅ Fine-tuned Qwen2.5-VL QLoRA adapter attached.")
        else:
            self.qwen_model = base_qwen

        self.qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        print("✅ Foggy Brain Engine initialization complete!\n")

    def retrieve_rag_context(self, user_query, top_k=5):
        """Fetches top-k relevant knowledge nodes using Hybrid (BM25 + Dense) Search."""
        return self.retriever.retrieve(user_query, top_k=top_k)

    def run_vision_inference(self, image_path):
        """Processes image with SigLIP + OOD detector."""
        start_time = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.vision_model.backbone(pixel_values=inputs["pixel_values"])
            pooled_emb = outputs.pooler_output.cpu().numpy().squeeze(0)
            logits = self.vision_model.classifier(outputs.pooler_output)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0

        is_ood = False
        if self.ood_detector:
            mean = self.ood_detector["class_means"][pred_idx]
            precision = self.ood_detector["precision_matrix"]
            diff = pooled_emb - mean
            dist = np.sqrt(diff.T @ precision @ diff)
            if dist > self.ood_detector["threshold"]:
                is_ood = True

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "label": pred_label,
            "confidence": confidence,
            "is_ood": is_ood,
            "latency_ms": latency_ms
        }
    
    def stream_chat(self, history_messages, active_image=None, vision_meta=None, user_query=""):
        """Streams multi-turn chat responses using RAG context, vision metadata, and Qwen generation."""
        rag_context = self.retrieve_rag_context(user_query, top_k=5)

        system_instruction = (
            "You are Foggy, a precision Black Soldier Fly farming AI assistant. "
            "Use the verified local dynamic context provided to give precise operational advice. "
            "Do NOT issue tool calls or use XML tool tags. Speak directly to the user in clean text. "
            "Provide complete, concise responses. If using numbered points, ensure every point has content—do not leave empty or incomplete numbered lines."
        )

        context_block = f"DYNAMIC LOCAL KNOWLEDGE NODES:\n{rag_context}"
        if vision_meta and active_image is not None:
            context_block += (
                f"\n\nLIVE SIGLIP CLASSIFICATION: Image classified as '{vision_meta['label']}' "
                f"with {vision_meta['confidence']:.2f}% confidence. (Out-Of-Domain: {vision_meta['is_ood']})"
            )

        prompt = f"<|im_start|>system\n{system_instruction}\n\n{context_block}<|im_end|>\n"

        for idx, msg in enumerate(history_messages):
            role = msg["role"]
            content = msg["content"]
            if role == "user" and active_image is not None and idx == len(history_messages) - 1:
                prompt += f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{content}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                if idx == len(history_messages) - 1 and role == "user":
                    prompt += "<|im_start|>assistant\n"

        inputs = self.qwen_processor(
            text=[prompt],
            images=[[active_image]] if active_image else None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        class StopOnSubstring(StoppingCriteria):
            def __init__(self, tokenizer, stop_string, prompt_len):
                self.tokenizer = tokenizer
                self.stop_string = stop_string
                self.prompt_len = prompt_len
            def __call__(self, input_ids, scores, **kwargs):
                text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
                return self.stop_string in text

        stopping_criteria = StoppingCriteriaList([
            StopOnSubstring(self.qwen_processor.tokenizer, "<tool_call>", prompt_len)
        ])

        streamer = TextIteratorStreamer(self.qwen_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Register <|im_end|> as a real stop token, alongside the default EOS
        im_end_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [im_end_id]
        default_eos = self.qwen_processor.tokenizer.eos_token_id
        if default_eos is not None and default_eos != im_end_id:
            eos_ids.append(default_eos)

        # Generation Kwargs with repetition control
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=600,
            temperature=0.7,           # Standard temperature for clear sampling
            top_p=0.85,                # Truncates tail probability artifacts
            repetition_penalty=1.1,    # Gentle penalty (prevents loops without killing syntax)
            eos_token_id=eos_ids,       # <-- now stops on <|im_end|> too
            pad_token_id=self.qwen_processor.tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria
       )

        thread = Thread(target=self.qwen_model.generate, kwargs=generation_kwargs)
        thread.start()

        raw_streamed_text = ""
        token_count = 0
        start_time = time.perf_counter()

        for new_text in streamer:
            raw_streamed_text += new_text
            
            clean_chunk = re.sub(r'spepulation\.\s*|speculation\.\s*|\*?angstrom\*?', '', new_text, flags=re.IGNORECASE)
            clean_chunk = clean_chunk.replace("<tool_call>", "").replace("</tool_call>", "")
            
            sys.stdout.write(clean_chunk)
            sys.stdout.flush()
            token_count += 1

        thread.join()
        total_latency = time.perf_counter() - start_time
        speed = token_count / total_latency if total_latency > 0 else 0.0

        print(f"\n\n⚡ [Benchmark] Latency: {total_latency:.2f}s | Tokens: {token_count} | Speed: {speed:.2f} tok/s")

        # Post-process full response to remove artifacts and empty orphan numbers
        full_response = re.sub(
            r'(\s*[\d_.\-*]*\s*\(?_?This message was generated.*|\/DoNotLetYour\/.*|The\'Black\'c\'solution.*|<tool_call>.*)',
            '',
            raw_streamed_text,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        # Remove orphaned list numbers followed by empty lines (e.g. "4.\n\n")
        full_response = re.sub(r'^\s*\d+\.\s*$', '', full_response, flags=re.MULTILINE)
        full_response = re.sub(r'\n{3,}', '\n\n', full_response)
        full_response = re.sub(r'spepulation\.\s*|speculation\.\s*|\*?angstrom\*?', '', full_response, flags=re.IGNORECASE).strip()

        return full_response

    
# ==========================================
# INTERACTIVE APPLICATION LOOP
# ==========================================
if __name__ == "__main__":
    engine = FoggyBrainEngine()

    print("=" * 68)
    print(" 🌱 Foggy Brain Multi-Modal Core Active")
    print(" Slash Commands:")
    print("  - /image <file_path>  : Load active image (e.g. /image test.jpeg)")
    print("  - /clear_image       : Clear active image (switch to text-only)")
    print("  - /index             : View stored RAG knowledge count")
    print("  - /clear or new      : Reset conversation history")
    print("  - /stats             : Show session and memory metrics")
    print("  - exit or quit       : Exit session")
    print("=" * 68 + "\n")

    conversation_history = []
    active_image = None
    active_image_path = None
    vision_meta = None

    while True:
        try:
            status_tag = f"[{Path(active_image_path).name}]" if active_image_path else "[Text-Only]"
            user_input = input(f"\nYou {status_tag} > ").strip()

            if not user_input:
                continue

            raw_input = user_input.strip()
            lower_input = raw_input.lower()

            # Command Handlers
            if lower_input in ["exit", "quit"]:
                break

            clear_image_triggers = [
                "/clear_image", "/clear-image", "/clean-image", "/clean_image",
                "/remove_image", "/remove-image", "clear image", "remove image", "clear_image"
            ]
            if any(lower_input == cmd for cmd in clear_image_triggers) or lower_input.startswith("clear_dataset"):
                active_image = None
                active_image_path = None
                vision_meta = None
                print("🗑️ Active image cleared. Switched to text-only mode.")
                continue

            if lower_input in ["/clear", "/reset", "new"]:
                print("🔄 Resetting conversation history...")
                conversation_history = []
                active_image = None
                active_image_path = None
                vision_meta = None
                continue

            if lower_input == "/index":
                print(f"📚 RAG Index contains {len(engine.retriever.chunks)} knowledge nodes.")
                continue

            if lower_input.startswith("/image"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip() not in ["<path>", "<path_to_image>"]:
                    img_path = parts[1].strip()
                    if os.path.exists(img_path):
                        active_image_path = img_path
                        active_image = Image.open(active_image_path).convert("RGB")
                        print(f"🖼️ Image loaded: {active_image_path}")
                        print("[SYSTEM: Running SigLIP 2 Vision Pass...]")
                        vision_meta = engine.run_vision_inference(active_image_path)
                        print(f"🎯 SigLIP Classification: {vision_meta['label']} ({vision_meta['confidence']:.2f}%)")
                        print(f"📌 Out-of-Domain: {vision_meta['is_ood']} | Latency: {vision_meta['latency_ms']:.2f}ms")

                        if vision_meta['is_ood']:
                            print("\n⚠️ [OOD Guardrail Intercept]: Uploaded image is out-of-domain.")
                            print("💡 Please provide a valid BSF stage image to receive advice.")
                            active_image = None
                            active_image_path = None
                            vision_meta = None
                    else:
                        print(f"❌ Error: File '{img_path}' not found on disk.")
                else:
                    print("⚠️ Example usage: /image dataset/1_eggs/eggs1.jpeg")
                continue

            if lower_input == "/stats":
                allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
                print(f"📊 Memory Allocated: {allocated:.2f} MB")
                print(f"💬 Conversation Turns: {len(conversation_history) // 2}")
                print(f"🖼️ Active Image: {active_image_path or 'None'}")
                print(f"📚 RAG Chunks: {len(engine.retriever.chunks)}")
                continue

            # Command Guardrail
            if raw_input.startswith("/"):
                print(f"❌ Unrecognized command '{raw_input}'.")
                print("💡 Available commands: /image <path>, /clear_image, /index, /clear, /stats")
                continue

            # Inline Image Check
            image_match = re.search(r'\b[\w/\\.-]+\.(?:jpeg|jpg|png)\b', user_input, re.IGNORECASE)
            image_failed = False
            is_ood_blocked = False

            if image_match:
                img_path = image_match.group(0)
                if os.path.exists(img_path):
                    active_image_path = img_path
                    active_image = Image.open(active_image_path).convert("RGB")
                    print(f"\n[SYSTEM: Analyzing image '{active_image_path}' via SigLIP 2 Classifier...]")
                    vision_meta = engine.run_vision_inference(active_image_path)
                    print(f"🎯 SigLIP Classification: {vision_meta['label']} ({vision_meta['confidence']:.2f}%)")
                    print(f"📌 Out-of-Domain: {vision_meta['is_ood']} | Latency: {vision_meta['latency_ms']:.2f}ms")

                    user_input = user_input.replace(img_path, "").strip()

                    if vision_meta['is_ood']:
                        print("\n⚠️ [OOD Guardrail Intercept]: Uploaded image does not belong to a recognized BSF life-cycle stage.")
                        print("💡 Please provide a clear BSF image to generate advice.")
                        is_ood_blocked = True
                        active_image = None
                        active_image_path = None
                        vision_meta = None
                else:
                    print(f"❌ Aborted: Specified image file '{img_path}' was not found on disk.")
                    image_failed = True

            if image_failed or is_ood_blocked:
                continue

            conversation_history.append({"role": "user", "content": user_input})

            print("\nFoggy: ", end="")
            sys.stdout.flush()

            assistant_reply = engine.stream_chat(
                history_messages=conversation_history,
                active_image=active_image,
                vision_meta=vision_meta,
                user_query=user_input
            )

            conversation_history.append({"role": "assistant", "content": assistant_reply})
            print("-" * 68)

        except KeyboardInterrupt:
            print("\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")