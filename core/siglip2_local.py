# Save as download_siglip2.py and run: python download_siglip2.py
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-224"

print(f"📥 Downloading SigLIP 2 model and processor from HuggingFace ({MODEL_ID})...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

print("✅ SigLIP 2 downloaded and cached successfully!")