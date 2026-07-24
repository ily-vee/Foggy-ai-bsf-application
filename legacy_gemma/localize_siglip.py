import os
from transformers import AutoProcessor, AutoModel

LOCAL_DIR = "./siglip2_local"

if __name__ == "__main__":
    print("⏳ Downloading and saving SigLIP 2 locally...")
    model_name = "google/siglip2-base-patch16-224"

    # Force load from Hub and write locally
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    processor.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)
    print(f"✅ SigLIP 2 is saved locally in '{LOCAL_DIR}'. You are fully offline-ready!")
