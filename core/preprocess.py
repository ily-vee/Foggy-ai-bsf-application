"""
preprocess.py
Preprocesses photos, extracts 768-dim SigLIP 2 embeddings, and provides bulk
dataset feature extraction for training & OOD calibration.
"""

import os
import json
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "bsf_images_dataset"
OUTPUT_DIR = "processed_dataset"

class PhotoValidationError(Exception):
    """Custom exception raised when an input file fails basic image checks."""
    pass


print("⚙️ Initializing SigLIP 2 Feature Extractor...")
_processor = AutoProcessor.from_pretrained(MODEL_ID)
_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()


def preprocess_and_embed(image_path: str) -> torch.Tensor:
    """
    Validates a single image file and extracts its 768-dim SigLIP 2 vision embedding.
    """
    if not os.path.exists(image_path):
        raise PhotoValidationError(f"File not found: '{image_path}'")

    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise PhotoValidationError("Uploaded file is corrupted or not a valid image format.")

    inputs = _processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Get output from SigLIP 2
        outputs = _model.get_image_features(**inputs)
        
        # Safely unpack PyTorch tensor if SigLIP 2 returns BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state.mean(dim=1)
        else:
            features = outputs

    return features.cpu()


def process_dataset_folder(dataset_dir: str = DATASET_DIR, output_dir: str = OUTPUT_DIR):
    """
    Iterates through class subfolders in dataset_dir, extracts embeddings,
    and saves them as NumPy arrays for training, K-Fold, and OOD calibration.
    """
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    embeddings = []
    labels = []
    image_paths = []

    class_folders = [f for f in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, f))]

    if not class_folders:
        print(f"❌ No subfolders found in '{dataset_dir}'.")
        return

    print(f"\n📂 Found {len(class_folders)} class folders: {class_folders}")
    print("⏳ Extracting SigLIP 2 embeddings...\n")

    for class_name in class_folders:
        folder_path = os.path.join(dataset_dir, class_name)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

        print(f"  --> Processing '{class_name}': {len(files)} images...")

        for file_name in files:
            img_path = os.path.join(folder_path, file_name)
            try:
                emb = preprocess_and_embed(img_path)
                
                # Normalize embedding to unit vector
                emb_norm = emb / emb.norm(dim=-1, keepdim=True)
                
                embeddings.append(emb_norm.squeeze().numpy())
                labels.append(class_name)
                image_paths.append(img_path)
            except PhotoValidationError as e:
                print(f"      ⚠️ Skipping corrupt image '{file_name}': {e}")
            except Exception as e:
                print(f"      ⚠️ Error processing '{file_name}': {e}")

    if not embeddings:
        print("❌ No embeddings extracted.")
        return

    embeddings_np = np.array(embeddings, dtype=np.float32)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    lbl_path = os.path.join(output_dir, "labels.json")
    paths_file = os.path.join(output_dir, "paths.json")

    np.save(emb_path, embeddings_np)
    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    with open(paths_file, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, indent=2)

    print(f"\n🎉 Extraction Complete!")
    print(f"  • Embeddings saved to: '{emb_path}' (Shape: {embeddings_np.shape})")
    print(f"  • Labels saved to:     '{lbl_path}' (Total: {len(labels)})")
    print(f"  • Paths saved to:      '{paths_file}'\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        process_dataset_folder()
    else:
        test_file = sys.argv[1]
        try:
            emb = preprocess_and_embed(test_file)
            print(f"✅ Extracted single embedding shape: {emb.shape}")
        except PhotoValidationError as e:
            print(f"❌ Validation failed: {e}")