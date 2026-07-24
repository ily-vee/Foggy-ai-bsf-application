import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 1. Configuration Setup
DATASET_DIR = "bsf_images_dataset"
OUTPUT_FEATURES_FILE = "bsf_features.npy"
OUTPUT_LABELS_FILE = "bsf_labels.npy"

# Automatically map folder names to numerical class indexes
CLASS_MAPPING = {
    "1_eggs": 0,
    "2_early_larvae": 1,
    "3_feeding_larvae": 2,
    "4_pupae": 3,
    "5_bsf_adult": 4
}


def main():
    # 2. Load the frozen SigLIP 2 model and processor from Hugging Face
    print("⏳ Loading SigLIP 2 feature extractor (frozen backend)...")
    model_name = "google/siglip2-base-patch16-224"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Ensure the model is completely frozen (no training gradients tracked)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    features_list = []
    labels_list = []

    print(f"🚀 Starting dataset traversal across directory: '{DATASET_DIR}'")

    # 3. Traverse through folders and extract vectors
    for root, dirs, files in os.walk(DATASET_DIR):
        folder_name = os.path.basename(root)

        # Skip if the folder isn't part of our explicitly defined target classes
        if folder_name not in CLASS_MAPPING:
            continue

        class_idx = CLASS_MAPPING[folder_name]
        print(f"\n📁 Processing folder: {folder_name} (Class ID: {class_idx})")

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)

                try:
                    # Open image and convert to clean RGB format
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Preprocess image to fit SigLIP's expected dimensions
                        inputs = processor(images=img, return_tensors="pt")

                        # Extract the raw vision embeddings
                        with torch.no_grad():
                            # We use model.vision_model to specifically access the image encoder
                            # and grab the 'pooler_output', which is the 768-dim embedding
                            outputs = model.vision_model(**inputs)
                            embedding = outputs.pooler_output.squeeze().cpu().numpy()

                        features_list.append(embedding)
                        labels_list.append(class_idx)
                        print(f"   ✅ Vectorized: {file} -> Array shape: {embedding.shape}")

                except Exception as e:
                    print(f"   ❌ Failed processing {file}: {str(e)}")

    # 4. Save computed datasets to disk as high-performance NumPy arrays
    if features_list:
        np.save(OUTPUT_FEATURES_FILE, np.array(features_list))
        np.save(OUTPUT_LABELS_FILE, np.array(labels_list))
        print(f"\n🎯 Feature extraction successful!")
        print(f"💾 Saved features array to: {OUTPUT_FEATURES_FILE} (Shape: {np.array(features_list).shape})")
        print(f"💾 Saved labels array to: {OUTPUT_LABELS_FILE} (Shape: {np.array(labels_list).shape})")
    else:
        print("❌ No images found or processed. Double check folder structures.")


if __name__ == "__main__":
    main()