import os
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Configuration
DATASET_DIR = "bsf_images_dataset"
OUTPUT_FEATURES = "bsf_qwen_features.npy"
OUTPUT_LABELS = "bsf_qwen_labels.npy"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CLASS_MAPPING = {
    "1_eggs": 0,
    "2_early_larvae": 1,
    "3_feeding_larvae": 2,
    "4_pupae": 3,
    "5_bsf_adult": 4
}

def extract_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Qwen 2.5-VL visual encoder on {device}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )

    # Access Qwen's native vision module
    if hasattr(model, "visual"):
        vision_encoder = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_encoder = model.model.visual
    else:
        raise AttributeError("Could not locate the visual encoder module in Qwen model.")

    vision_encoder.eval()

    features_list = []
    labels_list = []

    print("📸 Processing dataset and extracting visual embeddings...")
    
    for class_folder, class_idx in CLASS_MAPPING.items():
        folder_path = os.path.join(DATASET_DIR, class_folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Directory '{folder_path}' not found. Skipping.")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  --> Found {len(files)} images in '{class_folder}' (Label: {class_idx})")

        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                    output = vision_encoder(
                        inputs.pixel_values, 
                        grid_thw=inputs.image_grid_thw
                    )
                    
                    # Unpack the tensor from the output dataclass
                    if hasattr(output, "last_hidden_state"):
                        patch_embeds = output.last_hidden_state
                    elif isinstance(output, (tuple, list)):
                        patch_embeds = output[0]
                    else:
                        patch_embeds = output

                    # Mean pool spatial token vectors into one 1D array
                    pooled_vec = patch_embeds.mean(dim=0).squeeze().cpu().numpy()

                features_list.append(pooled_vec)
                labels_list.append(class_idx)

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\n✅ Feature extraction complete!")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape:   {y.shape}")

    np.save(OUTPUT_FEATURES, X)
    np.save(OUTPUT_LABELS, y)
    print(f"💾 Saved to '{OUTPUT_FEATURES}' and '{OUTPUT_LABELS}'.")

if __name__ == "__main__":
    extract_features()