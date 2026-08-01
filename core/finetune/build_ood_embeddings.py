"""
build_ood_embeddings.py

Extracts visual embeddings from the fine-tuned SigLIP 2 model 
and calculates per-class centroid/covariance metrics for OOD detection.
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model

# Config
MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_DIR = "models/siglip2_bsf_lora"
DATASET_DIR = "labeled_photos"  # Update with your actual dataset directory path
OUTPUT_OOD_FILE = "models/siglip2_bsf_lora/ood_embeddings.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification Wrapper matching train_siglip_lora.py
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
        return outputs.pooler_output

def generate_embeddings():
    print("🔹 Loading SigLIP 2 base model and applying LoRA wrapper...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    full_model = AutoModel.from_pretrained(MODEL_ID)
    base_vision_model = full_model.vision_model

    # 1. Apply identical LoRA configuration used during training
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    lora_vision_model = get_peft_model(base_vision_model, peft_config)

    # 2. Load class mapping
    with open(f"{SIGLIP2_DIR}/class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)

    # 3. Instantiate model wrapper and load saved weights safely
    hidden_dim = base_vision_model.config.hidden_size
    model = Siglip2BSFClassifier(lora_vision_model, num_classes=num_classes, hidden_dim=hidden_dim)
    
    state_dict = torch.load(f"{SIGLIP2_DIR}/siglip2_bsf_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("🔹 Loading image dataset for reference embeddings...")
    dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)["train"]
    
    embeddings_by_class = {label_name: [] for label_name in class_mapping.values()}

    print("🔹 Extracting feature embeddings...")
    with torch.no_grad():
        for item in dataset:
            img = item["image"].convert("RGB")
            label_idx = item["label"]
            label_name = class_mapping[str(label_idx)]

            inputs = processor(images=img, return_tensors="pt").to(device)
            pooled_emb = model(inputs["pixel_values"])
            
            # Normalize embedding vector
            norm_emb = pooled_emb / pooled_emb.norm(dim=-1, keepdim=True)
            embeddings_by_class[label_name].append(norm_emb.cpu().squeeze(0).numpy())

    # 4. Compute per-class centroids
    centroids = {}
    all_embeddings = []
    for stage, emb_list in embeddings_by_class.items():
        emb_arr = np.array(emb_list)
        centroids[stage] = np.mean(emb_arr, axis=0)
        all_embeddings.extend(emb_list)

    ood_data = {
        "centroids": centroids,
        "all_embeddings": np.array(all_embeddings),
        "class_mapping": class_mapping
    }

    torch.save(ood_data, OUTPUT_OOD_FILE)
    print(f"✅ Successfully calculated and saved OOD reference data -> {OUTPUT_OOD_FILE}")

if __name__ == "__main__":
    generate_embeddings()