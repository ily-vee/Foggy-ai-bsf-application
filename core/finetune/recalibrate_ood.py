import json
import torch
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from transformers import SiglipImageProcessor

from model_siglip import SiglipBSFClassifier

# Paths
MANIFEST_PATH = Path("core/finetune/dataset_vision.json")
LORA_DIR = Path("models/siglip_bsf_lora")
OOD_SAVE_PATH = Path("models/bsf_ood_detector.pkl")
MODEL_ID = "google/siglip-base-patch16-224"

def recalibrate_ood():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Recalibrating OOD Detector using device: {device}")

    # 1. Load Image Processor & Fine-Tuned Model Architecture
    local_path = Path("siglip2_local")
    if local_path.exists():
        processor = SiglipImageProcessor.from_pretrained(str(local_path))
    else:
        processor = SiglipImageProcessor.from_pretrained(MODEL_ID)

    model = SiglipBSFClassifier(num_classes=5)
    
    # Load trained LoRA vision adapter & classification head
    adapter_path = LORA_DIR / "vision_adapter"
    head_path = LORA_DIR / "classifier_head.pt"

    if adapter_path.exists():
        model.backbone.load_adapter(str(adapter_path), adapter_name="default")
        print(f"  -> Loaded fine-tuned LoRA adapter from {adapter_path}")
    if head_path.exists():
        model.classifier.load_state_dict(torch.load(head_path, map_location=device))
        print(f"  -> Loaded fine-tuned classifier head from {head_path}")

    model.to(device)
    model.eval()

    # 2. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    embeddings = []
    labels = []

    print(f"Extracting features across {len(samples)} images...")
    with torch.no_grad():
        for sample in samples:
            image_path = sample["image_path"]
            label_id = sample["label_id"]

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Extract pooled 768-dim embeddings from the fine-tuned backbone
            outputs = model.backbone(pixel_values=inputs["pixel_values"])
            pooled_emb = outputs.pooler_output.cpu().numpy().squeeze(0)

            embeddings.append(pooled_emb)
            labels.append(label_id)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # 3. Fit Mahalanobis Out-of-Distribution Detector per Class
    class_means = {}
    class_covariances = {}

    for c in range(5):
        class_embs = embeddings[labels == c]
        if len(class_embs) > 0:
            mean = np.mean(class_embs, axis=0)
            cov_estimator = EmpiricalCovariance().fit(class_embs)
            class_means[c] = mean
            class_covariances[c] = cov_estimator.precision_

    # Shared covariance calculation for global thresholding
    global_cov = EmpiricalCovariance().fit(embeddings)
    
    ood_data = {
        "class_means": class_means,
        "class_covariances": class_covariances,
        "precision_matrix": global_cov.precision_,
        "threshold": float(np.percentile([
            np.min([
                np.sqrt((emb - class_means[c]).T @ global_cov.precision_ @ (emb - class_means[c]))
                for c in class_means
            ]) for emb in embeddings
        ], 95)) * 1.5  # 95th percentile safety boundary
    }

    OOD_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ood_data, OOD_SAVE_PATH)
    print(f"OOD Detector successfully recalibrated and saved to {OOD_SAVE_PATH}")

if __name__ == "__main__":
    recalibrate_ood()