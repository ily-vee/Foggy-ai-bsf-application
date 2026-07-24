import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CLASSIFIER_PATH = "bsf_classifier.pth"

CLASS_NAMES = [
    "1_eggs",
    "2_early_larvae",
    "3_feeding_larvae",
    "4_pupae",
    "5_bsf_adult"
]

class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim=1280, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def predict_stage(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Loading models on {device}...")

    # 1. Load Qwen Processor and Model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=device
    )

    # Access visual encoder module
    if hasattr(model, "visual"):
        vision_encoder = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_encoder = model.model.visual
    else:
        raise AttributeError("Could not locate visual encoder.")

    vision_encoder.eval()

    # 2. Extract feature vector for test image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = vision_encoder(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
        patch_embeds = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        feature_vec = patch_embeds.mean(dim=0).squeeze()

        # Format feature vector for classifier batch input: [1, 1280]
        if feature_vec.ndim == 1:
            feature_vec = feature_vec.unsqueeze(0)
        feature_vec = feature_vec.to(dtype=torch.float32)

    # 3. Load Trained MLP Head
    classifier = BSFClassifierHead(input_dim=feature_vec.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.eval()

    # 4. Infer stage and probabilities
    with torch.no_grad():
        logits = classifier(feature_vec)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_class_idx = int(torch.argmax(logits, dim=1).item())

    # 5. Output results
    print("\n" + "=" * 45)
    print(f"📌 PREDICTED STAGE: {CLASS_NAMES[pred_class_idx]}")
    print(f"🎯 CONFIDENCE:      {probabilities[pred_class_idx] * 100:.2f}%")
    print("=" * 45)
    print("Class Probabilities:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  • {name:<18}: {probabilities[idx] * 100:6.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_stage(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_image>")