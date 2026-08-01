"""
train_siglip_lora.py

Fine-tunes SigLIP 2 (Vision Model) using LoRA for Black Soldier Fly stage classification.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model

# 1. Configuration
MODEL_ID = "google/siglip2-base-patch16-224"
DATASET_DIR = "labeled_photos"  # Update with your dataset path
OUTPUT_DIR = "models/siglip2_bsf_lora"
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. SigLIP 2 Classification Wrapper
class Siglip2BSFClassifier(nn.Module):
    def __init__(self, base_vision_model, num_classes, hidden_dim=768):
        super().__init__()
        self.vision_model = base_vision_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits, pooled_output

def train_siglip2():
    print(f"Loading SigLIP 2 processor and vision backbone ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load via AutoModel and extract the vision tower to prevent state_dict mismatches
    full_model = AutoModel.from_pretrained(MODEL_ID)
    base_vision_model = full_model.vision_model

    # 3. Apply LoRA to Vision Transformer Attention Layers
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    lora_vision_model = get_peft_model(base_vision_model, peft_config)
    lora_vision_model.print_trainable_parameters()

    # 4. Load Dataset & Class Mappings
    dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)
    labels = dataset["train"].features["label"].names
    num_classes = len(labels)
    print(f"Detected {num_classes} BSF stages: {labels}")

    def transform(example_batch):
        images = [x.convert("RGB") for x in example_batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["label"] = example_batch["label"]
        return inputs

    dataset.set_transform(transform)
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

    # 5. Model Initialization
    hidden_dim = base_vision_model.config.hidden_size
    model = Siglip2BSFClassifier(lora_vision_model, num_classes=num_classes, hidden_dim=hidden_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 6. Fine-Tuning Loop
    print("\nStarting SigLIP 2 LoRA Fine-Tuning...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            targets = torch.tensor(batch["label"]).to(device)

            optimizer.zero_grad()
            logits, _ = model(pixel_values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

    # 7. Save Model Weights & Mappings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/siglip2_bsf_model.pt")
    
    mapping = {i: label for i, label in enumerate(labels)}
    with open(f"{OUTPUT_DIR}/class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✅ Saved SigLIP 2 LoRA model and mappings to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train_siglip2()