import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

from dataset import get_dataloaders
from model_siglip import get_siglip_lora_model

# Training Configuration
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
OUTPUT_DIR = Path("models/siglip_bsf_lora")

def train_model():
    # Detect accelerator (CUDA GPU or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on device: {device}")

    # Load DataLoaders and Model
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = get_siglip_lora_model(num_classes=5)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_acc = 0.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting Vision Adapter Fine-Tuning...")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * pixel_values.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label_id"].to(device)

                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * pixel_values.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # Save Checkpoint if Validation Accuracy Improves
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            # Save Peft vision adapter weights
            model.backbone.save_pretrained(str(OUTPUT_DIR / "vision_adapter"))
            # Save classifier head weights
            torch.save(model.classifier.state_dict(), OUTPUT_DIR / "classifier_head.pt")
            print(f"  -> Saved best checkpoint (Val Acc: {best_val_acc:.2f}%) to {OUTPUT_DIR}")

    print("=" * 55)
    print(f"Fine-Tuning Complete! Best Validation Accuracy: {best_val_acc:.2f}%\n")

if __name__ == "__main__":
    train_model()