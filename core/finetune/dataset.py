import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor

# Path to the manifest generated in Step 2
MANIFEST_PATH = Path("core/finetune/dataset_vision.json")
MODEL_ID = "google/siglip-base-patch16-224"

class BSFVisionDataset(Dataset):
    def __init__(self, samples, processor, is_train=True):
        self.samples = samples
        self.processor = processor
        self.is_train = is_train

        # Spatial augmentations for training
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = item["image_path"]
        label_id = item["label_id"]

        image = Image.open(image_path).convert("RGB")

        if self.is_train:
            image = self.augmentations(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "label_id": torch.tensor(label_id, dtype=torch.long)
        }

def get_dataloaders(batch_size=8, val_ratio=0.2, seed=42):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found at {MANIFEST_PATH}. Run Step 2 first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    labels = [s["label_id"] for s in samples]
    train_samples, val_samples = train_test_split(
        samples, test_size=val_ratio, random_state=seed, stratify=labels
    )

    local_path = Path("siglip2_local")
    if local_path.exists():
        processor = AutoProcessor.from_pretrained(str(local_path))
    else:
        processor = AutoProcessor.from_pretrained(MODEL_ID)

    train_dataset = BSFVisionDataset(train_samples, processor, is_train=True)
    val_dataset = BSFVisionDataset(val_samples, processor, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders successfully constructed:")
    print(f"  - Training samples:   {len(train_samples)} ({len(train_loader)} batches)")
    print(f"  - Validation samples: {len(val_samples)} ({len(val_loader)} batches)")

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()