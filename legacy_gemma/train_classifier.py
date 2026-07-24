import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Configuration & Hyperparameters
FEATURES_FILE = "bsf_features.npy"
LABELS_FILE = "bsf_labels.npy"
MODEL_SAVE_PATH = "bsf_classifier.pth"

INPUT_DIM = 768  # SigLIP 2 output size
NUM_CLASSES = 5  # Eggs, Early Larvae, Feeding Larvae, Pupae, Adult
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001


# 2. Define the exact same MLP Architecture
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BSFClassifierHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def main():
    # Load dataset arrays
    try:
        X = np.load(FEATURES_FILE)
        y = np.load(LABELS_FILE)
    except FileNotFoundError:
        print("❌ Feature files not found. Run extract_features.py first.")
        return

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, criterion and optimizer
    model = BSFClassifierHead(INPUT_DIM, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Training final classifier head on all available samples...")

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss / len(X):.4f}")

    # Save the trained model parameters to disk
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎯 Model training complete! Saved production weights to: '{MODEL_SAVE_PATH}'")


if __name__ == "__main__":
    main()
