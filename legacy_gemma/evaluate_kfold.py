import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# 1. Configuration & Hyperparameters
FEATURES_FILE = "bsf_features.npy"
LABELS_FILE = "bsf_labels.npy"
INPUT_DIM = 768  # Output dimension of SigLIP 2
NUM_CLASSES = 5  # Eggs, Early Larvae, Feeding Larvae, Pupae, Adult
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
K_FOLDS = 5


# 2. Define the PyTorch MLP (Multi-Layer Perceptron)
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BSFClassifierHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevents overfitting on smaller datasets
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def train_and_validate_fold(fold_idx, train_loader, val_loader):
    """Trains a freshly initialized model for one fold and returns its validation accuracy."""
    # Reset model and optimizer states for this clean run
    model = BSFClassifierHead(INPUT_DIM, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation Loop
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"📊 Fold {fold_idx + 1}/{K_FOLDS} Complete - Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    # Load extracted features
    try:
        X = np.load(FEATURES_FILE)
        y = np.load(LABELS_FILE)
    except FileNotFoundError:
        print("❌ Could not find feature files. Please run extract_features.py first!")
        return

    print(f"Loaded {X.shape[0]} samples with feature size {X.shape[1]}")

    # Initialize Stratified K-Fold split configuration
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f"\n🚀 Running Stratified {K_FOLDS}-Fold Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split features and convert to PyTorch tensors
        X_train, y_train = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long)
        X_val, y_val = torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long)

        # Build DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Train and validate this fold
        fold_acc = train_and_validate_fold(fold_idx, train_loader, val_loader)
        fold_accuracies.append(fold_acc)

    # Final general summary
    mean_accuracy = np.mean(fold_accuracies)
    print(f"\n🎯 Cross-Validation Process Finished!")
    print(f"📈 Average Generalization Accuracy across all folds: {mean_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
