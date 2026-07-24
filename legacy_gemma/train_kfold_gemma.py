import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Hyperparameters & Constants
FEATURES_FILE = "bsf_qwen_features.npy"
LABELS_FILE = "bsf_qwen_labels.npy"
BEST_MODEL_PATH = "bsf_classifier.pth"

NUM_CLASSES = 5
N_SPLITS = 5
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# PyTorch MLP Head Architecture
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes=5):
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

def train_and_eval_kfold():
    # 1. Load cached embeddings
    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)

    input_dim = X.shape[1]
    print(f"📂 Loaded dataset: {X.shape[0]} samples with feature dimension {input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []
    best_overall_accuracy = 0.0

    print(f"\n⚡ Starting {N_SPLITS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} / {N_SPLITS} ---")

        # Prepare fold datasets & loaders
        X_train, y_train = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long)
        X_val, y_val = torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        # Initialize Model, Loss, and Optimizer
        model = BSFClassifierHead(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

        # Training Loop
        best_val_acc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluation phase
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(batch_y.numpy())

            acc = accuracy_score(val_targets, val_preds)
            if acc > best_val_acc:
                best_val_acc = acc

        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc * 100:.2f}%")

        # Save absolute best model state across all folds
        if best_val_acc > best_overall_accuracy:
            best_overall_accuracy = best_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 Best model weight updated and saved to '{BEST_MODEL_PATH}'")

            # Output confusion matrix for top model fold
            cm = confusion_matrix(val_targets, val_preds)
            print("Confusion Matrix:")
            print(cm)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print("\n" + "=" * 50)
    print(f"📊 Final K-Fold Results across {N_SPLITS} folds:")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    train_and_eval_kfold()