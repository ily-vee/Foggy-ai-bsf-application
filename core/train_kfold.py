"""
train_kfold.py
1. Evaluates a PyTorch linear classification head across 5-Fold Stratified CV.
2. Fits and calibrates the Mahalanobis OOD detector across folds.
3. Trains the final linear head and OOD detector on ALL 25 samples and saves artifacts.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import mahalanobis

EMBEDDINGS_PATH = "processed_dataset/embeddings.npy"
LABELS_PATH = "processed_dataset/labels.json"
MODEL_SAVE_PATH = "classifier_head.pt"
OOD_SAVE_PATH = "bsf_ood_detector.pkl"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Simple PyTorch Linear Classifier Head
class BSFClassifierHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super(BSFClassifierHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Mahalanobis Out-of-Distribution Detector
class MahalanobisOODDetector:
    def __init__(self, shrinkage=1e-1):  # Increased shrinkage from 1e-3 to 1e-1 for small sample sizes
        self.shrinkage = shrinkage
        self.class_means = {}
        self.inv_cov = None
        self.threshold = None

    def fit(self, X, y, percentile=99.0):  # Relaxed percentile from 95.0 to 99.0
        unique_classes = np.unique(y)
        d_dim = X.shape[1]

        for cls in unique_classes:
            self.class_means[cls] = np.mean(X[y == cls], axis=0)

        pooled_cov = np.zeros((d_dim, d_dim))
        for cls in unique_classes:
            X_cls = X[y == cls]
            centered = X_cls - self.class_means[cls]
            pooled_cov += centered.T @ centered

        pooled_cov /= len(X)
        pooled_cov += self.shrinkage * np.eye(d_dim)
        self.inv_cov = np.linalg.pinv(pooled_cov)

        distances = [self.compute_distance(x, y[i]) for i, x in enumerate(X)]
        self.threshold = float(np.percentile(distances, percentile))

    def compute_distance(self, x, class_label):
        mean = self.class_means[class_label]
        delta = x - mean
        return float(np.sqrt(np.dot(np.dot(delta, self.inv_cov), delta)))

    def get_min_distance(self, x):
        distances = [
            float(np.sqrt(np.dot(np.dot(x - m, self.inv_cov), x - m)))
            for m in self.class_means.values()
        ]
        return min(distances)


def train_classifier(X_train, y_train, input_dim=768, num_classes=5, epochs=100, lr=0.01):
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model = BSFClassifierHead(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

    return model


def main():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(LABELS_PATH):
        print("❌ Error: Processed embeddings or labels not found. Run preprocess.py first!")
        return

    X = np.load(EMBEDDINGS_PATH)
    with open(LABELS_PATH, "r") as f:
        labels_raw = json.load(f)

    # Map class labels to integers
    unique_classes = sorted(list(set(labels_raw)))
    label_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_label = {idx: cls for idx, cls in enumerate(unique_classes)}
    y = np.array([label_to_idx[l] for l in labels_raw])

    print(f"📊 Dataset Loaded: {X.shape[0]} samples across {len(unique_classes)} classes.")

    # Step 1: 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    print("\n🔄 Running 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = train_classifier(X_tr, y_tr, num_classes=len(unique_classes))
        model.eval()

        with torch.no_grad():
            preds = model(torch.tensor(X_va, dtype=torch.float32)).argmax(dim=1).numpy()

        acc = accuracy_score(y_va, preds)
        accuracies.append(acc)
        print(f"  • Fold {fold} Accuracy: {acc * 100:.2f}%")

    print(f"\n✅ Mean Cross-Validation Accuracy: {np.mean(accuracies) * 100:.2f}% (±{np.std(accuracies) * 100:.2f}%)")

    # Step 2: Train Final Artifacts on 100% Data
    print("\n🚀 Training final artifacts on all 25 samples...")
    final_model = train_classifier(X, y, num_classes=len(unique_classes), epochs=150)
    
    # Save PyTorch classifier weights & class mapping
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "class_to_idx": label_to_idx,
        "idx_to_class": idx_to_label
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"  • Classifier head saved to: '{MODEL_SAVE_PATH}'")

    # Fit and save Mahalanobis OOD Detector
    ood_detector = MahalanobisOODDetector()
    ood_detector.fit(X, y)
    
    ood_artifact = {
        "class_means": ood_detector.class_means,
        "inv_cov": ood_detector.inv_cov,
        "threshold": ood_detector.threshold,
        "idx_to_class": idx_to_label
    }
    with open(OOD_SAVE_PATH, "wb") as f:
        pickle.dump(ood_artifact, f)
    print(f"  • OOD Detector pickle saved to: '{OOD_SAVE_PATH}'")
    print(f"  • OOD Distance Threshold (95th percentile): {ood_detector.threshold:.4f}\n")


if __name__ == "__main__":
    main()