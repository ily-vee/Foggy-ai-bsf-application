"""
Extract SigLIP embeddings from labeled photos and train a lightweight
classifier head on top (SigLIP stays frozen), evaluated with
stratified k-fold cross-validation.
"""

import glob
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from collections import Counter

from preprocess import preprocess_and_embed, PhotoValidationError

LABELED_DIR = "labeled_photos"
CLASSES = ["egg", "larva", "prepupa", "pupa", "adult"]
DESIRED_K = 5

def get_embedding(path):
    emb = preprocess_and_embed(path)  # now includes validation + blur check!
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().detach().numpy()

X, y = [], []
for label_idx, label in enumerate(CLASSES):
    folder = os.path.join(LABELED_DIR, label)
    files = glob.glob(os.path.join(folder, "*.jpg")) + \
            glob.glob(os.path.join(folder, "*.jpeg")) + \
            glob.glob(os.path.join(folder, "*.png"))
    print(f"{label}: {len(files)} photos")
    for f in files:
        try:
            X.append(get_embedding(f))
            y.append(label_idx)
        except PhotoValidationError as e:
            print(f"  SKIPPED {f}: {e}")

X = np.array(X)
y = np.array(y)
print(f"\nTotal: {len(X)} embeddings, shape {X.shape}")

class_counts = Counter(y)
smallest_class = min(class_counts.values())
k = min(DESIRED_K, smallest_class)

if k < 2:
    print(f"\nERROR: your smallest class only has {smallest_class} photo(s).")
    print("Cross-validation needs at least 2 examples per class. Add more photos and rerun.")
    exit()

if k < DESIRED_K:
    print(f"\nSmallest class has {smallest_class} photos, so reducing k from {DESIRED_K} to {k}.")

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

all_true, all_pred = [], []

print(f"\nRunning {k}-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
    X_test_t = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test_t = torch.tensor(y[test_idx], dtype=torch.long)

    head = nn.Linear(X.shape[1], len(CLASSES))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        optimizer.zero_grad()
        out = head(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = head(X_test_t).argmax(dim=1).numpy()

    fold_acc = accuracy_score(y[test_idx], preds)
    print(f"  Fold {fold}: accuracy {fold_acc:.3f}")

    all_true.extend(y[test_idx])
    all_pred.extend(preds)

print("\n--- Overall cross-validated results ---")
print("Accuracy:", accuracy_score(all_true, all_pred))
print("Macro F1:", f1_score(all_true, all_pred, average="macro", zero_division=0))
print("\nConfusion matrix (rows=true, cols=predicted):")
print(confusion_matrix(all_true, all_pred, labels=list(range(len(CLASSES)))))
print("\nClassification report:")
print(classification_report(all_true, all_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0))

# Train one final head on ALL data for actual deployment
print("\nTraining final head on full dataset for deployment...")
X_all_t = torch.tensor(X, dtype=torch.float32)
y_all_t = torch.tensor(y, dtype=torch.long)
final_head = nn.Linear(X.shape[1], len(CLASSES))
optimizer = torch.optim.Adam(final_head.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(100):
    optimizer.zero_grad()
    out = final_head(X_all_t)
    loss = loss_fn(out, y_all_t)
    loss.backward()
    optimizer.step()

torch.save(final_head.state_dict(), "classifier_head.pt")
print("Saved final classifier head to classifier_head.pt")
