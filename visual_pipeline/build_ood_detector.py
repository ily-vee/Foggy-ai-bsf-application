"""
Builds the BSF out-of-distribution (wrong-image) detector from real
labeled photos. Produces two versions -- with and without PCA -- so you
can compare them before deciding which to keep.
"""

import glob
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from preprocess import preprocess_and_embed, PhotoValidationError
from bsf_ood_detector import MahalanobisOODDetector

LABELED_DIR = "labeled_photos"
CLASSES = ["egg", "larva", "prepupa", "pupa", "adult"]
CALIB_FRACTION = 0.3   # ~30% of each class held out for threshold calibration
PCA_DIMENSIONS = 20    # reduce 768 -> 20 dims for the PCA version


def load_embeddings():
    """Extract embeddings + labels from labeled_photos/, skipping bad photos."""
    X, y = [], []
    for label in CLASSES:
        folder = os.path.join(LABELED_DIR, label)
        files = (
            glob.glob(os.path.join(folder, "*.jpg"))
            + glob.glob(os.path.join(folder, "*.jpeg"))
            + glob.glob(os.path.join(folder, "*.png"))
        )
        for f in files:
            try:
                emb = preprocess_and_embed(f)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                X.append(emb.squeeze().detach().numpy())
                y.append(label)
            except PhotoValidationError as e:
                print(f"  SKIPPED {f}: {e}")
    return np.array(X), np.array(y)


def build_and_evaluate(X_fit, y_fit, X_calib, y_calib, label, use_pca=False, pca=None):
    detector = MahalanobisOODDetector()
    detector.fit(X_fit, list(y_fit))
    threshold = detector.calibrate(X_calib, list(y_calib), percentile=97.5)

    # Self-distances: how far genuine calibration photos sit from their OWN class
    self_distances = []
    for emb, cls in zip(X_calib, y_calib):
        self_distances.append(detector.distance_to_class(emb, cls))

    # A synthetic "wrong image" -- random noise, nowhere near any real cluster
    rng = np.random.default_rng(0)
    fake_wrong = rng.normal(loc=0, scale=X_fit.std(), size=X_fit.shape[1])
    _, nearest_cls, wrong_distance = detector.is_in_distribution(fake_wrong)

    print(f"\n--- {label} ---")
    print(f"Threshold (97.5th percentile): {threshold:.3f}")
    print(f"Genuine calib photos -- mean distance: {np.mean(self_distances):.3f}, "
          f"max: {np.max(self_distances):.3f}")
    print(f"Synthetic wrong image -- distance: {wrong_distance:.3f} "
          f"(nearest class: {nearest_cls})")
    print(f"Separation ratio (wrong / max genuine): "
          f"{wrong_distance / np.max(self_distances):.2f}x  "
          f"(higher = better separation)")

    return detector


if __name__ == "__main__":
    print("Loading embeddings from labeled_photos/ ...")
    X, y = load_embeddings()
    print(f"Total usable embeddings: {len(X)}")
    for cls in CLASSES:
        print(f"  {cls}: {(y == cls).sum()}")

    # Stratified split so every class is represented in both fit and calib sets
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X, y, test_size=CALIB_FRACTION, stratify=y, random_state=42
    )

    # --- Version 1: no PCA, raw 768-dim embeddings ---
    detector_plain = build_and_evaluate(
        X_fit, y_fit, X_calib, y_calib, label="WITHOUT PCA (768-dim)"
    )
    detector_plain.save("bsf_ood_detector.pkl")

    # --- Version 2: PCA-reduced embeddings ---
    pca = PCA(n_components=PCA_DIMENSIONS, random_state=42)
    X_fit_pca = pca.fit_transform(X_fit)
    X_calib_pca = pca.transform(X_calib)

    detector_pca = build_and_evaluate(
        X_fit_pca, y_fit, X_calib_pca, y_calib,
        label=f"WITH PCA ({PCA_DIMENSIONS}-dim)"
    )
    detector_pca.save("bsf_ood_detector_pca.pkl")

    # Save the PCA transform too -- needed to reduce new embeddings at inference time
    import pickle
    with open("bsf_ood_pca_transform.pkl", "wb") as f:
        pickle.dump(pca, f)

    print("\nSaved: bsf_ood_detector.pkl, bsf_ood_detector_pca.pkl, bsf_ood_pca_transform.pkl")