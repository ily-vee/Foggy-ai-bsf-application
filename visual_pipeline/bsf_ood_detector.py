"""
BSF "wrong image" detector using Mahalanobis distance in SigLIP2 embedding space.

Idea
----
We never collect negative photos. Instead we model how tightly each of the
5 BSF life-stage classes clusters in the 768-dim SigLIP2 embedding space
(mean + covariance per class). At inference time, a new photo's embedding
is scored against every class's cluster. If it's not reasonably close to
ANY class, it's flagged as "not a recognizable BSF photo" -- regardless of
what the classifier head would have guessed.

You already have everything this needs: the embeddings you extract for
training the classifier head.

Usage
-----
1. Fit the detector once on your training embeddings + labels.
2. Save it (pickle) alongside your classifier head.
3. At inference time, call `is_in_distribution(embedding)` BEFORE trusting
   the classifier head's prediction.
"""

import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class MahalanobisOODDetector:
    """Per-class mean + a single shared (tied) covariance, used for OOD scoring.

    Why a shared covariance instead of one per class: with 768-dim
    embeddings you'll typically have far fewer than 768 examples per
    class, so a per-class covariance matrix is badly under-determined
    and its inverse becomes unstable. Pooling residuals across all
    classes to estimate ONE shared covariance (standard technique from
    the Mahalanobis OOD-detection literature, e.g. Lee et al. 2018) uses
    your full dataset size for that estimate instead of one class's
    slice, which is far more stable.
    """

    class_names: List[str] = field(default_factory=list)
    means: Dict[str, np.ndarray] = field(default_factory=dict)
    inv_cov: np.ndarray = None
    threshold: float = None  # set during calibration

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, embeddings: np.ndarray, labels: List[str], shrinkage: float = 1.0):
        """
        embeddings: (N, D) array of SigLIP2 embeddings from your training set
        labels:     length-N list of class names, e.g.
                    "egg" / "larva" / "prepupa" / "pupa" / "adult"
        shrinkage:  value added to the covariance diagonal for numerical
                    stability. Tune down if you have hundreds+ of
                    examples per class; raise it if distances still look
                    unstable (e.g. wildly different scales per class).
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels)

        self.class_names = sorted(set(labels))
        self.means = {}

        # Pool mean-centered residuals across ALL classes to estimate one
        # shared covariance matrix (see class docstring for why).
        all_residuals = []
        for cls in self.class_names:
            cls_embeddings = embeddings[labels == cls]
            if cls_embeddings.shape[0] < 2:
                raise ValueError(
                    f"Need at least 2 examples for class '{cls}' to estimate covariance, "
                    f"got {cls_embeddings.shape[0]}."
                )
            mean = cls_embeddings.mean(axis=0)
            self.means[cls] = mean
            all_residuals.append(cls_embeddings - mean)

        pooled_residuals = np.vstack(all_residuals)
        cov = np.cov(pooled_residuals, rowvar=False)
        cov += np.eye(cov.shape[0]) * shrinkage
        self.inv_cov = np.linalg.inv(cov)

        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def distance_to_class(self, embedding: np.ndarray, cls: str) -> float:
        """Mahalanobis distance from `embedding` to class `cls`'s mean."""
        diff = embedding - self.means[cls]
        return float(np.sqrt(diff @ self.inv_cov @ diff.T))

    def nearest_class(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Returns (closest_class_name, distance_to_it)."""
        embedding = np.asarray(embedding, dtype=np.float64)
        distances = {cls: self.distance_to_class(embedding, cls) for cls in self.class_names}
        best_cls = min(distances, key=distances.get)
        return best_cls, distances[best_cls]

    def is_in_distribution(self, embedding: np.ndarray) -> Tuple[bool, str, float]:
        """
        Main entry point for the gate.

        Returns (is_in_distribution, nearest_class, distance).
        If threshold hasn't been calibrated yet, raises an error --
        you must run calibrate() first.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before using is_in_distribution().")

        nearest_cls, distance = self.nearest_class(embedding)
        return distance <= self.threshold, nearest_cls, distance

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def calibrate(self, embeddings: np.ndarray, labels: List[str], percentile: float = 97.5):
        """
        Sets the accept/reject threshold using your OWN training embeddings
        (no negative/random images required).

        For each training example, compute its Mahalanobis distance to its
        OWN class's centroid. That gives you a distribution of "how far a
        genuine BSF photo of this class typically sits from its centroid."
        Set the threshold at a high percentile of that distribution (e.g.
        97.5th) so that ~97.5% of genuine BSF photos pass, while embeddings
        far outside that spread (wrong images) get rejected.

        Tune `percentile` based on your tolerance for false rejections vs
        false accepts once you test on real farmer photos.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels)

        self_distances = []
        for cls in self.class_names:
            cls_embeddings = embeddings[labels == cls]
            for emb in cls_embeddings:
                self_distances.append(self.distance_to_class(emb, cls))

        self.threshold = float(np.percentile(self_distances, percentile))
        return self.threshold

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MahalanobisOODDetector":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- Replace this block with your real SigLIP2 embeddings + labels ---
    # embeddings.shape == (N, 768), labels is a length-N list like
    # ["egg", "egg", "larva", "adult", ...]
    #
    # IMPORTANT: split into a fit set (used to compute class means/covariance)
    # and a separate calibration set (used to set the threshold). Calibrating
    # on the same samples used to fit the means understates the true spread
    # (resubstitution bias) -- do this split on your real data too.
    rng = np.random.default_rng(0)
    fake_classes = ["egg", "larva", "prepupa", "pupa", "adult"]
    fit_embeddings, fit_labels = [], []
    calib_embeddings, calib_labels = [], []
    class_centers = {}
    for i, cls in enumerate(fake_classes):
        center = rng.normal(loc=i * 3, scale=1, size=768)
        class_centers[cls] = center
        fit_samples = center + rng.normal(scale=0.5, size=(40, 768))
        calib_samples = center + rng.normal(scale=0.5, size=(20, 768))
        fit_embeddings.append(fit_samples)
        fit_labels.extend([cls] * 40)
        calib_embeddings.append(calib_samples)
        calib_labels.extend([cls] * 20)
    fit_embeddings = np.vstack(fit_embeddings)
    calib_embeddings = np.vstack(calib_embeddings)
    # -----------------------------------------------------------------

    detector = MahalanobisOODDetector()
    detector.fit(fit_embeddings, fit_labels)
    threshold = detector.calibrate(calib_embeddings, calib_labels, percentile=97.5)
    print(f"Calibrated threshold: {threshold:.3f}")

    # A genuine-looking BSF larva photo: fresh noise draw around the same
    # true larva center (simulates a new, unseen larva photo)
    genuine_embedding = class_centers["larva"] + rng.normal(scale=0.5, size=768)
    ok, nearest, dist = detector.is_in_distribution(genuine_embedding)
    print(f"Genuine photo  -> in_distribution={ok}, nearest={nearest}, distance={dist:.3f}")

    # A "wrong image" (e.g. a chicken photo) -- far from every cluster
    wrong_embedding = rng.normal(loc=50, scale=5, size=768)
    ok, nearest, dist = detector.is_in_distribution(wrong_embedding)
    print(f"Wrong photo    -> in_distribution={ok}, nearest={nearest}, distance={dist:.3f}")

    detector.save("bsf_ood_detector.pkl")
