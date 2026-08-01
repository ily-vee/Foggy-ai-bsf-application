"""
Full BSF photo classification pipeline: validation -> embedding ->
out-of-distribution check -> life-stage prediction.

This is the single entry point everything else (WhatsApp webhook, etc.)
should call.
"""

import torch
import torch.nn as nn

from preprocess import preprocess_and_embed, PhotoValidationError
from bsf_ood_detector import MahalanobisOODDetector

CLASSES = ["egg", "larva", "prepupa", "pupa", "adult"]
CLASSIFIER_HEAD_PATH = "classifier_head.pt"
OOD_DETECTOR_PATH = "bsf_ood_detector.pkl"
EMBEDDING_DIM = 768


class ClassificationResult:
    """Represents the outcome of running a photo through the full pipeline."""

    def __init__(self, status: str, message: str, predicted_class: str = None,
                 confidence: float = None, ood_distance: float = None):
        self.status = status  # "success" | "validation_failed" | "out_of_distribution"
        self.message = message
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.ood_distance = ood_distance

    def __repr__(self):
        return (f"ClassificationResult(status={self.status!r}, "
                f"predicted_class={self.predicted_class!r}, "
                f"confidence={self.confidence}, message={self.message!r})")


# --- Load the trained classifier head once, at import time ---
_classifier_head = nn.Linear(EMBEDDING_DIM, len(CLASSES))
_classifier_head.load_state_dict(torch.load(CLASSIFIER_HEAD_PATH, map_location="cpu"))
_classifier_head.eval()

# --- Load the OOD detector once, at import time ---
_ood_detector = MahalanobisOODDetector.load(OOD_DETECTOR_PATH)


def classify_photo(image_path: str) -> ClassificationResult:
    """
    Runs a photo through the full pipeline:
    1. Validation + embedding (preprocess.py)
    2. Out-of-distribution check (bsf_ood_detector.py)
    3. Life-stage classification (classifier_head.pt)

    Returns a ClassificationResult with a status the caller (e.g. the
    WhatsApp webhook) can branch on to decide what to reply.
    """
    # Step 1: validation + embedding
    try:
        embedding = preprocess_and_embed(image_path)
    except PhotoValidationError as e:
        return ClassificationResult(
            status="validation_failed",
            message=str(e),
        )

    # Normalize the same way the OOD detector and classifier were trained on
    embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding_np = embedding_norm.squeeze().detach().numpy()

    # Step 2: out-of-distribution check
    is_ok, nearest_class, distance = _ood_detector.is_in_distribution(embedding_np)
    if not is_ok:
        return ClassificationResult(
            status="out_of_distribution",
            message=(
                "This photo doesn't look like a recognizable BSF life stage. "
                "Please retake a clear photo of the colony."
            ),
            ood_distance=distance,
        )

    # Step 3: classification
    with torch.no_grad():
        logits = _classifier_head(embedding_norm)
        probs = torch.softmax(logits, dim=-1).squeeze()
        predicted_idx = int(probs.argmax())
        confidence = float(probs[predicted_idx])

    return ClassificationResult(
        status="success",
        message=f"Predicted life stage: {CLASSES[predicted_idx]}",
        predicted_class=CLASSES[predicted_idx],
        confidence=confidence,
        ood_distance=distance,
    )


if __name__ == "__main__":
    import sys
    import os

    test_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/testimage3.jpeg"

    print(f"\n📷 Photo: {os.path.basename(test_path)}")
    print("-" * 50)

    result = classify_photo(test_path)

    if result.status == "success":
        print("✅ Status: Success")
        print(f"🐛 Predicted stage: {result.predicted_class}")
        print(f"📊 Confidence: {result.confidence * 100:.1f}%")
        print(f"🔍 OOD distance: {result.ood_distance:.3f} (in-distribution)")
    elif result.status == "out_of_distribution":
        print("🚫 Status: Rejected — not a recognizable BSF photo")
        print(f"🔍 OOD distance: {result.ood_distance:.3f} (too far from any known class)")
        print(f"💬 Message: {result.message}")
    elif result.status == "validation_failed":
        print("⚠️  Status: Validation failed")
        print(f"💬 Reason: {result.message}")

    print("-" * 50 + "\n")