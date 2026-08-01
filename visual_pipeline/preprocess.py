from transformers import AutoImageProcessor, AutoModel
from PIL import Image, UnidentifiedImageError
import torch
import cv2
import numpy as np
import os

CHECKPOINT = "google/siglip2-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT).eval()
model.to(DEVICE)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 10
MIN_DIMENSION = 50
BLUR_THRESHOLD = 100.0  # lower = blurrier; tune this against real farmer photos


class PhotoValidationError(Exception):
    """Raised when a photo fails validation before reaching the model."""
    pass


def validate_file(image_path: str):
    # 1. Extension check
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise PhotoValidationError(f"Unsupported file type '{ext}'. Please send a JPG or PNG photo.")

    # 2. File size check
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PhotoValidationError(f"File too large ({size_mb:.1f}MB). Please send a smaller photo.")


def check_blur(image_path: str):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise PhotoValidationError("Could not read image for blur check — file may be corrupted.")

    edges = cv2.Canny(gray, 50, 150)
    if edges.sum() == 0:
        raise PhotoValidationError("Photo appears too blurry. Please retake it in better focus/lighting.")

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_pixels = laplacian[edges > 0]

    if len(edge_pixels) == 0:
        raise PhotoValidationError("Photo appears too blurry. Please retake it in better focus/lighting.")

    sharpness = edge_pixels.var()
    if sharpness < BLUR_THRESHOLD:
        raise PhotoValidationError("Photo appears too blurry. Please retake it in better focus/lighting.")
    return sharpness

def preprocess_and_embed(image_path: str) -> torch.Tensor:
    # Validation layer 1: file-level checks
    validate_file(image_path)

    # Validation layer 2: can PIL actually open it as an image?
    try:
        img = Image.open(image_path)
        img.verify()  # checks file integrity without fully decoding
        img = Image.open(image_path)  # reopen after verify() closes the file pointer
    except (UnidentifiedImageError, OSError) as e:
        raise PhotoValidationError(f"File is not a valid image or is corrupted: {e}")

    # Validation layer 3: minimum resolution
    if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
        raise PhotoValidationError("Image too small — likely not usable for classification")

    # Validation layer 4: blur check
    check_blur(image_path)

    # Preprocessing + embedding (CUDA fix applied safely)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # Moves input tensors cleanly to GPU/CPU
    
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        
    embedding = output.pooler_output.cpu()  # Transferred to CPU tensor for numpy/scikit-learn downstream
    return embedding

if __name__ == "__main__":
    test_path = "photos_raw/WhatsApp Image 2026-07-08 at 11.03.45.jpeg"
    try:
        result = preprocess_and_embed(test_path)
        print("Success:", result.shape)
    except PhotoValidationError as e:
        print("Validation failed:", e)