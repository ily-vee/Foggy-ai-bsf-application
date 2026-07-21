import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 1. Configuration
MODEL_NAME = "google/siglip2-base-patch16-224"
CLASSIFIER_WEIGHTS = "bsf_classifier.pth"
INPUT_DIM = 768
NUM_CLASSES = 5

# Map our numerical class indexes back to readable English labels
CLASS_LABELS = {
    0: "1_eggs",
    1: "2_early_larvae",
    2: "3_feeding_larvae",
    3: "4_pupae",
    4: "5_bsf_adult"
}


# 2. Re-create the MLP Class so PyTorch can map the saved weights
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


def load_classifier():
    """Loads the trained PyTorch weights into our MLP architecture."""
    model = BSFClassifierHead(INPUT_DIM, NUM_CLASSES)
    if os.path.exists(CLASSIFIER_WEIGHTS):
        model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS))
        model.eval()
        return model
    else:
        print(f"❌ Error: Could not find '{CLASSIFIER_WEIGHTS}'. Please run train_classifier.py first!")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python classify_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found.")
        sys.exit(1)

    print("⏳ Initializing models...")

    # Load feature extractor and classifier
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    siglip_model = AutoModel.from_pretrained(MODEL_NAME)
    siglip_model.eval()

    classifier = load_classifier()

    # Process and classify
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 1. Extract feature embedding from SigLIP 2
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                siglip_outputs = siglip_model.vision_model(**inputs)
                embedding = siglip_outputs.pooler_output  # Shape: [1, 768]

                # 2. Pass embedding through our custom PyTorch MLP head
                classifier_outputs = classifier(embedding)

                # Apply Softmax to get clean, readable confidence percentages
                probabilities = torch.nn.functional.softmax(classifier_outputs, dim=1).squeeze().numpy()

            predicted_class_idx = np.argmax(probabilities)
            predicted_label = CLASS_LABELS[predicted_class_idx]
            confidence = probabilities[predicted_class_idx] * 100

            print("\n" + "=" * 40)
            print(f"🔍 ANALYSIS RESULTS FOR: {os.path.basename(image_path)}")
            print("=" * 40)
            print(f"🏆 Predicted Stage:  {predicted_label}")
            print(f"📈 Confidence Score: {confidence:.2f}%")
            print("-" * 40)
            print("Distribution breakdown:")
            for idx, prob in enumerate(probabilities):
                print(f"  * {CLASS_LABELS[idx]:<18}: {prob * 100:5.2f}%")
            print("=" * 40)

    except Exception as e:
        print(f"❌ Inference processing failure: {str(e)}")


if __name__ == "__main__":
    main()
