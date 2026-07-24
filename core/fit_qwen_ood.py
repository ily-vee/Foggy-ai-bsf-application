import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from bsf_ood_detector import MahalanobisOODDetector

VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

def generate_qwen_embeddings(image_paths, processor, vision_encoder, device):
    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output = vision_encoder(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
            patch_embeds = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            
            if patch_embeds.ndim == 3:
                feature_vec = patch_embeds.mean(dim=1)
            elif patch_embeds.ndim == 2:
                feature_vec = patch_embeds.mean(dim=0, keepdim=True)
            else:
                feature_vec = patch_embeds
                
            embeddings.append(feature_vec.to(dtype=torch.float32).squeeze().cpu().numpy())
    return np.array(embeddings, dtype=np.float64)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VISION_MODEL_ID, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    vision_encoder = model.visual if hasattr(model, "visual") else model.model.visual
    vision_encoder.eval()

    # TODO: Load your training image file paths & labels
    # train_paths = [...]
    # train_labels = [...]
    # calib_paths = [...]
    # calib_labels = [...]

    print("Extracting Qwen embeddings...")
    train_embeds = generate_qwen_embeddings(train_paths, processor, vision_encoder, device)
    calib_embeds = generate_qwen_embeddings(calib_paths, processor, vision_encoder, device)

    detector = MahalanobisOODDetector()
    detector.fit(train_embeds, train_labels)
    threshold = detector.calibrate(calib_embeds, calib_labels, percentile=97.5)
    
    detector.save("qwen_ood_detector.pkl")
    print(f" Saved qwen_ood_detector.pkl with calibrated threshold: {threshold:.3f}")