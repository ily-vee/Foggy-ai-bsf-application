import torch
import torch.nn as nn
from pathlib import Path
from transformers import SiglipVisionModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

MODEL_ID = "google/siglip-base-patch16-224"

class SiglipBSFClassifier(nn.Module):
    def __init__(self, num_classes=5, r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        
        # Load local backbone if cached, otherwise fetch from Hugging Face
        local_path = Path("siglip2_local")
        if local_path.exists():
            print(f"Loading base vision model from local cache: {local_path}")
            self.backbone = SiglipVisionModel.from_pretrained(str(local_path))
        else:
            print(f"Loading base vision model from Hugging Face: {MODEL_ID}")
            self.backbone = SiglipVisionModel.from_pretrained(MODEL_ID)

        # LoRA Adapter Configuration for Vision Encoder
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        # Apply LoRA to the vision backbone
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        # SigLIP hidden embedding size is 768
        hidden_size = self.backbone.config.hidden_size
        
        # Custom 5-class BSF Stage Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        # Extract features from SigLIP backbone
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Pooled visual representation (batch_size, hidden_size)
        pooled_output = outputs.pooler_output
        
        # Class logits
        logits = self.classifier(pooled_output)
        return logits

def get_siglip_lora_model(num_classes=5):
    model = SiglipBSFClassifier(num_classes=num_classes)
    
    # Print trainable vs total parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture Ready:")
    print(f"  - Total Parameters:     {all_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    return model

if __name__ == "__main__":
    model = get_siglip_lora_model()