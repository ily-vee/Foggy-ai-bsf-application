import json
import torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH = Path("core/finetune/dataset_llm.jsonl")
OUTPUT_DIR = Path("models/qwen_bsf_qlora")

# System instruction enforcing role identity and prohibiting boilerplate disclaimers
SYSTEM_PROMPT = (
    "You are Foggy, a precision Black Soldier Fly farming AI assistant. "
    "Provide concise, direct operational advice. Do not include canned AI disclaimers, signatures, or meta-commentary."
)

def train_qwen_qlora():
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID) 
    # converts images to pixel values and tokenizes text prompts, numerical formatting for model input

    # 4-bit Quantization Configuration for QLoRA
    # configures quantization, enabling 4-bit precision with nf4 quantization, and sets compute dtype based on GPU support
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # nf4 Normalized float 4-bit,stores weights more diligently, it is a newer method that reduces memory usage while maintaining model performance
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # The weights stay compressed in 4-bit for storage. But when calculations happen,they're temporarily computed in BF16 or FP16 because GPUs can't realistically do all neural network math directly in 4-bit with sufficient numerical stability.
        bnb_4bit_use_double_quant=True 
    )

    # this loads the Qwen2.5-VL model in 4-bit precision, applying the quantization configuration and setting the device map to auto for GPU allocation, figures out where every layer goes . It also sets the torch dtype based on GPU support for bfloat16 or float16 during computation.
    print(f"Loading model {MODEL_ID} in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    # Prepare the model for k-bit training, freezes the non-trainable parameters and sets up the model for efficient low-rank adaptation (LoRA) fine-tuning. This is crucial for QLoRA, which focuses on training only a small subset of parameters while keeping the rest frozen, allowing for efficient fine-tuning on limited resources.
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA targets across attention and MLP layers
    peft_config = LoraConfig(
        r=16, # rank, determines how much the low-rank matrices can adapt to the original weights. A higher r allows for more flexibility but increases the number of trainable parameters hemce how much information can be retained/learned.
        lora_alpha=32, #determines how strongly the learned information from the low-rank matrices influences the original weights. A higher alpha means the LoRA updates have a stronger effect on the model's predictions.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # specifies which layers of the model will be adapted using LoRA because not every layer contributes equally to learning. These are typically the projection layers in attention mechanisms and MLPs, where most of the model's learning capacity resides
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load multi-modal dataset using HF datasets
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

    def collate_fn(batch):
        texts = []
        images = []
        for sample in batch:
            messages = sample["messages"]
            # Extract image path and question/answer
            img_path = messages[0]["content"][0]["image"]
            user_text = messages[0]["content"][1]["text"]
            assistant_text = messages[1]["content"][0]["text"]

            img = Image.open(img_path).convert("RGB")
            
            # Format prompt with ChatML system header + vision tag + assistant target
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
            )
            texts.append(prompt)
            images.append([img])

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=30,
        optim="paged_adamw_8bit",
        remove_unused_columns=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    print("\nStarting Qwen2.5-VL QLoRA Fine-Tuning...")
    trainer.train()

    print(f"\nSaving fine-tuned QLoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("QLoRA Fine-Tuning Complete!")

if __name__ == "__main__":
    train_qwen_qlora()