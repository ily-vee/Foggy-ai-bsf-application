import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Text-only model
DATASET_PATH = Path("dataset_foggy_v1.jsonl")
OUTPUT_DIR = Path("models/qwen_bsf_qlora")

def train_qwen_qlora():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading text model {MODEL_ID} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,  # Higher batch size possible without vision embeddings
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting Qwen2.5-3B QLoRA Fine-Tuning...")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-Tuning Complete!")

if __name__ == "__main__":
    train_qwen_qlora()