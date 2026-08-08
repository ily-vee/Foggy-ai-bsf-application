import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info

# MUST match inference_pipeline_qwen_vlm.py's QWEN_BASE exactly — a LoRA
# trained against a different model (different size, different Qwen
# generation) will not attach to the inference pipeline's model at all.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH = Path("dataset_foggy_vlm_v1.jsonl")  # the VLM dataset, not the old text-only one
OUTPUT_DIR = Path("models/qwen_vlm_bsf_qlora")

# MUST match inference_pipeline_qwen_vlm.py's QWEN_VL_MIN_PIXELS/MAX_PIXELS.
# Training on a different image-resolution range than what's served at
# inference time means the model learns on a different vision-token-count
# distribution than it'll actually see in production — keep these identical.
QWEN_VL_MIN_PIXELS = 256 * 28 * 28
QWEN_VL_MAX_PIXELS = 768 * 28 * 28


class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor
        # Right-padding is required for the prompt-length-based masking
        # below to correctly line up with token position 0 of each row.
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        # Extract raw conversation lists from the batch dictionaries
        conversations = [example["messages"] for example in examples]
        
        # Process images/videos from raw conversations
        image_inputs, video_inputs = process_vision_info(conversations)
        
        # Format text prompts using the processor chat template
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in conversations
        ]
        
        # Tokenize and encode multimodal batch
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()

        # Mask pad tokens...
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # ...AND mask the prompt portion (system + user turn, including the
        # image and reference context) of each row. Without this, loss is
        # computed across the ENTIRE sequence, meaning the model partially
        # "learns" to predict the question and context back to itself —
        # wasted capacity at best, and it dilutes the actual signal you
        # want (how to answer), at worst. This finds each row's prompt
        # length by re-templating everything except the final assistant
        # turn, then masks exactly that many leading tokens.
        for i, conv in enumerate(conversations):
            prompt_only = conv[:-1]
            prompt_text = self.processor.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            prompt_image_inputs, _ = process_vision_info(prompt_only)
            prompt_len = self.processor(
                text=[prompt_text], images=prompt_image_inputs, return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        inputs["labels"] = labels
        
        return inputs


def train_qwen_vlm_qlora():
    print(f"Loading AutoProcessor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        min_pixels=QWEN_VL_MIN_PIXELS, max_pixels=QWEN_VL_MAX_PIXELS,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading Vision-Language Model {MODEL_ID} in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # Prepare quantized model for PEFT
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

    # Instantiate custom vision data collator
    data_collator = Qwen2VLDataCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,  # Lower batch size due to high VRAM overhead of pixel tensors
        gradient_accumulation_steps=8,  # Maintains effective batch size of 8
        warmup_steps=10,
        # max_steps=100 previously meant ~27 passes over the actual 29-row
        # dataset (100 steps * effective batch 8 / 29 examples) — real
        # overfitting risk on a set this small. num_train_epochs gives
        # direct, legible control instead: this is 8 full passes over your
        # 29 examples. Adjust once you've seen how the pilot behaves —
        # if outputs start reciting training examples verbatim on held-out
        # test images, that's a sign to lower this.
        num_train_epochs=8,
        max_steps=-1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        remove_unused_columns=False  # Required: prevents HF Trainer from dropping raw image path keys
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )

    print(f"\nStarting {MODEL_ID} QLoRA Fine-Tuning...")
    trainer.train()

    print(f"\nSaving fine-tuned adapter and processor to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("VLM Fine-Tuning Complete!")


if __name__ == "__main__":
    train_qwen_vlm_qlora()