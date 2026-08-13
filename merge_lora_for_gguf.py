"""
merge_lora_for_gguf.py

Merges the trained QLoRA adapter into the base Qwen2.5-VL-3B-Instruct model
and saves the result as a single, standalone model directory. This is a
required step before GGUF conversion — llama.cpp's convert_hf_to_gguf.py
expects one merged model, not a base model + separate adapter.

Usage:
  python3 merge_lora_for_gguf.py models/qwen_vlm_bsf_qlora models/qwen_vlm_bsf_merged
"""

import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Copied straight from the base model's original download rather than
# re-serialized by save_pretrained(). LoRA doesn't touch the vocabulary,
# so these should be byte-identical to the base model's — and this avoids
# a real, observed failure mode: save_pretrained() under one transformers
# version can write a tokenizer config format that a DIFFERENT transformers
# version (e.g. the older one llama.cpp's converter requires) fails to
# load, even though nothing about the actual tokenizer changed. Skipping
# the round-trip sidesteps this entirely.
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.json",
]


def merge(adapter_dir: str, output_dir: str):
    print(f"Loading base model {BASE_MODEL_ID}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)

    print("Overwriting tokenizer files with pristine copies from the base model cache...")
    base_snapshot_dir = Path(snapshot_download(repo_id=BASE_MODEL_ID))
    output_path = Path(output_dir)
    copied = []
    for filename in TOKENIZER_FILES:
        src = base_snapshot_dir / filename
        if src.exists():
            shutil.copy2(src, output_path / filename)
            copied.append(filename)
    print(f"Copied pristine tokenizer files: {copied}")

    print(f"\n✅ Merged model ready at {output_dir}")
    print("Next: convert this directory with llama.cpp's convert_hf_to_gguf.py")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 merge_lora_for_gguf.py <adapter_dir> <output_dir>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])