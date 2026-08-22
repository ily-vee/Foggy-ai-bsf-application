"""
train_qwenvlm_qlora_v3.py

Foggy — BSF Farming AI Assistant
Qwen2.5-VL-3B-Instruct QLoRA fine-tuning, v3: LANGUAGE DECODER + VISION TOWER.

WHY V3 (vs train_qwenvlm_qlora_v2.py)
-------------------------------------------
Purely a dataset-target change -- nothing about the model, quantization
strategy, LoRA setup, or training loop below differs from v2. The only
edits from that file are DATASET_PATH (now dataset_foggy_vlm_v3.jsonl,
built by build_foggy_vlm_dataset_v3.py) and OUTPUT_DIR (now
models/qwen_vlm_bsf_qlora_v3, kept separate from v2's adapter so it stays
around as a fallback/comparison point until v3 is validated, same reason
v2 didn't overwrite v1's output dir).

Why a new dataset version at all: testing the v2-trained adapter directly
surfaced three gaps no amount of retraining against the SAME data would
fix -- image-grounded answers not consistently naming the detected stage
(so the model sometimes contradicted SigLIP2's given label on an
out-of-training photo), missing coverage for plain generic FAQ-style
questions (causing raw-quoting of retrieved context instead of the
mandated structure), and a system prompt that needed real changes rather
than a live hand-edit (a live edit was tried and caused a measurable
"Here's how I know that: I'm quoting directly from..." degenerate pattern
in 100% of test turns -- see build_foggy_vlm_dataset_v3.py's own docstring
for the full account). v3's dataset and system prompt were built and are
being trained together, which is the safe way to change a prompt for a
lightly-tuned adapter like this one.

Everything else below -- the vision-tower QLoRA extension, the naming-
collision handling between vision and language module paths, the mixed-
precision quantization-skip-module discovery, the two-learning-rate
optimizer, the data collator -- is unchanged from v2 and still fully
applies, since none of it is dataset-specific. See train_qwenvlm_qlora_v2.py's
own docstring (preserved in git history / alongside this file) for the
full reasoning behind each of those, including the real crash this file
fixed (llm_int8_skip_modules=["visual"] silently giving up lm_head's
default quantization exclusion) and why that fix is version-agnostic
rather than hardcoded to one transformers release.

DATASET: dataset_foggy_vlm_v3.jsonl -- see that file's own generation
output for the current exact image/text-only/multi-turn/computed-values
breakdown; run `python build_foggy_vlm_dataset_v3.py --check_images` to
regenerate it and print current counts. No filtering to image-only
examples is done or needed: a text-only example's forward pass never
touches `model.visual`, so it naturally contributes zero gradient to the
vision LoRA weights while still teaching the language decoder tone/
structure/grounding -- the split happens on its own from the data.

WHAT THIS SCRIPT DOES NOT TOUCH: SigLIP2 (models/siglip2_bsf_lora) is a
completely separate model and training pipeline; nothing here loads or
affects it.

VERIFICATION CAVEAT: same as v2 -- this file was written and reasoned
through in a CPU-only dev environment without peft/bitsandbytes/trl/
qwen_vl_utils installed, verified structurally against the real installed
transformers version's actual module tree and signatures rather than
memory. Run with a tiny --max_steps first (see bottom of file) before a
full run.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_qwenvlm_qlora_v3")

# MUST match inference_pipeline_qwen_vlm.py's QWEN_BASE exactly.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH = Path("dataset_foggy_vlm_v3.jsonl")
# Separate output dir from v2's models/qwen_vlm_bsf_qlora_v2 -- keeps that
# adapter around as a fallback/comparison point until v3 is validated.
# Update inference_pipeline_qwen_vlm.py's QWEN_LORA_DIR (or wherever the
# GGUF conversion step reads the adapter from) once you're ready to switch.
OUTPUT_DIR = Path("models/qwen_vlm_bsf_qlora_v3")

# MUST match inference_pipeline_qwen_vlm.py's QWEN_VL_MIN_PIXELS/MAX_PIXELS --
# training on a different image-resolution range than what's served in production
# means learning on a different vision-token-count distribution than deployment.
QWEN_VL_MIN_PIXELS = 256 * 28 * 28
QWEN_VL_MAX_PIXELS = 768 * 28 * 28

# The vision tower's top-level submodule has been named "visual" since Qwen2-VL's
# original release and is depended on by GGUF conversion / mmproj tooling
# elsewhere in this project's deployment path, so it's about as stable a naming
# assumption as this codebase has -- unlike deeper internal layer names (attn vs.
# self_attn, proj vs. o_proj), which discover_vision_target_modules() below
# re-derives at runtime instead of assuming.
VISION_MODULE_MARKER = "visual"

LANGUAGE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LANGUAGE_LORA_R = 16
LANGUAGE_LORA_ALPHA = 32
LANGUAGE_LR = 2e-4

# Half the language rank/alpha, and roughly 4x lower learning rate: a smaller,
# gentler nudge to the pretrained vision representations rather than the same
# capacity/aggressiveness used for the language decoder -- see module docstring.
VISION_LORA_R = 8
VISION_LORA_ALPHA = 16
VISION_LR = 5e-5

LORA_DROPOUT = 0.05
EVAL_FRACTION = 0.10
SEED = 42


def discover_quantization_skip_modules(model_id):
    """Returns the llm_int8_skip_modules list to pass into BitsAndBytesConfig:
    transformers' own default exclusions (typically just lm_head -- see module
    docstring) UNION the vision tower's actual top-level path. Built from a
    torch.device("meta") skeleton constructed from just the config (no weight
    download) so this is correct regardless of how many levels this transformers
    version happens to nest the vision tower under, or what its default
    exclusions are -- neither is hardcoded here."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        skeleton = Qwen2_5_VLForConditionalGeneration(config)

    try:
        from transformers.quantizers.base import get_keys_to_not_convert
        default_skips = get_keys_to_not_convert(skeleton)
    except ImportError as e:
        logger.warning(f"Could not import transformers' get_keys_to_not_convert ({e}); "
                        f"falling back to a hardcoded ['lm_head']. Verify this is still "
                        f"correct for your installed transformers version.")
        default_skips = ["lm_head"]

    vision_prefix = next(
        (name for name, _ in skeleton.named_modules() if name.split(".")[-1] == VISION_MODULE_MARKER), None,
    )
    if vision_prefix is None:
        raise RuntimeError(
            f"Could not find a '{VISION_MODULE_MARKER}' submodule in {model_id}'s "
            f"architecture -- update VISION_MODULE_MARKER before training."
        )

    skip_modules = sorted(set(default_skips) | {vision_prefix})
    logger.info(f"Quantization skip modules: {skip_modules} (default exclusions: {default_skips}, "
                f"vision tower path: '{vision_prefix}')")
    return skip_modules


def discover_vision_target_modules(model, language_target_modules):
    """Find the vision-tower Linear submodules NOT already covered (by accidental
    name collision -- see module docstring) by language_target_modules, and return
    the shortest path suffix for each that provably cannot also match a
    LANGUAGE-decoder Linear layer. This is intentionally empirical rather than
    hardcoding "attn.qkv"/"attn.proj"/etc.: those names were verified once against
    one transformers version and could rename in another."""
    linear_paths = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    vision_paths = [p for p in linear_paths if VISION_MODULE_MARKER in p]
    language_paths = [p for p in linear_paths if VISION_MODULE_MARKER not in p]

    def safe_suffix(path):
        parts = path.split(".")
        for n in range(2, len(parts) + 1):  # start at 2 segments: never accept a
            candidate = ".".join(parts[-n:])  # single bare leaf like "0" or "proj"
            if not any(lp.endswith(candidate) for lp in language_paths):
                return candidate
        return path  # fully-qualified path is always unique as a last resort

    targets = set()
    for path in vision_paths:
        leaf = path.split(".")[-1]
        if leaf in language_target_modules:
            continue  # already covered via the shared-name collision; see rank_pattern
        targets.add(safe_suffix(path))
    return sorted(targets)


def verify_vision_stayed_unquantized(model):
    vision_dtypes = {p.dtype for n, p in model.named_parameters() if VISION_MODULE_MARKER in n}
    logger.info(f"Vision tower parameter dtypes after loading: {vision_dtypes}")
    if any(dt in (torch.int8, torch.uint8) for dt in vision_dtypes):
        raise RuntimeError(
            "Vision tower appears to have been 4-bit quantized despite "
            "llm_int8_skip_modules -- 'visual' no longer matches this "
            "transformers version's vision-tower module naming. Fix "
            "VISION_MODULE_MARKER before training."
        )


def verify_lora_ranks(model):
    """Best-effort check that rank_pattern/alpha_pattern actually gave the vision
    tower a different rank than the language decoder. Warns (doesn't crash) if the
    installed peft version stores per-layer rank somewhere this doesn't expect --
    better to flag that for a human than to assume silently either way."""
    try:
        vision_ranks, language_ranks = set(), set()
        for name, module in model.named_modules():
            r = getattr(module, "r", None)
            if not isinstance(r, dict) or "default" not in r:
                continue
            (vision_ranks if VISION_MODULE_MARKER in name else language_ranks).add(r["default"])
        logger.info(f"Vision-tower LoRA ranks in use: {vision_ranks} (expected {{{VISION_LORA_R}}})")
        logger.info(f"Language-decoder LoRA ranks in use: {language_ranks} (expected {{{LANGUAGE_LORA_R}}})")
        if vision_ranks and vision_ranks != {VISION_LORA_R}:
            raise RuntimeError(
                f"rank_pattern did not give the vision tower rank {VISION_LORA_R} "
                f"(found {vision_ranks} instead) -- check the rank_pattern regex "
                f"against this peft version's matching semantics before training."
            )
    except AttributeError as e:
        logger.warning(f"Could not verify per-module LoRA ranks ({e}) -- inspect model.named_modules() by hand.")


def build_optimizer(model, language_lr, vision_lr):
    """Two param groups (language vs. vision LoRA weights) at two learning rates --
    see module docstring for why. Falls back to plain torch.optim.AdamW if
    bitsandbytes' paged 8-bit optimizer isn't available in whatever environment
    this actually runs in; that costs more VRAM but is never a correctness issue,
    just a memory one (reduce per_device_train_batch_size or increase
    gradient_accumulation_steps if that fallback path OOMs)."""
    language_params, vision_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (vision_params if VISION_MODULE_MARKER in name else language_params).append(param)

    logger.info(f"Optimizer param groups: {len(language_params)} language tensors @ lr={language_lr}, "
                f"{len(vision_params)} vision tensors @ lr={vision_lr}")
    param_groups = [
        {"params": language_params, "lr": language_lr},
        {"params": vision_params, "lr": vision_lr},
    ]
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(param_groups)
    except (ImportError, AttributeError) as e:
        logger.warning(f"bitsandbytes PagedAdamW8bit unavailable ({e}); falling back to torch.optim.AdamW.")
        return torch.optim.AdamW(param_groups)


class Qwen2VLDataCollator:
    """Unchanged from v2 -- this class doesn't need to know or care which model
    parameters are trainable, only how to build (input_ids, labels) from raw
    conversations. It already handles the dataset's multi-turn grounding-via-
    history and conversation-memory examples correctly: prompt_only = conv[:-1]
    masks EVERYTHING except the final assistant turn regardless of how many
    turns precede it."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        conversations = [example["messages"] for example in examples]

        image_inputs, video_inputs = process_vision_info(conversations)
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]

        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for i, conv in enumerate(conversations):
            prompt_only = conv[:-1]
            prompt_text = self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
            prompt_image_inputs, _ = process_vision_info(prompt_only)
            prompt_len = self.processor(
                text=[prompt_text], images=prompt_image_inputs, return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def train_qwen_vlm_qlora_v3(max_steps: int = -1):
    logger.info(f"Loading AutoProcessor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, min_pixels=QWEN_VL_MIN_PIXELS, max_pixels=QWEN_VL_MAX_PIXELS,
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info("Resolving quantization skip modules (vision tower + transformers' own defaults)...")
    quant_skip_modules = discover_quantization_skip_modules(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # Keeps the vision tower in compute_dtype instead of 4-bit, AND preserves
        # transformers' own default exclusions (e.g. lm_head) that passing this
        # parameter at all would otherwise silently give up -- see module
        # docstring's "A REAL BUG THIS FILE HAD" section (train_qwenvlm_qlora_v2.py).
        llm_int8_skip_modules=quant_skip_modules,
    )

    logger.info(f"Loading {MODEL_ID} (language decoder in 4-bit, vision tower in {compute_dtype})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=compute_dtype,
    )
    verify_vision_stayed_unquantized(model)

    model = prepare_model_for_kbit_training(model)

    vision_targets = discover_vision_target_modules(model, LANGUAGE_TARGET_MODULES)
    if not vision_targets:
        raise RuntimeError(
            "No vision-tower Linear modules discovered -- VISION_MODULE_MARKER "
            f"('{VISION_MODULE_MARKER}') no longer matches this model's structure. "
            "Inspect model.named_modules() and update it before training."
        )
    logger.info(f"Discovered vision LoRA targets (beyond the shared-name ones already in "
                f"LANGUAGE_TARGET_MODULES): {vision_targets}")

    peft_config = LoraConfig(
        r=LANGUAGE_LORA_R,
        lora_alpha=LANGUAGE_LORA_ALPHA,
        target_modules=LANGUAGE_TARGET_MODULES + vision_targets,
        # Full-path regex, NOT the leaf-name-based target_modules matching above --
        # this is what actually separates vision's rank from language's despite
        # the shared gate_proj/up_proj/down_proj leaf names. See module docstring.
        rank_pattern={f".*{VISION_MODULE_MARKER}.*": VISION_LORA_R},
        alpha_pattern={f".*{VISION_MODULE_MARKER}.*": VISION_LORA_ALPHA},
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    verify_lora_ranks(model)
    model.print_trainable_parameters()

    full_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    split_dataset = full_dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
    logger.info(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    data_collator = Qwen2VLDataCollator(processor)
    optimizer = build_optimizer(model, LANGUAGE_LR, VISION_LR)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,  # per-image VRAM cost; see gradient_accumulation_steps
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch size 8
        warmup_steps=10,
        # 5 epochs is a starting point, not a fixed answer -- load_best_model_at_end
        # + EarlyStoppingCallback below exist specifically so this doesn't have to
        # be tuned by hand in advance. v2's own run confirmed this matters: eval_loss
        # bottomed out at epoch 2 and got worse for 2 epochs straight while
        # train_loss kept falling -- classic overfitting on a dataset this size, and
        # early stopping caught it and loaded the epoch-2 checkpoint instead of the
        # more-overfit final one.
        num_train_epochs=5,
        max_steps=max_steps,  # -1 lets num_train_epochs govern; override via --max_steps for a smoke test
        learning_rate=LANGUAGE_LR,  # informational only -- the optimizer above sets the real per-group LRs
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        remove_unused_columns=False,  # required: prevents HF Trainer from dropping raw image/message columns
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info(f"\nStarting {MODEL_ID} QLoRA fine-tuning (language decoder + vision tower)...")
    trainer.train()

    logger.info(f"\nSaving fine-tuned adapter and processor to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info("VLM fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Cap total training steps (e.g. 5) for a smoke test that exercises the "
             "full pipeline -- quantization, both LoRA targets, both optimizer "
             "groups, save/reload -- before committing to a full run. Default -1 "
             "means run the full num_train_epochs.",
    )
    args = parser.parse_args()
    if not DATASET_PATH.exists():
        sys.exit(
            f"{DATASET_PATH} not found. Run `python build_foggy_vlm_dataset_v3.py --check_images` "
            f"first (from this same directory) to generate it."
        )
    train_qwen_vlm_qlora_v3(max_steps=args.max_steps)
