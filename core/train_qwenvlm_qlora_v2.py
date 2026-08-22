"""
train_qwenvlm_qlora_v2.py

Foggy — BSF Farming AI Assistant
Qwen2.5-VL-3B-Instruct QLoRA fine-tuning, v2: LANGUAGE DECODER + VISION TOWER.

WHY V2 (vs train_qwenvlm_qlora (1).py)
-------------------------------------------
v1's target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
"down_proj"] only ever targeted the LANGUAGE decoder. This is a genuinely separate
job from SigLIP2 (models/siglip2_bsf_lora): SigLIP2 already owns stage
classification (egg/larva/prepupa/pupa/adult) — see inference_pipeline_qwen_vlm.py's
[Vision Analysis] block and its explicit instruction to the VLM to treat SigLIP2's
stage call as authoritative rather than re-deriving it. What SigLIP2 does NOT do is
answer open-ended follow-up questions about a photo's *condition* — "do they look
healthy," "are they overpopulated in the tray." That's squarely the VLM's job, and
it depends on Qwen2.5-VL's own vision tower (a separate ViT, unrelated to SigLIP2)
extracting features nuanced enough to support those judgments. A frozen,
off-the-shelf vision tower was never fine-tuned to notice BSF-specific visual cues
(larval density, moisture sheen, mold texture, discoloration); this script extends
QLoRA into that vision tower so it can learn to.

A NAMING COLLISION THIS FILE HANDLES DELIBERATELY: introspecting the actual
installed transformers' Qwen2_5_VLForConditionalGeneration (not guessed from
memory — see discover_vision_target_modules() below) shows the vision tower's MLP
reuses the exact same leaf names as the language decoder's MLP (gate_proj/up_proj/
down_proj), while its attention module is named "attn" (language: "self_attn") and
its output projection is bare "proj" (language: "o_proj"). Three consequences:
  1. v1's LANGUAGE_TARGET_MODULES list, unbeknownst to anyone, was ALREADY applying
     LoRA to the vision tower's MLP as a side effect of PEFT's suffix-based module
     matching — just not its attention or patch-merger layers.
  2. A naive addition of bare "proj" as a vision target would ALSO match the
     language decoder's q/k/v/o_proj (every one of those literally ends in "proj"),
     silently breaking the vision/language rank split below.
  3. Because gate_proj/up_proj/down_proj are shared names, target_modules alone
     can't give vision a different LoRA rank than language — that's what
     rank_pattern/alpha_pattern (regex over the full module path, not the leaf
     name) are for.
discover_vision_target_modules() resolves this empirically rather than by asserting
a specific transformers-version-dependent naming convention: it walks the model,
and for each vision Linear layer, grows the path suffix used as a target string
just until no LANGUAGE-decoder Linear also ends with that suffix. That makes this
script self-correcting if a future transformers release renames "attn" to
"self_attn" in the vision tower, or vice versa — it doesn't have to be told.

MIXED PRECISION BY DESIGN: BitsAndBytesConfig.llm_int8_skip_modules keeps the
vision tower OUT of 4-bit quantization (it's small relative to the 3B language
backbone, so the memory cost of keeping it in bf16 is minor) while the language
decoder is still real QLoRA (4-bit NF4 base + LoRA delta). This matters because
vision towers are reported to be more sensitive to aggressive quantization than
language decoders are, and the whole point of this run is visual judgment
quality. verify_vision_stayed_unquantized() checks this actually happened rather
than trusting it silently.

A REAL BUG THIS FILE HAD, FOUND VIA AN ACTUAL CRASH: an earlier version passed
llm_int8_skip_modules=["visual"] and crashed inside from_pretrained with
`AttributeError: 'weight' is not an nn.Module` during _initialize_missing_keys.
Root cause, confirmed by reading transformers' actual quantizer source and
reproducing the failure offline (see discover_quantization_skip_modules()):
transformers' should_convert_module() matches skip patterns via re.match (anchored
at the START of the module path) or str.endswith -- never "contains anywhere."
The vision tower's real path is "model.visual...", not "visual...", so a bare
"visual" pattern matched nothing and the vision tower got quantized regardless.
Worse, passing llm_int8_skip_modules AT ALL (even a working value) makes
transformers skip its own automatic default exclusions -- confirmed to be
['lm_head'] for this model via get_keys_to_not_convert() -- so this also silently
gave up the lm_head protection v1 gets for free just by never touching this
parameter, which is what actually produced the "weights are not tied" warning
and subsequent crash (a quantized lm_head fighting Qwen's weight-tying).
discover_quantization_skip_modules() fixes both: it builds a weight-free
meta-device skeleton of the real model (from config only, no multi-GB download)
to call transformers' own get_keys_to_not_convert() for the correct default list,
then finds the vision tower's actual top-level path by name rather than assuming
a nesting depth, and returns the union -- so llm_int8_skip_modules is always
correct regardless of how a given transformers version happens to nest things.

TWO LEARNING RATES: language and vision LoRA parameters get separate optimizer
param groups (VISION_LR well below LANGUAGE_LR). The vision tower's pretrained
representations are doing useful general-purpose work already; on a dataset this
size (180 examples, ~127 of which touch an image at all) large vision-tower
updates risk overwriting that with noise faster than they'd learn anything BSF-
specific. This needed a hand-built optimizer instead of TrainingArguments' single
`optim=`/`learning_rate=` — see build_optimizer().

DATASET: dataset_foggy_vlm_v2.jsonl (180 examples: 127 image-grounded, 53 text-
only, across all 5 life stages + 9 non-photo topics). No filtering to image-only
examples is done or needed: a text-only example's forward pass never touches
`model.visual`, so it naturally contributes zero gradient to the vision LoRA
weights while still teaching the language decoder tone/structure/grounding — the
split happens on its own from the data, not from code here.

WHAT THIS SCRIPT DOES NOT TOUCH: SigLIP2 (models/siglip2_bsf_lora) is a completely
separate model and training pipeline; nothing here loads or affects it.

VERIFICATION CAVEAT: this file was written and reasoned through in a CPU-only dev
environment without peft/bitsandbytes/trl/qwen_vl_utils installed. Every structural
claim about Qwen2.5-VL's module tree was verified by instantiating the real
`Qwen2_5_VLForConditionalGeneration` class (tiny random config, no weights
downloaded) against the transformers version actually installed there, and every
TrainingArguments/from_pretrained kwarg used below was checked against that same
installed version's real signature. The parts that could NOT be verified offline —
whether bitsandbytes' `PagedAdamW8bit` import path and llm_int8_skip_modules'
matching behavior hold on your GPU machine's package versions — are guarded with
runtime assertions and a fallback (see build_optimizer and the verify_* functions)
so a mismatch fails loudly near the top of a run instead of silently three hours
in. Run with a tiny --max_steps first (see bottom of file) before a full run.
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
logger = logging.getLogger("train_qwenvlm_qlora_v2")

# MUST match inference_pipeline_qwen_vlm.py's QWEN_BASE exactly.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH = Path("dataset_foggy_vlm_v2.jsonl")
# Separate output dir from v1's models/qwen_vlm_bsf_qlora -- keeps the proven
# language-only adapter around as a fallback until this one is validated.
# Update inference_pipeline_qwen_vlm.py's QWEN_LORA_DIR (or wherever the GGUF
# conversion step reads the adapter from) once you're ready to switch over.
OUTPUT_DIR = Path("models/qwen_vlm_bsf_qlora_v2")

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
EVAL_FRACTION = 0.10  # ~18 of 180 examples held out for eval-loss monitoring.
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
    """Unchanged from v1 -- this class doesn't need to know or care which model
    parameters are trainable, only how to build (input_ids, labels) from raw
    conversations. It already handles the dataset's multi-turn grounding-via-
    history examples correctly: prompt_only = conv[:-1] masks EVERYTHING except
    the final assistant turn regardless of how many turns precede it."""

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


def train_qwen_vlm_qlora_v2(max_steps: int = -1):
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
        # docstring's "A REAL BUG THIS FILE HAD" section.
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
        # 5 epochs over 180 examples (~6x v1's dataset) is a starting point, not a
        # fixed answer -- load_best_model_at_end + EarlyStoppingCallback below
        # exist specifically so this doesn't have to be tuned by hand in advance.
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
            f"{DATASET_PATH} not found. Run `python build_foggy_vlm_dataset_v2.py` "
            f"first (from this same directory) to generate it."
        )
    train_qwen_vlm_qlora_v2(max_steps=args.max_steps)
