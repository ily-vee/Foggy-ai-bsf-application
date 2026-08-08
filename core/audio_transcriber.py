"""
audio_transcriber.py

Wraps faster-whisper for turning a WhatsApp voice note into text. Load the
model ONCE at process startup (it's a few hundred MB-1GB+ depending on size)
and reuse it — never construct WhisperModel per-request.
"""

import os
# Must be set BEFORE torch/onnxruntime/ctranslate2 are imported anywhere in
# the process. On Windows, PyTorch and faster-whisper's dependencies
# (onnxruntime/ctranslate2) each bundle their own Intel OpenMP runtime
# (libiomp5md.dll); loading both crashes with "OMP: Error #15" otherwise.
# This is the standard workaround for inference-only workloads like this —
# it's officially "unsupported" for heavy multi-threaded numeric work where
# two conflicting thread pools could race, but for straightforward
# transcription + generation calls like this pipeline runs, it's safe in
# practice and widely used for exactly this faster-whisper/PyTorch conflict.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from faster_whisper import WhisperModel

logger = logging.getLogger("FoggyAudio")

# "medium" gives meaningfully better accuracy than "small" (domain terms,
# accents, background noise), at real added VRAM/latency cost on a GPU
# already hosting SigLIP2 + Qwen2.5-3B. Override via env var to tune this
# after measuring actual headroom, without a code change + redeploy:
#   FOGGY_WHISPER_MODEL=small      (lower VRAM/latency, worse accuracy)
#   FOGGY_WHISPER_MODEL=medium     (default)
#   FOGGY_WHISPER_MODEL=large-v3   (best accuracy, significant VRAM cost)
WHISPER_MODEL_SIZE = os.environ.get("FOGGY_WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.environ.get("FOGGY_WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("FOGGY_WHISPER_COMPUTE_TYPE", "float16")

MIN_AVG_LOGPROB = -1.0   # below this, treat transcription as low-confidence
MIN_TRANSCRIPT_CHARS = 3  # near-empty output = likely silence/noise

_whisper_model = None


def load_whisper_model(device: str = None, compute_type: str = None):
    global _whisper_model
    device = device or WHISPER_DEVICE
    compute_type = compute_type or WHISPER_COMPUTE_TYPE
    if _whisper_model is None:
        logger.info(f"Loading faster-whisper ({WHISPER_MODEL_SIZE}) on {device} ({compute_type})...")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    return _whisper_model


# Domain vocabulary hint — generic Whisper models have never seen terms
# like "prepupa" or "instar" in typical training data and will often
# mis-transcribe them as the nearest common word. initial_prompt biases
# decoding toward this vocabulary without requiring any fine-tuning.
DOMAIN_VOCAB_PROMPT = (
    "Black Soldier Fly, BSF, larvae, larva, prepupa, prepupae, pupa, pupae, "
    "instar, oviposition, substrate, moisture, feedstock."
)


def transcribe_audio(audio_path: str, translate_to_english: bool = False) -> dict:
    """Returns:
        {
          "text": str,               # transcript, "" if nothing usable
          "language": str,           # detected language code
          "language_probability": float,
          "low_confidence": bool,    # True if avg_logprob suggests unreliable transcription
          "empty": bool,             # True if transcript is effectively blank
        }
    Never raises for "bad audio" cases — callers should check "empty" /
    "low_confidence" rather than relying on exceptions to catch garbage input.
    """
    model = load_whisper_model()
    task = "translate" if translate_to_english else "transcribe"

    segments, info = model.transcribe(audio_path, beam_size=5, task=task, initial_prompt=DOMAIN_VOCAB_PROMPT)

    texts = []
    logprobs = []
    for seg in segments:
        texts.append(seg.text.strip())
        logprobs.append(seg.avg_logprob)

    full_text = " ".join(t for t in texts if t).strip()
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -999.0

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "low_confidence": avg_logprob < MIN_AVG_LOGPROB,
        "empty": len(full_text) < MIN_TRANSCRIPT_CHARS,
    }