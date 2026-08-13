"""
test_gguf_server.py

Tests the quantized GGUF model through llama-server's OpenAI-compatible API,
using the SAME system prompt and structured [Vision Analysis]/[Reference
Context] format the model was actually fine-tuned on. This is a fair test,
unlike the bare-question llama-mtmd-cli call — if quality is still off
with this, that's real signal; if it recovers, the earlier result was
purely a testing-conditions artifact.

Prerequisite: llama-server running with --mmproj and --image-min-tokens 1024
(see command in the conversation).

Usage:
  python3 test_gguf_server.py path/to/photo.jpg
"""

import sys
import base64
import requests

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Same system prompt used in training (build_foggy_vlm_dataset.py /
# inference_pipeline_qwen_vlm.py) — copy exactly, don't paraphrase, since
# the model was fine-tuned conditioned on this specific text.
SYSTEM_PROMPT = (
    "You are Foggy, a precision Black Soldier Fly (BSF) farming AI assistant. "
    "Use the verified local knowledge and vision classification provided to give thorough, actionable advice. "
    "Base every specific number only on the reference context given. If the reference context doesn't cover "
    "something, say so plainly instead of inventing a figure. Never include citations, DOIs, or footnote "
    "markers — you have no access to external sources beyond the reference context provided. When an image is "
    "included, describe only what you can actually observe."
)

# Example structured blocks, matching what the real engine constructs —
# swap the stage/confidence/context for whatever the real SigLIP2
# classifier + retriever would actually produce for this specific image
# in production.
STRUCTURED_PROMPT_TEMPLATE = """[Vision Analysis]
Detected Stage: Prepupa (80.8% confidence)

[Reference Context]
Non-feeding stage. Ramps needed at 30-45 degree incline. Pupation medium depth: 15-20cm, 60% moisture. Wandering prepupal duration: 7-10 days.

User Question: {question}"""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test(image_path: str, question: str = "Identify the life stage shown and give care advice."):
    image_b64 = encode_image(image_path)
    prompt_text = STRUCTURED_PROMPT_TEMPLATE.format(question=question)

    payload = {
        "model": "qwen_bsf",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    print("Sending request to llama-server...")
    resp = requests.post(SERVER_URL, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    print("=" * 60)
    print(result["choices"][0]["message"]["content"])
    print("=" * 60)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_gguf_server.py path/to/photo.jpg")
        sys.exit(1)
    test(sys.argv[1])