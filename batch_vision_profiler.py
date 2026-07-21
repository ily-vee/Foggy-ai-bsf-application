import os
import base64
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image

# Configuration setup matching your workspace
DATASET_DIR = "bsf_images_dataset"
OUTPUT_LOG = "bsf_vision_profile_results.txt"
SERVER_URL = "http://localhost:8080/v1/chat/completions"
MAX_IMAGE_SIZE = (800, 800)  # Safe processing dimension for your hardware link


def encode_and_resize_image(image_path):
    """Opens an image file, automatically resizes it in memory, and returns base64 text."""
    with Image.open(image_path) as img:
        # Convert to RGB color mode if image is in RGBA or palette mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Downscale image using high-quality resampling if it exceeds target dimensions
        img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Save output bytes directly to an in-memory stream buffer
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_bsf_stage(image_path, expected_stage):
    """Packages the optimized image payload and pushes it to the active llama-server process."""
    try:
        base64_image = encode_and_resize_image(image_path)
    except Exception as e:
        return f"Image processing error: {str(e)}"

    prompt = (
        f"You are a computer vision tool optimizing a Black Soldier Fly (BSF) farming automation engine. "
        f"This image is categorized from the folder structure as: '{expected_stage}'. "
        f"Analyze this image carefully. Note down the structural layout, count densities if applicable, "
        f"identify potential contamination or structural anomalies, and verify if the contents align visually "
        f"with the '{expected_stage}' growth stage."
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    try:
        # Increased timeout to accommodate CPU array matrix generation cycles
        response = requests.post(SERVER_URL, json=payload, timeout=180)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error [{response.status_code}]: {response.text}"
    except Exception as e:
        return f"Pipeline connection failure: {str(e)}"


def run_pipeline():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Directory '{DATASET_DIR}' not found. Please double-check folder positioning.")
        return

    print(f"🚀 Starting Automated Batch Vision Profiling Pipeline on dataset folder: {DATASET_DIR}")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== BSF MULTIMODAL PROFILING REPORT - {datetime.now()} ===\n\n")

        # Traverse the BSF category matrix folder structure
        for root, dirs, files in os.walk(DATASET_DIR):
            stage_name = os.path.basename(root)

            if stage_name == DATASET_DIR:
                continue

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_image_path = os.path.join(root, file)
                    print(f"\n🔬 Processing: [{stage_name}] -> {file}...")

                    analysis_output = analyze_bsf_stage(full_image_path, stage_name)

                    log_entry = (
                        f"--------------------------------------------------\n"
                        f"IMAGE PATH: {full_image_path}\n"
                        f"EXPECTED LABELED STAGE: {stage_name}\n"
                        f"ANALYSIS TIME: {datetime.now()}\n"
                        f"--------------------------------------------------\n"
                        f"{analysis_output}\n\n"
                    )
                    log_file.write(log_entry)
                    log_file.flush()  # Sync buffer data to disk immediately

                    print(f"✅ Processed {file}. Analysis logged.")

    print(f"\n🎯 Pipeline execution complete! Detailed results saved to: '{OUTPUT_LOG}'")


if __name__ == "__main__":
    run_pipeline()
