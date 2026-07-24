# automated downloader designed to grab the essential files of an AI model—specifically Google's Gemma 3
# (4B Instruct)—and save them directly to a local computer.

import os
from huggingface_hub import snapshot_download

print("🚀 Foggy Core Weight Ingestion Pipeline — Active")
print("Target: google/gemma-3-4b-it (Pristine 16-bit Parameter Arrays)")

# Target directory matches the folder you just created in Git Bash
target_directory = r"C:\Users\VIAMAISH\Documents\Foggy AI Project\gemma-3-raw-weights"

print(f"\nStreaming data segments into: {target_directory}")
print("This may take some time depending on your network connection...\n")

# Establishes connection to the Hugging Face repository to pull raw tensor blocks
snapshot_download(
    repo_id="google/gemma-3-4b-it",
    local_dir=target_directory,
    allow_patterns=["*.json", "*.safetensors", "*.model"]
)

print("\n✅ Weight Ingestion Complete! All raw parameter blocks safely secured.")