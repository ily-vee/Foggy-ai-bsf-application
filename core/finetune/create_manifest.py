import json
from pathlib import Path

# Target BSF stages mapped to fixed class indices
CLASS_MAPPING = {
    "egg": 0,
    "eggs": 0,
    "early": 1,
    "early_larvae": 1,
    "feeding": 2,
    "feeding_larvae": 2,
    "active": 2,
    "pupa": 3,
    "pupae": 3,
    "adult": 4,
    "adult_bsf": 4,
    "bsf": 4
}

DATASET_DIR = Path("dataset/bsf_images_dataset")
OUTPUT_MANIFEST = Path("core/finetune/dataset_vision.json")

def generate_manifest():
    manifest = []
    
    if not DATASET_DIR.exists():
        print(f"Error: Directory '{DATASET_DIR}' not found.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Scan all subdirectories inside bsf_images_dataset
    subfolders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]
    
    for folder in subfolders:
        folder_name_lower = folder.name.lower()
        
        # Determine class index by matching folder name keywords
        assigned_idx = None
        matched_stage_key = None
        
        for key, idx in CLASS_MAPPING.items():
            if key in folder_name_lower:
                assigned_idx = idx
                matched_stage_key = key
                break
                
        if assigned_idx is None:
            print(f"Skipping folder (no stage match): {folder.name}")
            continue

        count = 0
        for img_path in folder.rglob("*"):
            if img_path.suffix.lower() in valid_extensions:
                manifest.append({
                    "image_path": str(img_path.as_posix()),
                    "label_name": folder.name,
                    "label_id": assigned_idx
                })
                count += 1
                
        print(f"Mapped folder '{folder.name}' (Class ID {assigned_idx}): {count} images found.")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nSuccessfully generated dataset manifest: {OUTPUT_MANIFEST} ({len(manifest)} total images mapped)")

if __name__ == "__main__":
    generate_manifest()