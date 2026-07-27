import json
import random
from pathlib import Path

MANIFEST_PATH = Path("core/finetune/dataset_vision.json")
OUTPUT_LLM_JSONL = Path("core/finetune/dataset_llm.jsonl")

# Stage-specific domain knowledge templates for Foggy Brain
STAGE_KNOWLEDGE = {
    "eggs": [
        ("What stage of the Black Soldier Fly life cycle is shown here, and how should they be maintained?",
         "This image shows BSF **Eggs**. They require high relative humidity (around 60-80%) and temperatures between 27°C and 30°C. Keep them suspended directly above a moist attractant substrate so neonates can fall safely into feed upon hatching."),
        ("Identify the stage and state the optimal hatching conditions.",
         "Identified: **BSF Egg Clutches**. Ensure egg straps remain dry and elevated above the attractant. Hatching typically occurs within 3 to 4 days under warm, humid conditions.")
    ],
    "early_larvae": [
        ("Analyze this BSF sample and provide management advice.",
         "This image depicts **Early Larvae (Neonates)**. Feed them finely ground, highly digestible feed (e.g., wheat bran slurry at 65-70% moisture). Avoid heavy or dense waste at this stage as young larvae can easily suffocate."),
        ("What stage is present and what is the feeding protocol?",
         "Stage: **Early Larvae**. Maintain substrate moisture strictly around 65-70% and keep ambient temperatures around 28°C to maximize survival rates during nursery rearing.")
    ],
    "feeding_larvae": [
        ("What is the current stage of this BSF culture and its bio-conversion performance?",
         "These are **Feeding Larvae (5th Instar)**. This is the peak bio-conversion stage where larvae consume organic waste rapidly. Maintain feed moisture at 65-70% and ensure adequate tray ventilation to prevent heat accumulation."),
        ("Identify the life cycle stage and describe waste processing parameters.",
         "Stage: **Active Feeding Larvae**. Optimal processing requires a diet rich in carbohydrates and proteins. Substrate temperature should stay below 35°C; elevated heat indicates excessive metabolic activity or overstocking.")
    ],
    "pupae": [
        ("Identify the stage shown in the image and explain harvesting requirements.",
         "This image shows **Pupae / Pre-pupae**. During this stage, larvae darken, stop feeding, and seek dry, dark areas to pupate. Harvest them immediately for protein processing or migrate them to an emergence cage for breeding."),
        ("What stage is this, and is any feed required?",
         "Identified: **BSF Pupae**. No feed or moisture is required at this stage. Keep the pupae in a dark, dry, well-ventilated enclosure until adult fly emergence.")
    ],
    "adult_bsf": [
        ("What stage of Black Soldier Fly is pictured, and what are their habitat needs?",
         "These are **Adult Black Soldier Flies**. Adult flies do not consume solid waste and rely solely on water or light sugar solutions. Provide natural sunlight or full-spectrum LED lighting to encourage mating and oviposition."),
        ("Identify this fly species/stage and state breeding conditions.",
         "Identified: **Adult Black Soldier Fly (Hermetia illucens)**. Maintain ambient temperature at 28-32°C with adequate humidity. Place clean egg oviposition structures nearby for egg laying.")
    ]
}

def generate_llm_dataset():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Complete Step 2 first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        vision_samples = json.load(f)

    llm_samples = []

    for item in vision_samples:
        img_path = item["image_path"]
        raw_label = item["label_name"].lower()

        # Match label string to key
        matched_key = None
        for key in STAGE_KNOWLEDGE:
            if key in raw_label:
                matched_key = key
                break

        if not matched_key:
            continue

        # Select Q&A pairs for multi-modal instruction tuning
        for question, answer in STAGE_KNOWLEDGE[matched_key]:
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
            }
            llm_samples.append(conversation)

    random.shuffle(llm_samples)

    OUTPUT_LLM_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LLM_JSONL, "w", encoding="utf-8") as f:
        for sample in llm_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Successfully generated multi-modal instruction dataset:")
    print(f"  - Target path: {OUTPUT_LLM_JSONL}")
    print(f"  - Total conversational samples: {len(llm_samples)}")

if __name__ == "__main__":
    generate_llm_dataset()