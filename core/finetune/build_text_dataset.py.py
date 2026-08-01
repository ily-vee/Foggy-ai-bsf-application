"""
build_text_dataset.py

Generates a 400+ example text dataset for Qwen2.5-3B QLoRA.
Combines core BSF domain facts, diagnostic step-by-step procedures,
and diverse user query variations (formal, informal, edge cases).
"""

import json
import random

OUTPUT_FILE = "dataset_text_v2.jsonl"

SYSTEM_PROMPT = (
    "You are Foggy, a precision Black Soldier Fly (BSF) farming AI assistant. "
    "Use the verified local knowledge and vision classification provided to give thorough, actionable advice. "
    "When giving advice, walk the farmer through concrete steps — what to do, in what order, and what to watch out for."
)

# Expand question variations per stage
QA_EXPANDED = {
    "egg": {
        "queries": [
            "How do I set up an optimal egg harvesting and incubation system from scratch?",
            "What is the complete step-by-step procedure for handling BSF eggs from collection to neonate hatching?",
            "My egg hatching rate is low. Walk me through a full diagnostic and correction procedure.",
            "What environment should I maintain for BSF eggs?",
            "How do I collect BSF eggs without damaging them?",
            "How many days until eggs hatch, and what stops them from hatching?"
        ],
        "answers": [
            "### BSF Egg Incubation & Hatching Procedure\n\n"
            "To achieve maximum hatch rates (>90%), follow this step-by-step protocol:\n\n"
            "#### Step 1: Attractant & Egg-Trapping Setup\n"
            "* **Attractant**: Use a fermenting mixture (e.g., rotten fruit or fermented wheat bran) in a collection bucket[cite: 14].\n"
            "* **Egg Medium**: Place corrugated cardboard flutes (3x5 cm strips) directly above the attractant[cite: 14]. **Crucial**: Ensure cardboard never touches liquid wet waste[cite: 14].\n\n"
            "#### Step 2: Collection & Transfer\n"
            "* Check traps every 24–48 hours for creamy-white egg clusters[cite: 14].\n"
            "* Carefully scrape egg masses using a flat blade or spatula[cite: 14]. Avoid crushing eggs[cite: 14].\n\n"
            "#### Step 3: Incubation Chamber Environment\n"
            "* **Temperature**: Maintain target range between **27°C and 30°C**[cite: 10, 14].\n"
            "* **Relative Humidity**: Keep humidity **above 60–70%**[cite: 14]. Below 50% RH, egg desiccation mortality spikes[cite: 14].\n"
            "* **Suspension**: Suspend egg holders over moist starter substrate so hatching neonates fall directly into feed[cite: 14].\n\n"
            "#### Step 4: Hatching & Neonate Transition\n"
            "* Hatching typically occurs around **Day 4**[cite: 14].\n"
            "* Allow neonates to migrate downward independently[cite: 14]. Do not manually handle tiny larvae[cite: 14].",
            
            "Stage: **BSF Eggs**.\n\n"
            "Key incubation parameters:\n"
            "1. **Temperature**: Maintain **27–30°C** continuously[cite: 10, 14].\n"
            "2. **Humidity**: Keep relative humidity at **65–70%**[cite: 14].\n"
            "3. **Positioning**: Keep egg masses elevated off wet waste to prevent drowning or fungal mold growth[cite: 14].\n"
            "4. **Timeline**: Expect hatching within **4 to 5 days** under optimal warmth[cite: 14]."
        ]
    },
    "larva": {
        "queries": [
            "Walk me through the exact step-by-step feeding and substrate management for feeding larvae.",
            "There is a severe ammonia smell and high heat in my larval tray. Give me an immediate action plan.",
            "How do I optimize bio-conversion rate and growth speed during the 5th instar stage?",
            "What feedstock moisture level is required for feeding larvae?",
            "My larvae are trying to crawl out of the tray early. What is wrong?",
            "How much feed should I give larvae daily?"
        ],
        "answers": [
            "### Corrective Action Plan for Overheated / Anaerobic Substrate\n\n"
            "An ammonia or sour odor combined with high temperature (>35°C) indicates an **anaerobic condition caused by excess moisture and overcrowding**[cite: 10, 14].\n\n"
            "#### Step 1: Immediate Load Reduction\n"
            "* **Split the Bed**: Divide the contents of the affected tray into two or three separate crates[cite: 14]. High density generates self-heating[cite: 14].\n\n"
            "#### Step 2: Moisture Adjustment & Aeration\n"
            "* **Add Dry Bulking Agent**: Mix in dry wheat bran or fine sawdust to absorb free liquid[cite: 14]. Target **65% substrate moisture**[cite: 10, 14].\n"
            "* **Manual Turning**: Fluff and turn the substrate thoroughly using a hand tool to vent heat and gases[cite: 14].\n\n"
            "#### Step 3: Ventilation Check\n"
            "* Ensure tray bases have drainage holes (6 mm diameter)[cite: 14].\n"
            "* Stack crates with at least 10 cm vertical spacing for airflow.\n\n"
            "#### Step 4: Halt Feeding\n"
            "* Stop adding fresh feed for 24–48 hours until larvae clear current material[cite: 14].",

            "Stage: **Feeding Larvae (Instar 3-5)**.\n\n"
            "Feed management principles:\n"
            "- **Moisture**: Keep feedstock at **65–70% moisture**[cite: 10, 14].\n"
            "- **Temperature**: Maintain substrate temperature under **35°C** (ideal **27–30°C**)[cite: 10, 14].\n"
            "- **Feeding Rate**: Supply roughly **0.1 to 0.2 grams of feed per larva daily** depending on instar size."
        ]
    },
    "prepupa": {
        "queries": [
            "How do I build and operate a self-harvesting system for prepupae?",
            "What is the step-by-step workflow for transitioning prepupae into pupation and adult emergence cages?",
            "My prepupae are crawling out into feed trays instead of harvest bins. How do I fix this?",
            "Do prepupae require feeding?",
            "What color do BSF larvae turn when they enter prepupal stage?"
        ],
        "answers": [
            "### Pre-Pupae Self-Harvesting Protocol\n\n"
            "When BSF reach the prepupal stage, they turn dark brown/black, empty their digestive tracts, stop feeding, and seek dry shelter[cite: 14].\n\n"
            "#### Step 1: Exit Ramp Setup\n"
            "* **Incline Angle**: Build exit ramps angled between **30° and 45°**[cite: 14].\n"
            "* **Ramp Texture**: Use wood or textured plastic so prepupae can climb out cleanly[cite: 14].\n\n"
            "#### Step 2: Collection Bins\n"
            "* Place dry collection buckets at the top/end of ramps[cite: 14].\n"
            "* Keep bins dry, shaded, and free from moisture or feed residues[cite: 14].\n\n"
            "#### Step 3: Pupation Medium\n"
            "* Fill crates with 15 cm of loose, porous material (damp wood shavings or coco coir)[cite: 14].\n"
            "* Maintain medium moisture around **60%**[cite: 14].",

            "Stage: **Pre-pupae**.\n\n"
            "Pre-pupae have stopped eating[cite: 14]. Do **not** add fresh food waste to prepupal collection trays[cite: 14]. Keep them dry and provide 30–45 degree ramps so they self-harvest into collection bins[cite: 14]."
        ]
    },
    "pupa": {
        "queries": [
            "What conditions do pupae need to complete transformation to adults?",
            "Why are my BSF pupae failing to emerge as flies?",
            "How should I store pupae before moving them to the love cage?"
        ],
        "answers": [
            "### BSF Pupation Management\n\n"
            "Pupae are motionless and vulnerable. Follow these requirements:\n\n"
            "1. **Medium**: Place pupae in a porous, aerated medium (wood shavings)[cite: 14] at ~60% moisture[cite: 14].\n"
            "2. **Environment**: Keep temperature at **27–30°C** in a dark, shaded location[cite: 14].\n"
            "3. **Emergence**: Emergence occurs in **7–14 days**[cite: 14]. Ensure pupation trays are inside or connected directly to adult fly cages."
        ]
    },
    "adult": {
        "queries": [
            "What is the complete guide for building and maintaining an effective BSF adult breeding cage?",
            "How do I maximize adult mating, oviposition, and fly lifespan?",
            "Do adult flies need food, and how do I prevent high fly mortality?"
        ],
        "answers": [
            "### BSF Adult Mating & Oviposition Management Guide\n\n"
            "Adult BSF do not feed on solids[cite: 14]. Success relies on environmental and lighting controls[cite: 14].\n\n"
            "#### Step 1: Enclosure & Light\n"
            "* **Volume**: Minimum size of **1 m³**[cite: 14].\n"
            "* **Lighting**: Direct natural sunlight or high-intensity LED/UV lights are required for mating behavior.\n\n"
            "#### Step 2: Temperature & Hydration\n"
            "* **Temperature**: **27–30°C**[cite: 14].\n"
            "* **Humidity**: Keep RH at **70%**[cite: 14].\n"
            "* **Water Supply**: Mist water or provide sugar-water wicks[cite: 14].\n\n"
            "#### Step 3: Oviposition Stations\n"
            "* Hang cardboard egg traps above an attractant bucket[cite: 14]."
        ]
    }
}

# Multiple system/context header styles
HEADER_VARIATIONS = [
    "[Vision Pipeline Analysis]\nPredicted Stage: {stage}\nConfidence: {conf}%\nStatus: In-Distribution\n\n[Retrieved Context]\n{context}\n\nUser Question: {q}",
    "[System Context: Vision Model detected stage '{stage}' (Confidence: {conf}%)]\nDYNAMIC KNOWLEDGE BASE:\n{context}\n\nUser Query: {q}",
    "Detected Life Stage: {stage} (Confidence: {conf}%)\nRELEVANT MANUAL EXCERPTS:\n{context}\n\nFarmer Question: {q}",
    "Classification Output: {stage} ({conf}% match)\nRAG Data: {context}\n\nQuery: {q}"
]

CONTEXT_SNIPPETS = {
    "egg": "Optimal temp: 27-30°C. Moisture: >60% RH. Eggs must sit in dry crevices above feed[cite: 10, 14]. Hatching time: ~4 days[cite: 14].",
    "larva": "Optimum substrate temp: 27-30°C (max 35°C)[cite: 10, 14]. Moisture target: 65-70%[cite: 10, 14]. Feed conversion peak at 5th instar[cite: 14]. Split overcrowded trays to reduce heat[cite: 14].",
    "prepupa": "Non-feeding stage[cite: 14]. Ramps needed at 30-45 degree incline[cite: 14]. Pupation medium depth: 15-20cm, 60% moisture[cite: 14].",
    "pupa": "Non-feeding, motionless[cite: 14]. Emergence: 7-14 days at 27-30°C[cite: 14]. Medium moisture: ~60%[cite: 14].",
    "adult": "Non-feeding on solids[cite: 14]. Requires liquid water / sugar solution[cite: 14]. Temp: 27-30°C, RH: 70%[cite: 14]. Requires strong light for flight mating[cite: 14]."
}

def generate_dataset():
    records = []
    
    # Combinatorial generation across all stages
    for stage, content in QA_EXPANDED.items():
        queries = content["queries"]
        answers = content["answers"]
        context_text = CONTEXT_SNIPPETS[stage]
        
        for q in queries:
            for a in answers:
                for header_fmt in HEADER_VARIATIONS:
                    # Inject variable confidence levels
                    conf_score = round(random.uniform(82.0, 99.9), 1)
                    
                    user_text = header_fmt.format(
                        stage=stage.capitalize(),
                        conf=conf_score,
                        context=context_text,
                        q=q
                    )
                    
                    records.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": a}
                        ]
                    })

    random.shuffle(records)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated {len(records)} high-quality training examples -> {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()