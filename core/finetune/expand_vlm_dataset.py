"""
expand_vlm_dataset.py

Regenerates a richer dataset_vlm_v2.jsonl from your existing BSF images,
using facts drawn from your actual knowledge_docs (Park 2015 UMass manual,
ECHO Technical Note 99, icipe field guide, Shumo et al. 2019 x2, Prihartantyo
et al. 2026) instead of the 8 fixed one-line templates the original file used.

WHAT THIS FIXES (see conversation for full diagnosis):
  1. Only 8 unique answer templates existed across the whole dataset -> the
     model learned "short canned label + one fact" as its default style for
     EVERY question, including ones nothing like the training examples.
  2. Zero examples existed for the adult stage -> added here.
  3. No troubleshooting-style examples existed (temperature spikes, odor,
     humidity crashes, feed problems) -> added here, paired with real images
     from the relevant stage.
  4. No length variety existed -> this script mixes short/medium/long
     answers per class instead of one fixed length for everything.

HOW IT WORKS
  - Glob your existing image folders (same structure as before).
  - For each image, pair it with 2 different (question, answer) variants
    drawn from the fact bank below -- same approach your original file used
    (reuse each image across a couple of question framings), but now with
    real variety in both content and length instead of 2 fixed templates.
  - Append troubleshooting entries, each paired with a real image from the
    relevant stage so the model still receives a photo the way it will in
    production.
  - Deduplicates and writes dataset_vlm_v2.jsonl.

BEFORE RUNNING: adjust CLASS_DIRS below if your folder naming differs, and
make sure you actually HAVE images in a 5th "adult" folder -- if you don't,
this script will just skip that class and print a warning. You'll need to
collect at least a handful of adult BSF photos for that gap to close for
real; this script can't invent images.

USAGE
    python expand_vlm_dataset.py --out dataset_vlm_v2.jsonl
"""

import argparse
import glob
import itertools
import json
import os
import random

CLASS_DIRS = {
    "eggs": "dataset/bsf_images_dataset/1_eggs",
    "early_larvae": "dataset/bsf_images_dataset/2_early_larvae",
    "feeding_larvae": "dataset/bsf_images_dataset/3_feeding_larvae",
    "pupae": "dataset/bsf_images_dataset/4_pupae",
    "adult": "dataset/bsf_images_dataset/5_adult_bsf",
}
VALID_EXT = (".jpg", ".jpeg", ".png")

# ----------------------------------------------------------------------
# FACT BANK -- paraphrased from your knowledge_docs, organized by stage.
# Mixes short / medium / long answers on purpose (this is the main fix).
# ----------------------------------------------------------------------

QA_BANK = {
    "eggs": {
        "questions": [
            "Identify the stage and state the optimal hatching conditions.",
            "What stage of the Black Soldier Fly life cycle is shown here, and how should they be maintained?",
            "Are these eggs healthy, and what happens if humidity is too low?",
            "How should I collect and handle eggs at this stage?",
        ],
        "answers": [
            # short
            "Stage: **BSF Eggs**. Keep them at 27-30°C with 60%+ relative humidity, suspended above (never directly on) a moist attractant. Hatching takes about 4 days.",
            # medium
            "Stage: **BSF Eggs**.\n\nA few things matter most here:\n- **Temperature**: hold 27-30°C for reliable hatching.\n- **Humidity**: keep it above 60% RH -- research on this species shows egg viability drops sharply below that, with mortality spiking under very dry conditions (as low as 25% RH).\n- **Placement**: eggs should sit above or beside the food source, never directly in it -- that's how the female lays them naturally, and it protects the eggs from getting waterlogged or smothered.\n\nExpect hatching in roughly 4 days if these conditions hold.",
            # long, troubleshooting-flavored
            "Stage: **BSF Eggs**.\n\nHandling steps:\n1. Check that the egg mass sits on or near a dry surface (cardboard flutes work well) positioned just above your feed/attractant -- not touching it.\n2. Maintain 27-30°C and at least 60% relative humidity around the eggs.\n3. If you're collecting eggs from an attractant trap, scrape them off gently with a flat tool once you see clusters -- avoid crushing them, they're fragile.\n4. Hatching normally takes about 4 days. If you're past day 5-6 with no hatching, suspect the humidity has been too low -- dry conditions are the most common cause of failed hatches, not temperature.\n5. Once hatched, the neonate larvae will migrate down into the feed on their own -- you don't need to move them by hand.",
        ],
    },
    "early_larvae": {
        "questions": [
            "What stage is present and what is the feeding protocol?",
            "Analyze this BSF sample and provide management advice.",
            "The larvae in this tray seem to be dying off -- what could be wrong?",
            "How much feed should I be giving larvae at this stage?",
        ],
        "answers": [
            "Stage: **Early Larvae (Neonates)**. Keep substrate moisture at 65-70% and temperature near 27-28°C. Feed a finely ground, easily digestible starter diet -- avoid dense or chunky waste, which can suffocate larvae this small.",
            "Stage: **Early Larvae (Neonates)**.\n\nThis is the highest-mortality window in the whole life cycle, so a few things matter more here than at any later stage:\n- **Moisture**: 65-70% in the substrate. Too dry and neonates desiccate quickly given how small they are (under 2mm at hatching); too wet and the substrate turns anaerobic and smells.\n- **Feed texture**: use something fine and easy to move through, like a wheat bran slurry. Chunky or dense waste can physically trap tiny larvae.\n- **Temperature**: hold around 27-28°C -- within the range where larval development proceeds efficiently without stressing them.\n\nCheck the tray daily during this stage; problems compound fast at this size.",
            "Stage: **Early Larvae (Neonates)**.\n\nIf larvae are dying off at this stage, walk through these in order:\n1. **Moisture first** -- pinch a bit of substrate; if it crumbles dry, that's almost always the cause. Target 65-70%.\n2. **Feed texture** -- overly chunky or fibrous waste can physically block or suffocate larvae this small. Switch to a finer starter feed (bran-based slurry).\n3. **Overcrowding** -- too many neonates on too little feed causes competition and stress; thin them into additional trays if density looks high.\n4. **Temperature swings** -- confirm you're not dipping below ~25°C overnight, which slows their ability to feed and recover.\n\nMost early-larvae die-offs trace back to moisture or feed texture rather than disease.",
        ],
    },
    "feeding_larvae": {
        "questions": [
            "What is the current stage of this BSF culture and its bio-conversion performance?",
            "Identify the life cycle stage and describe waste processing parameters.",
            "How do I handle high temperature spikes in crowded larval beds?",
            "What feedstock will get the best growth out of larvae at this stage?",
        ],
        "answers": [
            "Stage: **Feeding Larvae (5th Instar)**. This is peak bio-conversion -- larvae are consuming roughly their own body weight in waste daily. Keep feed moisture at 65-70% and substrate temperature below 35°C; ensure good tray ventilation.",
            "Stage: **Feeding Larvae (5th Instar)**.\n\nThis is your highest-value stage for waste processing -- larvae here can consume close to twice their body weight per day. Keep these in range:\n- **Moisture**: 65-70% in the feed/substrate.\n- **Temperature**: below 35°C. Larvae generate their own heat through activity in a dense bed, so overcrowded trays can run hotter than ambient air suggests.\n- **Feedstock quality**: diets richer in protein and fat (manure, fish waste, brewery spent grain) push crude protein and fat higher in the harvested larvae than fruit/vegetable waste alone, though the latter is lower-odor and easier to source.\n\nSpent grain in particular has been shown to produce faster-developing, heavier larvae than substrates like cow dung across a wide range of temperatures.",
            "Stage: **Feeding Larvae (5th Instar)**.\n\nFor a temperature spike in a crowded bed, work through this:\n1. **Spread the load** -- split an overcrowded tray into two; larval activity itself generates heat, and density is often the real driver, not just ambient temperature.\n2. **Improve ventilation** -- make sure trays aren't sealed; stagnant air traps metabolic heat.\n3. **Turn the substrate** -- mixing exposes trapped heat and moisture to the surface, cooling the mass.\n4. **Check for anaerobic odor** -- if the substrate smells sour/ammonia-like alongside the heat, moisture is also too high; add a dry bulking material (bran, sawdust) to fix both at once.\n5. **Target range**: keep the substrate below 35°C -- development is fastest and healthiest between 25-30°C, and larvae reared at cooler ends of that range tend to reach a larger final size, even if it takes a few more days.\n\nA sustained spike above 35°C, uncorrected, will push larvae to migrate/crawl off prematurely as if they were ready to pupate, even though they aren't mature -- that's often the first visible sign farmers notice.",
        ],
    },
    "pupae": {
        "questions": [
            "What stage is this, and is any feed required?",
            "Identify the stage shown in the image and explain harvesting requirements.",
            "When should I move pupae to the mating cage?",
            "The pupation medium seems too wet -- is that a problem?",
        ],
        "answers": [
            "Stage: **BSF Pupae**. No feed or added moisture is needed here. Keep them in a dark, dry, well-ventilated enclosure until adult flies emerge, typically 1-2 weeks later.",
            "Stage: **Pupae / Pre-pupae**.\n\nBy this point larvae have darkened, stopped feeding, and migrated to a dry area to pupate -- this is a non-feeding stage. A few practical points:\n- **Pupation medium**: something porous and loose, 15-20cm deep, works best. Too shallow and emerging adults may not reach the surface; too deep and they may not bother trying.\n- **Moisture**: moist wood shavings (60-70%) reduce desiccation-related mortality compared to a bare dry surface, but avoid soaking it -- waterlogged medium can suffocate developing pupae just as easily as bone-dry medium can dry them out.\n- **Timing**: pupation typically runs 1-2 weeks before adult emergence.\n\nIf you're harvesting for feed rather than breeding, collect at the pre-pupal (dark, still-mobile) stage instead of waiting for full pupation -- the skin gets tougher and higher in chitin the longer you wait, which reduces digestibility for some livestock.",
            "Stage: **Pupae / Pre-pupae**.\n\nOn moving pupae to the mating cage: once you see pre-pupae reliably crawling toward drier, darker corners of the tray (the self-harvesting behavior), that's your signal they're ready to transfer. Keep the destination mating enclosure at roughly 27-30°C and around 70% relative humidity, with some water source and ideally a sugar source (a dilute honey solution measurably extends adult lifespan). Adults typically emerge 1-2 weeks after pupation begins, so plan the transfer with that lead time in mind.",
        ],
    },
    "adult": {
        "questions": [
            "What stage is shown here, and what care do adults need?",
            "Is this adult fly a pest risk, and what should the colony conditions be?",
            "How do I set up breeding for adults at this stage?",
        ],
        "answers": [
            "Stage: **Adult Black Soldier Fly**. Adults don't feed on solids -- they lack functional chewing mouthparts and rely on fat reserves built up as larvae, though water or a dilute sugar solution extends their short lifespan (roughly 5-16 days).",
            "Stage: **Adult Black Soldier Fly**.\n\nA few things worth knowing about adults specifically:\n- They **don't bite, sting, or spread disease** -- they lack the mouthparts to chew and only sip liquids (water, or dilute sugar solution) via a sponge-like mouthpart.\n- Only **females visit the feed area**, and only to lay eggs above or beside it -- never directly on the waste itself.\n- Mating requires **strong light** (ideally direct sunlight) and enough cage volume for flight -- cages under roughly 1m³ struggle to produce reliable mating.\n- A mating enclosure should hold **27-30°C and ~70% relative humidity**, with water and a sugar source available; the sugar source alone can meaningfully extend adult lifespan.\n\nBecause adults are so short-lived and don't feed on solids, they pose no pest or contamination risk to stored feed the way houseflies do.",
        ],
    },
}

# ----------------------------------------------------------------------
# SHARED QUESTION POOL -- used across EVERY class, not tied to one stage.
#
# WHY THIS EXISTS: if a question phrasing only ever appears paired with one
# class during training (e.g. "Identify the stage..." only ever paired with
# eggs), the model can learn a phrase->class shortcut and ignore the actual
# image/SigLIP context entirely -- exactly what happened in production when
# a feeding-larvae photo classified at 99.96% confidence still got answered
# as "Eggs" because the question echoed the eggs-only training phrase.
#
# Pairing the SAME generic questions with images from ALL five classes
# breaks that shortcut: the wording can no longer predict the answer, so
# the model is forced to actually condition on the image and the injected
# SigLIP classification instead.
# ----------------------------------------------------------------------
GENERIC_QUESTIONS = [
    "Identify the stage shown and give appropriate care advice.",
    "What life-cycle stage is this, and what should I do next?",
    "Based on this photo, what stage is it and how should I manage it?",
    "What do you see in this image, and what action is needed?",
    "Please identify the stage and advise on next steps.",
]

# Troubleshooting entries not tied to a single simple label -- each gets
# paired with a real image from the listed stage so the model still sees
# an actual photo, matching how it will be used in production.
TROUBLESHOOTING_BANK = [
    {
        "stage": "feeding_larvae",
        "question": "There's a strong ammonia/sour smell coming from the larval bed -- what's going on?",
        "answer": (
            "That smell almost always means the substrate has gone anaerobic from excess moisture. "
            "Fixes, in order:\n"
            "1. Mix in a dry bulking material -- rice bran, sawdust, or dry rice husks absorb excess "
            "water fast.\n"
            "2. Check drainage -- trays should have small holes (roughly 6mm) so liquid doesn't pool.\n"
            "3. Reduce feed volume temporarily until the smell clears -- you're likely overfeeding "
            "relative to what the larvae can process before it rots.\n"
            "4. Turn the substrate to expose trapped pockets of wet material to air.\n\n"
            "Left unaddressed, anaerobic conditions also attract houseflies to lay eggs in the same "
            "feed, which competes with your BSF colony."
        ),
    },
    {
        "stage": "eggs",
        "question": "I haven't found any eggs on my attractant trap in several days -- what should I try?",
        "answer": (
            "A few likely causes, in rough order of how common they are:\n"
            "1. **Weak attractant** -- rotting fruit (pineapple in particular has outperformed banana "
            "or fermented bran in trials) draws wild females in more reliably than bland waste.\n"
            "2. **No egg-laying surface** -- females need dry cracks or crevices near the attractant, "
            "not on it. Stacked wood blocks or corrugated cardboard with small gaps work well.\n"
            "3. **Low wild population nearby** -- try relocating the trap closer to existing organic "
            "waste (compost piles, market waste dumps) where wild BSF are more likely to be active.\n"
            "4. **Season/temperature** -- BSF are far less active in cool weather; if it's a cold "
            "period, expect lower activity regardless of setup."
        ),
    },
    {
        "stage": "pupae",
        "question": "Some pupae look shriveled and aren't emerging as adults -- what happened?",
        "answer": (
            "Shriveled, non-emerging pupae usually point to a pupation medium that was too dry. "
            "Pupae need a moist (60-70%) but not waterlogged medium -- bare dry surfaces cause "
            "desiccation and meaningfully raise mortality before adults emerge. Check the depth too: "
            "medium shallower than about 15cm or deeper than 20cm can also disrupt emergence, either "
            "because the medium doesn't hold moisture evenly or because emerging adults struggle to "
            "reach the surface."
        ),
    },
]

def load_bank(fp, out_records, existing_keys):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset_vlm_v2.jsonl")
    parser.add_argument("--pairs-per-image", type=int, default=2,
                         help="How many different (question, answer) variants to pair with each image")
    parser.add_argument("--troubleshooting-per-class", type=int, default=3,
                         help="How many images per relevant stage to attach each troubleshooting Q&A to")
    args = parser.parse_args()

    random.seed(42)  # reproducible output
    records = []
    seen = set()

    for stage, folder in CLASS_DIRS.items():
        images = []
        for ext in VALID_EXT:
            images.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        images.sort()

        if not images:
            print(f"⚠️  No images found for '{stage}' in {folder} -- skipping. "
                  f"{'This is your missing adult-stage gap -- collect some photos!' if stage == 'adult' else ''}")
            continue

        bank = QA_BANK[stage]
        q_cycle = itertools.cycle(bank["questions"])
        a_cycle = itertools.cycle(bank["answers"])
        generic_q_cycle = itertools.cycle(GENERIC_QUESTIONS)

        print(f"📸 {stage}: {len(images)} images found")
        for img_path in images:
            img_path_norm = img_path.replace("\\", "/")

            # Class-specific phrasing(s) -- gives the model natural variety
            # in how a farmer might ask about this particular stage.
            for _ in range(args.pairs_per_image):
                question = next(q_cycle)
                answer = next(a_cycle)
                key = (img_path_norm, question, answer)
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "image": img_path_norm},
                            {"type": "text", "text": question},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": answer},
                        ]},
                    ]
                })

            # Generic phrasing from the SHARED pool -- same wording gets
            # reused across every class, so it can never be a reliable
            # shortcut to the answer on its own.
            generic_question = next(generic_q_cycle)
            generic_answer = next(a_cycle)
            key = (img_path_norm, generic_question, generic_answer)
            if key not in seen:
                seen.add(key)
                records.append({
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "image": img_path_norm},
                            {"type": "text", "text": generic_question},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": generic_answer},
                        ]},
                    ]
                })

    # Troubleshooting entries, paired with real images from the relevant stage
    for entry in TROUBLESHOOTING_BANK:
        folder = CLASS_DIRS.get(entry["stage"])
        if not folder:
            continue
        images = []
        for ext in VALID_EXT:
            images.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        images.sort()
        if not images:
            continue

        sample = images[:args.troubleshooting_per_class]
        for img_path in sample:
            img_path_norm = img_path.replace("\\", "/")
            key = (img_path_norm, entry["question"], entry["answer"])
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path_norm},
                        {"type": "text", "text": entry["question"]},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": entry["answer"]},
                    ]},
                ]
            })

    random.shuffle(records)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(records)} training examples to {args.out}")
    print("Next: review a sample of these for accuracy, then point train_qwen_qlora.py at this file.")


if __name__ == "__main__":
    main()