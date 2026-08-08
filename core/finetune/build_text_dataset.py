"""
build_foggy_dataset_v1.py

Foggy — BSF Farming AI Assistant
QLoRA fine-tuning dataset generator (Qwen chat format), v1.

SOURCE MATERIAL
----------------
Every fact/number used below traces back to one of these six verified
documents (all peer-reviewed papers or extension-service manuals):

  [PARK]   Park, H. 2015. "The Black Soldier Fly Larvae Manual." UMass
           Amherst / Sustainable UMass Student Showcase.
  [ECHO]   Chalermliamthong, Trail, Walle, Motis. 2023. "Black Soldier
           Fly Larvae Production." ECHO Technical Note No. 99.
  [ICIPE]  Tanga et al. 2017. "Black Soldier Fly Manual Guide" — icipe /
           NWO field guide for commercial BSF production.
  [JPCS]   Prihartantyo et al. 2026. "Effect of Feed Combination
           Variation on BSF Larvae Weight and Extraction Method on BSF
           Larvae Oil Yield in Biodiesel Production." J. Phys. Conf. Ser.
  [SHUMO1] Shumo et al. 2019. "Influence of Temperature on Selected
           Life-History Traits of Black Soldier Fly Reared on Two Common
           Urban Organic Waste Streams in Kenya." Animals 9(3):79.
  [SHUMO2] Shumo et al. 2019. "The Nutritive Value of Black Soldier Fly
           Larvae Reared on Common Organic Waste Streams in Kenya."
           Scientific Reports 9:10110.

A document tagged as "unreliable content-farm text" (unattributed
testimonials, round unverifiable numbers, off-topic affiliate links) was
excluded entirely — no fact in this script derives from it.

DESIGN PRINCIPLES (carried over / extended from dataset_text_v3)
------------------------------------------------------------------
1. No "[cite: ...]" or DOI-shaped tokens are ever baked into a target
   answer. Numbers are stated as facts, not as footnoted citations.
2. Index-aligned query/answer pairs only — no cross-product reuse.
3. Three answer LENGTHS are represented on purpose:
     - LONG:   multi-step numbered procedure, ends by asking the farmer
               a concrete next-action / diagnostic question.
     - MEDIUM: a tight paragraph (3-5 sentences), no headers.
     - SHORT:  1-2 sentences, direct answer only.
   All three lengths appear for a mix of vision-header, text-only, and
   real-time/troubleshooting framings.
4. TEXT-ONLY examples (no vision/stage/confidence header) are included
   so the model does not force that framing onto plain chat.
5. GROUNDING-PRIORITY examples: reference context intentionally states a
   figure that conflicts with commonly "known" BSF trivia (e.g. total
   life cycle length, egg count) and the target answer sides with the
   given reference, not parametric memory.
6. INSUFFICIENT-CONTEXT examples: the farmer asks for a number the
   supplied reference snippet does not contain; the target answer says
   so plainly and offers the closest thing it *does* know, rather than
   inventing a figure.
7. REAL-TIME / TROUBLESHOOTING examples simulate a farmer messaging
   Foggy about something happening right now in their unit (bad smell,
   larvae crawling out, low hatch rate, etc.), paired with procedural
   answers that end in a next-action question so the assistant can keep
   diagnosing across turns.
"""

import json
import random

OUTPUT_FILE = "dataset_foggy_v1.jsonl"

SYSTEM_PROMPT = (
    "You are Foggy, a precision Black Soldier Fly (BSF) farming AI assistant. "
    "Use the verified local knowledge and vision classification provided to give thorough, actionable advice "
    "to farmers. Base every specific number only on the reference context given. If the reference context "
    "doesn't cover something, say so plainly instead of inventing a figure. When useful, end your response by "
    "asking the farmer a concrete next-action or diagnostic question. Never include citations, DOIs, or "
    "footnote markers — you have no access to external sources beyond the reference context provided."
)

# ---------------------------------------------------------------------------
# Reference context snippets, keyed by topic. These are what gets injected
# into the "[Reference Context]" block of each training example.
# ---------------------------------------------------------------------------
CONTEXT_SNIPPETS = {
    "egg": (
        "Adult females lay clusters of roughly 500-1200 eggs (individual clutches sometimes reported as low "
        "as 206) in dry cracks and crevices 2-5mm from moist decaying organic matter, never directly on it. "
        "Ideal egg-laying/incubation conditions: 27C, relative humidity 60% or higher. Eggs hatch in about "
        "4 days under these conditions. Corrugated cardboard flutes (3 flutes/cm) or grooved rubber tubes "
        "placed above an attractant are standard oviposition-collection surfaces."
    ),
    "larva": (
        "Newly hatched larvae are about 1.8mm long, dull white to cream colored. Optimal feed moisture: "
        "60-90% (many operations target 65-70%). Optimal processing temperature: 27-33C; larvae generate "
        "some of their own heat through feeding activity. Larvae pass through 5 instars and take "
        "approximately 13-18 days to reach the prepupal stage under favorable conditions (this can extend "
        "up to several weeks, or in hostile conditions much longer, since BSF larvae can delay maturity). "
        "Mature larvae reach roughly 25-27mm in length. Larvae are photophobic and bury themselves away "
        "from light. BSFL require 4.5 to 10 kg of organic waste to produce 1 kg of larval biomass, versus "
        "roughly 10 kg of feed per 1 kg of beef in cattle."
    ),
    "prepupa": (
        "The wandering (non-feeding) prepupal stage lasts about 7-10 days at 27C. Prepupae stop feeding, "
        "empty their gut, darken from white toward brown/black, and migrate away from the food source toward "
        "a dry, sheltered place to pupate. This self-harvesting behavior is exploited with 40-45 degree exit "
        "ramps leading to a dry collection bucket. Ideal pupation-medium depth is 15-20cm at around 60% "
        "moisture; too deep and emerging adults cannot reach the surface, too shallow and prepupae keep "
        "wandering and waste fat reserves."
    ),
    "pupa": (
        "Pupae develop inside a darkened, hardened exoskeleton in a porous, loose, well-aerated medium "
        "(e.g. moist wood shavings at 60-70% moisture) kept dry rather than wet to avoid desiccation-driven "
        "mortality. Pupation commonly takes about 7-14 days (some sources report up to two weeks) at 27C. "
        "Adults emerge from the pupal case once pupation completes."
    ),
    "adult": (
        "Adult BSF lack functional chewing mouthparts and do not feed on solids; they rely on fat reserves "
        "built up as larvae and take up only water or a sugar solution, which can extend adult lifespan from "
        "roughly 5-8 days minimum up to 16-40+ days. Mating occurs 2 days after emergence, requires strong "
        "light (natural sunlight or a comparable artificial source) and adequate flight volume in the cage, "
        "and typically happens on or near the ground with males intercepting females in flight. Egg-laying "
        "follows about 2 days after mating. Adults do not bite, sting, or transmit disease."
    ),
    "life_cycle_overall": (
        "Total egg-to-adult life cycle is commonly reported as roughly 38-45 days under favorable rearing "
        "conditions (egg ~4 days, larva ~13-18 days feeding, prepupa ~7-10 days, pupa ~7-14 days), though "
        "under hostile or resource-poor conditions larvae can delay maturity for months."
    ),
    "environment": (
        "BSF development is most efficient around 27C; the upper tolerable range for development is roughly "
        "30-36C, above which development is inhibited. Egg viability peaks near 30C and drops sharply below "
        "15C or above 40C. Relative humidity tolerance for larvae spans roughly 30-90%, with 50-70% considered "
        "optimal; overly wet substrate goes anaerobic and smells foul, overly dry substrate causes desiccation "
        "mortality. Strong light (ideally direct sunlight) is required to trigger adult mating; low-light "
        "indoor setups need supplemental high-intensity lighting in the 450-700nm range."
    ),
    "feedstock": (
        "BSF larvae will eat most municipal, market, kitchen, and manure-based organic waste but cannot "
        "efficiently digest high-lignin plant material (cow manure ~10% lignin, wheat straw ~23% lignin) or "
        "diets that are entirely liver/fish rendering (mortality reported as high as 98-99.8% on those diets "
        "alone). Feeding rate benchmarks: larvae can consume roughly 15 kg/m^2/day of feeding surface at scale; "
        "about 20% of organic food matter by weight converts into larval biomass. Mixing waste streams (e.g. "
        "chicken manure with plant waste) is used to balance protein content. Feedstock high in fat "
        "(fish waste, poultry/pig/cattle manure) yields larvae with 40%+ protein and 20%+ fat; fruit and "
        "vegetable waste alone yields close to 40% protein but lower fat (<10%)."
    ),
    "nutrition_composition": (
        "Dried BSF larvae typically contain 37-63% crude protein and 7-35% fat depending on the rearing "
        "substrate, plus vitamins and minerals; a commonly cited composition is around 42% crude protein, "
        "~35% ether extract (fat), ~14.6% ash, ~7.9% moisture, ~7% crude fiber. Larvae reared on brewers' "
        "spent grain (SG) generally show higher crude protein and fat than those reared on cow dung (CD) or "
        "chicken manure (CM); kitchen waste (KW) substrates in one controlled study gave the highest crude "
        "protein among three tested substrates. BSFL amino acid profile is broadly comparable to fishmeal, "
        "including methionine and lysine levels that can meet or exceed FAO reference values for poultry feed."
    ),
    "disease_food_safety": (
        "The mid-gut of BSF larvae is highly acidic (pH <= 3) and shows activity against pathogens such as "
        "Salmonella and Staphylococcus aureus, but pathogen survival in the gut has still been documented, so "
        "post-harvest treatment matters when larvae will be used for sensitive feed or human consumption. "
        "Effective pathogen-reduction treatments include: boiling larvae 5 minutes; toasting over open flame "
        "at 150C for 5 minutes with regular turning; oven-drying at 60C until weight loss stops (2-3 days); "
        "or oven-drying 22 minutes at 150C. No aflatoxin traces were detected in BSFL reared on chicken "
        "manure, kitchen waste, or spent grain in controlled testing, though low levels of heavy metals "
        "(cadmium and lead) can accumulate in larvae depending on substrate contamination."
    ),
    "harvesting": (
        "Larvae/prepupae self-harvest by crawling up 40-45 degree exit ramps toward a dry collection point "
        "once they reach the wandering prepupal stage, which is the simplest and most passive harvest method. "
        "For larvae still in the feeding stage, wet harvesting (washing through progressively finer screens) "
        "or dry harvesting (shaking/sieving) are used to separate larvae from residual feed. Sorting/sizing is "
        "labor-intensive at scale; transferring larvae to a finer-textured feed (e.g. bran) near the end of "
        "the cycle makes separation easier."
    ),
    "economics_value": (
        "Feed costs commonly make up roughly 60-70% of total livestock production cost in a smallholder "
        "operation, which is the main economic driver for BSF adoption as an alternative protein source to "
        "fishmeal and soybean meal. Larval frass (residue after bioconversion) is itself a saleable organic "
        "fertilizer, with an N-P-K value cited around 5%-3%-2% and organic matter content above 80%; frass is "
        "often diluted before field application because of its nutrient concentration."
    ),
    "oil_biodiesel": (
        "In a controlled feeding trial, BSF larvae fed a 1:1 combination of high-fat feeds (skipjack tuna, "
        "avocado, tofu) under 70-80% feedstock moisture and a 2-day pre-fermentation period produced the "
        "highest larval weight gain with the avocado:tofu combination (21.96% w/w increase). Oil extraction "
        "from harvested larvae via Soxhlet extraction (n-hexane, ~69C) yielded 44.11% w/w oil versus 17.64% "
        "w/w via room-temperature maceration, and Soxhlet extraction also gave better selectivity for lauric, "
        "myristic, and oleic acids, which are useful for biodiesel. The resulting biodiesel met the SNI "
        "7128:2015 standard (density 868 kg/m^3, kinematic viscosity 4.26 cSt, acid number 0.32 mg-KOH/g)."
    ),
}

# ---------------------------------------------------------------------------
# Index-aligned query/answer sets. Each topic has a mix of LONG / MEDIUM /
# SHORT answers, each tagged with its length for reporting purposes.
# ---------------------------------------------------------------------------
QA_SETS = {
    "egg": [
        {
            "length": "long",
            "query": "Walk me through the full process for collecting BSF eggs from an attractant trap and getting them safely into the starter feed.",
            "answer": (
                "Here's the full workflow:\n\n"
                "1. Set up the attractant. Use rotting fruit, fermented bran, or manure in an open container placed outdoors "
                "where BSF are active.\n"
                "2. Provide a laying surface above it. Stack corrugated cardboard flutes, wooden blocks with small gaps, or "
                "grooved rubber tubes directly over the attractant, 2-5mm above the moist material, not touching it.\n"
                "3. Check daily. Female BSF lay clusters of roughly 500-1200 eggs in the cracks. Multiple females often lay "
                "in the same flute.\n"
                "4. Collect gently. Remove the cardboard or blocks, separate them, and scrape the creamy-white egg clusters "
                "off with a toothpick or flat blade rather than scraping hard against the surface.\n"
                "5. Transfer to incubation. Place eggs on or just above a moist starter substrate (using a mesh screen to "
                "keep them from direct contact with wet feed), and hold at 27C and at least 60% relative humidity.\n"
                "6. Wait for hatch. Eggs hatch in about 4 days under these conditions, and neonate larvae will migrate down "
                "into the starter feed on their own.\n\n"
                "What's your current setup for the attractant — outdoor open trap, or an enclosed mating cage with an internal food source?"
            ),
        },
        {
            "length": "medium",
            "query": "What conditions do BSF eggs actually need to hatch well?",
            "answer": (
                "Eggs need to sit in dry cracks or crevices close to — but not touching — moist decaying organic matter, "
                "since females always lay near a food source rather than on it. For best hatch rates, hold the egg-laying "
                "and incubation area at around 27C with relative humidity of 60% or higher. Under those conditions eggs "
                "hatch in about 4 days. Below-optimal humidity is the most common cause of poor hatch rates, since eggs "
                "are small and vulnerable to desiccation."
            ),
        },
        {
            "length": "short",
            "query": "How long do BSF eggs take to hatch?",
            "answer": "About 4 days, provided the eggs are held near 27C with relative humidity of 60% or above.",
        },
        {
            "length": "medium",
            "query": "My egg traps have been empty for over a week even though I see adult flies nearby. What's going on?",
            "answer": (
                "A few likely causes: the attractant may not be strong-smelling enough to draw egg-laying females in — "
                "rotting fruit or fermented bran close to a wooden-block or cardboard-flute laying surface tends to work "
                "well. Check that the laying surface is positioned just above the attractant rather than touching or too "
                "far from it, since females lay near the food source but never directly on it. Also confirm your site gets "
                "enough warmth and light; BSF activity, including oviposition, drops off in cool or shaded conditions. "
                "Have you checked whether the flies you're seeing near the trap are actually landing and laying, or just passing through?"
            ),
        },
    ],
    "larva": [
        {
            "length": "long",
            "query": "I've got a strong ammonia smell and the substrate feels hot to the touch in one of my larval trays. Give me an action plan.",
            "answer": (
                "That combination points to overcrowding and excess moisture pushing the substrate anaerobic. Here's what to do:\n\n"
                "1. Stop adding fresh feed to that tray immediately — let the larvae work through what's already there for "
                "24-48 hours.\n"
                "2. Reduce density. Split the tray's contents across two or three containers so larvae aren't packed too "
                "tightly.\n"
                "3. Correct moisture. Mix in a dry bulking material (bran, sawdust, or dry rice husk) to bring the substrate "
                "back toward the 60-70% moisture range rather than the waterlogged state that's driving the smell.\n"
                "4. Turn the substrate. Mixing it thoroughly vents trapped heat and gas and reintroduces oxygen, since "
                "anaerobic conditions are what produce the ammonia odor.\n"
                "5. Check drainage. If the container doesn't already have drainage holes to remove excess liquid, add them "
                "so this doesn't recur.\n\n"
                "Roughly how deep is the feed layer in that tray right now, and does the container have any drainage at all?"
            ),
        },
        {
            "length": "medium",
            "query": "What moisture level should I be targeting for the larval feedstock?",
            "answer": (
                "Aim for roughly 60-70% moisture in the feedstock, which sits within the broader 60-90% range larvae can "
                "tolerate. Processing temperature should stay in the 27-33C range for efficient feeding. Too much moisture "
                "pushes the substrate anaerobic and causes foul odors and reduced growth; too little moisture stalls the "
                "larvae's ability to process the feed efficiently."
            ),
        },
        {
            "length": "short",
            "query": "How long does the larval feeding stage last?",
            "answer": "Roughly 13-18 days under favorable temperature and feed conditions, though it can run much longer if conditions are poor.",
        },
        {
            "length": "short",
            "query": "How many kilograms of waste does it take to produce 1kg of larvae?",
            "answer": "Around 4.5 to 10 kg of organic waste per 1 kg of larval biomass, depending on the feedstock quality.",
        },
        {
            "length": "medium",
            "query": "My larvae seem to be growing really slowly compared to what I expected. What should I check first?",
            "answer": (
                "Slow growth usually traces back to temperature, moisture, or feedstock quality. First check that the "
                "substrate is holding in the 27-33C range — larvae outside that window process food much more slowly. "
                "Next check moisture; both overly wet (anaerobic, foul-smelling) and overly dry substrate will slow "
                "feeding. Finally, consider the feedstock itself — high-lignin material like cow manure or straw is much "
                "harder for larvae to digest than manure from omnivores, fish waste, or mixed kitchen scraps. What are you currently feeding them, and have you measured the substrate temperature directly?"
            ),
        },
    ],
    "prepupa": [
        {
            "length": "long",
            "query": "How do I build a self-harvesting setup so I don't have to manually pull prepupae out of the feed?",
            "answer": (
                "This relies on the prepupae's natural wandering instinct once they stop feeding. Build it like this:\n\n"
                "1. Identify the trigger. Prepupae stop eating, empty their gut, and darken toward brown/black — that's "
                "your signal they're ready to migrate.\n"
                "2. Install exit ramps. Angle ramps at 40-45 degrees from inside the feeding container up and out, using "
                "wood or textured plastic so prepupae can get traction.\n"
                "3. Position a dry collection point. Place a bucket or bin at the top of the ramp, kept completely free of "
                "moisture and feed residue — prepupae are seeking dry shelter and will keep wandering if the collection "
                "point looks like more wet substrate.\n"
                "4. Provide a proper pupation medium in the collector. Loose, porous material 15-20cm deep at around 60% "
                "moisture works best; too shallow and they'll keep wandering and burn fat reserves, too deep and "
                "emerging adults may fail to reach the surface later.\n"
                "5. Check the collector daily so pupae don't sit past their emergence window before being moved to your "
                "mating cage.\n\n"
                "Do you want this feeding directly into your mating enclosure, or into a separate pupation container you'll transfer by hand?"
            ),
        },
        {
            "length": "medium",
            "query": "How long does the prepupal wandering stage last?",
            "answer": (
                "About 7-10 days at 27C. During this window the prepupae aren't feeding at all — they've emptied their "
                "gut and are actively searching for a dry, sheltered place to pupate, which is exactly the behavior "
                "self-harvesting ramp systems are designed to exploit."
            ),
        },
        {
            "length": "short",
            "query": "Do prepupae need to be fed?",
            "answer": "No — prepupae are a non-feeding stage; don't add fresh feed to a prepupal collection area.",
        },
        {
            "length": "medium",
            "query": "My prepupae are ending up back in the feed tray instead of the collection bucket. How do I fix that?",
            "answer": (
                "That usually means the exit ramp isn't the easiest dry path out. Check the incline first — it should be "
                "40-45 degrees; anything shallower or too smooth to grip makes prepupae give up and wander back toward "
                "the substrate. Also make sure the collection point at the top is genuinely dry and free of feed residue, "
                "since prepupae are seeking dry shelter and will reject a collector that still smells like the feed tray. "
                "Is the ramp surface textured enough to climb, or is it a smooth material like bare plastic?"
            ),
        },
    ],
    "pupa": [
        {
            "length": "long",
            "query": "What does the pupal stage need to successfully produce adult flies, step by step?",
            "answer": (
                "1. Provide the right medium. Use a porous, loose material — moist wood shavings are a common choice — "
                "held at around 60-70% moisture. It needs to stay damp enough to prevent desiccation but not so wet it "
                "goes anaerobic.\n"
                "2. Set the depth correctly. 15-20cm is the target; too deep and emerging flies can't reach the surface, "
                "too shallow and prepupae may not settle to pupate at all.\n"
                "3. Hold temperature near 27C. This is the same optimum as the rest of the life cycle.\n"
                "4. Protect from light and disturbance. Pupae are stationary and vulnerable during this stage, so keep "
                "the container shaded and undisturbed.\n"
                "5. Wait it out. Pupation commonly takes about 7-14 days before adults emerge from the case.\n"
                "6. Move emerging adults promptly. Once you see emergence starting, transfer pupae to your mating cage if "
                "they aren't already there, so newly emerged flies have immediate access to light and space to mate.\n\n"
                "Is your pupation container currently indoors or outdoors, and does it get natural daylight?"
            ),
        },
        {
            "length": "medium",
            "query": "How long does pupation take before adults emerge?",
            "answer": (
                "Roughly 7-14 days at around 27C. During this window the exoskeleton darkens and hardens and the insect "
                "doesn't feed or move, so the main job on your end is just keeping the medium at the right moisture and "
                "leaving it undisturbed."
            ),
        },
        {
            "length": "short",
            "query": "What moisture level should the pupation medium be?",
            "answer": "Around 60-70% moisture in a loose, porous medium like wood shavings — dry enough to avoid anaerobic conditions but damp enough to prevent desiccation.",
        },
    ],
    "adult": [
        {
            "length": "long",
            "query": "How do I set up my mating cage to actually get consistent egg production from adult BSF?",
            "answer": (
                "1. Prioritize light. Strong light — ideally direct sunlight, or a comparable high-intensity artificial "
                "source — is required to trigger mating behavior. This is the single most common failure point for "
                "indoor setups.\n"
                "2. Give them room. Mating typically happens near the ground with males intercepting females mid-flight, "
                "so the cage needs enough open flight volume, not just floor space.\n"
                "3. Hold temperature near 27C and keep humidity reasonably high — adults benefit from moist conditions "
                "similar to the rest of the life cycle.\n"
                "4. Provide only liquid food. Adults have no functional chewing mouthparts and don't eat solids; supply "
                "water or a sugar solution, which can extend their lifespan well beyond the 5-8 day minimum.\n"
                "5. Give females a place to lay. Hang cardboard flutes or wooden blocks above a mild attractant food "
                "source inside the cage — mating starts about 2 days after emergence, and egg-laying follows roughly 2 "
                "days after mating.\n\n"
                "Is your mating cage currently getting direct sunlight, or is it indoors under artificial light?"
            ),
        },
        {
            "length": "medium",
            "query": "Do adult BSF need to eat solid food?",
            "answer": (
                "No — adults lack functional chewing mouthparts and don't feed on solids at all. They survive on fat "
                "reserves built up during the larval stage and only take up liquid water or a sugar solution, which can "
                "extend their lifespan from a bare minimum of around 5-8 days out to 16-40+ days."
            ),
        },
        {
            "length": "short",
            "query": "How soon after emergence do adult BSF mate?",
            "answer": "About 2 days after emerging from the pupal case.",
        },
        {
            "length": "short",
            "query": "What's the single biggest requirement for getting adult BSF to mate?",
            "answer": "Strong light — ideally direct sunlight, or an equivalent high-intensity artificial light source; without it, mating activity drops off sharply even if temperature and humidity are fine.",
        },
    ],
    "life_cycle_overall": [
        {
            "length": "medium",
            "query": "How long is the complete BSF life cycle from egg to adult?",
            "answer": (
                "Roughly 38-45 days under favorable rearing conditions: about 4 days as an egg, 13-18 days feeding as a "
                "larva, 7-10 days as a wandering prepupa, and 7-14 days pupating before an adult emerges. That total can "
                "stretch out much longer — larvae are able to delay maturity for extended periods if conditions turn "
                "unfavorable, so treat 38-45 days as the well-managed baseline, not a guarantee."
            ),
        },
        {
            "length": "short",
            "query": "Give me the quick life-cycle stage order for BSF.",
            "answer": "Egg, then larva (5 feeding instars), then prepupa (non-feeding, migrating), then pupa, then adult.",
        },
    ],
    "environment": [
        {
            "length": "long",
            "query": "I'm designing a new rearing room from scratch. What environmental setup should I plan for across the whole colony?",
            "answer": (
                "1. Temperature: hold the room around 27C for eggs, larvae, prepupae, and pupae alike — that's the shared "
                "optimum across stages. Avoid letting any zone drift above roughly 30-36C, where development starts to "
                "get inhibited.\n"
                "2. Humidity: target 60-70% relative humidity for eggs and mating, and 60-90% substrate moisture "
                "tolerance for larvae (aim toward the 60-70% part of that range in practice). Anything sustained above "
                "90% or below 30% starts causing problems — anaerobic odor at the wet end, desiccation at the dry end.\n"
                "3. Light: make sure your adult mating area gets strong light — direct sunlight is ideal; if indoors, "
                "plan supplemental lighting in the 450-700nm range at meaningful intensity, since low light is a common "
                "reason mating fails in enclosed rooms.\n"
                "4. Airflow and drainage: plan ventilation to prevent ammonia buildup in larval trays, and drainage holes "
                "or sloped trays so excess moisture doesn't pool.\n"
                "5. Zoning: separate egg-laying/mating space from larval feeding trays and from the pupation area, since "
                "each has a slightly different ideal humidity/light profile.\n\n"
                "Roughly how many trays or what production scale are you planning for — that changes how much I'd lean on active climate control versus passive shading and ventilation."
            ),
        },
        {
            "length": "medium",
            "query": "What temperature range should I be worried about as too hot for BSF?",
            "answer": (
                "Development is most efficient around 27C, and things start to get inhibited once temperatures climb "
                "into roughly the 30-36C range. Egg viability in particular drops off sharply above 40C, so if you're "
                "seeing temperatures creep that high in direct sun or a poorly ventilated container, that's worth "
                "addressing with shading or airflow before it affects hatch rates or larval survival."
            ),
        },
        {
            "length": "short",
            "query": "What's the optimal humidity range for BSF larvae?",
            "answer": "Larvae tolerate roughly 30-90% relative humidity, with 50-70% considered optimal for healthy development.",
        },
    ],
    "feedstock": [
        {
            "length": "long",
            "query": "I have access to chicken manure, fruit and vegetable market waste, and some fish scraps. How should I think about combining these as feedstock?",
            "answer": (
                "1. Understand what each brings. Fish waste is high in fat but can carry heavy metals if the source "
                "fish is contaminated. Chicken manure tends to boost protein content relative to plant waste alone. "
                "Fruit and vegetable waste gives close to 40% protein in the resulting larvae but comes with lower fat "
                "(under 10%).\n"
                "2. Avoid feeding any single stream at the extreme. Diets that are entirely liver or fish rendering have "
                "shown very high larval mortality (98%+) in trials, so don't rely on fish scraps alone.\n"
                "3. Blend for balance. Mixing manure with the fruit/vegetable waste is a standard way to boost protein "
                "without sacrificing the lower odor and easier handling of plant waste.\n"
                "4. Watch moisture and particle size. Chop or otherwise reduce particle size so larvae (which have no "
                "chewing mouthparts as adults but do have functional mouthparts as larvae) can access it efficiently, "
                "and keep the mixed feedstock in a workable moisture range rather than waterlogged.\n"
                "5. Avoid high-lignin material. None of your three streams are high-lignin, which is good — that "
                "category (straw, mature plant stems) is poorly digested by BSFL.\n\n"
                "What ratio were you thinking of mixing these in, and do you know the fish scraps' source well enough to rule out heavy-metal contamination?"
            ),
        },
        {
            "length": "medium",
            "query": "Can BSF larvae eat straw or other fibrous plant material efficiently?",
            "answer": (
                "Not very well on its own. BSFL struggle to digest lignin, the tough structural component of plant "
                "cell walls — wheat straw runs about 23% lignin versus roughly 10% in cow manure and only about 2% in "
                "pig manure. That's a big part of why cow manure is considered a lower-performing substrate compared to "
                "poultry or pig manure, and why straw alone isn't a good primary feedstock."
            ),
        },
        {
            "length": "short",
            "query": "Roughly how much of the feed by weight actually converts into larval biomass?",
            "answer": "About 20% of organic food matter by weight typically converts into larval biomass.",
        },
    ],
    "nutrition_composition": [
        {
            "length": "medium",
            "query": "What's the typical protein and fat content of dried BSF larvae?",
            "answer": (
                "It varies with the rearing substrate, but dried BSFL commonly run 37-63% crude protein and 7-35% fat. "
                "A frequently cited baseline composition is around 42% crude protein and roughly 35% fat by dry weight, "
                "plus meaningful ash, fiber, and moisture content. Larvae reared on nutrient-dense substrates like "
                "brewers' spent grain or manure tend to land toward the higher end of both ranges."
            ),
        },
        {
            "length": "short",
            "query": "Is BSFL amino acid quality comparable to fishmeal?",
            "answer": "Broadly yes — BSFL amino acid profiles, including methionine and lysine, are comparable to fishmeal and can meet or exceed FAO reference values for poultry feed.",
        },
        {
            "length": "long",
            "query": "I'm choosing between chicken manure, kitchen waste, and brewers' spent grain as my substrate and want the best nutritional outcome. Walk me through the tradeoffs.",
            "answer": (
                "Here's how these three commonly compare:\n\n"
                "1. Brewers' spent grain (SG): generally produces larvae with strong crude protein and fat content, and "
                "in fatty-acid work tends to give the highest overall fatty acid levels among common substrates.\n"
                "2. Chicken manure (CM): also a strong performer for protein and fat, with the tradeoff that it can "
                "drive higher mineral turnover in the larvae (both a benefit and something to monitor depending on your "
                "target feed use).\n"
                "3. Kitchen waste (KW): in controlled comparisons this produced the highest crude protein of the three, "
                "though with somewhat different fat and fiber characteristics than SG or CM.\n"
                "4. None of the three showed detectable aflatoxin contamination in controlled testing, which is "
                "reassuring, but heavy metal accumulation (cadmium, lead) is still substrate-dependent, so sourcing "
                "matters.\n\n"
                "What's this feed ultimately going to — poultry, fish, or pigs? That affects which of these tradeoffs matters most for you."
            ),
        },
    ],
    "disease_food_safety": [
        {
            "length": "long",
            "query": "I want to process my harvested larvae to be as safe as possible before selling them as feed. What's the procedure?",
            "answer": (
                "1. Understand the risk. BSF larvae have a highly acidic mid-gut that suppresses some pathogens, but "
                "pathogen survival has still been documented in the gut, so don't skip post-harvest treatment, "
                "especially for anything beyond direct livestock feeding.\n"
                "2. Pick a heat treatment. Effective, tested options include: boiling for 5 minutes; toasting over an "
                "open flame at 150C for 5 minutes with regular turning to prevent sticking or burning; or oven-drying at "
                "60C until weight loss stops (typically 2-3 days).\n"
                "3. For maximum pathogen reduction, consider oven-drying at 150C for about 22 minutes, which has shown "
                "strong reductions in human-relevant pathogens.\n"
                "4. Dry and store. Drying also concentrates nutrients and extends shelf life, which is useful "
                "regardless of the pathogen-control angle.\n"
                "5. Screen your substrate source. Since heavy metals can accumulate in larvae depending on what they "
                "were fed, avoid known-contaminated waste streams as feedstock in the first place.\n\n"
                "Is this feed headed for livestock, or are you looking at eventual human-consumption-grade processing? The bar for caution is higher for the latter."
            ),
        },
        {
            "length": "medium",
            "query": "Do BSF larvae accumulate aflatoxins from contaminated feed?",
            "answer": (
                "In controlled testing on chicken manure, kitchen waste, and brewers' spent grain substrates, no "
                "aflatoxin traces were detected in the larvae. Heavy metals like cadmium and lead are a separate "
                "concern, though — those can accumulate in larvae depending on contamination in the substrate, so "
                "screening your feedstock source still matters even though aflatoxin risk looks low."
            ),
        },
        {
            "length": "short",
            "query": "What's a simple way to reduce pathogen load in harvested larvae?",
            "answer": "Boiling them for 5 minutes is a simple, effective treatment; toasting at 150C for 5 minutes or oven-drying are other validated options.",
        },
    ],
    "harvesting": [
        {
            "length": "medium",
            "query": "What's the easiest way to harvest larvae at scale?",
            "answer": (
                "Let the prepupae self-harvest. Once larvae reach the wandering prepupal stage, they naturally climb "
                "40-45 degree exit ramps toward dry ground, which you can direct into a collection bucket with almost no "
                "manual labor. If you need to harvest earlier, while larvae are still feeding, wet harvesting (washing "
                "through progressively finer screens) or dry harvesting (shaking/sieving) will separate larvae from "
                "residual feed, though both are more labor-intensive than self-harvesting."
            ),
        },
        {
            "length": "short",
            "query": "What angle should a self-harvesting exit ramp be?",
            "answer": "Around 40-45 degrees, with a textured surface prepupae can grip to climb.",
        },
    ],
    "economics_value": [
        {
            "length": "medium",
            "query": "Why does BSF farming make economic sense for a smallholder?",
            "answer": (
                "Feed typically accounts for roughly 60-70% of total livestock production cost, so a low-cost protein "
                "source that can be grown on waste materially changes the economics. Beyond the larvae themselves, the "
                "leftover frass is a saleable organic fertilizer in its own right — it carries a meaningful N-P-K "
                "value (commonly cited around 5%-3%-2%) and is usually diluted before use given how concentrated it is."
            ),
        },
        {
            "length": "short",
            "query": "Is the leftover frass actually worth anything?",
            "answer": "Yes — it's a saleable organic fertilizer with meaningful N-P-K content, though it's usually diluted before field application because it's quite concentrated.",
        },
    ],
    "oil_biodiesel": [
        {
            "length": "long",
            "query": "I want to try extracting oil from BSF larvae for biodiesel. What does a proven feeding and extraction setup look like?",
            "answer": (
                "1. Choose a high-fat feed combination. A controlled trial using a 1:1 mix of skipjack tuna, avocado, "
                "and tofu — fed after a 2-day pre-fermentation period and at 70-80% feedstock moisture — produced strong "
                "larval weight gains, with the avocado:tofu combination performing best (a 21.96% w/w weight increase).\n"
                "2. Harvest and process the larvae as usual once they reach maturity.\n"
                "3. Extract the oil. Soxhlet extraction (using n-hexane at around 69C) significantly outperformed "
                "room-temperature maceration — 44.11% w/w oil yield versus 17.64% w/w.\n"
                "4. Expect better fatty acid selectivity from Soxhlet too, particularly for lauric, myristic, and oleic "
                "acids, which matter for biodiesel quality.\n"
                "5. Convert to biodiesel via standard transesterification. In the referenced study this produced "
                "biodiesel meeting the SNI 7128:2015 standard.\n\n"
                "Do you have access to Soxhlet extraction equipment, or would you need to start with maceration despite the lower yield?"
            ),
        },
        {
            "length": "short",
            "query": "Which extraction method gives more oil from BSF larvae — Soxhlet or maceration?",
            "answer": "Soxhlet extraction, by a wide margin — 44.11% w/w oil yield versus 17.64% w/w for room-temperature maceration in controlled testing.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Header variations for the "vision pipeline" framing (image + RAG context).
# ---------------------------------------------------------------------------
HEADER_VARIATIONS = [
    "[Vision Pipeline Analysis]\nPredicted Stage: {stage}\nConfidence: {conf}%\nStatus: In-Distribution\n\n"
    "[Reference Context]\n{context}\n\nUser Question: {q}",
    "[System Context: Vision Model detected stage '{stage}' (Confidence: {conf}%)]\n"
    "Reference Context:\n{context}\n\nUser Query: {q}",
    "Detected Life Stage: {stage} (Confidence: {conf}%)\nReference Context:\n{context}\n\nFarmer Question: {q}",
]

TEXT_ONLY_TEMPLATE = "User Question: {q}"

CONTEXT_ONLY_TEMPLATE = "[Reference Context]\n{context}\n\nUser Question: {q}"

# A stage label is only meaningful for the five life-cycle topics; other
# topics (environment, feedstock, economics, etc.) get a vision header of
# "General" so the model also learns the vision block isn't always a life
# stage classification.
STAGE_LABELS = {
    "egg": "Egg",
    "larva": "Larva",
    "prepupa": "Prepupa",
    "pupa": "Pupa",
    "adult": "Adult",
}

# ---------------------------------------------------------------------------
# GROUNDING-PRIORITY examples: the reference number is stated explicitly and
# conflicts with generic BSF trivia a base model might "recall" instead.
# ---------------------------------------------------------------------------
GROUNDING_PRIORITY_EXAMPLES = [
    {
        "context": "Total egg-to-adult life cycle for this colony: 38 days flat under current conditions. "
        "(Use this figure, not a different total.)",
        "q": "How long is the full BSF life cycle for my colony?",
        "a": "Based on the reference for your colony's current conditions, the full egg-to-adult life cycle is 38 days.",
    },
    {
        "context": "Egg clutch size for this colony has been logged at 900 eggs per female on average. "
        "(Use this figure, not a different average.)",
        "q": "About how many eggs does one female lay?",
        "a": "For your colony, the logged average is about 900 eggs per female.",
    },
    {
        "context": "Larval development in this unit has been running 21 days from hatch to prepupal stage under "
        "the current feed and temperature regime. (Use this figure, not a different duration.)",
        "q": "How many days does the larval feeding stage take in my setup?",
        "a": "Under your unit's current feed and temperature regime, larval development is running about 21 days from hatch to the prepupal stage.",
    },
]

# ---------------------------------------------------------------------------
# INSUFFICIENT-CONTEXT examples: the question asks for a detail genuinely
# not covered by the paired reference snippet.
# ---------------------------------------------------------------------------
INSUFFICIENT_CONTEXT_EXAMPLES = [
    {
        "topic": "larva",
        "q": "Exactly how many grams of feed should I give per larva per day at 3rd instar?",
        "a": (
            "The reference data doesn't give a precise gram-per-larva figure for 3rd instar specifically — that "
            "depends on larval density and your specific feedstock, which vary by setup. What it does confirm is the "
            "moisture (60-70%) and temperature (27-33C) targets, and that larvae can consume roughly 15 kg/m^2/day of "
            "feeding surface at scale. I'd rather flag the exact per-larva number as unanswered here than guess at it."
        ),
    },
    {
        "topic": "adult",
        "q": "What's the exact maximum lifespan a BSF adult can reach?",
        "a": (
            "The reference gives a range — adults commonly live 16-40+ days when supplied with water or a sugar "
            "solution, with 5-8 days as a bare minimum without feeding — but it doesn't specify a hard maximum beyond "
            "that. I wouldn't want to state a firm ceiling the data doesn't actually support."
        ),
    },
    {
        "topic": "economics_value",
        "q": "What's the current market price per kilogram for dried BSF larvae in my region?",
        "a": (
            "The reference context doesn't include current market pricing — that varies by region and changes over "
            "time, and I don't have a verified local price to give you. What I can tell you is that feed typically runs "
            "60-70% of total livestock production cost, which is the main economic case for producing your own BSFL "
            "rather than buying protein feed. For actual current pricing, I'd check with local BSF producers or "
            "livestock feed suppliers directly."
        ),
    },
]

# ---------------------------------------------------------------------------
# REAL-TIME / TROUBLESHOOTING examples: farmer messaging Foggy about
# something happening right now, no vision header, procedural answer ending
# in a next-action question.
# ---------------------------------------------------------------------------
REALTIME_EXAMPLES = [
    {
        "context": CONTEXT_SNIPPETS["larva"] + " " + CONTEXT_SNIPPETS["environment"],
        "q": "It's been raining for three days straight and my outdoor larval bins are soaked. What do I do right now?",
        "a": (
            "Act on this today, not later:\n\n"
            "1. Check for standing water. If any bin has pooled liquid, tip it to drain immediately — larvae will "
            "leave the substrate to escape waterlogged conditions.\n"
            "2. Add a dry bulking agent. Mix in bran, sawdust, or another dry absorbent material to pull the substrate "
            "back toward the 60-70% moisture range.\n"
            "3. Improve drainage now. If bins don't have drainage holes, add them (roughly 6mm is a workable size) so "
            "this doesn't repeat with the next rain.\n"
            "4. Cover what you can. A simple roof or tarp over the bins prevents direct rain entry while still "
            "allowing airflow.\n"
            "5. Watch for anaerobic odor over the next 24 hours — that's your sign moisture is still too high.\n\n"
            "Do you have dry bulking material on hand right now, or do you need a substitute suggestion based on what's locally available to you?"
        ),
    },
    {
        "context": CONTEXT_SNIPPETS["prepupa"] + " " + CONTEXT_SNIPPETS["harvesting"],
        "q": "I'm seeing prepupae crawling out of the bin and dying on the floor instead of reaching the collection bucket. What's happening?",
        "a": (
            "This is almost always a ramp or collection-point problem, not a colony health problem. Quick checklist:\n\n"
            "1. Check the ramp incline — it should be 40-45 degrees. Too shallow, and prepupae overshoot or fall off "
            "before reaching the top.\n"
            "2. Check ramp texture — smooth surfaces make it hard for prepupae to grip and climb consistently.\n"
            "3. Confirm the collection bucket is actually at the top of the ramp and not gapped or disconnected, since "
            "even a small gap lets prepupae fall short.\n"
            "4. Check for dehydration risk — prepupae wandering outside the intended path for too long can dry out "
            "before finding shelter, since this is a non-feeding stage running on stored fat.\n\n"
            "Can you tell whether the ones dying on the floor made it to the top of the ramp and missed the bucket, or never made it up the ramp at all?"
        ),
    },
    {
        "context": CONTEXT_SNIPPETS["adult"] + " " + CONTEXT_SNIPPETS["environment"],
        "q": "My mating cage has been up for two weeks and I still haven't seen a single egg cluster. What should I check?",
        "a": (
            "Two weeks with zero eggs points to the mating trigger, not the egg-laying setup. Work through this order:\n\n"
            "1. Light: this is the most common failure point. Mating requires strong light, ideally direct sunlight. If "
            "the cage is indoors or shaded, that alone can fully suppress mating.\n"
            "2. Cage volume: mating happens with males intercepting females in flight, so a cramped cage with little "
            "open flight space will suppress mating even with good light.\n"
            "3. Adult age: mating starts about 2 days after emergence, and egg-laying follows about 2 days after that "
            "— confirm your adults have actually had time to reach that window.\n"
            "4. Water/sugar source: adults need liquid food to survive long enough to mate; confirm one is present and "
            "not dried out.\n"
            "5. Laying surface: even with successful mating, females need cardboard flutes or wooden blocks near an "
            "attractant to actually deposit eggs where you'll see them.\n\n"
            "Is the cage getting direct sunlight at any point in the day, or is it fully indoors under artificial light?"
        ),
    },
    {
        "context": CONTEXT_SNIPPETS["disease_food_safety"],
        "q": "A buyer is asking whether my dried larvae batch is safe — what should I tell them about how it was processed?",
        "a": (
            "Be specific about the treatment method, since that's what determines safety here. If you boiled the "
            "batch for 5 minutes, toasted it at 150C for 5 minutes with regular turning, or oven-dried it at 60C until "
            "weight loss stopped (or at 150C for about 22 minutes for maximum pathogen reduction), any of those are "
            "validated pathogen-reduction methods you can state directly. If you skipped a heat treatment step, say so "
            "plainly rather than assuming the larvae's naturally acidic gut alone makes them safe — pathogen survival "
            "in the gut has still been documented even with that natural defense. Also worth mentioning: aflatoxin "
            "testing on common substrates found no traces, but heavy metal levels depend on your specific feedstock "
            "source, so if you know it was clean waste, that's worth stating too.\n\n"
            "Which treatment method did you actually use on this batch, so I can help you phrase it accurately for the buyer?"
        ),
    },
]


def format_context_header(topic: str, context_text: str, query: str, header_fmt: str) -> str:
    stage = STAGE_LABELS.get(topic, "General")
    conf_score = round(random.uniform(82.0, 99.9), 1)
    return header_fmt.format(stage=stage, conf=conf_score, context=context_text, q=query)


def generate_dataset():
    records = []

    # --- Main topic QA sets: each entry rendered in a rotating mix of
    # framings so the model sees vision-header, context-only, and
    # text-only versions across the dataset without every example being
    # multiplied against every framing (keeps index-alignment honest).
    framings = ["vision", "context_only", "text_only"]
    for topic, qa_list in QA_SETS.items():
        context_text = CONTEXT_SNIPPETS[topic]
        for i, item in enumerate(qa_list):
            q = item["query"]
            a = item["answer"]

            # Rotate framing per item so every topic gets a spread, but
            # also always include a text-only rendering for at least one
            # variant so plain-chat framing is present for every item.
            primary_framing = framings[i % len(framings)]

            if primary_framing == "vision":
                header_fmt = random.choice(HEADER_VARIATIONS)
                user_text = format_context_header(topic, context_text, q, header_fmt)
            elif primary_framing == "context_only":
                user_text = CONTEXT_ONLY_TEMPLATE.format(context=context_text, q=q)
            else:
                user_text = TEXT_ONLY_TEMPLATE.format(q=q)

            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": a},
                ]
            })

            # Also add a plain text-only rendering of every item (in
            # addition to whatever primary framing it got) so text-only
            # coverage is dense, matching the fix from v3.
            if primary_framing != "text_only":
                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": TEXT_ONLY_TEMPLATE.format(q=q)},
                        {"role": "assistant", "content": a},
                    ]
                })

    # --- Grounding-priority examples ---
    for ex in GROUNDING_PRIORITY_EXAMPLES:
        user_text = CONTEXT_ONLY_TEMPLATE.format(context=ex["context"], q=ex["q"])
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ex["a"]},
            ]
        })

    # --- Insufficient-context examples ---
    for ex in INSUFFICIENT_CONTEXT_EXAMPLES:
        context_text = CONTEXT_SNIPPETS[ex["topic"]]
        user_text = CONTEXT_ONLY_TEMPLATE.format(context=context_text, q=ex["q"])
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ex["a"]},
            ]
        })

    # --- Real-time / troubleshooting examples ---
    for ex in REALTIME_EXAMPLES:
        user_text = CONTEXT_ONLY_TEMPLATE.format(context=ex["context"], q=ex["q"])
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ex["a"]},
            ]
        })
        # Also include a text-only version of the same troubleshooting
        # question (farmer chatting without RAG context attached).
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TEXT_ONLY_TEMPLATE.format(q=ex["q"])},
                {"role": "assistant", "content": ex["a"]},
            ]
        })

    random.shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick composition report
    n_long = sum(1 for topic_qa in QA_SETS.values() for it in topic_qa if it["length"] == "long")
    n_medium = sum(1 for topic_qa in QA_SETS.values() for it in topic_qa if it["length"] == "medium")
    n_short = sum(1 for topic_qa in QA_SETS.values() for it in topic_qa if it["length"] == "short")

    print(f"Generated {len(records)} training examples -> {OUTPUT_FILE}")
    print(f"Base QA items by length -> long: {n_long}, medium: {n_medium}, short: {n_short}")
    print(f"Grounding-priority examples: {len(GROUNDING_PRIORITY_EXAMPLES)}")
    print(f"Insufficient-context examples: {len(INSUFFICIENT_CONTEXT_EXAMPLES)}")
    print(f"Real-time/troubleshooting examples: {len(REALTIME_EXAMPLES)} (x2 framings each)")


if __name__ == "__main__":
    generate_dataset()