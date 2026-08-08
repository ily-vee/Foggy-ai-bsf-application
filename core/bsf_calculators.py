"""
bsf_calculators.py

Deterministic replacements for the arithmetic the LLM was previously doing
in free text — which is where fabricated-looking-but-wrong numbers like
"7.14 cubic centimeters" and an invented "6:1 C:N ratio" came from earlier
in this project. These are plain Python functions with no model involved;
the inference pipeline calls them directly and hands the LLM the exact
result to restate, rather than asking it to compute anything itself.

The constants below (feed rate, stage durations) mirror the same figures
already baked into RAG_KNOWLEDGE_BASE / the training data, so computed
values stay consistent with everything else the model has been grounded on.
"""

from typing import Optional, Dict, Tuple

# Feed rate per larva per day, grams — matches "0.1 to 0.2 grams of feed
# per larva daily" used in the training data.
FEED_RATE_MIN_G_PER_LARVA_DAY = 0.10
FEED_RATE_MAX_G_PER_LARVA_DAY = 0.20

# Rough average larva weight at mid-feeding stage, used only to convert a
# biomass figure (kg) into an approximate larva count. This is a coarse
# assumption, not a measurement — flagged as such wherever it's used.
APPROX_LARVA_WEIGHT_G = 0.15

# Typical duration ranges (days) per stage — same figures used in
# RAG_KNOWLEDGE_BASE and the QLoRA training data.
STAGE_DURATION_DAYS: Dict[str, Tuple[float, float]] = {
    "egg": (4, 4),
    "larva": (13, 18),
    "prepupa": (7, 10),
    "pupa": (7, 14),
    "adult": (5, 40),
}


def calculate_feed_quantity(larvae_count: Optional[float] = None, biomass_kg: Optional[float] = None) -> Dict:
    """Daily feed quantity range (grams) for a given larvae count or biomass."""
    if larvae_count is None and biomass_kg is None:
        raise ValueError("Provide either larvae_count or biomass_kg")
    if larvae_count is None:
        larvae_count = biomass_kg * 1000 / APPROX_LARVA_WEIGHT_G

    min_g = larvae_count * FEED_RATE_MIN_G_PER_LARVA_DAY
    max_g = larvae_count * FEED_RATE_MAX_G_PER_LARVA_DAY

    return {
        "larvae_count_used": round(larvae_count),
        "feed_min_g_per_day": round(min_g, 1),
        "feed_max_g_per_day": round(max_g, 1),
        "feed_min_kg_per_day": round(min_g / 1000, 3),
        "feed_max_kg_per_day": round(max_g / 1000, 3),
    }


def estimate_stage_transition(stage: str, days_in_stage: float) -> Dict:
    """Estimated remaining-day range before an individual/batch transitions
    out of its current stage, given how long it's already been tracked."""
    stage = stage.lower()
    if stage not in STAGE_DURATION_DAYS:
        raise ValueError(f"Unknown stage '{stage}'")

    min_days, max_days = STAGE_DURATION_DAYS[stage]
    remaining_min = max(min_days - days_in_stage, 0)
    remaining_max = max(max_days - days_in_stage, 0)

    return {
        "stage": stage,
        "days_in_stage": round(days_in_stage, 1),
        "typical_duration_range_days": (min_days, max_days),
        "estimated_remaining_days_range": (round(remaining_min, 1), round(remaining_max, 1)),
    }


def calculate_moisture_dilution(dry_mass_kg: float, current_moisture_pct: float, target_moisture_pct: float) -> Dict:
    """Water (kg, ~1kg/L) needed to raise a substrate from current to
    target moisture percentage."""
    if not (0 <= current_moisture_pct < 100) or not (0 <= target_moisture_pct < 100):
        raise ValueError("Moisture percentages must be between 0 and 100")
    if target_moisture_pct <= current_moisture_pct:
        raise ValueError("Target moisture must exceed current moisture for a dilution calc")

    solids_kg = dry_mass_kg * (1 - current_moisture_pct / 100)
    total_mass_needed_kg = solids_kg / (1 - target_moisture_pct / 100)
    water_to_add_kg = total_mass_needed_kg - dry_mass_kg

    return {
        "water_to_add_kg": round(water_to_add_kg, 2),
        "water_to_add_liters": round(water_to_add_kg, 2),
        "resulting_total_mass_kg": round(total_mass_needed_kg, 2),
    }