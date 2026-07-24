"""
foggy_core.py
Unified application core for Foggy AI.
Interfaces the SigLIP 2 + Mahalanobis vision engine with application logic for WhatsApp/Chatbot.
"""

import json
from foggy_engine_qwen import FoggyBSFEngine

# Farming advice lookup table based on life cycle stage
STAGE_GUIDANCE = {
    "1_eggs": (
        "🥚 **Stage Identified: BSF Eggs**\n\n"
        "• **Moisture:** Keep relative humidity high (60–70%) near dark, dry egg-laying strips.\n"
        "• **Temperature:** Maintain optimal incubation at 27°C – 30°C.\n"
        "• **Action:** Ensure freshly hatched neonates have immediate access to moist, high-protein starter feed."
    ),
    "2_early_larvae": (
        "🐛 **Stage Identified: Early Larvae (Neonates / Small Larvae)**\n\n"
        "• **Feed Texture:** Provide finely blended, highly digestible organic waste.\n"
        "• **Moisture:** Substrate moisture should be around 65–70%.\n"
        "• **Density:** Avoid over-crowding in young trays to prevent heat buildup."
    ),
    "3_feeding_larvae": (
        "🐛 **Stage Identified: Active Feeding Larvae (3rd - 5th Instar)**\n\n"
        "• **Feed Conversion:** Peak waste processing stage! Feed kitchen waste, fruit peels, or brewer's waste.\n"
        "• **Temperature Control:** Watch out for self-heating substrate; keep below 38°C.\n"
        "• **Harvest Prep:** Prepare separation screens as larvae near full size."
    ),
    "4_pupae": (
        "🪵 **Stage Identified: Prepupae / Pupae**\n\n"
        "• **Environment:** Dark, dry, non-toxic substrate (e.g., dry sawdust or wood shavings).\n"
        "• **Behavior:** Feeding has stopped; larvae seek dry ground to pupate.\n"
        "• **Harvest/Emergence:** Harvest prepupae for animal feed or place in dark emergence cages for breeding."
    ),
    "5_bsf_adult": (
        "🪰 **Stage Identified: Adult Black Soldier Fly**\n\n"
        "• **Mating Conditions:** Require direct sunlight or specialized LED lighting (350–450 nm wavelength).\n"
        "• **Hydration:** Adults do not eat, but require clean water misting on cage walls to stay hydrated.\n"
        "• **Egg Attractants:** Place fermenting attractant bait below egg-laying cardboard strips."
    )
}


class FoggyCoreApp:
    def __init__(self):
        self.engine = FoggyBSFEngine()

    def setup(self):
        """Initializes the underlying vision inference engine."""
        self.engine.initialize()

    def process_user_media(self, image_path: str) -> dict:
        """
        Main entrypoint for processing user-uploaded images.
        """
        result = self.engine.predict(image_path)

        # 1. Error / Corrupt File
        if result["status"] == "error":
            return {
                "success": False,
                "user_message": f"⚠️ Could not process image: {result['message']}"
            }

        # 2. Out-of-Distribution Image (e.g., Kenyan bracelet, non-BSF object)
        if result.get("is_ood", False):
            return {
                "success": False,
                "is_ood": True,
                "user_message": (
                    "⚠️ **Unknown / Non-BSF Image Detected**\n\n"
                    "The uploaded image does not match any recognized Black Soldier Fly life cycle stage. "
                    "Please ensure the photo is clear and focused on BSF eggs, larvae, pupae, or adults."
                ),
                "details": result
            }

        # 3. Successful Classification
        predicted_stage = result["predicted_stage"]
        confidence_pct = round(result["confidence"] * 100, 1)
        advice = STAGE_GUIDANCE.get(predicted_stage, "No specific advice available for this stage.")

        user_msg = (
            f"✅ **Analysis Complete** (Confidence: {confidence_pct}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{advice}"
        )

        return {
            "success": True,
            "stage": predicted_stage,
            "confidence": result["confidence"],
            "user_message": user_msg,
            "full_details": result
        }


if __name__ == "__main__":
    import sys

    app = FoggyCoreApp()
    app.setup()

    # Test with feeding larvae image or CLI arg
    test_img = sys.argv[1] if len(sys.argv) > 1 else "testimage1.jpeg"
    
    print(f"\n📱 Simulating user message with image: '{test_img}'...\n")
    response = app.process_user_media(test_img)
    print(response["user_message"])