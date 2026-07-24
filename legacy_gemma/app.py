import os
import re
import requests
from flask import Flask, request, jsonify

# Import your working pipeline directly from foggy_engine.py
from foggy_engine import run_vision_inference, stream_conversation

app = Flask(__name__)

# Production session registry to track separate users by phone number
whatsapp_sessions = {}


def get_session_history(phone_number):
    """Retrieves or initializes chat history for an isolated phone number."""
    if phone_number not in whatsapp_sessions:
        whatsapp_sessions[phone_number] = []
    return whatsapp_sessions[phone_number]


# 1. Define the Incoming Webhook Route
@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    # Parse incoming JSON payload from the WhatsApp API Gateway
    data = request.get_json() or {}

    # Extract structural details (Universal keys across Twilio / Meta structures)
    sender_id = data.get("from")  # e.g., "+254712345678"
    user_prompt = data.get("text", "")  # The text message string
    media_url = data.get("media_url")  # Temporary internet URL of uploaded photo

    if not sender_id:
        return jsonify({"status": "error", "message": "Missing sender identifier"}), 400

    print(f"\n📥 Incoming WhatsApp from {sender_id}: '{user_prompt}'")

    image_info = None

    # 2. Handle Incoming Image Media if Present
    if media_url:
        print(f"🖼️ Media URL detected: {media_url}. Downloading temporary image...")
        try:
            # Download the binary file from WhatsApp secure servers
            response = requests.get(media_url, stream=True)
            if response.status_code == 200:
                local_temp_path = f"temp_{sender_id.replace('+', '')}.jpg"

                with open(local_temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)

                print(f"   ✅ Saved temp image to disk: {local_temp_path}")

                # Run your SigLIP 2 + PyTorch MLP model on the downloaded image
                print("   🔍 Analyzing via local SigLIP 2 vision network...")
                label, confidence = run_vision_inference(local_temp_path)
                image_info = {"label": label, "confidence": confidence}
                print(f"   🎯 Vision Classification: {label} ({confidence:.2f}%)")

                # Safely delete the temporary file after processing to preserve disk space
                if os.path.exists(local_temp_path):
                    os.remove(local_temp_path)
            else:
                print("   ❌ Failed to download media from provided gateway URL.")
        except Exception as e:
            print(f"   ❌ Media handler exception: {str(e)}")

    # 3. Retrieve Isolated Conversation History
    history = get_session_history(sender_id)

    # 4. Process response using your existing conversational RAG engine
    print("🤖 Processing context and compiling streaming response...")

    # Capture the streamed response into a clean string for WhatsApp transmission
    reply_text = stream_conversation(user_prompt, history, image_info)

    # 5. Fine-Tune Styling specifically for WhatsApp mobile viewports
    # Convert standard markdown bold (**) to WhatsApp markdown (*)
    reply_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', reply_text)
    # Strip out any markdown headers (#) completely
    reply_text = re.sub(r'#+\s*', '', reply_text)

    # Update history for this specific sender
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": reply_text})

    # Keep history bounded to last 10 turns to avoid exploding prompt contexts
    if len(history) > 10:
        whatsapp_sessions[sender_id] = history[-10:]

    # 6. Response Payload sent back to the WhatsApp Gateway to deliver to the user
    return jsonify({
        "to": sender_id,
        "reply": reply_text
    }), 200


if __name__ == "__main__":
    # Boot Flask on port 5000
    print("🚀 Flask Webhook listener booting up locally...")
    app.run(host="0.0.0.0", port=5000, debug=False)
