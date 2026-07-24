import os
import base64
import requests
from io import BytesIO
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"
MAX_IMAGE_SIZE = (800, 800)


def encode_and_resize_image(image_path):
    """Opens a local image, optimizes its size for the CPU, and base64 encodes it."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def ask_about_image(image_path, user_question):
    """Sends both the image and a custom text question to the integrated server."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    print("\n⏳ Processing image and thinking...")
    base64_image = encode_and_resize_image(image_path)

    # Constructing the message payload containing both text and image
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=600)
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            print("\n🤖 Foggy's Answer:")
            print(answer)
        else:
            print(f"❌ Server Error [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failure: {str(e)}")


if __name__ == "__main__":
    # --- CHANGE THESE VALUES TO TEST ---
    # Pick one image from your new dataset folder
    test_image = "bsf_images_dataset/2_early_larvae/your_test_image.jpg"

    # Type whatever custom question you want to ask about that specific image
    my_question = "What stage of development is visible here, and how is the moisture level of the feed substrate?"

    # Run the test
    ask_about_image("test_image.jpeg", "Are there any signs of mold here?")
