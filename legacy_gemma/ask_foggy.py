import os
import sys
import base64
import requests
import argparse
from io import BytesIO
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"
MAX_IMAGE_SIZE = (800, 800)


def encode_and_resize_image(image_path):
    """Opens a local image, scales it for CPU performance, and encodes to base64."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def query_foggy(image_path, prompt):
    """Sends the optimized image along with the user's terminal prompt."""
    if not os.path.exists(image_path):
        print(f"❌ Error: The file '{image_path}' does not exist.")
        sys.exit(1)

    print(f"⏳ Processing '{image_path}' and consulting Foggy AI...")

    try:
        base64_image = encode_and_resize_image(image_path)
    except Exception as e:
        print(f"❌ Error reading/resizing image: {str(e)}")
        sys.exit(1)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    try:
        # Long timeout to allow your CPU to complete the full matrix attention math
        response = requests.post(SERVER_URL, json=payload, timeout=600)
        if response.status_code == 200:
            print("\n🤖 Foggy AI:")
            print(response.json()['choices'][0]['message']['content'].strip())
        else:
            print(f"❌ Server Error [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failure: {str(e)}")


def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Interact with Foggy's multimodal vision system directly from the terminal."
    )

    # Required positional arguments
    parser.add_argument("image", help="Path to the image file (e.g., test_image.jpeg)")
    parser.add_argument("prompt", help="The question or command you want to ask about the image")

    args = parser.parse_args()

    # Run the query
    query_foggy(args.image, args.prompt)


if __name__ == "__main__":
    main()
