import os
import requests
import json
import sys

print("🌱 Foggy BSF Assistant (Gemma 3 4B - Real-Time Streaming Mode)")
print("Type 'exit' to quit\n")

knowledge_path = "bsf_knowledge.txt"
if os.path.exists(knowledge_path):
    with open(knowledge_path, "r", encoding="utf-8") as file:
        bsf_context = file.read()
else:
    bsf_context = "No local data available."

conversation = [
    {
        "role": "system",
        "content": f"You are Foggy, an AI assistant for BSF farmers. Give practical advice based on these metrics:\n{bsf_context}"
    }
]

API_URL = "http://localhost:8080/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Foggy shutting down...")
        break

    conversation.append({"role": "user", "content": user_input})
    print("\nFoggy: ", end="")
    sys.stdout.flush()

    # CRITICAL CHANGE: We switch "stream" to True
    payload = {
        "messages": conversation,
        "temperature": 0.3,
        "stream": True
    }

    try:
        # Send a streaming request to llama-server
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), stream=True)

        full_reply = ""

        # Parse the incoming network chunks line-by-line as they generate
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').lstrip('data: ')
                if decoded_line.strip() == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(decoded_line)
                    # Extract the fractional token character text
                    delta_content = chunk_json['choices'][0]['delta'].get('content') or ''

                    # Print out each character chunk instantly without newline breaks
                    sys.stdout.write(delta_content)  # writes that tiny fragment of text (a single word or part of
                    # a word) directly to the screen
                    sys.stdout.flush()  # forces the terminal to show it instantly
                    full_reply += delta_content
                except json.JSONDecodeError:
                    continue
        print("\n")

        # Save complete statement to context memory history
        conversation.append({"role": "assistant", "content": full_reply})

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the local inference server. Check port 8080.")
        break
