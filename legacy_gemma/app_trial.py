import requests
import json


def query_foggy_brain(prompt):
    url = "http://localhost:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Structuring the payload with system guidelines for precision agriculture
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are Project Foggy's core AI engine. Provide precise, brief, and actionable advice for "
                           "Black Soldier Fly (BSF) farmers based on provided data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,  # Low temperature ensures focused, hallucination-free metrics
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the local Foggy inference engine. Ensure llama-server is running."


# Local testing handshake
if __name__ == "__main__":
    test_prompt = ("A farmer reports their BSF breeding room crate humidity is hitting 45%. What is the quick remedial "
                   "advice?")
    print("Sending data payload to local model...")
    print("-" * 50)
    ai_insight = query_foggy_brain(test_prompt)
    print(ai_insight)