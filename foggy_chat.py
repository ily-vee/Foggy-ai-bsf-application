import os
import ollama

print("🌱 Foggy BSF Assistant (Gemma 3 4B Engine)")
print("Type 'exit' to quit\n")

# 1. Read the BSF context file dynamically
knowledge_path = "bsf_knowledge.txt"
if os.path.exists(knowledge_path):
    with open(knowledge_path, "r") as file:
        bsf_context = file.read()
else:
    bsf_context = "No local data available."

# 2. Inject the data straight into the System Personality
conversation = [
    {
        "role": "system",
        "content": f"""
        You are Foggy, an AI assistant for Black Soldier Fly farmers.
        Give practical farming advice based on the reference metrics provided below.
        Explain reasoning clearly.
        If a user query asks for metrics not listed in the data or if you don't know, state clearly that you don't have that local data point.

        REFERENCE METRICS:
        {bsf_context}
        """
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Foggy shutting down...")
        break

    conversation.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    print("\nFoggy is thinking (CPU Mode)...")

    response = ollama.chat(
        model="gemma3:4b",
        messages=conversation
    )

    answer = response["message"]["content"]

    print("\nFoggy:")
    print(answer)
    print()

    conversation.append(
        {
            "role": "assistant",
            "content": answer
        }
    )