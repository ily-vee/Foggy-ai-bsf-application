import time
import ollama

print("🌱 Foggy BSF Core Controller")
print("Enforcing 2048 Context Truncation to save CPU cycles...\n")

# Minimal system prompt initialization
conversation = [
    {
        "role": "system",
        "content": "You are Foggy, a concise BSF farming advisor. Answer in 2-3 clear sentences."
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Shutting down core controller...")
        break

    conversation.append({
        "role": "user",
        "content": user_input
    })

    print("\n[Computing Matrix Multiplications on CPU...]")
    start_time = time.time()

    # Advanced options tuning to optimize for CPU bottlenecks
    response = ollama.chat(
        model="gemma3:4b",
        messages=conversation,
        options={
            "num_ctx": 2048,  # Force-shrinks memory overhead from 4096 to 2048 tokens
            "num_predict": 100,  # Caps output response length to prevent rambling delays
            "temperature": 0.3  # Lowers computational randomness for faster settling
        }
    )

    end_time = time.time()
    reply = response["message"]["content"]

    print(f"\nFoggy:\n{reply}")
    print(f"\n⏱️ Turn Latency: {end_time - start_time:.2f} seconds\n")

    conversation.append({
        "role": "assistant",
        "content": reply
    })