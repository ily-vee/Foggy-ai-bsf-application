import time
import ollama

start = time.time()

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content":
                "Explain BSF larvae feeding requirements"
        }
    ]
)

end = time.time()

print(response["message"]["content"])

print("\nTime:")
print(end - start)
