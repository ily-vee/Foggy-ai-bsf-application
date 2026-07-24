import ollama

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain Black Soldier Fly (BSF) larvae growth factors"
        }
    ]
)

print(response["message"]["content"])