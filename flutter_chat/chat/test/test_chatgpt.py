from openai import OpenAI
client = OpenAI()

text_input = "인공지능에 대해 알려줘"

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": text_input
        }
    ]
)

print(completion.choices[0].message)