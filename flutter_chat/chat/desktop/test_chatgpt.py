from openai import OpenAI
client = OpenAI()

text_input = '인공지능에 대해 한줄로 알려줘'

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": text_input}
  ]
)

print(completion.choices[0].message.content)
print(completion)