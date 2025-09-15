from openai import OpenAI

client = OpenAI(api_key="8cefb70606f3472d8731bd65661ce409", base_url="http://8289.model.mingxingtech.com:10032/v1")
response = client.chat.completions.create(
  model="qwen2.5:72b",
  messages=[{"role": "user", "content": "Who are you?"}],
  temperature=0.0,
)

response = response.choices[0].message.content
print(response)