from openai import OpenAI

client = OpenAI(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjUzMTYyOTRkLTk4YTItNDc1OS05ZTE1LWI5MzNkN2FjNmM3NSJ9.VYT-yIGtYXQwXB8G45dqRGtN8cUggl73ZsvrNx-9Pss", base_url="http://123.57.228.132:8285/api")
response = client.chat.completions.create(
  model="deepseek-v3.2-20251201-160k-local",
  messages=[{"role": "user", "content": "Who are you?"}],
  temperature=0.0,
)

response = response.choices[0].message.content
print(response)