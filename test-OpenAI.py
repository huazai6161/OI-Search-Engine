from openai import OpenAI

api_key = ""

client = OpenAI(api_key=api_key)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Read this web article 'https://www.luogu.com.cn/article/92rdgo6f' and give a summary."
        }
    ]
)

print(completion.choices[0].message)