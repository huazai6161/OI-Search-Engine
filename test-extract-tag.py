import openai, json

# Open and load a JSON file
with open('data/IOI_outline/NOI.json', 'r', encoding='utf-8') as file:
    syllabus = json.load(file)

with open('data/questions/Luogu/statement/P6784 「EZEC-3」造房子.md', 'r', encoding='utf-8') as f:
    statement = f.read()

with open('data/questions/Luogu/solution/P6784 「EZEC-3」造房子.md', 'r', encoding='utf-8') as f:
    solution = f.read()

client = openai.OpenAI(api_key="")

prompt = f"""知识点大纲：{syllabus}
题目描述：{statement}
题解：{solution}
请阅读题目、题解、以及知识点大纲，分析该题目所考察的大纲知识点，提取这些知识点，并用 Json 格式输出。
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"你是一个信息学竞赛专家。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
)

a = response.choices[0].message.content

print(a)