import ollama

response = ollama.chat(
        model='deepseek-r1:32b',
        messages=[
            {'role': 'user', 'content': '我想知道什么是系统科学?'},
            ],
        stream=True
)

for chunk in response:
    if 'message' in chunk:
        print(chunk['message']['content'], end='', flush=True)
