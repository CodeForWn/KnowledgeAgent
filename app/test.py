"""
    Description:
   大模型流式输出客户端
"""

import json
import requests
import time

url = "http://192.168.121.245:30361/stream"  # 测试K8s， qwen2.5-14B

header = {"Content-Type": "application/json; charset=utf-8"}

prompt = "你好，请用多行介绍你。"
messages = [
    {"role": "system", "content": "你是亿语通航，是亿通国际自主研发的航贸大语言模型，专注于航运贸易领域的各类问答，为用户创造价值。"},
    {"role": "user", "content": prompt}
]

d = {
    "messages": messages,
    "temperature": 0.1,
    "top_p": 0.9,
    "request_id": "16461"
}

d = json.dumps(d)

s = time.time()

response = requests.post(url=url, headers=header, data=d, stream=True)

all_text = []

# 设置event_stream
for chunk in response.iter_content(chunk_size=5000, decode_unicode=True):
    if "[DONE]" not in chunk:
        chunk = json.loads(chunk.replace("data:", ""))
        print(chunk)
        all_text.append(chunk["message"])
        e = time.time()
        print("Cost　Time:", e - s)

print(all_text)
all_text = "".join(all_text)
print(all_text)
print(len(all_text))
