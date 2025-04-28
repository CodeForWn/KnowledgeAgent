import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-072837472de74c139551100f63906bd8",   # 直接写key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=[
        {"role": "system",
         "content": [{"type": "text","text": "You are a helpful assistant."}]},
        {"role": "user","content": [{
            # 直接传入视频文件时，请将type的值设置为video_url
            # 使用OpenAI SDK时，视频文件默认每间隔0.5秒抽取一帧，且不支持修改，如需自定义抽帧频率，请使用DashScope SDK.
            "type": "video_url",
            "video_url": {"url": "http://119.45.164.254/resource/%E4%B8%AD%E5%B0%8F%E5%AD%A6%E8%AF%BE%E7%A8%8B/%E9%AB%98%E4%B8%AD%20%E5%9C%B0%E7%90%86/%E9%80%89%E4%BF%AE2-2/%E4%B8%BB%E9%A2%985/%E9%98%9C%E6%96%B0%E5%B8%82.mp4"}},
            {"type": "text","text": "这段视频的内容是什么?"}]
         }]
)
print(completion.choices[0].message.content)