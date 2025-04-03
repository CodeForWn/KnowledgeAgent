import time
import hmac
import base64
import hashlib
import requests

API_KEY = "7ecfe70d70c4"
SECRET_KEY = "P0YpttKRvnnWMskbjrB0jK00WoHF5xYO"
API_HOST = "https://open.aippt.cn"
API_PATH = "/open-api/v1/ppt/gen-by-md"
API_URL = API_HOST + API_PATH

def generate_signature(secret_key, method, path, timestamp):
    sign_str = f"{method}\n{path}\n{timestamp}"
    hmac_obj = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1)
    return base64.b64encode(hmac_obj.digest()).decode()

def render_markdown_to_ppt(markdown_text, title):
    timestamp = str(int(time.time()))
    signature = generate_signature(SECRET_KEY, "POST", API_PATH, timestamp)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "x-timestamp": timestamp,
        "x-signature": signature
    }

    payload = {
        "title": title,
        "content": markdown_text,
        "outputType": "pptx"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        res = response.json()
        if res.get("code") == 0:
            return res["data"]["url"]
        else:
            raise Exception("渲染失败：" + res.get("message", "未知错误"))
    else:
        raise Exception("请求失败：" + response.text)
