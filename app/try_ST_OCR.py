import requests
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test')


def try_indexing(json_file_path):
    url = "http://localhost:5555/api/ST_OCR"
    headers = {'Content-Type': 'application/json'}

    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        logger.error(f"文件不存在: {json_file_path}")
        return

    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 发送 POST 请求
    response = requests.post(url, headers=headers, json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")


if __name__ == '__main__':
    # JSON 文件路径
    json_file_path = "/work/kmc/kmcGPT/model/16000ocr.json"

    # 调用测试函数
    try_indexing(json_file_path)

