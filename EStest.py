import requests
from elasticsearch import Elasticsearch
import json


# 测试'build_file_index'接口
def test_build_file_index():
    url = 'http://localhost:5777/api/build_file_index'

    # 准备测试数据
    data = {
        'user_id': 'admin',
        'assistant_id': '0001',
        'file_id': '2204',
        'tenant_id': '同济',
        'file_name': '孙家栋.pdf',
        'download_path': 'http://kmcgpt.sundeinfo.cn/api/dev/file/download?id=1714247829270659074'
    }

    # 创建一个Session对象
    session = requests.Session()

    # 设置请求头，指定编码方式
    session.headers['Content-Type'] = 'application/json; charset=utf-8'

    # 发送 POST 请求
    response = session.post(url, json=data)
    response.encoding = 'utf-8'
    # 解析JSON字符串
    decoded_content = json.loads(response.content)

    # 打印响应内容
    print("Response Content:", decoded_content)
    # 检查响应状态码
    if response.status_code == 200:
        print("Test Passed: Status Code is 200")
    else:
        print(f"Test Failed: Status Code is {response.status_code}")


def read_index_content():
    es = Elasticsearch(hosts=['https://127.0.0.1:9200/'],
                       verify_certs=False,
                       basic_auth=('elastic', 'hnUB+cMBlrpxWfeLOuqz'),
                       ).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )

    # 查询索引内容
    index_name = '0001_2204'
    query = {
        "query": {
            "match_all": {}  # 查询所有文档
        }
    }

    results = es.search(index=index_name, body=query)

    # 打印查询结果
    for hit in results['hits']['hits']:
        print(hit['_source'])


def test_delete_index(index_name):
    url = f'http://localhost:5777/api/delete_index/{index_name}'
    # 发送 POST 请求删除索引
    headers = {'Authorization': 'Sundeinfollm_KmcGPT'}
    response = requests.post(url, headers=headers)
    # 打印响应内容
    print(response.text)


def test_get_answer():
    # 定义接口的URL
    url = "http://127.0.0.1:5777/api/get_answer"  # 替换为你的接口的实际URL

    # 定义要发送的数据，包含 assistant_id、query 等字段
    data = {
        "assistant_id": "0001",
        "query": '两弹一星',
        "func": "embed",
        "ref_num": 3
    }

    # 发送POST请求到问答接口
    response = requests.post(url, json=data)

    # 检查响应状态码
    if response.status_code == 200:
        print("Test Passed: Status Code is 200")
    else:
        print(f"Test Failed: Status Code is {response.status_code}")

    # 解析响应JSON内容
    try:
        response_data = response.json()
        print(response_data)
    except Exception as e:
        print(f"获取问答失败: {str(e)}")


def test_get_open_answer():
    # 定义接口的URL
    url = "http://127.0.0.1:5777/api/get_open_ans"  # 确保URL是正确的，对应Flask应用的地址

    # 定义要发送的数据，包含query字段
    data = {
        "query": '两弹一星是什么？',
    }

    # 发送POST请求到问答接口
    response = requests.post(url, json=data)
    # 解析响应JSON内容
    try:
        response_data = response.text
        print(response_data)
    except Exception as e:
        print(f"获取问答失败: {str(e)}")


if __name__ == '__main__':
    test_build_file_index()
    # read_index_content()
    # test_get_answer()
    # test_get_open_answer()
    # test_delete_index("0001_2204")

