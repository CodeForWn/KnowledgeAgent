import requests
from elasticsearch import Elasticsearch
import json


# 测试'build_file_index'接口
def test_build_file_index():
    url = 'http://127.0.0.1:5777/api/build_file_index'

    # 准备测试数据
    data = {
        'user_id': 'admin',
        'assistant_id': '0001',
        'file_id': '2207',
        'tenant_id': '党办',
        'file_name': '四个意识.pdf',
        'download_path': 'http://172.16.20.154:39250/api//dev/file/download?id=1731510987309432834'
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
    es = Elasticsearch(hosts=['http://127.0.0.1:9200/'],
                       verify_certs=False,
                       basic_auth=('elastic', '=3KtbmJug4-Skm2n3oV*'),
                       ).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )

    # 查询索引内容
    index_name = '1732060769857597441_1732060822315757570'
    query = {
        "query": {
            "match_all": {}  # 查询所有文档
        }
    }

    results = es.search(index=index_name, body=query)

    # 打印查询结果
    for hit in results['hits']['hits']:
        print(hit['_source']['file_name'])


def test_delete_index(index_name):
    url = f'http://127.0.0.1:5777/api/delete_index/{index_name}'
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
        "query": '校园一卡通的分类和功能',
        "func": "embed",
        "ref_num": 3,
        "llm": 'cutegpt'
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
        "query": '帮我生成一段生病请假的假条',
        "llm": 'cutegpt'
    }

    # 发送POST请求到问答接口
    response = requests.post(url, json=data)
    # 解析响应JSON内容
    try:
        response_data = response.json()
        print(response_data)
    except Exception as e:
        print(f"获取问答失败: {str(e)}")


def test_summary(file_id):
    # 定义接口的URL
    url = "http://127.0.0.1:5777/api/generate_summary_and_questions"  # 确保URL是正确的，对应Flask应用的地址

    # 定义要发送的数据，包含query字段
    data = {
        "file_id": file_id,
    }

    # 发送POST请求到问答接口
    response = requests.post(url, json=data)
    # 解析响应JSON内容
    try:
        response_data = response.json()
        print(response_data)
    except Exception as e:
        print(f"获取问答失败: {str(e)}")


def test_indexing_api(documents):
    # 定义接口的URL
    url = "http://127.0.0.1:5777/api/kmc/indexing"  # 确保URL是正确的，对应Flask应用的地址

    # 定义要发送的数据
    data = documents

    # 发送POST请求到索引接口
    try:
        response = requests.post(url, json=data)
        # 解析响应JSON内容
        response_data = response.json()
        print("API响应:", response_data)
    except Exception as e:
        print(f"发送请求失败: {str(e)}")

# 准备测试数据
test_data = [
    {
        "documentId": "12345",
        "documentTitle": "示例文档标题",
        "documentContent": "同济大学2023本科专业概览——数学、计算机科学、物理学(拔尖学生培养)数学与应用数学（数学拔尖学生培养基地）同济大学数学与应用数学（数学拔尖学生培养）致力于培养具备扎实数学理论基础，具有原创意识，放眼学科发展、瞄准数学前沿问题的一流数学专业人才。人工智能的根基在数学，2019年成立“智能计算与应用”同济大学数学中心，2021年获批上海市智能计算前沿科学研究基地，数学拔尖基地引领前沿数学和基础算法的理论研究，培养人工智能、建模及算法的新型应用数学家，助力我国的人工智能应用实现重大突破。同济大学数学科学学院始建于1945年，众多知名学者曾在此任教。新中国成立后几经调整，于1980年恢复应用数学系，2006年定名为数学系，2016年成立数学科学学院。"
    },
    {
        "documentId": "67890",
        "documentTitle": "另一个示例文档标题",
        "documentContent": "这是另一个文档的内容。每个文档都有唯一的ID、标题和内容。"
    }
]

if __name__ == '__main__':
    # test_build_file_index()
    # read_index_content()
    # test_get_answer()
    # test_get_open_answer()
    # test_summary(1312)
    test_delete_index("1781319034160123905")
    # test_indexing_api(test_data)

