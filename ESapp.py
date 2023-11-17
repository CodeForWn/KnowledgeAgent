from flask import Flask, request, jsonify
import re
import urllib3
import shutil
import pickle
from pdf2markdown import PDF
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import jieba.posseg as pseg
import tempfile
import os
from pdf2markdown import *
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import logging
from logging.handlers import RotatingFileHandler
import PyPDF2
import pypdf
import pikepdf
from traceback import format_exc
import warnings
import elasticsearch.exceptions
import warnings
from urllib3.exceptions import InsecureRequestWarning
from sentence_transformers import SentenceTransformer
import json
from ltp import LTP


# 加载配置文件
with open('es_config.json') as config_file:
    config = json.load(config_file)

# 使用配置
elasticsearch_hosts = config['elasticsearch']['hosts']
basic_auth_username = config['elasticsearch']['basic_auth_username']
basic_auth_password = config['elasticsearch']['basic_auth_password']
log_file = config['log_file']
stop_words = config['stopwords']
model_path = config['model_path']
secret_token = config['secret_token']
llm_ans_api = config['external_api']['llm_ans']
backend_notify_api = config['external_api']['backend_notify']
file_storage_path = config['file_storage_path']
record_path = config['record_path']
# 创建日志记录器，并设置日志级别
logger = logging.getLogger('myapp')
logger.setLevel(logging.INFO)

# 创建一个循环文件处理器，设置文件名和最大文件大小
file_handler = RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024 * 1024, backupCount=20)
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别

# 创建日志记录的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器添加到日志记录器
logger.addHandler(file_handler)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 忽略因不检查 SSL 证书而产生的警告
warnings.simplefilter('ignore', InsecureRequestWarning)

app = Flask(__name__)
CORS(app)
# 初始化 BGE 模型
model_path = model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

predefined_qa = {
    "你是谁？": "我是小孔，是双地信息有限公司基于大模型的办公助手。",
    "你好？": "您好，我是小孔，是双地信息有限公司基于大模型的办公助手，欢迎向我提问",
    "你叫什么？": "我是小孔，是双地信息有限公司基于大模型的办公助手。",
    "如何使用此服务？": "您可以在选定的助手中向我提问，我将帮您检索您想找的内容。",
    "我要怎么开始？": "您可以在选定的助手中向我提问，我将帮您检索您想找的内容。",
    "如何提问？": "在提问框中输入您的问题，然后点击“发送”图标即可。",
    "如何上传文件？": "在提问框上方选择‘上传文件’选项，然后按照提示操作即可轻松上传。",
    "为什么我的文件没有上传成功？": "请检查您的网络连接或文件格式，如有必要，请联系技术支持。",
    "我的数据安全吗？": "我们高度重视用户数据的安全性，采取多重加密和安全措施来保护您的数据。",
    "用户协议是什么？": "用户协议详细说明了使用我们服务的条款和条件，请在注册前仔细阅读。",
    "你们的公司是什么？": "我们的公司叫双地信息有限公司，专注于提供高质量的技术服务。",
    "你们的公司叫什么？": "我们的公司叫双地信息有限公司，专注于提供高质量的技术服务。"
    # 根据需要添加更多的问题和答案
}


# 意图识别&意图修复
def extend_query(query):
    words = list(pseg.cut(query))
    # print(words)
    # 检查所有词的词性是否为名词
    if all(flag.startswith('n') for word, flag in words):
        print("进行问题补充。。。")
        return f"{query}是什么？"
    else:
        return query


def load_stopwords(filepath):
    """读取停用词文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


# 读取所有类型文件
def process_pdf_file(pdf_path):
    """处理单个PDF文件"""
    doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        logger.info("加载 %d 页，文件源：%s", len(pages), pdf_path)
        logger.info("按页处理开始。。。")
        stopwords = load_stopwords("stopwords.txt")
        text_splitter = CharacterTextSplitter(chunk_size=350, chunk_overlap=0, separator='\n', length_function=len)
        document_texts = set()
        filtered_texts = set()  # 存储处理后的文本
        for page_index, page in enumerate(pages, start=1):
            split_text = text_splitter.split_text(page.page_content)
            for text in split_text:
                if text not in document_texts:
                    document_texts.add(text)
                    words = pseg.cut(text)
                    filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                    if filtered_text not in filtered_texts:
                        filtered_texts.add(filtered_text)
                        doc_list.append({'page': page_index, 'text': filtered_text})
                        print(f"Added text (Page {page_index}): {filtered_text[:30]}...")  # 打印文本的前30个字符
    except Exception as e:
        logger.error("PDF文件 %s 处理过程出现错误: %s", pdf_path, str(e))
        print(f"PDF文件{pdf_path}处理过程出现错误: {str(e)}")
    finally:
        try:
            os.remove(pdf_path)
            logger.info("成功删除文件：%s", pdf_path)
        except OSError as e:
            logger.error("删除文件失败：%s，错误：%s", pdf_path, str(e))
    return doc_list


def cal_passage_embed(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.inference_mode():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()[0].tolist()


def cal_query_embed(query):
    instruction = "为这个句子生成表示以用于检索相关文章："
    return cal_passage_embed(instruction + query)


def count_chinese_chars(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return len(pattern.findall(text))


def check_if_pdf_valid(filename):
    """
    判断pdf文件是否可用
    :param filename:
    :return:
    """
    try:
        PyPDF2.PdfFileReader(open(filename, "rb"))
    except:
        try:
            pikepdf.Pdf.open(filename)
        except:
            try:
                pypdf.PdfReader(filename)
            except:
                print(format_exc())
                return False
            else:
                return True
        else:
            return True
    else:
        return True


def download_pdf(download_url, file_id):
    try:
        # 使用requests库下载PDF文件
        headers = {'token': file_id}
        response = requests.get(download_url, headers=headers)

        if response.status_code != 200:
            return {
                "status_code": response.status_code,
                "status": "FAILURE",
                "message": f"下载PDF失败，状态码：{response.status_code}, URL: {download_url}"
            }
        # 构建 PDF 文件的完整路径
        pdf_path = os.path.join(file_storage_path, f"{file_id}.pdf")

        # 保存下载的PDF文件
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)

        return pdf_path

    except Exception as e:
        logger.error("%s 下载PDF失败，URL: %s, 错误: %s", file_id, download_url, str(e))
        return {
            "status_code": 500,
            "status": "FAILURE",
            "message": f"下载PDF失败，URL: {download_url}, 错误: {str(e)}"
        }


def readPDF(pdf_path, file_id):
    # 构建保存TXT的路径
    txt_path = f"{file_storage_path}/{file_id}.txt"
    print('开始读取PDF...')
    # 调用 pdf2markdown 模块中的 PDF 类来解析 PDF 文件，将其转换为文本，并保存为 .txt 文件
    PDF(pdf_path, txt_path).parsePDF()
    print('读取PDF完成, 保存txt完成')
    ISZH = False
    doc = None
    try:
        with open(txt_path, encoding='utf-8') as f:
            doc = f.read()
            cnt = count_chinese_chars(doc)
            if cnt > 20:
                ISZH = True
            doc = doc.split('\f')
    except UnicodeDecodeError as e:
        print(f"解码文件时出错：{e}")

    doc_list = []
    if ISZH:
        for idx, txt in enumerate(doc):
            txt = txt.replace('\n', "").replace('\r', '')
            if len(txt) <= 20:
                continue
            if txt.count('.') > 20:
                continue
            text_split = []
            for i in range(0, len(txt), 350):
                text_split.append(txt[i:i + 400])
            for i in text_split:
                doc_list.append({'page': idx + 1, 'text': i})
    else:
        for idx, txt in enumerate(doc):
            txt = txt.replace('\n', "").replace('\r', '')
            txt = word_tokenize(txt)
            if len(txt) <= 20:
                continue
            text_split = []
            for i in range(0, len(txt), 250):
                text_split.append(txt[i:i + 300])
            for i in text_split:
                doc_list.append({'page': idx + 1, 'text': " ".join(i)})
    print('文本分割完成.')
    return doc_list


def create_es_index(user_id, tenant_id, assistant_id, file_id, file_name, download_path, doc_list):
    index_name = f'{assistant_id}_{file_id}'  # 根据助手ID和文件ID创建索引名称
    es = Elasticsearch(hosts=elasticsearch_hosts,
                       verify_certs=False,
                       basic_auth=(basic_auth_username, basic_auth_password),
                       ).options(
        request_timeout=20,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )

    # 检测是否成功连接到ES
    if es.ping():
        print("成功连接到Elasticsearch")
    else:
        print("无法连接到Elasticsearch")
        logger.error("无法连接到Elasticsearch")
        return {file_id, "FAILURE", "无法连接到Elasticsearch"}

    if es.indices.exists(index=index_name):
        logger.info("索引已存在，删除索引")
        print("索引已存在，删除索引")
        es.indices.delete(index=index_name)
    logger.info("开始创建索引")
    print("开始创建索引")
    mappings = {
        "properties": {
            "text": {"type": "text", "analyzer": "standard"},
            "embed": {"type": "dense_vector", "dims": 1024},
            "user_id": {"type": "keyword"},
            "assistant_id": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "file_id": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "download_path": {"type": "keyword"}
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    # logger.info("开始插入数据")
    # print("开始插入数据")
    for item in doc_list:
        page = item['page']
        text = item['text']
        embed = cal_passage_embed(text)
        document = {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "file_id": file_id,
            "file_name": file_name,
            "tenant_id": tenant_id,
            "download_path": download_path,
            "page": page,
            "text": text,
            "embed": embed
        }
        es.index(index=index_name, document=document)

    logger.info("索引建立完成")
    print("索引建立完成")
    return "success"


def notify_backend(file_id, result, failure_reason=None):
    """通知后端接口处理结果"""
    url = backend_notify_api  # 更新后的后端接口URL
    headers = {'token': file_id}
    payload = {
        'id': file_id,
        'result': result
    }
    if failure_reason:
        payload['failureReason'] = failure_reason

    response = requests.post(url, json=payload, headers=headers)
    return response.status_code


@app.route('/api/build_file_index', methods=['POST'])
def build_file_index():
    data = request.json  # 获取前端传来的json数据
    user_id = data.get('user_id')
    assistant_id = data.get('assistant_id')
    file_id = data.get('file_id')
    file_name = data.get('file_name')
    download_path = data.get('download_path')
    tenant_id = data.get('tenant_id')

    try:
        # 下载文件并处理
        logger.info("开始下载文件并处理: %s", file_name)
        print("开始下载PDF文件：", file_name)
        pdf_path = download_pdf(download_path, file_id)
        doc_list = process_pdf_file(pdf_path)

        if not doc_list:
            notify_backend(file_id, "FAILURE", "未能成功处理PDF文件")
            logger.error("未能成功处理PDF文件: %s", file_id)
            print("未能成功处理PDF文件:", file_id)
            return jsonify({"status": "error", "message": "未能成功处理PDF文件"})

        # 建立ES索引
        logger.info("开始建立索引")
        print("开始建立索引")
        result = create_es_index(user_id, tenant_id, assistant_id, file_id, file_name, download_path, doc_list)

        if result == "success":
            notify_backend(file_id, "SUCCESS")
            logger.info("建立索引成功: %s_%s", assistant_id, file_id)
            print("建立索引成功:", f'{assistant_id}_{file_id}')
            return jsonify({"status": "成功建立索引"})
        else:
            notify_backend(file_id, "FAILURE", result)
            logger.error("建立索引失败: %s_%s", assistant_id, file_id)
            print("建立索引失败:", f'{assistant_id}_{file_id}')
            return jsonify({"status": "建立索引失败", "message": result})

    except Exception as e:
        notify_backend(file_id, "FAILURE", str(e))
        logger.error("建立索引失败: %s_%s, 错误: %s", assistant_id, file_id, e)
        print("建立索引失败:", f'{assistant_id}_{file_id}', e)
        return jsonify({"status": "error", "message": str(e)})


# 获得开放性回答
@app.route('/api/get_open_ans', methods=['POST'])
def get_open_ans():
    data = request.json
    query = data.get('query')
    response = requests.post(llm_ans_api, json={'query': query, 'loratype': 'qa'}).json()
    ans = response['ans']
    return ans


def get_ans(query, refs):
    ref_list = [k['text'] for k in refs]
    prex = f"参考这一篇文章里与问题相关的以下{len(ref_list)}段文本，然后基于这些内容，回答后面的问题：\n"
    for i, ref in enumerate(ref_list):
        prex += f"[{i + 1}]:{ref}\n"

    query = extend_query(query)
    query = f"{prex}\n问题：{query}\n：你应当尽量用原文回答。若文本中缺乏相关信息，则回答“没有足够信息来回答”。如果回答内容在30字以内，请在回答之后将该段原文总结，不要太短。"
    print("最后的prompt:", query)
    logger.info("prompt: %s", query)
    # print("prompt:", query)
    response = requests.post(llm_ans_api, json={'query': query, 'loratype': 'qa'}).json()
    ans = response['ans']
    return ans


@app.route('/api/get_answer', methods=['POST'])
def answer_question():
    query_body = {}
    refs = []
    es = Elasticsearch(hosts=elasticsearch_hosts,
                       verify_certs=False,
                       basic_auth=(basic_auth_username, basic_auth_password),
                       ).options(
        request_timeout=20,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )
    # 检测是否成功连接到ES
    if es.ping():
        print("成功连接到Elasticsearch")
    else:
        print("无法连接到Elasticsearch")
        return {"FAILURE", "无法连接到Elasticsearch"}
    try:
        data = request.json
        assistant_id = data.get('assistant_id')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 3)
        print("接收到的参数：", assistant_id, query, func, ref_num)
        # 检查问题是否匹配预设的问题
        if query in predefined_qa:
            return jsonify({'answer': predefined_qa[query], 'matches': []}), 200

        if not assistant_id or not query:
            logger.error("参数不完整")
            print("参数不完整")
            return jsonify({'error': '参数不完整'}), 400

        if func == 'embed':
            logger.info("使用向量数据库匹配。。。。")
            print("使用向量数据库匹配。。。。")
            query_embed = cal_query_embed(query)
            query_body = {
                "query": {
                    "script_score": {
                        "query": {"term": {"assistant_id": assistant_id}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                            "params": {"query_vector": query_embed}
                        }
                    }}}
        elif func == 'bm25':
            logger.info("使用bm25匹配。。。。")
            print("使用bm25匹配。。。。")

            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"text": query}},
                            {"term": {"assistant_id": assistant_id}}
                        ]
                    }
                }
            }

        # 在所有符合条件的ES索引中查询
        index_pattern = f'{assistant_id}_*'
        top_k = ref_num
        result = es.search(index=index_pattern, body=query_body)
        # print("查询结果：", result)
        if 'hits' in result and 'hits' in result['hits']:
            print("命中结果")
            hits = result['hits']['hits'][:top_k]
            refs = [{'text': hit['_source']['text'],
                     'page': hit['_source']['page'],
                     'file_id': hit['_source']['file_id'],
                     'file_name': hit['_source']['file_name'],
                     'score': hit['_score'],
                     'download_Path': hit['_source']['download_path']} for hit in hits]

        if not result['hits']['hits']:
            logger.error("未找到相关文本片段")
            print("文本匹配未命中结果")
            return jsonify({'error': '未找到相关文本片段'})

        ans = get_ans(query, refs)
        # 返回包含匹配得分的结果
        # logger.info("回答: %s 匹配文本: %s", ans, refs)
        print('回答:', ans, '匹配文本:', refs)
        log_data = {
            'question': query,
            'answer': ans,
            'matches': refs
        }

        # 将回答和匹配文本保存到JSON文件中
        with open(record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

        return jsonify({'answer': ans, 'matches': refs}), 200

    except Exception as e:
        logger.error("获取回答失败: %s", str(e))
        return jsonify({'error': str(e)})


# 设置一个静态token
SECRET_TOKEN = secret_token


@app.route('/api/delete_index/<index_name>', methods=['POST'])
def delete_index(index_name):
    try:
        # 获取请求头中的token
        token = request.headers.get('Authorization')
        # print("token:", token)

        # 验证token
        if token != SECRET_TOKEN:
            logger.error("token验证失败")
            return jsonify({"code": 403, "msg": "无权限"}), 403
        # 验证参数
        if not index_name:
            logger.error("错误：缺少索引名称参数")
            return jsonify({"code": 500, "msg": "错误：缺少索引名称参数"})
        # Elasticsearch 服务器的地址和认证信息
        es = Elasticsearch(hosts=elasticsearch_hosts,
                           verify_certs=False,
                           basic_auth=(basic_auth_username, basic_auth_password),
                           ).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )

        # 检查索引是否存在，发送 DELETE 请求删除索引
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f'成功删除索引 {index_name}')
            logger.info("成功删除索引 %s", index_name)
            return jsonify({"code": 200, "msg": f"成功删除索引 {index_name}"})
        else:
            print(f'索引 {index_name} 不存在')
            logger.error("索引 %s 不存在", index_name)
            return jsonify({"code": 500, "msg": f"索引 {index_name} 不存在"})

    except Exception as e:
        logger.error("删除索引 %s 失败，错误信息：%s", index_name, str(e))
        print(f'删除索引 {index_name} 失败，错误信息：{str(e)}')
        return jsonify({"code": 500, "msg": f"删除索引 {index_name} 失败，错误信息：{str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5777, debug=False)
