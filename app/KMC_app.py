# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import threading
import queue
import urllib3
import logging
import requests
from logging.handlers import RotatingFileHandler
import sys
import re
import uuid
import jieba.posseg as pseg
from FlagEmbedding import FlagReranker
sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)
CORS(app)
# 加载配置
# 使用环境变量指定环境并加载配置
config = Config(env='production')
config.load_config()  # 指定配置文件的路径
config.load_predefined_qa()
logger = config.logger
record_path = config.record_path
backend_notify_api = config.external_api_backend_notify
# 创建 FileManager 实例
file_manager = FileManager(config)
# 创建ElasticSearchHandler实例
es_handler = ElasticSearchHandler(config)
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)
model_path = "/work/kmc/kmcGPT/model/bge-reranker-base"

# 创建队列
file_queue = queue.Queue()
index_lock = threading.Lock()
logger.info('服务启动中。。。')


def generate_assistant_id():
    # 生成一个随机的UUID并转换为字符串
    return str(uuid.uuid4())


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
    logger.info("后端接口返回状态码：%s", response.status_code)
    return response.status_code


def pull_file_data():
    # 模拟从服务器获取文件数据
    return []


def _push(file_data):
    global file_queue
    file_queue.put(file_data)


def _process_file_data(data):
    with app.app_context():
        user_id = data.get('user_id')
        assistant_id = data.get('assistant_id')
        file_id = data.get('file_id')
        file_name = data.get('file_name')
        download_path = data.get('download_path')
        tenant_id = data.get('tenant_id')

        try:
            # 处理文件并创建索引
            logger.info("开始下载文件并处理: %s，文件路径：%s", file_name, download_path)
            file_path = file_manager.download_pdf(download_path, file_id)
            doc_list = file_manager.process_pdf_file(file_path, file_name)

            if not doc_list:
                notify_backend(file_id, "FAILURE", "未能成功处理PDF文件")
                logger.error("未能成功处理PDF文件: %s", file_id)
                return jsonify({"status": "error", "message": "未能成功处理PDF文件"})

            index_name = assistant_id
            es_handler.create_index(index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path)
            es_handler.notify_backend(file_id, "SUCCESS")
        except Exception as e:
            logger.error("处理文件数据失败: {}".format(e))
            es_handler.notify_backend(file_id, "FAILURE", str(e))


def _thread_index_func(isFirst):
    while True:
        try:
            _index_func(isFirst)
        except Exception as e:
            print("索引处理失败:", e)


def _index_func(isFirst):
    global file_queue
    try:
        file_data = file_queue.get(timeout=5)
        logger.info("获取到队列语料")
        _process_file_data(file_data)
        logger.info("队列语料处理完毕")
        return True
    except queue.Empty as e:
        if isFirst:
            files = pull_file_data()
            for file_data in files:
                _push(file_data)
            if len(files) == 0:
                # get failed files from server
                pass
        return False
    except Exception as e:
        logger.error("索引功能异常: {}".format(e))


@app.route('/api/build_file_index', methods=['POST'])
def build_file_index():
    data = request.json
    _push(data)
    return jsonify({"status": "success", "message": "文件数据已接收，准备处理"})


# 获得开放性回答
@app.route('/api/get_open_ans', methods=['POST'])
def get_open_ans():
    data = request.json
    session_id = data.get('session_id')
    token = data.get('token')
    query = data.get('query')
    llm = data.get('llm', 'qwen')  # 默认使用CuteGPT
    # 获取历史对话内容
    history = prompt_builder.get_history(session_id, token)
    logger.info(f"历史对话：{history}")
    # 构建新的prompt
    prompt = prompt_builder.generate_open_answer_prompt(query, history)
    logger.info(f"prompt:{prompt}")
    answer = ''
    if llm.lower() == 'cutegpt':
        answer = large_model_service.get_answer_from_Tyqwen(prompt)
    elif llm.lower() == 'chatglm':
        task_id = large_model_service.async_invoke_chatglm(prompt)
        answer = large_model_service.query_async_result_chatglm(task_id)
    elif llm.lower() == 'chatgpt':
        answer = large_model_service.get_answer_from_chatgpt(prompt)
    elif llm.lower() == 'qwen':
        answer = large_model_service.get_answer_from_Tyqwen(prompt)
    return jsonify({'answer': answer, 'matches': []}), 200


@app.route('/api/get_answer', methods=['POST'])
def answer_question():
    try:
        # 初始化重排模型
        reranker = FlagReranker(model_path, use_fp16=True)
        # 收集所有检索到的文本片段
        all_refs = []
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'qwen').lower()

        if not assistant_id or not query:
            return jsonify({'error': '参数不完整'}), 400

        # 检查问题是否在预定义的问答中
        predefined_answer = config.predefined_qa.get(query)
        if predefined_answer:
            # 如果预设的答案是一个字符串，意味着没有匹配信息，返回答案和空的matches列表
            if isinstance(predefined_answer, str):
                return jsonify({'answer': predefined_answer, 'matches': []}), 200
            # 如果预设的答案是一个字典，包含'answer'和'matches'键，返回相应的内容
            elif isinstance(predefined_answer,
                            dict) and 'answer' in predefined_answer and 'matches' in predefined_answer:
                return jsonify({
                    'answer': predefined_answer['answer'],
                    'matches': predefined_answer['matches']
                }), 200

        if func == 'bm25' or 'embed':
            bm25_refs = es_handler.search_bm25(assistant_id, query, ref_num)
            embed_refs = es_handler.search_embed(assistant_id, query, ref_num)
            all_refs = bm25_refs + embed_refs

        if not all_refs:
            ans = large_model_service.get_answer_from_Tyqwen(query)
            ans = "您的问题没有在文本片段中找到答案，正在使用预训练知识库为您解答：" + ans
            return jsonify({'answer': ans, 'matches': all_refs}), 200

        # 使用重排模型进行重排并归一化得分
        # 提取文本并构造查询-引用对
        ref_pairs = [[query, ref['text']] for ref in all_refs]  # 为每个参考文档与查询组合成对
        scores = reranker.compute_score(ref_pairs, normalize=True)  # 计算每对的得分并归一化
        # 根据得分排序，并选择得分最高的引用
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        # 提取前五个最高的分数
        top_list = sorted_refs[:5]
        top_scores = [score for _, score in sorted_refs[:5]]
        # 打印出这些分数
        print("Top 5 scores:", top_scores)
        top_refs = [ref for ref, score in sorted_refs[:5]]  # 假设您只需要前5个最相关的引用
        prompt = prompt_builder.generate_answer_prompt(query, top_refs)
        if llm == 'cutegpt':
            # old_answer = large_model_service.get_answer_from_Tyqwen(prompt)
            # beauty_prompt = prompt_builder.generate_beauty_prompt(old_answer)
            # ans = large_model_service.get_answer_from_Tyqwen(beauty_prompt)
            ans = large_model_service.get_answer_from_Tyqwen(prompt)
        elif llm == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
        elif llm == 'chatgpt':
            ans = large_model_service.get_answer_from_chatgpt(prompt)
        elif llm == 'qwen':
            ans = large_model_service.get_answer_from_Tyqwen(prompt)
        else:
            return jsonify({'error': '未知的大模型服务'}), 400

        ans = ans.replace("\n", "<br/>")
        log_data = {'question': query,
                    'answer': ans,
                    'matches': [{
                        'text': ref['text'],
                        'original_text': ref['original_text'],
                        'page': ref['page'],
                        'file_id': ref['file_id'],
                        'file_name': ref['file_name'],
                        'download_path': ref['download_path'],
                        'score': ref['score'],
                        'rerank_score': score
                    } for ref, score in top_list]}

        # 记录日志
        logger.info(f"问答记录: {log_data}")
        # 将回答和匹配文本保存到JSON文件中
        with open(record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        return jsonify({'answer': ans, 'matches': log_data['matches']}), 200

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_summary_and_questions', methods=['POST'])
def generate_summary_and_questions():
    try:
        data = request.json
        file_id = data['file_id']
        ref_num = data.get('ref_num', 5)
        es_handler.create_answers_index()
        # 检查是否已有存储的答案
        existing_answer = es_handler._search_("answers_index", {"query": {"term": {"file_id": file_id}}})
        if 'hits' in existing_answer and 'hits' in existing_answer['hits'] and existing_answer['hits']['hits']:
            logger.info(f"找到了文件ID {file_id} 的存储答案")
            stored_answer = existing_answer['hits']['hits'][0]['_source']['sum_rec']
            stored_answer = stored_answer.replace("\n", "<br>")
            logger.info(f"存储答案为 {stored_answer}")
            return jsonify({'answer': stored_answer, 'matches': []}), 200

        logger.info(f"正在查询文件ID {file_id} 的前 {ref_num} 段文本")
        query_body = {
            "query": {
                "term": {
                    "file_id": file_id  # 确保file_id匹配
                }
            },
            "sort": [
                {"page": {"order": "asc"}}  # 按照page字段升序排序
            ]
        }
        results = es_handler._search_('_all', query_body, ref_num)

        if 'hits' in results and 'hits' in results['hits']:
            ref_list = [hit['_source']['text'] for hit in results['hits']['hits']]
            prompt = prompt_builder.generate_summary_and_questions_prompt(ref_list)
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
            # 存储新生成的答案
            es_handler.index("answers_index", {"file_id": file_id, "sum_rec": ans})
        else:
            logger.error("未找到文件ID {file_id} 的文本段落")
            ans = "未找到相关信息"

        # 将字符串中的\n替换为<br>
        ans = ans.replace("\n", "<br>")
        logger.info(f"总结或推荐问题： {ans}")
        return jsonify({'answer': ans, 'matches': []}), 200

    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        return jsonify({'error': '内部错误，请联系管理员'}), 500


@app.route('/api/delete_index/<index_name>', methods=['POST'])
def delete_index_route(index_name):
    try:
        # 设置一个静态token
        SECRET_TOKEN = config.secret_token
        # 获取请求头中的token
        token = request.headers.get('Authorization')

        # 验证token
        if token != SECRET_TOKEN:
            logger.error("token验证失败")
            return jsonify({"code": 403, "msg": "无权限"}), 4032

        # 验证参数
        if not index_name:
            logger.error("错误：缺少索引名称参数")
            return jsonify({"code": 500, "msg": "错误：缺少索引名称参数"})

        # 使用 ElasticSearchHandler 的 delete_index 方法
        if es_handler.delete_index(index_name):
            logger.info(f"成功删除索引 {index_name}")
            return jsonify({"code": 200, "msg": f"成功删除索引 {index_name}"})
        else:
            logger.error(f"索引 {index_name} 不存在或删除失败")
            return jsonify({"code": 500, "msg": f"索引 {index_name} 不存在或删除失败"})

    except Exception as e:
        logger.error(f"删除索引 {index_name} 失败，错误信息：{str(e)}")
        return jsonify({"code": 500, "msg": f"删除索引 {index_name} 失败，错误信息：{str(e)}"})


@app.route('/api/delete_file_index/<assistant_id>/<file_id>', methods=['POST'])
def delete_file_from_index(assistant_id, file_id):
    try:
        # 设置一个静态token
        SECRET_TOKEN = config.secret_token
        # 获取请求头中的token
        token = request.headers.get('Authorization')

        # 验证token
        if token != SECRET_TOKEN:
            logger.error("token验证失败")
            return jsonify({"code": 403, "msg": "无权限"}), 403

        # 验证参数
        if not assistant_id or not file_id:
            logger.error("错误：缺少索引名称参数")
            return jsonify({"code": 500, "msg": "错误：缺少索引名称参数"})

        # 构建查询条件
        query_body = {"query": {"term": {"file_id": file_id}}}

        # 删除存储答案
        es_handler.delete_summary_answers(file_id)

        # 执行删除操作
        response = es_handler.delete_by_query(assistant_id, query_body)
        if response['deleted'] > 0:
            logger.info(f"文档 {file_id}片段删除成功")
            return jsonify({"code": 200, "msg": "文档片段删除成功"})
        else:
            logger.error(f"文档 {file_id}删除失败")
            return jsonify({"code": 500, "msg": "文档片段删除失败"})

    except Exception as e:
        logger.error(f"删除文档片段失败，错误信息：{str(e)}")
        return jsonify({"code": 500, "msg": "删除文档片段失败"}), 500


@app.route('/api/kmc/indexing', methods=['POST'])
def indexing():
    try:
        data = request.json
        documents = data

        # 随机生成 assistantId
        assistant_id = generate_assistant_id()
        doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
        for document in documents:
            doc_id = document['documentId']
            doc_title = document['documentTitle']
            doc_content = document['documentContent']

            # 对内容进行分割
            stopwords = file_manager.load_stopwords()
            filtered_texts = set()  # 存储处理后的文本
            split_text = file_manager.spacy_chinese_text_splitter(doc_content, max_length=400)
            for text in split_text:
                words = pseg.cut(text)
                filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                if filtered_text and filtered_text not in filtered_texts:
                    filtered_texts.add(filtered_text)
                    doc_list.append({
                        'text': filtered_text,
                        'file_id': doc_id,
                        'file_name': doc_title
                    })

        # 创建索引并存储到ES
        index_name = assistant_id
        mappings = {
            "properties": {
                "text": {"type": "text", "analyzer": "standard"},
                "embed": {"type": "dense_vector", "dims": 1024},
                "assistant_id": {"type": "keyword"},
                "file_id": {"type": "keyword"},
                "file_name": {"type": "keyword"},
            }
        }
        if es_handler.index_exists(index_name):
            logger.info("索引已存在，删除索引")
            es_handler.delete_index(index_name)

        logger.info("开始创建索引")
        es_handler.es.indices.create(index=index_name, mappings=mappings)
        # 插入文档
        for item in doc_list:
            embed = es_handler.cal_passage_embed(item['text'])
            document = {
                "assistant_id": assistant_id,
                "file_id": item['file_id'],
                "file_name": item['file_name'],
                "text": item['text'],
                "embed": embed
            }
            es_handler.es.index(index=index_name, document=document)

        logger.info(f"索引 {index_name} 创建并插入索引成功")
        return jsonify({'status': 'success', 'body': {'assistantId': assistant_id}}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ST_get_answer', methods=['POST'])
def ST_answer_question():
    try:
        data = request.json
        assistant_id = data.get('assistant_id')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 3)
        llm = data.get('llm', 'cutegpt').lower()

        if not assistant_id or not query:
            return jsonify({'error': '参数不完整'}), 400
        if func == 'bm25':
            refs = es_handler.ST_search_bm25(assistant_id, query, ref_num)
        if func == 'embed':
            refs = es_handler.ST_search_embed(assistant_id, query, ref_num)

        if not refs:
            return jsonify({'error': '未找到相关文本片段'})

        prompt = prompt_builder.generate_answer_prompt(query, refs)
        if llm == 'cutegpt':
            ans = large_model_service.get_answer_from_Tyqwen(prompt)
        elif llm == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
        elif llm == 'chatgpt':
            ans = large_model_service.get_answer_from_chatgpt(prompt)
        else:
            return jsonify({'error': '未知的大模型服务'}), 400

        log_data = {'question': query,
                    'answer': ans,
                    'matches': refs}

        # 记录日志
        logger.info(f"Query processed: {log_data}")

        return jsonify({'answer': ans, 'matches': refs}), 200

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(2)]
    for t in threads:
        t.start()

    app.run(host='0.0.0.0', port=5777, debug=False)
