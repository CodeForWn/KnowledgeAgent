# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder
from transformers import AutoTokenizer, AutoModel
import json
import threading
import queue
import urllib3
import logging
import requests
from logging.handlers import RotatingFileHandler
sys.path.append("E:\\工作\\KmcGPT\\KmcGPT")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)
CORS(app)
# 加载配置
# 使用环境变量指定环境并加载配置
config = Config(env='development')
config.load_config()  # 指定配置文件的路径
logger = config.logger
backend_notify_api = config.external_api_backend_notify
# 创建 FileManager 实例
file_manager = FileManager(config)
# 创建ElasticSearchHandler实例
es_handler = ElasticSearchHandler(config)
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)
# 创建队列
file_queue = queue.Queue()
logger.info('服务启动中。。。')


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
            logger.info("开始下载文件并处理: %s", file_name)
            pdf_path = file_manager.download_pdf(download_path, file_id)
            doc_list = file_manager.process_pdf_file(pdf_path)

            if not doc_list:
                notify_backend(file_id, "FAILURE", "未能成功处理PDF文件")
                logger.error("未能成功处理PDF文件: %s", file_id)
                return jsonify({"status": "error", "message": "未能成功处理PDF文件"})

            index_name = f'{assistant_id}_{file_id}'
            es_handler.create_index(index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path)
            es_handler.notify_backend(file_id, "SUCCESS")
        except Exception as e:
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
    llm = data.get('llm', 'cutegpt')  # 默认使用CuteGPT
    # 获取历史对话内容
    history = prompt_builder.get_history(session_id, token)
    # 构建新的prompt
    prompt = prompt_builder.generate_open_answer_prompt(query, history)

    answer = ''
    if llm.lower() == 'cutegpt':
        answer = large_model_service.get_answer_from_cute_gpt(prompt)
    elif llm.lower() == 'chatglm':
        task_id = large_model_service.async_invoke_chatglm(prompt)
        answer = large_model_service.query_async_result_chatglm(task_id)

    return jsonify({'answer': answer, 'matches': []}), 200


@app.route('/api/get_answer', methods=['POST'])
def answer_question():
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
            refs = es_handler.search_bm25(assistant_id, query, ref_num)
        if func == 'embed':
            refs = es_handler.search_embed(assistant_id, query, ref_num)

        if not refs:
            return jsonify({'error': '未找到相关文本片段'})

        prompt = prompt_builder.generate_answer_prompt(query, refs)
        if llm == 'cutegpt':
            ans = large_model_service.get_answer_from_cute_gpt(prompt)
        elif llm == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
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


@app.route('/api/generate_summary_and_questions', methods=['POST'])
def generate_summary_and_questions():
    try:
        data = request.json
        file_id = data['file_id']
        ref_num = data.get('ref_num', 5)

        # 检查是否已有存储的答案
        existing_answer = es_handler._search_("answers_index", {"query": {"term": {"file_id": file_id}}})
        if 'hits' in existing_answer and 'hits' in existing_answer['hits'] and existing_answer['hits']['hits']:
            logger.info(f"找到了文件ID {file_id} 的存储答案")
            stored_answer = existing_answer['hits']['hits'][0]['_source']['sum_rec']
            logger.info(f"存储答案为 {stored_answer}")
            return jsonify({'answer': stored_answer, 'matches': []}), 200

        logger.info(f"正在查询文件ID {file_id} 的前 {ref_num} 段文本")
        results = es_handler._search_('_all', {"query": {"term": {"file_id": file_id}}}, ref_num)

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
            return jsonify({"code": 403, "msg": "无权限"}), 403

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


if __name__ == '__main__':
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(2)]
    for t in threads:
        t.start()

    app.run(host='0.0.0.0', port=5777, debug=False)