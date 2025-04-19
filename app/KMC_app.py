# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder
import types
import time
import os


class NoRequestStatusFilter(logging.Filter):
    def filter(self, record):
        return "/api/request_status" not in record.getMessage()


# 为 werkzeug 添加过滤器
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.addFilter(NoRequestStatusFilter())


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
rerank_model_path = config.rerank_model_path

# 创建队列
file_queue = queue.Queue()
# 创建全局锁，确保一次只有一个文件处理
file_processing_lock = threading.Lock()
index_lock = threading.Lock()
logger.info('服务启动中。。。')

# 初始化请求状态字典
request_status = {}
request_lock = threading.Lock()


def cleanup_request_status():
    with request_lock:
        current_time = time.time()
        for req_id in list(request_status.keys()):
            if current_time - request_status[req_id]['start_time'] > 120:  # 超过一小时的记录
                del request_status[req_id]


@app.before_request
def before_request():
    cleanup_request_status()
    request_id = str(uuid.uuid4())  # 生成唯一请求ID
    request.environ['REQUEST_ID'] = request_id  # 将请求ID存储在请求环境中
    with request_lock:
        request_status[request_id] = {
            "start_time": time.time(),
            "status": "processing",
            "url": request.url
        }
    # logger.info(f"开始处理请求: {request_id} for {request.url}")


@app.after_request
def after_request(response):
    request_id = request.environ.get('REQUEST_ID')
    with request_lock:
        if request_id in request_status:
            # 如果请求仍然处于 "processing" 状态，则将其更新为 "completed"
            if request_status[request_id]["status"] == "processing":
                request_status[request_id]["end_time"] = time.time()
                request_status[request_id]["status"] = "completed"
                # 日志记录（可选）
                # logger.info(f"完成请求: {request_id} for {request.url}")
    return response


@app.teardown_request
def teardown_request(exception):
    request_id = request.environ.get('REQUEST_ID')
    with request_lock:
        # 检查请求状态是否仍然是 "processing"，以防止重复更新
        if request_id in request_status and request_status[request_id]["status"] == "processing":
            request_status[request_id]["end_time"] = time.time()
            request_status[request_id]["status"] = "failed" if exception else "completed"
            # 日志记录失败的请求
            logger.info(f"请求失败: {request_id} for {request.url} due to {exception}")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/request_status', methods=['GET'])
def get_request_status():
    with request_lock:
        # 过滤 active_requests 中的 /api/request_status 请求
        active_requests = {k: v for k, v in request_status.items() if v['status'] == 'processing' and '/api/get_answer_stream' in v['url']}
        # 过滤 completed_requests 中的 /api/request_status 请求
        completed_requests = {k: v for k, v in request_status.items() if v['status'] == 'completed' and '/api/get_answer_stream' in v['url']}
    return jsonify({
        'active_requests': active_requests,
        'completed_requests': completed_requests
    }), 200


@app.route('/api/monitor')
def serve_monitor_page():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'request_monitor.html')


def generate_assistant_id():
    # 生成一个随机的UUID并转换为字符串
    return str(uuid.uuid4())


@app.route('/')
def hello_world():
    return 'Hello, World!'


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

processed_files = set()  # 用于追踪已经处理的文件
def _push(file_data):
    global file_queue, processed_files
    if file_data['file_id'] not in processed_files:
        file_queue.put(file_data)
        processed_files.add(file_data['file_id'])
    else:
        logger.info(f"文件 {file_data['file_id']} 已经处理过，跳过")


def _process_file_data(data):
    try:
        with file_processing_lock:
            with app.app_context():
                user_id = data.get('user_id')
                assistant_id = data.get('assistant_id')
                file_id = data.get('file_id')
                file_name = data.get('file_name')
                download_path = data.get('download_path')
                tenant_id = data.get('tenant_id')
                tag = data.get('tag')
                createTime = data.get('createTime')

                logger.info("开始下载文件并处理: %s，文件路径：%s", file_name, download_path)
                file_path = file_manager.download_pdf(download_path, file_id)
                doc_list = file_manager.process_pdf_file(file_path, file_name)

                if not doc_list:
                    notify_backend(file_id, "FAILURE", "未能成功处理PDF文件")
                    logger.error("未能成功处理PDF文件: %s", file_id)
                    return

                index_name = assistant_id
                try:
                    createTime_int = int(createTime)
                except ValueError as ve:
                    logger.error(f"Invalid createTime value: {createTime}. Error: {ve}")
                    es_handler.notify_backend(file_id, "FAILURE", f"Invalid createTime value: {createTime}")
                    return

                es_handler.create_index(index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path, tag, createTime_int)
                es_handler.notify_backend(file_id, "SUCCESS")
                logger.info("文件处理完成并创建索引: %s", file_name)

            # 在文件处理完成后添加 3 秒间隔
            time.sleep(3)

    except Exception as e:
        logger.error("处理文件数据失败: %s", e)
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
    logger.info("文件数据已接收，准备处理: %s", data)
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


@app.route('/api/get_open_ans_stream', methods=['POST'])
def get_open_ans_stream():
    data = request.json
    session_id = data.get('session_id')
    token = data.get('token')
    query = data.get('query')
    llm = data.get('llm', 'qwen')  # 默认使用 qwen
    top_p = data.get('top_p', 0.8)
    temperature = data.get('temperature', 0)
    # 获取历史对话内容
    history = prompt_builder.get_history(session_id, token)
    logger.info(f"历史对话：{history}")
    # 构建新的 prompt
    prompt = prompt_builder.generate_open_answer_prompt(query, history)
    logger.info(f"prompt:{prompt}")

    if llm.lower() == 'qwen':
        response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
        return Response(response_generator, content_type='text/plain; charset=utf-8')

    if llm.lower() == 'deepseek':
        response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
        return Response(response_generator, content_type='text/plain; charset=utf-8')
    else:
        # 非流式模型或其他模型的处理
        if llm.lower() == 'cutegpt':
            answer = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
        elif llm.lower() == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            answer = large_model_service.query_async_result_chatglm(task_id)
        elif llm.lower() == 'chatgpt':
            answer = large_model_service.get_answer_from_chatgpt(prompt)
        else:
            answer = "Unsupported model"

        return jsonify({'answer': answer, 'matches': []}), 200


@app.route('/api/get_answer_stream_old', methods=['POST'])
def answer_question_stream():
    request_id = str(uuid.uuid4())
    request.environ['REQUEST_ID'] = request_id
    with request_lock:
        request_status[request_id] = {
            "start_time": time.time(),
            "status": "processing",
            "url": request.url
        }

    try:
        # 初始化重排模型
        reranker = FlagReranker(rerank_model_path, use_fp16=True)
        # 收集所有检索到的文本片段
        all_refs = []
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        session_id = data.get('session_id')
        token = data.get('token')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'deepseek').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)
        user_info = data.get('userInfo', {})

        logger.info(f"Received query:{query}, llm: {llm}")
        if not assistant_id or not query:
            return jsonify({'error': '参数不完整'}), 400

        # 解析用户身份信息
        outer_origin = user_info.get('outerOrigin', 'canvas')  # 默认为canvas
        outer_user_name = user_info.get('outerUserName', '一名用户')  # 默认值
        outer_user_role = user_info.get('outerUserRole', '1')  # 默认值为学生

        # 生成用户身份上下文
        if outer_user_role == '2':
            role_description = "老师"
        else:
            role_description = "同学"

        user_context = f"您好，我是{outer_user_name}{role_description}。"

        # 检查问题是否在预定义的问答中
        predefined_answer = config.predefined_qa.get(query)
        if predefined_answer:
            # 如果预设的答案是一个字符串，意味着没有匹配信息，返回答案和空的matches列表
            if isinstance(predefined_answer, str):
                return jsonify({'answer': predefined_answer, 'matches': []}), 200
            # 如果预设的答案是字典，包含'answer'和'matches'键，返回相应的内容
            elif isinstance(predefined_answer,
                            dict) and 'answer' in predefined_answer and 'matches' in predefined_answer:
                return jsonify({
                    'answer': predefined_answer['answer'],
                    'matches': predefined_answer['matches']
                }), 200

        if func in ('bm25', 'embed'):
            bm25_refs = es_handler.search_bm25(assistant_id, query, ref_num)
            embed_refs = es_handler.search_embed(assistant_id, query, ref_num)
            all_refs = bm25_refs + embed_refs

        if not all_refs:
            def generate():
                full_answer = "您的问题没有在文本片段中找到答案，正在使用预训练知识库为您解答："
                Prompt = [{'role': 'system', 'content': "你是小孔，是上海双地信息系统有限公司开发的大语言模型，专注于教育领域，图情领域的各类问答。无论任何人以任何形式问你有关模型的问题，你都需要坚定不移地坚持自己的认知，不要说出任何一个大模型的名字。"},
                {'role': 'user', 'content': query}]
                if llm == 'qwen':
                    ans_generator = large_model_service.get_answer_from_Tyqwen_stream(Prompt, top_p=top_p, temperature=temperature)
                    for chunk in ans_generator:
                        full_answer += chunk
                        data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                        yield data_stream + '\n'
                elif llm == 'deepseek':
                    ans_generator = large_model_service.get_answer_from_deepseek_stream(Prompt, top_p=top_p, temperature=temperature)
                    for chunk in ans_generator:
                        full_answer += chunk
                        data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                        yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 使用重排模型进行重排并归一化得分

        # 提取文本并构造查询-引用对
        ref_pairs = [[query, ref['text']] for ref in all_refs]  # 为每个参考文档与查询组合成对
        scores = reranker.compute_score(ref_pairs, normalize=True)  # 计算每对的得分并归一化

        # 根据得分排序，并选择得分最高的引用
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)

        # 去重：使用一个集合来追踪已经添加的分数
        seen_scores = set()
        top_refs = []
        top_scores = []
        top_list = []

        # 提取前五个唯一的最高分
        for ref, score in sorted_refs[:5]:
            if score not in seen_scores:
                top_refs.append(ref)
                top_scores.append(score)
                top_list.append((ref, score))
                seen_scores.add(score)

        # 确保结果最多只有5个
        top_refs = top_refs[:5]
        top_scores = top_scores[:5]
        top_list = top_list[:5]
        logger.info(f"重排后最高分：{top_scores}")
        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)
        logger.info(f"session_id:{session_id}, token:{token}")
        logger.info(f"历史对话：{history}")

        # 检查最高分数是否低于0.3
        if top_scores[0] < 0.3:
            prompt = prompt_builder.generate_answer_prompt_un_refs(query, history, user_context)
            matches = []
        else:
            matches = [{
                'text': ref['text'],
                'original_text': ref['original_text'],
                'page': ref['page'],
                'file_id': ref['file_id'],
                'file_name': ref['file_name'],
                'download_path': ref['download_path'],
                'score': ref['score'],
                'rerank_score': score
            } for ref, score in top_list]

            # 🔥 文件级全文召回逻辑
            file_ids = list({ref['file_id'] for ref in top_refs})
            full_context_refs = []
            seen_chars = 0

            logger.info(f"[全文召回] 即将召回以下 file_id 的全文内容：{file_ids}")

            for file_id in file_ids:
                content = es_handler.get_full_text_by_file_id(assistant_id.strip(), file_id.strip())
                if not content:
                    logger.warning(f"[全文召回] file_id={file_id} 内容为空，跳过。")
                    continue

                content_len = len(content)
                if seen_chars + content_len > 10000:
                    logger.info(
                        f"[全文召回] file_id={file_id} 内容超限（{content_len}字），当前已用{seen_chars}字，跳过该文件。")
                    break

                logger.info(f"[全文召回] file_id={file_id} 内容长度：{content_len}，当前累计：{seen_chars}，添加中。")
                full_context_refs.append({
                    'text': f"以下是文件（{file_id}）的完整内容：\n{content}",
                    'file_id': file_id
                })
                seen_chars += content_len

            logger.info(f"[全文召回] 最终用于提示词拼接的全文数：{len(full_context_refs)}，总长度：{seen_chars}")

            prompt = prompt_builder.generate_answer_prompt(
                query=query,
                refs=full_context_refs,
                history=history,
                user_context=user_context
            )

        if llm == 'qwen':
            logger.info("正在使用通义千问......")
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    try:
                        data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    except Exception as e:
                        yield f"Error during JSON encoding: {e}\n"
                        continue  # 跳过错误，继续处理下一个 chunk

                # 流结束后，执行日志记录
                log_data = {'question': query, 'answer': full_answer, 'matches': matches}
                logger.info(f"问答记录: {log_data}")
                with open(record_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

                # 更新请求状态为完成
                with request_lock:
                    if request_id in request_status:
                        request_status[request_id]["end_time"] = time.time()
                        request_status[request_id]["status"] = "completed"
                    yield data_stream + '\n'  # 每次生成一个新的 chunk

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        elif llm == 'deepseek':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    try:
                        data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    except Exception as e:
                        yield f"Error during JSON encoding: {e}\n"
                        continue  # 跳过错误，继续处理下一个 chunk

                    yield data_stream + '\n'  # 每次生成一个新的 chunk

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

        log_data = {'question': query,
                    'answer': full_answer,
                    'matches': matches}

        logger.info(f"问答记录: {log_data}")
        with open(record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

        # 更新请求状态为完成
        with request_lock:
            if request_id in request_status:
                request_status[request_id]["end_time"] = time.time()
                request_status[request_id]["status"] = "completed"

        return jsonify({'answer': full_answer, 'matches': matches}), 200

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        # 更新请求状态为失败
        with request_lock:
            if request_id in request_status:
                request_status[request_id]["end_time"] = time.time()
                request_status[request_id]["status"] = "failed"
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_answer_stream', methods=['POST'])
def answer_question_stream_new():
    request_id = str(uuid.uuid4())
    request.environ['REQUEST_ID'] = request_id

    with request_lock:
        request_status[request_id] = {
            "start_time": time.time(),
            "status": "processing",
            "url": request.url
        }

    try:
        data = request.json
        assistant_id = data.get('assistant_id')
        session_id = data.get('session_id')
        token = data.get('token')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'deepseek').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)
        user_info = data.get('userInfo', {})

        if not assistant_id or not query:
            return jsonify({'error': '参数不完整'}), 400

        outer_user_name = user_info.get('outerUserName', '一名用户')
        outer_user_role = user_info.get('outerUserRole', '1')
        role_description = "老师" if outer_user_role == '2' else "同学"
        user_context = f"您好，我是{outer_user_name}{role_description}。"

        predefined_answer = config.predefined_qa.get(query)
        if predefined_answer:
            if isinstance(predefined_answer, str):
                return jsonify({'answer': predefined_answer, 'matches': []}), 200
            elif isinstance(predefined_answer, dict):
                return jsonify(predefined_answer), 200

        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)

        all_refs = []
        if func in ('bm25', 'embed'):
            bm25_refs = es_handler.search_bm25(assistant_id, query, ref_num)
            embed_refs = es_handler.search_embed(assistant_id, query, ref_num)
            all_refs = bm25_refs + embed_refs

        if not all_refs:
            matches = []
            prompt = prompt_builder.generate_answer_prompt_un_refs(query, history, user_context)
        else:
            reranker = FlagReranker(rerank_model_path, use_fp16=True)
            ref_pairs = [[query, ref['text']] for ref in all_refs]
            scores = reranker.compute_score(ref_pairs, normalize=True)
            sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)

            seen_texts = set()
            top_list = []
            for ref, score in sorted_refs:
                if ref['text'] not in seen_texts:
                    seen_texts.add(ref['text'])
                    top_list.append((ref, score))
                if len(top_list) >= 5:
                    break

            top_refs, top_scores = zip(*top_list) if top_list else ([], [])

            matches = [{
                'text': ref['text'],
                'original_text': ref['original_text'],
                'page': ref['page'],
                'file_id': ref['file_id'],
                'file_name': ref['file_name'],
                'download_path': ref['download_path'],
                'score': ref['score'],
                'rerank_score': score
            } for ref, score in top_list]

            if not top_scores or top_scores[0] < 0.3:
                prompt = prompt_builder.generate_answer_prompt_un_refs(query, history, user_context)
                matches = []
            else:
                # ✅ 将前5个ref的file_id去重，构造成 ref 结构列表
                file_ids = list({ref['file_id'] for ref in top_refs})
                full_context_refs = []
                seen_chars = 0

                logger.info(f"即将召回以下 file_id 的全文内容：{file_ids}")

                for file_id in file_ids:
                    content = es_handler.get_full_text_by_file_id(assistant_id.strip(), file_id.strip())
                    if not content:
                        logger.warning(f"file_id = {file_id} 的全文为空，跳过。")
                        continue

                    content_len = len(content)
                    if seen_chars + len(content) > 15000:
                        logger.info(f"file_id = {file_id} 的全文超出总长度限制（已用 {seen_chars} 字符，将跳过该全文）")
                        break

                    logger.info(f"file_id = {file_id} 的全文长度为 {content_len} 字符，当前累计 {seen_chars}，即将添加")
                    full_context_refs.append({
                        'text': f"以下是文件（{file_id}）的完整内容：\n{content}",
                        'file_id': file_id
                    })
                    seen_chars += len(content)

                # ✅ 用全文作为上下文构建prompt（结构保持和top_refs一致）
                prompt = prompt_builder.generate_answer_prompt(
                    query=query,
                    refs=full_context_refs,
                    history=history,
                    user_context=user_context
                )

        def generate_stream(model_name, prompt, matches, top_p, temperature, query, request_id):
            full_answer = ""
            ans_generator = (large_model_service.get_answer_from_Tyqwen_stream if model_name == 'qwen' else large_model_service.get_answer_from_deepseek_stream)(prompt, top_p, temperature)

            for chunk in ans_generator:
                full_answer += chunk
                yield json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False) + '\n'

            log_data = {'question': query, 'answer': full_answer, 'matches': matches}
            logger.info(f"问题: {query}, 答案: {full_answer}")
            with open(record_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

            with request_lock:
                request_status[request_id].update({"end_time": time.time(), "status": "completed"})

        return Response(
            stream_with_context(generate_stream(llm, prompt, matches, top_p, temperature, query, request_id)),
            content_type='application/json; charset=utf-8'
        )

    except Exception as e:
        logger.error(f"Error in answer_question_stream: {e}")
        with request_lock:
            request_status[request_id].update({"end_time": time.time(), "status": "failed"})
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_answer_by_file_id', methods=['POST'])
def answer_question_by_file_id():
    try:
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        session_id = data.get('session_id')
        token = data.get('token')
        query = data.get('query')
        file_id_list = data.get('file_id')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)
        memory_time = data.get('memory_time', 3)  # 新增参数

        if not assistant_id or not query or not file_id_list:
            return jsonify({'error': '参数不完整'}), 400

        # 检查问题是否在预定义的问答中
        predefined_answer = config.predefined_qa.get(query)
        if predefined_answer:
            if isinstance(predefined_answer, str):
                return jsonify({'answer': predefined_answer, 'matches': []}), 200
            elif isinstance(predefined_answer, dict) and 'answer' in predefined_answer and 'matches' in predefined_answer:
                return jsonify({
                    'answer': predefined_answer['answer'],
                    'matches': predefined_answer['matches']
                }), 200

        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)
        if len(history) > memory_time:
            trimmed_history = history[-memory_time:]
        else:
            trimmed_history = history

        # 🔥 直接按 file_id_list 召回全文，作为上下文
        full_context_refs = []
        seen_chars = 0

        logger.info(f"开始按 file_id_list 直接召回全文内容：{file_id_list}")

        for file_id in file_id_list:
            content = es_handler.get_full_text_by_file_id(assistant_id.strip(), file_id.strip())
            if not content:
                logger.warning(f"file_id = {file_id} 的全文内容为空，跳过。")
                continue

            content_len = len(content)
            if seen_chars + content_len > 10000:
                logger.info(f"file_id = {file_id} 的全文超限（当前累计 {seen_chars} 字符，跳过）。")
                break

            logger.info(f"file_id = {file_id} 的全文长度为 {content_len}，将加入上下文。")
            full_context_refs.append({
                'text': f"以下是文件（{file_id}）的完整内容：\n{content}",
                'file_id': file_id
            })
            seen_chars += content_len

        logger.info(f"最终用于prompt构建的全文数：{len(full_context_refs)}，累计字符数：{seen_chars}")

        # 构造提示词
        prompt = prompt_builder.generate_prompt_for_file_id(query, full_context_refs, trimmed_history)

        if llm.lower() == 'qwen':
            response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

        if llm.lower() == 'deepseek':
            response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.error(f"Error in answer_question_by_file_id: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_answer', methods=['POST'])
def answer_question():
    try:
        # 初始化重排模型
        reranker = FlagReranker(rerank_model_path, use_fp16=True)
        # 收集所有检索到的文本片段
        all_refs = []
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        query = data.get('query')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

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
            prompt = [{'role': 'user', 'content': query}]
            ans = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
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
            # old_answer = large_model_service.get_answer_from_Qwen(prompt)
            # beauty_prompt = prompt_builder.generate_beauty_prompt(old_answer)
            # ans = large_model_service.get_answer_from_Qwen(beauty_prompt)
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
            logger.info(f"生成的总结prompt为 {prompt}")
            ans = large_model_service.get_answer_from_Tyqwen(prompt)
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


@app.route('/api/kmc/ST_indexing_by_step', methods=['POST'])
def indexing_by_step():
    try:
        data = request.json
        documents = data.get('documents', [])
        assistant_id = data.get('assistantId')

        # 检查是否传入了 assistantId，如果没有则生成一个新的
        if not assistant_id:
            assistant_id = generate_assistant_id()

            # 创建新的索引
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
            # 检查索引是否存在
            if not es_handler.index_exists(index_name):
                logger.info("开始创建索引")
                es_handler.es.indices.create(index=index_name, mappings=mappings)
        else:
            index_name = assistant_id

        doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
        stopwords = file_manager.load_stopwords()

        # 文档元数据的mapping
        metadata_index = 'document_metadata'
        metadata_mappings = {
            "properties": {
                "file_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "abstract": {"type": "text", "analyzer": "standard"},
                "year": {"type": "keyword"},
                "publisher": {"type": "keyword"},
                "author": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "abstract_embed": {"type": "dense_vector", "dims": 1024}
            }
        }

        # 建立文档元数据索引
        if not es_handler.index_exists(metadata_index):
            logger.info("创建文档元数据索引")
            es_handler.es.indices.create(index=metadata_index, mappings=metadata_mappings)

        for document in documents:
            file_id = document['documentId']
            doc_titles = document.get('TI', [])
            doc_content = document.get('documentContent', '')
            doc_abstract = document.get('Abstract_F', '')
            split_text = file_manager.spacy_chinese_text_splitter(doc_content, max_length=600)
            # 如果摘要为空，从文件的前两段和后两段生成摘要
            if not doc_abstract:
                logger.info("摘要为空，生成摘要")
                if len(split_text) >= 4:
                    ref_list = split_text[:2] + split_text[-2:]
                else:
                    ref_list = [doc_content]  # 不足四段使用全文

                abstract_prompt = prompt_builder.generate_abstract_prompt(ref_list)
                doc_abstract = large_model_service.get_answer_from_Tyqwen(abstract_prompt)  # 使用大模型生成摘要
                logger.info(f"生成摘要成功，文档ID: {file_id}, 摘要: {doc_abstract}")
            year = document.get('Year', '')
            publisher = document.get('LiteratureTitle_F', '')
            author = document.get('Author_1', '')

            doc_title = ' '.join(doc_titles)
            logger.info(f"处理文档 {file_id}, 标题: {doc_title}")

            # 生成摘要的嵌入向量
            abstract_embed = es_handler.cal_passage_embed(doc_abstract)

            # 插入文档元数据
            metadata_document = {
                "file_id": file_id,
                "title": doc_title,
                "abstract": doc_abstract,
                "year": year,
                "publisher": publisher,
                "author": author,
                "content": doc_content,
                "abstract_embed": abstract_embed
            }
            es_handler.es.index(index=metadata_index, id=file_id, document=metadata_document)
            # logger.info(f"插入文档元数据: {metadata_document}")

            # 对内容进行分割
            filtered_texts = set()  # 存储处理后的文本

            for text in split_text:
                words = pseg.cut(text)
                filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                if filtered_text and filtered_text not in filtered_texts:
                    filtered_texts.add(filtered_text)
                    doc_list.append({
                        'text': filtered_text,
                        'file_id': file_id,
                        'file_name': doc_title
                    })

        # 插入文档片段
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

        logger.info(f"索引 {index_name} 插入文档成功")
        return jsonify({'status': 'success', 'body': {'assistantId': assistant_id}}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/kmc/ST_indexing', methods=['POST'])
def indexing():
    try:
        data = request.json
        documents = data

        # 随机生成 assistantId
        assistant_id = generate_assistant_id()
        doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
        stopwords = file_manager.load_stopwords()

        # document索引名称建立
        metadata_index = 'document_metadata'

        # 文档元数据的mapping
        metadata_mappings = {
            "properties": {
                "file_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "abstract": {"type": "text", "analyzer": "standard"},
                "year": {"type": "keyword"},
                "publisher": {"type": "keyword"},
                "author": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "abstract_embed": {"type": "dense_vector", "dims": 1024}
            }
        }

        # 文本片段索引mapping
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

        # 建立文档元数据索引
        if not es_handler.index_exists(metadata_index):
            logger.info("创建文档元数据索引")
            es_handler.es.indices.create(index=metadata_index, mappings=metadata_mappings)
        # 建立文本片段索引
        if es_handler.index_exists(index_name):
            logger.info("索引已存在，删除索引")
            es_handler.delete_index(index_name)

        logger.info("开始创建索引")
        es_handler.es.indices.create(index=index_name, mappings=mappings)

        for document in documents:
            file_id = document['documentId']
            doc_titles = document.get('TI', [])
            doc_content = document.get('documentContent', '')
            doc_abstract = document.get('Abstract_F', '')
            split_text = file_manager.spacy_chinese_text_splitter(doc_content, max_length=600)
            # 如果摘要为空，从文件的前两段和后两段生成摘要
            if not doc_abstract:
                logger.info("摘要为空，生成摘要")
                if len(split_text) >= 4:
                    ref_list = split_text[:2] + split_text[-2:]
                else:
                    ref_list = [doc_content]  # 不足四段使用全文

                abstract_prompt = prompt_builder.generate_abstract_prompt(ref_list)
                doc_abstract = large_model_service.get_answer_from_Tyqwen(abstract_prompt)  # 使用大模型生成摘要
                logger.info(f"生成摘要成功，文档ID: {file_id}, 摘要: {doc_abstract}")
            year = document.get('Year', '')
            publisher = document.get('LiteratureTitle_F', '')
            author = document.get('Author_1', '')

            doc_title = ' '.join(doc_titles)
            logger.info(f"处理文档 {file_id}, 标题: {doc_title}")

            # 生成摘要的嵌入向量
            abstract_embed = es_handler.cal_passage_embed(doc_abstract)

            # 插入文档元数据
            metadata_document = {
                "file_id": file_id,
                "title": doc_title,
                "abstract": doc_abstract,
                "year": year,
                "publisher": publisher,
                "author": author,
                "content": doc_content,
                "abstract_embed": abstract_embed
            }
            es_handler.es.index(index=metadata_index, id=file_id, document=metadata_document)
            # logger.info(f"插入文档元数据: {metadata_document}")

            # 对内容进行分割
            filtered_texts = set()  # 存储处理后的文本

            for text in split_text:
                words = pseg.cut(text)
                filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                if filtered_text and filtered_text not in filtered_texts:
                    filtered_texts.add(filtered_text)
                    doc_list.append({
                        'text': filtered_text,
                        'file_id': file_id,
                        'file_name': doc_title
                    })

        # 插入文档片段
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
            refs = es_handler.ST_search_abstract_bm25(query, ref_num)
        if func == 'embed':
            refs = es_handler.ST_search_abstract_embed(query, ref_num)

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


@app.route('/api/ST_get_file', methods=['POST'])
def ST_search_file():
    try:
        # 初始化重排模型
        reranker = FlagReranker(rerank_model_path, use_fp16=True)
        data = request.json
        query = data.get('query')
        ref_num = data.get('ref_num', 5)

        if not query:
            return jsonify({'status': 'error', 'message': '缺少问题'}), 400

        # 在摘要字段中进行BM25和嵌入搜索，获取匹配的文档ID列表
        bm25_file_ids = es_handler.ST_search_abstract_bm25(query, ref_num)
        logger.info(f"BM25 file IDs: {bm25_file_ids}")
        embed_file_ids = es_handler.ST_search_abstract_embed(query, ref_num)
        logger.info(f"Embed file IDs: {embed_file_ids}")

        all_file_ids = list(set(bm25_file_ids + embed_file_ids))  # 去重

        if not all_file_ids:
            return jsonify({'file_ids': all_file_ids}), 200

        # 增加日志记录
        logger.info(f"All file IDs: {all_file_ids}")

        # 获取所有匹配文档的详细信息
        all_refs = []
        for file_id in all_file_ids:
            try:
                doc = es_handler.es.get(index='document_metadata', id=file_id)
                if doc and '_source' in doc:
                    source = doc['_source']
                    all_refs.append({
                        'file_id': source.get('file_id', ''),
                        'title': source.get('title', ''),
                        'abstract': source.get('abstract', ''),
                        'year': source.get('year', ''),
                        'publisher': source.get('publisher', ''),
                        'author': source.get('author', ''),
                        'content': source.get('content', '')
                    })
                else:
                    logger.error(f"Document with file_id {file_id} not found in index 'document_metadata'")
            except Exception as e:
                logger.error(f"Error retrieving document with file_id {file_id}: {e}")

        if not all_refs:
            return jsonify({'file_ids': all_file_ids}), 200

        # 如果只有一个引用对，直接返回结果
        if len(all_refs) == 1:
            return jsonify({
                'status': 'success',
                'file_ids': [all_refs[0]['file_id']],
                'details': all_refs
            }), 200

        # 使用重排模型进行重排并归一化得分
        ref_pairs = [[query, ref['abstract']] for ref in all_refs]  # 为每个参考文档与查询组合成对
        scores = reranker.compute_score(ref_pairs, normalize=True)  # 计算每对的得分并归一化
        if not isinstance(scores, list):
            logger.error(f"Scores should be a list but got {type(scores)}: {scores}")
            return jsonify({'status': 'error', 'message': 'Invalid score format'}), 500

        # 根据得分排序，并选择得分最高的引用
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        top_refs = [ref for ref, _ in sorted_refs[:ref_num]]  # 假设您只需要前ref_num个最相关的引用
        top_scores = [score for _, score in sorted_refs[:ref_num]]

        # 打印出这些分数
        logger.info(f"Top scores: {top_scores}")

        # 返回结果
        return jsonify({
            'status': 'success',
            'file_ids': [ref['file_id'] for ref in top_refs],
            'details': top_refs
        }), 200

    except Exception as e:
        logger.error(f"Error in ST_search_file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/title_rewrite', methods=['POST'])
def title_generation():
    data = request.json
    session_id = data.get('sessionId')
    question = data.get('question')
    answer = data.get('answer')
    try:
        content = f"问题：{question}\n回答：{answer}\n"
        final_prompt = prompt_builder.generate_title_prompt(content)
        title = large_model_service.get_answer_from_Tyqwen(final_prompt)
        logger.info(f"Title generated: {title}")
        return jsonify({
            "code": 200,  # 状态码
            "data": {
                "label": title
            }
        })
    except Exception as e:
        logger.error(f'Error in session {session_id} during title_generation: {e}')
        return jsonify({
            "code": 500,  # 服务器内部错误状态码
            "data": {
                "label": question
            }
        })


@app.route('/SnoopIE', methods=['POST'])
def get_snoopIE_ans():
    data = request.json
    query = data.get('query')
    data = {
        "messages": [
            {"role": "user", "content": query},
        ]
    }
    # 发送POST请求到接口
    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 将数据转换为JSON格式
    json_data = json.dumps(data)

    # 发送POST请求
    response = requests.post(api_url, headers=headers, data=json_data)
    # 检查响应状态码
    if response.status_code == 200:
        # 获取响应数据
        response_data = response.json()
        # 提取模型回答的content字段
        result_data = response_data['choices'][0]['message']['content']
        print(f"模型回答: {result_data}")
        return result_data
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


@app.route('/api/ST_get_answer_by_file_id', methods=['POST'])
def ST_answer_question_by_file_id():
    try:
        # 初始化重排模型
        reranker = FlagReranker(rerank_model_path, use_fp16=True)
        # 收集所有检索到的文本片段
        metadata_refs = []
        all_refs = []
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        session_id = data.get('session_id')
        token = data.get('token')
        query = data.get('query')
        file_id_list = data.get('file_id')
        ref_num = data.get('ref_num', 3)
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        if not assistant_id or not query or not file_id_list:
            return jsonify({'error': '参数不完整'}), 400

        # 检查问题是否在预定义的问答中
        predefined_answer = config.predefined_qa.get(query)
        if predefined_answer:
            if isinstance(predefined_answer, str):
                return jsonify({'answer': predefined_answer, 'matches': []}), 200
            elif isinstance(predefined_answer, dict) and 'answer' in predefined_answer and 'matches' in predefined_answer:
                return jsonify({
                    'answer': predefined_answer['answer'],
                    'matches': predefined_answer['matches']
                }), 200

        # 搜索只在给定的 file_id 的文件内容中进行
        bm25_refs = es_handler.ST_search_bm25(assistant_id, query, ref_num, file_id_list=file_id_list)
        embed_refs = es_handler.ST_search_embed(assistant_id, query, ref_num, file_id_list=file_id_list)
        all_refs = bm25_refs + embed_refs

        if not all_refs:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                Prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(Prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 获取所有匹配文档的详细信息
        for file_id in file_id_list:
            try:
                doc = es_handler.es.get(index='document_metadata', id=file_id)
                if doc:
                    source = doc['_source']
                    metadata_refs.append({
                        'file_id': str(file_id),
                        'title': source.get('title', ''),
                        'year': source.get('year', ''),
                        'publisher': source.get('publisher', ''),
                        'author': source.get('author', ''),
                    })
            except Exception as e:
                logger.error(f"没有找到该文献的文本信息 {file_id}: {e}")

        logger.info(f"Metadata references collected: {metadata_refs}")

        # 使用重排模型进行重排并归一化得分
        ref_pairs = [[query, ref['text']] for ref in all_refs]
        scores = reranker.compute_score(ref_pairs, normalize=True)
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        top_list = sorted_refs[:ref_num]
        top_scores = [score for _, score in sorted_refs[:5]]
        logger.info(f"Top scores: {top_scores}")
        top_refs = [ref for ref, score in top_list]

        # 检查 top_refs 中的 file_id
        for ref in top_refs:
            if 'file_id' not in ref:
                logger.error(f"Missing file_id in reference: {ref}")

        # 将元数据与文本片段信息合并
        merged_refs = []
        for ref in top_refs:
            file_id = str(ref.get('file_id'))  # 确保 file_id 是字符串类型
            if not file_id:
                logger.error(f"No file_id found in reference: {ref}")
                continue
            # 找到对应的元数据
            metadata = next((metadata for metadata in metadata_refs if metadata['file_id'] == file_id), None)
            if metadata:
                # 合并元数据和文本片段信息
                merged_ref = {**ref, **metadata}
                merged_refs.append(merged_ref)
            else:
                logger.warning(f"No metadata found for file_id {file_id}")

        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)
        # logger.info(f"Session ID: {session_id}")
        # logger.info(f"Token: {token}")
        logger.info(f"History: {history}")
        # 初始化默认的prompt和matches
        prompt = None
        matches = []

        # 获取之前的查询和对应的结果
        previous_queries = []
        if history:
            for item in history:
                if 'question' in item and 'documents' in item:
                    previous_queries.append((item['question'], item['documents']))

        # 检查最高分数是否低于0.3
        if top_scores[0] < 0.3:
            logger.info("问题与查询文献无关，属于继续提问")
            last_query = None
            last_matches = []

            # 找到与当前问题相关的上一次有效查询
            if previous_queries:
                for prev_query, prev_matches in reversed(previous_queries):
                    if prev_matches:
                        last_query = prev_query
                        last_matches = prev_matches
                        break

            if last_query and last_matches:
                prompt = prompt_builder.generate_ST_answer_prompt(query, last_matches, history)
                matches = [{
                    'text': doc['text'],
                    'file_id': doc['file_id'],
                    'file_name': doc.get('file_name', ''),
                    'score': doc.get('score', 0)
                } for doc in last_matches]
                logger.info(f"使用了generate_ST_answer_prompt，生成的prompt：{prompt}")
            else:
                prompt = prompt_builder.generate_answer_prompt_un_refs(query, history)
                matches = []
        else:
            prompt = prompt_builder.generate_ST_answer_prompt(query, merged_refs, history)
            matches = [{
                'text': ref['text'],
                'file_id': ref['file_id'],
                'file_name': ref['file_name'],
                'score': ref['score'],
                'rerank_score': score
            } for ref, score in top_list]
            logger.info(f"使用了generate_ST_answer_prompt，生成的prompt：{prompt}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

                logger.info(f"问题：{query}")
                logger.info(f"生成的答案：{full_answer}")

            # logger.info(f"命中文档：{matches}")
            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    large_model_service.shutdown()
    return jsonify({'status': 'success', 'message': 'Service shutdown'})


@app.route('/api/ST_OCR', methods=['POST'])
def ocr_indexing():
    try:
        data = request.json

        # 检查数据中是否包含docs字段
        if 'docs' not in data:
            raise KeyError("'docs'字段不存在")

        documents = data['docs']
        # 索引名称
        index_name = 'st_ocr'

        # 索引mapping
        mappings = {
            "properties": {
                "AB": {"type": "text", "analyzer": "standard"},
                "CT": {"type": "text", "analyzer": "standard"},
                "Id": {"type": "keyword"},
                "Issue_F": {"type": "keyword"},
                "KW": {"type": "keyword"},
                "JTI": {"type": "keyword"},
                "Pid": {"type": "keyword"},
                "Piid": {"type": "keyword"},
                "TI": {"type": "text", "analyzer": "standard"},
                "Year": {"type": "integer"},
                "AB_embed": {"type": "dense_vector", "dims": 1024},
                "CT_embed": {"type": "dense_vector", "dims": 1024}
            }
        }

        # 检查索引是否存在，如果不存在则创建索引
        if not es_handler.index_exists(index_name):
            logger.info(f"创建索引 {index_name}")
            es_handler.es.indices.create(index=index_name, mappings=mappings)

        for document in documents:
            doc_id = document.get('Id')
            doc_abstract = document.get('AB', None)
            doc_content = document.get('CT', '')

            # 如果摘要为空，生成摘要
            if not doc_abstract or not any(doc_abstract):
                logger.info("摘要为空，生成摘要")
                abstract_prompt = prompt_builder.generate_abstract_prompt(doc_content)
                doc_abstract = large_model_service.get_answer_from_Tyqwen(abstract_prompt)  # 使用大模型生成摘要
                logger.info(f"生成摘要成功，文档ID: {doc_id}, 摘要: {doc_abstract}")
                document['AB'] = [doc_abstract]

            # 计算嵌入
            ab_embed = es_handler.cal_passage_embed(doc_abstract)
            ct_embed = es_handler.cal_passage_embed(doc_content)

            # 添加嵌入到文档中
            document['AB_embed'] = ab_embed
            document['CT_embed'] = ct_embed

            es_handler.es.index(index=index_name, id=doc_id, document=document)
            logger.info(f"插入文档 {doc_id} 到索引 {index_name}")

        logger.info(f"索引 {index_name} 创建并插入数据成功")
        return jsonify({'status': 'success'}), 200

    except KeyError as e:
        logger.error(f"索引创建或插入数据失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

    except Exception as e:
        logger.error(f"索引创建或插入数据失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/Canvas_chatpdf', methods=['POST'])
def Canvas_chatpdf():
    try:
        # 读取请求参数
        data = request.json
        query = data.get('query')
        session_id = data.get('session_id')
        token = data.get('token')
        file_id = data.get('file_id')
        assistant_id = data.get('assistant_id')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.1)
        user_info = data.get('userInfo', {})

        if not query or not file_id:
            return jsonify({'error': '参数不完整'}), 400

        # 解析用户身份信息
        outer_origin = user_info.get('outerOrigin', 'canvas')  # 默认为canvas
        outer_user_name = user_info.get('outerUserName', '用户')  # 默认值
        outer_user_role = user_info.get('outerUserRole', '1')  # 默认值为学生

        # 生成用户身份上下文
        if outer_user_role == '2':
            role_description = "老师"
        else:
            role_description = "学生"

        user_context = f"您好，我是{outer_user_name}{role_description}。"

        try:
            history = prompt_builder.get_history(session_id, token)
        except Exception as e:
            history = []  # 或者你可以选择返回默认的空历史
            logger.warning(f"获取历史对话内容时出错: {e}")

        # 从 Elasticsearch 中获取全文内容
        all_content = ""
        try:
            all_content = es_handler.get_full_text_by_file_id(assistant_id.strip(), file_id.strip())
            if all_content:
                logger.info("检索到全文内容")
            else:
                logger.info(f"未能检索到全文内容 for Assistant ID: {assistant_id}, File ID: {file_id}")
        except Exception as e:
            logger.error(f"检索文本内容时出错 {file_id}: {e}")

        if not all_content:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p,
                                                                                  temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 生成 prompt
        prompt_messages = prompt_builder.generate_canvas_prompt(query, all_content, history, user_context)
        logger.info(f"Prompt: {prompt_messages}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

                logger.info(f"问题：{query}")
                logger.info(f"生成的答案：{full_answer}")

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')
        if llm == 'deepseek':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt_messages, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

                logger.info(f"问题：{query}")
                logger.info(f"生成的答案：{full_answer}")

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


# 定义 Flask 路由
@app.route('/api/web_search', methods=['POST'])
def web_search():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # 调用 large_model_service 的方法
    result = large_model_service.web_search_glm4(query)

    return jsonify(result)


def start_background_threads():
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(32)]
    for t in threads:
        t.daemon = True  # 将线程设置为守护线程
        t.start()


@app.route('/api/canvas_qa', methods=['POST'])
def canvas_chatpdf():
    try:
        # 读取请求参数
        data = request.json
        # 打印原始请求数据
        query = data.get('query')
        session_id = data.get('session_id')
        download_path = data.get('download_path')
        file_name = data.get('file_name')
        file_id = data.get('file_id')
        token = data.get('token')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.7)
        temperature = data.get('temperature', 0.1)
        user_info = data.get('userInfo', {})

        if not query or not download_path:
            return jsonify({'error': '参数不完整'}), 400

        # 解析用户身份信息
        outer_origin = user_info.get('outerOrigin', 'canvas')  # 默认为canvas
        outer_user_name = user_info.get('outerUserName', '用户')  # 默认值
        outer_user_role = user_info.get('outerUserRole', '1')  # 默认值为学生

        # 生成用户身份上下文
        if outer_user_role == '2':
            role_description = "老师"
        else:
            role_description = "学生"

        user_context = f"您好，我是{outer_user_name}{role_description}。"

        try:
            history = prompt_builder.get_history(session_id, token)
        except Exception as e:
            history = []  # 或者你可以选择返回默认的空历史
            logger.warning(f"获取历史对话内容时出错: {e}")

        logger.info("开始下载文件并处理: %s，文件路径：%s", file_name, download_path)
        file_path = file_manager.download_pdf(download_path, file_id)
        full_text = file_manager.process_Canvas_file(file_path, file_name)

        # 生成 prompt
        prompt_messages = prompt_builder.generate_canvas_prompt(query, full_text, history, user_context)
        logger.info(f"Prompt: {prompt_messages}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')
        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/image2text_stream', methods=['POST'])
def image2text_stream():
    try:
        data = request.json
        query = data.get('query')
        llm = data.get('llm', 'internvl')  # 默认使用 internvl2.5-latest
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        # 判断用户输入是文字还是图片
        messages = []
        if isinstance(query, list):
            for item in query:
                message_content = []

                # 如果有文字部分
                if 'text' in item and isinstance(item['text'], str):
                    message_content.append({"type": "text", "text": item['text']})

                # 如果有图片部分
                if 'image' in item and isinstance(item['image'], str):
                    # 如果只有图片，自动填充 text
                    if not any(c['type'] == 'text' for c in message_content):
                        message_content.append({"type": "text", "text": "请描述这张图片"})
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": item['image']}
                    })

                # 如果 message_content 只包含文字而没有图片
                if message_content and len(message_content) == 1 and 'text' in message_content[0]:
                    messages.append({
                        "role": "user",
                        "content": message_content[0]["text"]
                    })
                elif message_content:
                    # 如果有图片和文字，返回数组形式
                    messages.append({
                        "role": "user",
                        "content": message_content
                    })
                else:
                    return jsonify({"error": "每个条目必须包含有效的 text 或 image 字段"}), 400
        else:
            return jsonify({"error": "query 必须是一个列表"}), 400

        logger.info(f"构建的提示词: {messages}")

        # 调用模型获取回答并返回流式响应
        def generate_stream_response():
            # 使用你的方法获取流式输出
            for part in large_model_service.get_answer_from_internvl_stream(messages, top_p, temperature):
                yield part  # 返回流式内容

        return Response(generate_stream_response(), content_type='application/json; charset=utf-8')

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": "内部服务器错误", "details": str(e)}), 500


@app.route('/api/beautify_prompt', methods=['POST'])
def prompt_rewrite():
    try:
        data = request.json
        query = data.get('query')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.6)

        if llm == 'qwen':
            prompt = prompt_builder.generate_prompt_for_qwen(query)
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')
        if llm == 'deepseek':
            prompt = prompt_builder.generate_prompt_for_deepseek(query)
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.error(f"Error in prompt_rewrite: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/Text_polish', methods=['POST'])
def polish_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.6)
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_polish_prompt(question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'文本润色时发生错误: {str(e)}'
        }), 500


@app.route('/api/text_expansion', methods=['POST'])
def expand_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.6)
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_expand_prompt(question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'文本扩写时发生错误: {str(e)}'
        }), 500


@app.route('/api/text_translation', methods=['POST'])
def transaltion_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.4)
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_translation_prompt(question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'文本扩写时发生错误: {str(e)}'
        }), 500


@app.route('/api/polite_language', methods=['POST'])
def polite_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.7)
        style = data.get('style', '商务礼仪')
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_politeness_prompt(question, style)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'礼貌化文本时发生错误: {str(e)}'
        }), 500


@app.route('/api/how_to_say_no', methods=['POST'])
def refuse_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.7)
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_refusal_prompt(question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'拒绝别人时发生错误: {str(e)}'
        }), 500


@app.route('/api/write_email', methods=['POST'])
def email_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.7)
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_email_prompt(question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'生成邮件时发生错误: {str(e)}'
        }), 500


@app.route('/api/poem_writer', methods=['POST'])
def poem_text():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.7)
        poetry_style = data.get('poetry_style', '俳句')
        if not question :
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_poem_prompt(poetry_style, question)

        if llm == 'qwen':
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

        if llm == 'deepseek':
            ans_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p=top_p, temperature=temperature)
            return Response(ans_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'写诗时发生错误: {str(e)}'
        }), 500


# 确保线程在 Flask 应用启动前就启动
start_background_threads()


if __name__ == '__main__':
    # 在本地调试时可以继续使用 Flask 内置服务器
    app.run(host='0.0.0.0', port=5777, debug=False)



