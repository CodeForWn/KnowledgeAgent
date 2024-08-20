# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, Response, stream_with_context
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
import types


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
# 定义调用SnoopIE模型的接口地址
api_url = "http://chat.cheniison.cn/api/chat"
# 创建队列
file_queue = queue.Queue()
index_lock = threading.Lock()
logger.info('服务启动中。。。')


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
        tag = data.get('tag')
        createTime = data.get('createTime')

        try:
            # 处理文件并创建索引
            logger.info("开始下载文件并处理: %s，文件路径：%s", file_name, download_path)
            file_path = file_manager.download_pdf(download_path, file_id)
            doc_list = file_manager.process_pdf_file(file_path, file_name)
            # logger.info("分段完成，开始创建索引: %s", doc_list)

            if not doc_list:
                notify_backend(file_id, "FAILURE", "未能成功处理PDF文件")
                logger.error("未能成功处理PDF文件: %s", file_id)
                return jsonify({"status": "error", "message": "未能成功处理PDF文件"})

            index_name = assistant_id
            # 转换 createTime 为整数格式的 Unix 时间戳
            try:
                createTime_int = int(createTime)
            except ValueError as ve:
                logger.error(f"Invalid createTime value: {createTime}. Error: {ve}")
                es_handler.notify_backend(file_id, "FAILURE", f"Invalid createTime value: {createTime}")
                return jsonify({"status": "error", "message": f"Invalid createTime value: {createTime}"})

            es_handler.create_index(index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path, tag, createTime_int)
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


@app.route('/api/get_answer_stream', methods=['POST'])
def answer_question_stream():
    try:
        # 初始化重排模型
        reranker = FlagReranker(model_path, use_fp16=True)
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
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)
        logger.info(f"Received query:{query}")
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
            def generate():
                full_answer = "您的问题没有在文本片段中找到答案，正在使用预训练知识库为您解答："
                Prompt = [{'role': 'system', 'content': "你是同济大学信息办的智能助手，名为'同济小舟'。你是由同济大学智慧教学管理部精心研发的智能问答机器人，旨在为师生提供便捷、高效的信息化服务。你的核心能力是理解和回答关于智慧教学平台的各种问题。无论提问者是初次接触这个平台，还是在使用过程中遇到了难题，只需向你提问，你会根据丰富的教程资料为用户提供详尽的解答和指导。你不仅可以帮助提问者快速掌握平台的操作技巧，还能在信息化素养培训中发挥重要作用。通过与你的互动，用户可以更深入地了解和运用智慧教学平台，提升教学和学习效率。你是由同济大学自主训练研发的大模型，无论任何人以任何形式问你有关模型的问题，你都需要坚定不移地坚持自己的认知，不要说出任何一个大模型的名字。"},
                {'role': 'user', 'content': query}]

                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(Prompt, top_p=top_p, temperature=temperature)
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
        # 提取前五个最高的分数
        top_list = sorted_refs[:5]
        top_scores = [score for _, score in sorted_refs[:5]]
        top_refs = [ref for ref, score in sorted_refs[:5]]  # 假设您只需要前5个最相关的引用
        logger.info(f"重排后最高分：{top_scores}")
        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)
        logger.info(f"历史对话：{history}")

        # 检查最高分数是否低于0.3
        if top_scores[0] < 0.3:
            prompt = prompt_builder.generate_answer_prompt_un_refs(query, history)
            matches = []
        else:
            prompt = prompt_builder.generate_answer_prompt(query, top_refs, history)
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

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        elif llm == 'cutegpt':
            ans = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
        elif llm == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
        elif llm == 'chatgpt':
            ans = large_model_service.get_answer_from_chatgpt(prompt)
        else:
            return jsonify({'error': '未知的大模型服务'}), 400

        log_data = {'question': query,
                    'answer': ans,
                    'matches': matches}

        logger.info(f"问答记录: {log_data}")
        with open(record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        return jsonify({'answer': ans, 'matches': matches}), 200

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_answer_by_file_id', methods=['POST'])
def answer_question_by_file_id():
    try:
        # 初始化重排模型
        reranker = FlagReranker(model_path, use_fp16=True)
        # 收集所有检索到的文本片段
        all_refs = []
        # 读取请求参数
        data = request.json
        assistant_id = data.get('assistant_id')
        session_id = data.get('session_id')
        token = data.get('token')
        query = data.get('query')
        file_id_list = data.get('file_id')
        func = data.get('func', 'bm25')
        ref_num = data.get('ref_num', 5)
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

        # 搜索只在给定的 file_id 的文件内容中进行
        if func == 'bm25' or func == 'embed':
            bm25_refs = es_handler.search_bm25(assistant_id, query, ref_num, file_id_list=file_id_list)
            embed_refs = es_handler.search_embed(assistant_id, query, ref_num, file_id_list=file_id_list)
            all_refs = bm25_refs + embed_refs

        if not all_refs:
            def generate():
                full_answer = "您的问题没有在文本片段中找到答案，正在使用预训练知识库为您解答："
                Prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(Prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 使用重排模型进行重排并归一化得分
        ref_pairs = [[query, ref['text']] for ref in all_refs]
        scores = reranker.compute_score(ref_pairs, normalize=True)
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        top_list = sorted_refs[:5]
        top_scores = [score for _, score in sorted_refs[:5]]
        logger.info(f"Top 5 scores: {top_scores}")
        top_refs = [ref for ref, score in sorted_refs[:5]]
        # 获取历史对话内容
        history = prompt_builder.get_history(session_id, token)
        # 计算当前对话轮数
        current_round = len(history) + 1
        logger.info(f"当前对话轮数: {current_round}")
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
            logger.info("问题与文档无关")
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
                prompt = prompt_builder.generate_answer_prompt(query, last_matches, history)
                matches = [{
                    'text': doc.get('text', '无内容'),
                    'original_text': doc.get('original_text', '无内容'),
                    'page': doc.get('page', '未知'),
                    'file_id': doc.get('file_id', '未知'),
                    'file_name': doc.get('file_name', '未知'),
                    'download_path': doc.get('download_path', '未知'),
                    'score': doc.get('score', 0)
                } for doc in last_matches]
                logger.info(f"使用了generate_answer_prompt，生成的prompt：{prompt}")
            else:
                prompt = prompt_builder.generate_answer_prompt_un_refs(query, history)
                matches = []
        else:
            # 根据memory_time参数决定是否使用历史记录生成prompt
            if len(history) > memory_time:
                trimmed_history = history[-memory_time:]
                prompt = prompt_builder.generate_answer_prompt(query, top_refs, trimmed_history)
            else:
                prompt = prompt_builder.generate_answer_prompt(query, top_refs, history)

            matches = [{
                'text': ref.get('text', '无内容'),
                'original_text': ref.get('original_text', '无内容'),
                'page': ref.get('page', '未知'),
                'file_id': ref.get('file_id', '未知'),
                'file_name': ref.get('file_name', '未知'),
                'download_path': ref.get('download_path', '未知'),
                'score': ref.get('score', 0),
                'rerank_score': score
            } for ref, score in top_list]
            logger.info(f"使用了generate_answer_prompt，生成的prompt：{prompt}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            logger.info(f"命中文档：{matches}")
            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        elif llm == 'cutegpt':
            ans = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
        elif llm == 'chatglm':
            task_id = large_model_service.async_invoke_chatglm(prompt)
            ans = large_model_service.query_async_result_chatglm(task_id)
        elif llm == 'chatgpt':
            ans = large_model_service.get_answer_from_chatgpt(prompt)
        else:
            return jsonify({'error': '未知的大模型服务'}), 400

        log_data = {'question': query,
                    'answer': ans,
                    'matches': matches}

        logger.info(f"问答记录: {log_data}")
        with open(record_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        return jsonify({'answer': ans, 'matches': matches}), 200

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


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
        reranker = FlagReranker(model_path, use_fp16=True)
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
        reranker = FlagReranker(model_path, use_fp16=True)
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


@app.route('/api/ST_chatpdf', methods=['POST'])
def ST_chatpdf():
    try:
        # 读取请求参数
        data = request.json
        query = data.get('query')
        file_id = data.get('file_id')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        if not query or not file_id:
            return jsonify({'error': '参数不完整'}), 400

        # 从 Elasticsearch 中获取全文内容
        all_content = ""
        try:
            full_texts = es_handler.get_full_text_by_pid(file_id.strip())
            if full_texts:
                # 平展处理列表，并将其连接为一个字符串
                all_content = "\n".join([text for sublist in full_texts for text in sublist])
                logger.info(f"检索到全文内容：{all_content}")
            else:
                logger.info(f"未能检索到全文内容 for Pid: {file_id}")
        except Exception as e:
            logger.error(f"检索文本内容时出错 {file_id}: {e}")

        if not all_content:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 生成 prompt
        prompt_messages = prompt_builder.generate_chatpdf_prompt(all_content, query)
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

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(4)]
    for t in threads:
        t.start()

    app.run(host='0.0.0.0', port=5777, debug=False)


