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


def pull_file_data():
    # 模拟从服务器获取文件数据
    return []


def _push(file_data):
    global file_queue
    file_queue.put(file_data)


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


@app.route('/api/ST_Get_Answer', methods=['POST'])
def ST_Get_Answer():
    try:
        # 初始化重排模型
        reranker = FlagReranker(model_path, use_fp16=True)
        # 读取请求参数
        data = request.json
        query = data.get('query')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)
        logger.info(f"Received query: {query}")
        if not query:
            return jsonify({'error': '参数不完整'}), 400

        # 执行混合检索
        bm25_refs = es_handler.ST_search_bm25(query, ref_num)
        embed_refs = es_handler.ST_search_embed(query, ref_num)
        all_refs = bm25_refs + embed_refs

        if not all_refs:
            def generate():
                full_answer = "您的问题没有在文献中找到答案，正在使用预训练知识库为您解答："
                Prompt = [{'role': 'system', 'content': "你是一个近代历史文献研究专家"},
                          {'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(Prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 使用重排模型进行重排并归一化得分
        ref_pairs = [[query, str(ref['CT'])] for ref in all_refs]  # 为每个参考文档与查询组合成对
        scores = reranker.compute_score(ref_pairs, normalize=True)  # 计算每对的得分并归一化
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        top_list = sorted_refs[:3]
        top_scores = [score for _, score in top_list]
        top_refs = [ref for ref, _ in top_list]
        logger.info(f"重排后最高分：{top_scores}")

        prompt_messages = PromptBuilder.generate_ST_answer_prompt(query, top_refs)
        logger.info(f"Prompt: {prompt_messages}")

        matches = [{
            'CT': ref['CT'],
            'Pid': ref['Pid'],
            'TI': ref.get('TI', '无标题'),
            'score': ref['score'],
            'rerank_score': score,
            'AB': ref.get('AB', ''),
            'Id': ref.get('Id', ''),
            'Issue_F': ref.get('Issue_F', ''),
            'KW': ref.get('KW', ''),
            'JTI': ref.get('JTI', ''),
            'Piid': ref.get('Piid', ''),
            'Year': ref.get('Year', '')
        } for ref, score in top_list]

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in get_answer_stream: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(2)]
    for t in threads:
        t.start()

    app.run(host='0.0.0.0', port=5555, debug=False)
