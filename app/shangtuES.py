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
from threading import Thread
import time

sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)
CORS(app)

# 配置日志记录
logger = logging.getLogger('myapp')
logger.setLevel(logging.INFO)

# 创建控制台处理程序并设置级别为INFO
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# 创建格式化器并将其添加到处理程序
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将处理程序添加到记录器
logger.addHandler(console_handler)

# 使用环境变量指定环境并加载配置
config = Config(env='production')
config.load_config()  # 指定配置文件的路径
# 创建ElasticSearchHandler实例
es_handler = ElasticSearchHandler(config)
prompt_builder = PromptBuilder(config)
large_model_service = LargeModelAPIService(config)


@app.route('/api/ST_OCR', methods=['POST'])
def indexing():
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
def ST_answer_question_by_file_id():
    try:
        # 读取请求参数
        data = request.json
        query = data.get('query')
        file_id_list = data.get('file_id', '').split(',')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        if not query or not file_id_list:
            return jsonify({'error': '参数不完整'}), 400

        all_content = ""

        # 搜索只在给定的 file_id 的文件内容中进行
        for file_id in file_id_list:
            try:
                doc = es_handler.es.search(index='st_ocr', body={
                    "query": {
                        "term": {
                            "Pid.keyword": file_id.strip()
                        }
                    }
                })
                if doc['hits']['hits']:
                    for hit in doc['hits']['hits']:
                        all_content += hit['_source']['CT'] + "\n"
            except Exception as e:
                logger.error(f"检索文本内容时出错 {file_id}: {e}")

        if not all_content:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(query, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 生成 prompt
        prompt_messages = prompt_builder.generate_chatpdf_prompt(all_content, query)

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
    app.run(host='0.0.0.0', port=5555, debug=False)
