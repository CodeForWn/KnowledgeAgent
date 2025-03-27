# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import traceback
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
from Neo4j.KMC_neo4j import KMCNeo4jHandler
from neo4j import GraphDatabase, basic_auth
from MongoDB.KMC_Mongo import KMCMongoDBHandler
import types
import time
import os


app = Flask(__name__)
CORS(app)
# 加载配置
# 使用环境变量指定环境并加载配置
config = Config(env='production')
config.load_config()  # 指定配置文件的路径
config.load_predefined_qa()
logger = config.logger
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)
neo4j_handler = KMCNeo4jHandler(config)
file_manager = FileManager(config)
mongo_handler = KMCMongoDBHandler(config)
es_handler = ElasticSearchHandler(config)


@app.route("/api/question_agent", methods=["POST"])
def get_question_agent():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        query = knowledge_point + "的地理意义是什么？"
        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 调用 get_entity_details 方法获取该知识点的详细信息
        result = neo4j_handler.get_entity_details(knowledge_point)
        resources = result.get("resources", [])

        # 关闭数据库连接
        neo4j_handler.close()
        # 返回 JSON 格式的结果
        # logger.info(f"候选子图为：{result}")
        # 创建一个空列表，用于存储所有资源处理后的 doc_list
        combined_doc_list = []
        index_name = f"temp_kb_{knowledge_point}_{int(time.time())}"
        for res in resources:
            docID = res.get("docID")
            # 从 MongoDB 查询该资源的详细信息（例如 file_path 等）
            resource_detail = mongo_handler.get_resource_by_docID(docID)
            if resource_detail:
                # 提取文件路径和文件名（注意，文件名可以直接从 neo4j 的返回值中获取）
                file_path = resource_detail.get("file_path", "")
                file_name = resource_detail.get("file_name")
                subject = resource_detail.get("subject")
                resource_type = resource_detail.get("resource_type")
                metadata = resource_detail.get("metadata")

                # 判断文件是否存在
                if not os.path.exists(file_path):
                    logger.error(f"文件 {file_path} 不存在，跳过资源 {file_name}")
                    continue

                try:
                    # 使用文件处理模块对该文件进行分段处理
                    doc_list = file_manager.process_pdf_file(file_path, file_name)
                except Exception as e:
                    logger.error(f"处理文件 {file_name} 时发生异常: {e}", exc_info=True)
                    continue

                if doc_list:
                    success = es_handler.create_temp_index(index_name, doc_list, docID, file_name, file_path, subject, resource_type, metadata)
                    if success:
                        logger.info(f"临时索引 {index_name} 创建成功")
                        # 此处可以执行 ES 检索操作，例如：
                        bm25_hits = es_handler.agent_search_bm25(index_name, query, ref_num=10)

                        # 测试 Embed 检索：对索引中存储的文档进行向量检索
                        embed_hits = es_handler.agent_search_embed(index_name, query, ref_num=10)

                        combined_doc_list.extend(bm25_hits)
                        combined_doc_list.extend(embed_hits)
                    else:
                        logger.error("临时索引创建失败")
                        continue
                else:
                    logger.error(f"文件处理失败：{file_name}")
            else:
                logger.error(f"未在MongoDB中找到 docID: {docID} 的资源信息")

        mongo_handler.close()
        es_handler.delete_index(index_name)
        logger.info(f"索引 {index_name} 已删除")
        if not combined_doc_list:
            return jsonify({"error": "没有成功检索到任何资源数据"}), 404

        # 返回候选子图信息以及 ES 检索结果
        result_data = {
            "candidate_subgraph": result,
            "retrieval_results": combined_doc_list
        }
        return jsonify(result_data)

    except Exception as e:
        logger.error(f"Error in get_question_agent: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7777)