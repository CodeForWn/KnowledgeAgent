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


@app.route("/api/question_agent", methods=["POST"])
def get_question_agent():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 调用 get_entity_details 方法获取该知识点的详细信息
        result = neo4j_handler.get_entity_details(knowledge_point)

        # 关闭数据库连接
        neo4j_handler.close()

        # 返回 JSON 格式的结果
        logger.info(f"候选子图为：{result}")
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7777)