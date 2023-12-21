# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import jieba.posseg as pseg
import tempfile
import os
from File_manager.pdf2markdown import *
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import logging
from logging.handlers import RotatingFileHandler
import PyPDF2
import pypdf
import pikepdf
from traceback import format_exc
import elasticsearch.exceptions
import warnings
from sentence_transformers import SentenceTransformer
import json
import sys
# from ltp import LTP
import queue
import threading
import spacy
from config.KMC_config import Config
import logging
from logging.handlers import RotatingFileHandler
sys.path.append(r"E:\工作\KmcGPT\KmcGPT")
# config = Config(env='development')
# config.load_config()  # 指定配置文件的路径


class ElasticSearchHandler:
    def __init__(self, config):
        # 从配置中提取Elasticsearch相关配置
        self.es_hosts = config.elasticsearch_hosts
        self.es_username = config.elasticsearch_basic_auth_username
        self.es_password = config.elasticsearch_basic_auth_password
        self.backend_notify_api = config.external_api_backend_notify
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModel.from_pretrained(config.model_path)
        self.model.eval()
        self.config = config

        # 使用全局logger实例
        self.logger = self.config.logger

        # 连接到Elasticsearch
        self.es = Elasticsearch(
            hosts=self.es_hosts,
            basic_auth=(self.es_username, self.es_password),
            verify_certs=False
        )
        # 检测是否成功连接到ES
        if self.es.ping():
            self.logger.info("成功连接到Elasticsearch")
        else:
            self.logger.error("无法连接到Elasticsearch")

    def notify_backend(self, file_id, result, failure_reason=None):
        """通知后端接口处理结果"""
        url = self.backend_notify_api  # 更新后的后端接口URL
        headers = {'token': file_id}
        payload = {
            'id': file_id,
            'result': result
        }
        if failure_reason:
            payload['failureReason'] = failure_reason

        response = requests.post(url, json=payload, headers=headers)
        self.logger.info("后端接口返回状态码：%s", response.status_code)
        return response.status_code

    def index_exists(self, index_name):
        # 查询索引是否存在
        return self.es.indices.exists(index=index_name)

    def cal_passage_embed(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy()[0].tolist()

    def cal_query_embed(self, query):
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self.cal_passage_embed(instruction + query)

    def create_index(self, index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path):
        try:
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
            if self.index_exists(index_name):
                self.logger.info("索引已存在，删除索引")
                self.delete_index(index_name)
            self.logger.info("开始创建索引")
            self.es.indices.create(index=index_name, body={'mappings': mappings})
            # 插入文档
            for item in doc_list:
                embed = self.cal_passage_embed(item['text'])
                document = {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "file_id": file_id,
                    "file_name": file_name,
                    "tenant_id": tenant_id,
                    "download_path": download_path,
                    "page": item['page'],
                    "text": item['text'],
                    "original_text": item['original_text'],
                    "embed": embed
                }
                self.es.index(index=index_name, document=document)

            self.logger.info(f"索引 {index_name} 创建并插入索引成功")
            return True
        except Exception as e:
            self.logger.error(f"创建索引 {index_name} 或插入索引失败: {e}")
            return False

    def _search_(self, index_pattern, query_body, size=10):
        try:
            result = self.es.search(index=index_pattern, body=query_body, size=size)
            return result
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return None

    def delete_index(self, index_name):
        try:
            # 删除索引
            if self.index_exists(index_name):
                self.es.indices.delete(index=index_name)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting index: {e}")
            return False

    def delete_summary_answers(self, file_id):
        try:
            # 构建删除请求的查询
            query = {
                "query": {
                    "term": {"file_id": file_id}
                }
            }
            # 执行删除操作
            self.es.delete_by_query(index="answers_index", body=query)
            self.logger.info(f"成功删除文件ID {file_id} 的相关答案")
            return True
        except Exception as e:
            self.logger.info(f"删除文件ID {file_id} 的相关答案失败: {e}")
            return False

    def index(self, index_name, document):
        try:
            self.es.index(index=index_name, document=document)
            self.logger.info(f"文档已成功插入到索引 {index_name}")
        except Exception as e:
            self.logger.error(f"文档插入失败: {e}")

    def search(self, assistant_id, query_body, ref_num=10):
        try:
            # 在所有符合条件的ES索引中查询
            index_pattern = f'{assistant_id}_*'
            result = self.es.search(index=index_pattern, body=query_body, size=ref_num)

            if 'hits' in result and 'hits' in result['hits']:
                # 命中结果
                hits = result['hits']['hits']
                refs = [{
                    'text': hit['_source']['text'],
                    'original_text': hit['_source']['original_text'],
                    'page': hit['_source']['page'],
                    'file_id': hit['_source']['file_id'],
                    'file_name': hit['_source']['file_name'],
                    'score': hit['_score'],
                    'download_path': hit['_source']['download_path']
                } for hit in hits]
                return refs

        except Exception as e:
            # 如果未找到相关文本片段
            self.logger.error(f"Error during search: {e}")
            return []

    def search_bm25(self, assistant_id, query, ref_num=10):
        # 使用BM25方法搜索索引
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
        return self.search(assistant_id, query_body, ref_num)

    def search_embed(self, assistant_id, query, ref_num=10):
        # 使用Embed方法搜索索引
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "script_score": {
                    "query": {"term": {"assistant_id": assistant_id}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                        "params": {"query_vector": query_embed}
                    }
                }}}
        return self.search(assistant_id, query_body, ref_num)


# # 加载配置
# # 使用环境变量指定环境并加载配置
# config = Config(env='development')
# config.load_config('config\\config.json')  # 指定配置文件的路径
#
# # 创建ElasticSearchHandler实例
# es_handler = ElasticSearchHandler(config)
