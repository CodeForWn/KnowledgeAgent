from flask import Flask, request, jsonify
import re
import urllib3
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
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
from sentence_transformers import SentenceTransformer
import json
# from ltp import LTP
import queue
import threading
import spacy
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上两级目录
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)


class NoRequestStatusFilter(logging.Filter):
    def filter(self, record):
        return "/api/request_status" not in record.getMessage()


class Config(object):

    def __init__(self, env='production'):
        self.predefined_qa = {}
        self.secret_token = None
        self.external_api_backend_notify = "http://127.0.0.1:8888/sync/syncCallback"
        self.env = env
        # 设置默认值
        self.threads = 2
        self.elasticsearch_hosts = 'http://localhost:9200/'
        # 初始化日志处理器
        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger('myapp')
        self.logger.setLevel(logging.INFO)
        # 创建控制台日志处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        self.logger.addHandler(stream_handler)

        log_file = 'myapp.log'  # 替换为日志文件路径
        file_handler = RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024 * 1024, backupCount=20)
        file_handler.setLevel(logging.INFO)

        file_handler.setFormatter(formatter)

        # 添加过滤器以过滤掉 /api/request_status 的日志
        no_request_status_filter = NoRequestStatusFilter()
        self.logger.addFilter(no_request_status_filter)

        # 为所有处理器添加过滤器
        stream_handler.addFilter(no_request_status_filter)
        file_handler.addFilter(no_request_status_filter)

        self.logger.addHandler(file_handler)

    def _set(self, attr, value):
        if value is not None:
            setattr(self, attr, value)

    def _read_attr(self, conf, attr):
        if attr in conf:
            self._set(attr, conf[attr])

    def load_config(self, file_path=os.path.join(current_dir, 'config.json')):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_conf = json.load(f)
                conf = all_conf.get(self.env, {})
                # 读取通用配置
                self._read_attr(conf, 'rerank_model_path')
                self._read_attr(conf, 'model_path')
                self._read_attr(conf, 'embed_model_path')
                self._read_attr(conf, 'threads')
                self._read_attr(conf, 'log_file')
                self._read_attr(conf, 'spacy_model')
                self._read_attr(conf, 'history_api_url')
                self._read_attr(conf, 'stopwords')
                self._read_attr(conf, 'secret_token')
                self._read_attr(conf, 'Tyqwen_api_key')
                self._read_attr(conf, 'file_storage_path')
                self._read_attr(conf, 'record_path')
                self._read_attr(conf, 'chatgpt_api')
                self._read_attr(conf, 'mongodb_host')
                self._read_attr(conf, 'mongodb_port')
                self._read_attr(conf, 'mongodb_database')
                self._read_attr(conf, 'mongodb_username')
                self._read_attr(conf, 'mongodb_password')
                self._read_attr(conf, 'neo4j_max_depth_down')
                self._read_attr(conf, 'neo4j_include_relations')
                # Neo4j配置
                self._read_attr(conf, 'neo4j_uri')
                self._read_attr(conf, 'neo4j_username')
                self._read_attr(conf, 'neo4j_password')

                # 读取Elasticsearch相关配置
                es_config = conf.get('elasticsearch', {})
                for key in ['hosts', 'basic_auth_username', 'basic_auth_password']:
                    self._set('elasticsearch_' + key, es_config.get(key))

                # 读取外部api相关配置
                es_config = conf.get('external_api', {})
                for key in ['llm_ans', 'backend_notify']:
                    self._set('external_api_' + key, es_config.get(key))

        except Exception as e:
            print(f"Error loading config: {e}")

    def load_predefined_qa(self):
        try:
            with open(os.path.join(current_dir, 'predefined_qa.json'), 'r', encoding='utf-8') as file:
                self.predefined_qa = json.load(file)
        except Exception as e:
            self.logger.error(f"无法加载 predefined_qa.json: {e}")
            self.predefined_qa = {}


# # 使用环境变量指定环境并加载配置
# config = Config(env='development')
# config.load_config('config\\config.json')  # 指定配置文件的路径
# # 打印出所有配置信息
# config_attrs = vars(config)
# for key, value in config_attrs.items():
#     print(f"{key}: {value}")

