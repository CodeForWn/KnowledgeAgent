# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
from pdf2markdown import PDF
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import jieba.posseg as pseg
import tempfile
import os
from pdf2markdown import *
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
# from ltp import LTP
import queue
import threading
import spacy
import sys
sys.path.append("/pro_work/docker_home/work/kmc/KmcGPT/KMC")
from config.KMC_config import Config

# 处理查询语句类
class QueryProcessor:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModel.from_pretrained(config.model_path)
        self.model.eval()
        self.config = config

        # 使用全局logger实例
        self.logger = self.config.logger

    # 意图识别&意图修复
    @staticmethod
    def extend_query(query):
        words = list(pseg.cut(query))
        # print(words)
        # 检查所有词的词性是否为名词
        if all(flag.startswith('n') for word, flag in words):
            self.logger.info("进行问题补充。。。")
            return f"{query}是什么？"
        else:
            return query

    def cal_passage_embed(self, text):
        instruction = "为这个句子生成表示以用于检索相关文章："
        text = instruction + text
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy()[0].tolist()


# # 加载配置
# # 使用环境变量指定环境并加载配置
# config = Config(env='development')
# config.load_config('config\\config.json')  # 指定配置文件的路径
# query_processor = QueryProcessor(config)
#
# # 测试
# sample_query = "图谱"
# extended_query = query_processor.extend_query(sample_query)
# print(f"扩展的查询: {extended_query}")
# query_embedding = query_processor.cal_passage_embed(extended_query)
# print(f"查询嵌入向量: {query_embedding}")
