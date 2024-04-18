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
# from ltp import LTP
import queue
import threading
import spacy
import sys
sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config


# prompts类
class PromptBuilder:
    def __init__(self, config):
        self.history_api_url = config.history_api_url
        self.config = config

        # 使用全局logger实例
        self.logger = self.config.logger

    def get_history(self, session_id, token):
        url = f"{self.history_api_url}{session_id}"
        headers = {"Token": token}
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            return []

    # 构建开开放式问答prompt
    @staticmethod
    def generate_open_answer_prompt(query, history):
        overall_instruction = "你是商业AI智能助理。请根据给定任务描述，请给出对应请求的回答。\n"
        prompt = overall_instruction

        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    prompt += f"问：{item['question']}\n答：{item['content']}\n"

        prompt += f"问：{query}\n答：\n"
        return prompt

    # 构建总结文本和推荐问题的prompt
    @staticmethod
    def generate_summary_and_questions_prompt(ref_list):
        prex = f"参考这一篇文章的前{len(ref_list)}段文本，简要地多方面地概括文章提到了哪些内容，并生成3个推荐问题并用序号列出（推荐问题应该能根据文章的内容回答）：\n"
        for i, ref in enumerate(ref_list):
            prex += f"{i+1}:{ref}\n"
        return prex

    # 构建文档问答prompt
    @staticmethod
    def generate_answer_prompt(query, refs):
        # 构建参考文本部分
        ref_list = [ref['text'] for ref in refs]
        refs_prompt = f"参考这一篇文章里与问题相关的以下{len(ref_list)}段文本，然后回答后面的问题：\n"
        for i, ref in enumerate(ref_list):
            refs_prompt += f"[{i + 1}]:{ref}\n"

        # 构建最终的prompt
        final_prompt = f"{refs_prompt}\n你应当尽量用原文回答，并对回答的结构和内容进行完善和润色，让提问者感到你非常认真地在解决他的问题。问题：{query}\n"

        return final_prompt

    @staticmethod
    def generate_beauty_prompt(query):
        # 构建最终的prompt
        beauty_prompt = f"对于回答：{query}，请在不改变原文的基础上，对回答的结构和语言进行美化和完善，使人感到更加回答更加全面和贴切使用者提问的语境，仅输出修改后的回答，不要输出任何其他内容："

        return beauty_prompt
#
# # 加载配置
# # 使用环境变量指定环境并加载配置
# config = Config(env='development')
# config.load_config('config\\config.json')  # 指定配置文件的路径
# prompt_builder = PromptBuilder(config)
# # 获取历史数据和生成提示的示例
# history = prompt_builder.get_history(session_id="123", token="token123")
# prompt = prompt_builder.generate_open_answer_prompt("你是谁？", history)
# print("多轮对话prompt:", prompt)
# query = "同济大学的历史"
# refs = [{"text": "同济大学成立于1907年..."}, {"text": "该校是中国最早的..."}]
#
#
# final_prompt = prompt_builder.generate_answer_prompt(query, refs)
# print(final_prompt)
#
# # 假设有一组引用文本和一个查询问题
# refs = [
#     """十六大以来，以
# 胡锦涛同志为主要代表的中国共产党人，坚持以\n邓小平理论和“三个代表”重要思想为指导，根据新的发展要求，\n深刻认识和回答了新
# 形势下实现什么样的发展、怎样发展等重大问\n题，形成了以人为本、全面协调可持续发展的科学发展观。科学发展\n观是同马克思列宁
# 主义、毛泽东思想、邓小平理论、“三个代表”重\n要思想既一脉相承又与时俱进的科学理论，是马克思主义关于发展\n的世界观和方法论
# 的集中体现，是马克思主义中国化重大成果，是中\n国共产党集体智慧的结晶，是发展中国特色社会主义必须长期坚持\n的指导思想。十八大以来，以习近平同志为主要代表的中国共产党人，坚持把\n马克思主义基本原理同中国具体实际相结合、同中华优秀传统文化\n相结
# 合，科学回答了新时代坚持和发展什么样的中国特色社会主义、\n怎样坚持和发展中国特色社会主义等重大时代课题，创立了习近平\n新时代中国特色社会主义思想。""",
#     """十三届四中全会以来，以江泽民同志为主要代表的中国共产党\n人，$建设中国特色社会主义的实践中，加深了对什么是社会主义、\n怎样建设社会主义和建设什么样的党、怎样建设党的认识，积累了治\n $治国新的宝贵经验，形成了“三个代表”重要思想。“三个代\n表”重要思想是对马克思列宁主义、毛泽东思想、邓小平理论的继承\n和发展，反映了当代世界和中国的发展变化对党和国家工作的新要\n求，是加强和改进党的建设、推进我国社会主义自我完善和发展的强\n大理
# 论武器，是中国共产党集体智慧的结晶，是党必须长期坚持的\n指导思想。始终做到“三个代表”，是我们党的立党之本、执政之基、\n力
# 量之源。""",
#     """【中国共产党的中心任务】\n党的二十大报告指出，$现在起，中国共产党的中心任务就是团\n结带领全国各族人民全面建成社会主义现代化强国、实现第二个百年\n奋斗目标，以中国式现代
# 化全面推进中华民族伟大复兴。【中国式现代化】\n中国式现代化，是中国共产党领导的社会主义现代化，既有各国\n现代化的共同特征
# ，更有基于自己国情的中国特色。——中国式现代化是人口规模巨大的现代化。——中国式现代化是全体人民共同富裕的现代化。——中国式现
# 代化是物质文明和精神文明相协调的现代化。——中国式现代化是人与自然和谐共生的现代化。——中国式现代化是走和平发展道路的现代化
# 。【全过程人民民主】"""
# ]
#
# # 构建prompt
# summary_prompt = prompt_builder.generate_summary_and_questions_prompt(refs)
#
# # 打印或者处理summary_prompt
# print(summary_prompt)
