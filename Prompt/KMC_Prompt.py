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
        overall_instruction = "你是同济大学AI智能助理小舟。请根据给定问题描述，给出答案并以Markdown形式输出。\n"
        prompt = overall_instruction
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    prompt += f"问：{item['question']}\n答：{item['content']}\n"

        prompt += f"问：{query}\n答：\n"
        return prompt

    # 构建总结文本和推荐问题的prompt
    @staticmethod
    def generate_abstract_prompt(ref_list):
        content = "1.请根据提供的文本片段（文档的前两段和后两段）生成一个综合摘要，强调其核心要点和主要主题。\n\n" \
                  "3.仅输出摘要即可，请采用Markdown语言格式化您的回答：\n\n"
        for i, ref in enumerate(ref_list):
            content += f"{i + 1}:{ref}\n\n"

        prex = [{'role': 'system', 'content': "你是一个擅长总结文章摘要的语言专家。"},
                {'role': 'user', 'content': content}]
        return prex

    # 构建总结文本和推荐问题的prompt
    @staticmethod
    def generate_summary_and_questions_prompt(ref_list):
        content = "1.请深入分析这篇文章的开头几段，总结出其覆盖的关键要点，并确保涵盖多个维度。\n\n" \
                  "2.随后，请依据这些要点生成三个推荐问题，每个问题应直接关联文章内容且在文中能找到答案。\n\n" \
                  "3.仅输出总结和推荐问题即可，并使用序号清晰标注每个问题，输出的格式请用Markdown语言：\n\n"
        for i, ref in enumerate(ref_list):
            content += f"{i + 1}:{ref}\n\n"

        prex = [{'role': 'system', 'content': "你是一个总结文章的语言专家。"},
                {'role': 'user', 'content': content}]
        return prex

    # 构建文档问答prompt
    @staticmethod
    def generate_answer_prompt(query, refs, history):
        # 构建参考文本部分
        ref_list = [ref['text'] for ref in refs]
        refs_prompt = f"参考这一篇文章里与问题相关的以下几段文本，然后回答后面的问题：\n"
        for i, ref in enumerate(ref_list):
            refs_prompt += f"[{i + 1}]:{ref}\n"

        # 构建历史对话部分
        history_prompt = ""
        if history:
            history_prompt = "参考以下历史对话内容：\n"
            for item in history:
                if 'question' in item and 'content' in item:
                    history_prompt += f"用户：{item['question']}\n助手：{item['content']}\n"

        # 构建最终的prompt
        final_prompt = (f"{history_prompt}\n{refs_prompt}\n你应当尽量用原文回答，并对回答的结构和内容进行完善和润色，以markdown"
                        f"语言输出，语言风格更加贴合人们之间的日常交流。问题：{query}\n")

        return final_prompt

    @staticmethod
    def generate_answer_prompt_un_refs(query, history):

        # 构建历史对话部分
        history_prompt = ""
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    history_prompt += f"用户：{item['question']}\n助手：{item['content']}\n"

        # 构建最终的prompt
        final_prompt = f"{history_prompt}\n用户：{query}\n助手："

        return final_prompt

    # 构建上图问答prompt
    @staticmethod
    def generate_ST_answer_prompt(query, refs, history):
        # 构建历史对话部分
        history_prompt = ""
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    history_prompt += f"根据之前的对话记录：\n用户：{item['question']}\n助手：{item['content']}\n"

        # 构建参考文本部分，包括文档元数据和文本内容
        refs_prompt = ""
        for i, ref in enumerate(refs):
            metadata = f"标题: {ref['title']}，作者: {ref['author']}，年份: {ref['year']}，出版社: {ref['publisher']}\n"
            ref_text = f"{metadata}{ref['text']}\n"
            refs_prompt += f"[{i + 1}]: {ref_text}"

        # 构建最终的 prompt
        final_prompt = f"{history_prompt}参考以下文本片段，它们来自不同的文档，请根据这些信息和历史记录回答问题。\n{refs_prompt}\n问题：{query}\n"
        print("最终的prompt：", final_prompt)

        return final_prompt

    @staticmethod
    def generate_beauty_prompt(query):
        # 构建最终的prompt
        beauty_prompt = f"对于回答：{query}，请在不改变原文的基础上，对回答的结构和语言进行美化和完善，使人感到更加回答更加全面和贴切使用者提问的语境，仅输出修改后的回答，不要输出任何其他内容："

        return beauty_prompt

    @staticmethod
    def generate_title_prompt(content):
        # 构建标题重写prompt
        prex = [
            {'role': 'system', 'content': "你是一个擅长为对话生成标题的语言专家。"},
            {'role': 'user',
             'content': f"请根据下述内容生成一个标题，标题应简洁、准确地反映对话主题，且不超过10个字：\n{content}\n请直接输出标题。"}
        ]
        return prex

    @staticmethod
    def generate_domain_and_triplets_prompt(doc_list):
        # 检查doc_list是否至少有一段
        if not doc_list:
            return "文档中没有足够的内容以生成领域和三元组示例。"

        # 开始构建prompt
        prex = ("你是一个专门从文本中识别学科领域并构建知识点三元组的专家。\n\n"
                "任务：\n"
                "1. 读取并分析下面的文本段落，确定它们所讨论的主要学科领域（如物理、化学、政治等）。\n"
                "2. 确定领域后，基于这个领域构造三个示例知识点三元组，每个三元组包括两个知识点和它们之间的关系（如“前置”、“包含”、“相关”）。\n"
                "3. 回答仅输出结果，并按照下文的格式输出。\n\n"
                "知识点1, 关系, 知识点2,"
                "知识点3, 关系, 知识点4,"
                "文本段落：\n")

        # 如果有至少一段内容，加入到提示中
        number_of_paragraphs = min(len(doc_list), 2)  # 取前两段或更少
        for i, paragraph in enumerate(doc_list[:number_of_paragraphs], 1):
            text_content = paragraph['text']  # 从字典中提取text字段
            prex += f"{i}. {text_content}\n"

        return prex

    @staticmethod
    def generate_extract_prompt(domain_example, text_segment):
        prompt = (
            f"以下是从某领域的教材中提取的一段信息：{domain_example}。请基于这部分信息抽取提供的文本段落中该领域知识点之间的三元组关系。你需要注意的是：\n"
            "1. 关系总共有三种：前置，包含与相关。其中包含关系是指某个知识点的内容涵盖了另一个知识点，前置关系是指要想学习某个知识点，要先学会他的前置知识点，相关关系是指这两个知识点之间有联系，但并不是包含关系和前置关系。\n"
            "2. 同一对知识点之间只能存在一种关系。\n"
            "3. 你只需要返回json形式的知识点关系，不要有其他的任何文字。\n"
            "4. 关系数不能少于知识点数。\n"
            "5. 回答仅输出结果，并务必使用以下的json格式进行输出。\n\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"id\": 1, \"name\": \"知识点1\"},\n"
            "    {\"id\": 2, \"name\": \"知识点2\"}\n"
            "    // 添加更多知识点\n"
            "  ],\n"
            "  \"links\": [\n"
            "    {\"source\": 1, \"target\": 2, \"name\": \"前置\"}\n"
            "    // 添加更多关系\n"
            "  ]\n"
            "}\n"
            "提供的文本段落如下：\n"
            f"{text_segment}\n\n"
            )
        return prompt
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
