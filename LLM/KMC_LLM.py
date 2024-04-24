# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
from File_manager.pdf2markdown import PDF
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
import zhipuai
import time
import sys
sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config
import dashscope
from http import HTTPStatus
# 设置您的API密钥
zhipuai.api_key = "b415a5e9089d4bcae6c287890e3073eb.9BDiJukUgt1KPOmA"


# config = Config(env='development')
# config.load_config()  # 指定配置文件的路径


# LLMs 类
class LargeModelAPIService:
    def __init__(self, config):
        self.cute_gpt_api = config.external_api_llm_ans
        self.config = config
        self.chatgpt_api = config.chatgpt_api
        self.Tyqwen_api_key = config.Tyqwen_api_key
        # 使用全局logger实例
        self.logger = self.config.logger

    def get_answer_from_chatgpt(self, query):
        response = requests.post(self.chatgpt_api, json={'query': query})
        self.logger.info("正在请求GPT-4>>>>>>>>>>>>>>")
        if response.status_code == 200:
            ans = response.text  # 或者 response.text，取决于响应的内容类型
            self.logger.info(f"ChatGPT回答: {ans}")
            return ans
        else:
            self.logger.error(f"请求失败，状态码: {response.status_code}")
            return None

    def get_answer_from_cute_gpt(self, prompt, loratype='qa', max_length=2048):
        response = requests.post(self.cute_gpt_api, json={'query': prompt, 'loratype': loratype}).json()
        self.logger.info("正在请求CuteGPT>>>>>>>>>>>>>>")
        ans = response['ans']
        if len(ans) >= max_length:
            ans = "没有足够的信息进行推理，很抱歉没有帮助到您。"

        self.logger.info(f"Cute-GPT回答: {ans}")
        return ans

    def get_answer_from_Tyqwen(self, prompt):
        dashscope.api_key = self.Tyqwen_api_key
        resp = dashscope.Generation.call(
            model='qwen-14b-chat',
            prompt=prompt
        )

        if resp.status_code == HTTPStatus.OK:

            # 提取并返回文本部分
            text_response = resp.output['text'] if 'text' in resp.output else 'No text available'
            self.logger.info(text_response)
            return text_response
        else:
            # 记录错误代码和错误消息
            self.logger.error(resp.code)  # 错误代码
            self.logger.error(resp.message)  # 错误消息
            return resp.message  # 返回错误消息
    def get_answer_from_Qwen(self, prompt):
        url = 'http://kmc.sundeinfo.cn/model'

        # 要发送给模型的数据
        data = {
            'text': prompt
        }
        
        # 将数据转换为JSON格式
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return(response.text)
        
        
        
    def get_answer_from_Qwen(self, prompt):
        url = 'http://kmc.sundeinfo.cn/model'

        # 要发送给模型的数据
        data = {
            'text': prompt
        }

        # 将数据转换为JSON格式并设置请求头
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # 根据响应状态码进行错误处理或数据提取
        if response.status_code == HTTPStatus.OK:
            try:
                # 提取并返回文本部分
                text_response = response.text
                self.logger.info(text_response)
                return text_response
            except json.JSONDecodeError:
                error_message = "JSON decoding failed"
                self.logger.error(error_message)
                return error_message
            
        else:
            # 记录错误代码和错误消息
            error_message = f"HTTP error {response.status_code}: {response.text}"
            self.logger.error(error_message)
            return error_message
            

    def async_invoke_chatglm(self, prompt, top_p=0.7, temperature=0.9):
        response = zhipuai.model_api.async_invoke(
            model="chatglm_turbo",
            prompt=[{"role": "user", "content": prompt}],
            top_p=top_p,
            temperature=temperature
        )
        self.logger.info("正在请求Chat-GLM>>>>>>>>>>>>>>")
        if response['code'] == 200 and response['success']:
            task_id = response['data']['task_id']
            self.logger.info(f"任务ID: {task_id}")
            return task_id
        else:
            self.logger.error(f"异步调用失败: {response['msg']}")
            return None

    def query_async_result_chatglm(self, task_id):
        # 检查任务ID是否有效
        if task_id is None:
            self.logger.error("无效的任务ID")
            return

        # 循环查询直到获取结果
        while True:
            result = zhipuai.model_api.query_async_invoke_result(task_id)
            # 检查响应的code和success键来确定调用状态
            if result['code'] == 200 and result['success']:
                # 从data字典中获取任务状态
                task_status = result['data']['task_status']

                if task_status == "SUCCESS":
                    # 任务完成，打印结果
                    self.logger.info("异步调用结果：")
                    for choice in result['data']['choices']:
                        self.logger.info(f"{choice['content']}")
                        return choice['content']
                    break
                elif task_status == "FAILED":
                    self.logger.error("异步调用失败")
                    return "调用失败或无法获取结果"  # 返回失败信息
                else:
                    self.logger.info(f"当前任务状态: {task_status}")
                    self.logger.info("等待结果...")
                    time.sleep(5)  # 每5秒查询一次
            else:
                self.logger.error(f"查询失败: {result['msg']}")
                return "查询过程出错"  # 返回错误信息

    def tong_bu(self, query):
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[{"role": "user", "content": query}],
        )
        self.logger.info(f"{response.choices[0].message}")
        return response.choices[0].message

    def text2image(self, query):
        response = client.images.generations(
            model="cogview",  # 填写需要调用的模型名称
            prompt=query,
        )

        # 保存图片到本地
        img_data = requests.get(response.data[0].url).content
        with open('ai_pic.jpg', 'wb') as handler:
            handler.write(img_data)
        self.logger.info(img_data)
        return img_data
