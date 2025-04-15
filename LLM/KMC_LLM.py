# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import jieba.posseg as pseg
import tempfile
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
from langchain_core.tools import tool
import threading
import spacy
from config.KMC_config import Config
import dashscope
from dashscope import Generation
from http import HTTPStatus
import random
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import time
# 设置您的API密钥
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="5911de66da26821dc50121429e6ce856.M3ZIx6kKM9WbpfiO") # 填写您自己的APIKey
os.environ["HUNYUAN_API_KEY"] = "sk-Sf4nFin9Pl6UjmCxGF7zQS1VF1XL8T47aWPnyk7cqX19XH68"

# LLMs 类
class LargeModelAPIService:
    def __init__(self, config):
        self.cute_gpt_api = config.external_api_llm_ans
        self.config = config
        self.chatgpt_api = config.chatgpt_api
        self.Tyqwen_api_key = config.Tyqwen_api_key
        # 使用全局logger实例
        self.logger = self.config.logger

        # 初始化队列和线程
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()
        self.yt_model_url = "http://192.168.121.245:30342/stream"

    def worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            method, prompt, result_queue, *optional_params = task
            if len(optional_params) == 2:
                top_p, temperature = optional_params
            else:
                top_p, temperature = 0.8, 0  # 默认值
            # self.logger.info(f"Processing task: method={method}")
            if method == 'stream':
                result = self._get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
            else:
                result = self._get_answer_from_Tyqwen(prompt)
            result_queue.put(result)
            self.task_queue.task_done()
            # self.logger.info(f"Task completed: method={method}")
            time.sleep(5)

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

    def _get_answer_from_Tyqwen(self, prompt):
        try:
            with requests.Session() as session:
                session.keep_alive = False  # 每次请求后关闭连接
                dashscope.api_key = self.Tyqwen_api_key
                resp = dashscope.Generation.call(
                    model='qwen-plus',
                    messages=prompt,
                    # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                    seed=random.randint(1, 10000),
                    # 将输出设置为"message"格式
                    result_format='message')

                if resp.status_code == HTTPStatus.OK:
                    return resp["output"]["choices"][0]["message"]["content"]
                else:
                    self.logger.error(
                        f"Request id: {resp.request_id}, Status code: {resp.status_code}, Error code: {resp.code}, "
                        f"Error message: {resp.message}")
                    return None  # 如果请求失败，返回 None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求失败: {e}")
            return f"请求失败: {e}"

    def _get_answer_from_Tyqwen_stream(self, prompt, top_p=0.8, temperature=0):
        dashscope.api_key = self.Tyqwen_api_key
        try:
            response_generator = dashscope.Generation.call(
                model='qwen-max',
                top_p=top_p,
                temperature=temperature,
                messages=prompt,
                enable_search=True,
                result_format='message',  # 设置输出为 'message' 格式
                stream=True  # 开启流式输出
            )
            previous_output = ""

            for response in response_generator:
                with requests.Session() as session:
                    session.keep_alive = False  # 每次请求后关闭连接
                    if response.status_code == HTTPStatus.OK:
                        current_output = response.output.choices[0]['message']['content']
                        if len(current_output) >= len(previous_output):
                            new_output = current_output[len(previous_output):]
                        else:
                            # 文本回退的情况
                            new_output = current_output
                        previous_output = current_output
                        yield new_output
                    else:
                        yield f"Error {response.code}: {response.message}\n"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"大模型流式输出失败: {prompt}, error: {e}")
            yield f"请求模型失败: {e}"

    def _get_answer_from_yt_model_stream(self, prompt, top_p=0.8, temperature=0):
        headers = {"Content-Type": "application/json; charset=utf-8"}

        # 准备请求数据
        data = {
            "messages": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "request_id": "16461"
        }

        # 将数据转换为JSON格式
        data = json.dumps(data)

        try:
            # 发起流式请求
            response = requests.post(self.yt_model_url, headers=headers, data=data, stream=True)

            # 检查响应是否成功
            if response.status_code != HTTPStatus.OK:
                yield f"Error {response.status_code}: {response.text}\n"
                return

            full_answer = ""  # 用于拼接完整的回答
            previous_output = ""  # 用于保存上一块的输出，检查回退情况

            # 逐块处理响应数据
            for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                if chunk:
                    # 移除数据前缀
                    chunk = chunk.replace("data:", "").strip()

                    # 如果是结束标记 [DONE]，则停止
                    if "[DONE]" in chunk:
                        break

                    try:
                        # 解析JSON数据
                        response_data = json.loads(chunk)

                        # 处理模型返回的消息
                        if "message" in response_data:
                            current_output = response_data["message"]

                            # 增量拼接结果
                            if len(current_output) >= len(previous_output):
                                new_output = current_output[len(previous_output):]
                            else:
                                # 如果发生回退（文本变短），直接返回完整内容
                                new_output = current_output

                            # 更新之前的输出
                            previous_output = current_output
                            full_answer += new_output

                            # 返回每个增量数据
                            yield new_output

                    except json.JSONDecodeError:
                        continue  # 如果无法解析JSON数据，跳过该块

        except requests.exceptions.RequestException as e:
            yield f"请求模型失败: {e}\n"
    
    # def get_answer_from_yt_model_stream(self, prompt, top_p=0.8, temperature=0):
    #     """
    #     公共方法：从模型获取流式响应并打印结果
    #     """
    #     # 调用私有方法获取流式响应
    #     for result in self._get_answer_from_yt_model_stream(prompt, top_p, temperature):
    #         # 输出流式结果
    #         return result

    # def _get_answer_from_yt_model(self, prompt, top_p=0.8, temperature=0):
    #     """
    #     非流式方法：从模型获取响应数据
    #     直接获取完整响应数据并返回
    #     """
    #     header = {"Content-Type": "application/json; charset=utf-8"}
    #     # 构建请求数据
    #     data = {
    #         "messages": prompt,
    #         "temperature": temperature,
    #         "top_p": top_p,
    #         "request_id": "16461",
    #         "result_format": 'message',  # 设置输出为 'message' 格式
    #         "stream": False  # 非流式请求
    #     }
    #     d = json.dumps(data)
    #
    #     start = time.time()
    #     try:
    #         # 发送POST请求，获取完整响应数据
    #         response = requests.post(self.yt_model_url, data=d, headers=header)
    #         response.raise_for_status()  # 如果响应错误，抛出异常
    #     except requests.exceptions.RequestException as e:
    #         self.logger.error(f"Request failed: {e}")
    #         return None
    #
    #     end = time.time()
    #     print("Time taken:", end - start)
    #
    #     # 解析并返回响应内容
    #     try:
    #         res = response.json()
    #         if res["code"] == 200:
    #             return res["message"]
    #         else:
    #             self.logger.error(f"Error in response: {res}")
    #             return None
    #     except ValueError:
    #         self.logger.error(f"Failed to decode response: {response.text}")
    #         return None

    def get_answer_from_Yitong_stream(self, prompt, top_p=0.9, temperature=0.1):
        header = {"Content-Type": "application/json; charset=utf-8"}

        messages = [
            {"role": "system",
             "content": "你是亿语通航，是亿通国际自主研发的航贸大语言模型，专注于航运贸易领域的各类问答，为用户创造价值。"},
            {"role": "user", "content": prompt}
        ]

        data = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "request_id": "16461"  # request_id 可以根据需要动态生成
        }

        payload = json.dumps(data)

        try:
            response = requests.post(
                url="http://192.168.121.245:30342/stream",
                headers=header,
                data=payload,
                stream=True
            )
            # 若响应状态不为200，则抛出HTTPError
            response.raise_for_status()

            previous_output = ""
            # 逐块读取响应数据
            for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                # 如果包含结束标识，则退出循环
                if "[DONE]" in chunk:
                    break

                chunk = chunk.strip()
                if not chunk:
                    continue

                # 去除可能的 "data:" 前缀
                if chunk.startswith("data:"):
                    chunk = chunk[len("data:"):].strip()

                try:
                    data_chunk = json.loads(chunk)
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}, chunk: {chunk}")
                    continue

                # 从解析后的数据中获取 message 字段
                message = data_chunk.get("message", "")
                if not message:
                    continue

                # 如果返回的内容是累计型的，则仅提取新增部分
                if message.startswith(previous_output):
                    new_text = message[len(previous_output):]
                else:
                    new_text = message

                previous_output = message

                # 实时yield新增的文本块
                yield new_text

        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
            yield f"HTTP error occurred: {http_err}"
        except requests.exceptions.ConnectionError as conn_err:
            self.logger.error(f"Connection error occurred: {conn_err}")
            yield f"Connection error occurred: {conn_err}"
        except requests.exceptions.Timeout as timeout_err:
            self.logger.error(f"Timeout error occurred: {timeout_err}")
            yield f"Timeout error occurred: {timeout_err}"
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request failed: {req_err}")
            yield f"Request failed: {req_err}"
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            yield f"Unexpected error: {e}"

    def get_answer_from_Tyqwen(self, prompt):
        result_queue = queue.Queue()
        self.task_queue.put(('non_stream', prompt, result_queue))
        result = result_queue.get()
        return result

    def get_answer_from_Tyqwen_stream(self, prompt, top_p=0.8, temperature=0):
        result_queue = queue.Queue()
        self.task_queue.put(('stream', prompt, result_queue, top_p, temperature))
        result = result_queue.get()
        return result

    def get_answer_from_deepseek(self, prompt, top_p=0.8, temperature=0.6):
        """
        使用 ollama 调用 deepseek-r1:32b 模型，非流式，一次性返回完整结果（适合 JSON 格式响应）。
        """
        url = "http://106.14.20.122/37-11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek-r1:32b",
            "messages": prompt,
            "top_p": top_p,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            try:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    return {"error": "模型未返回结果"}
            except Exception as e:
                return {"error": f"解析模型响应失败: {str(e)}"}
        else:
            return {"error": f"请求失败，状态码：{response.status_code}"}

    def get_answer_from_deepseek_stream(self, prompt, top_p=0.8, temperature=0.6):
        """
        使用 ollama 调用 deepseek-r1:32b 模型实现流式输出。
        """
        url = "http://106.14.20.122/37-11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek-r1:32b",
            "messages": prompt,
            "top_p": top_p,
            "temperature": temperature,
            "stream": True
        }
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    # 处理每一行数据
                    line = line.decode('utf-8').strip()
                    # 跳过包含 <think> 的数据行
                    if "<think>" in line:
                        yield line

                    if line.startswith('data:'):
                        # 提取 JSON 部分
                        json_str = line[5:].strip()
                        # 如果返回 [DONE]，则退出循环
                        if json_str == "[DONE]":
                            break

                        if json_str:
                            try:
                                response_data = json.loads(json_str)
                                if 'choices' in response_data:
                                    for choice in response_data['choices']:
                                        if 'delta' in choice and 'content' in choice['delta']:
                                            yield choice['delta']['content']
                            except json.JSONDecodeError:
                                yield "解析 JSON 时出错"
        else:
            yield f"请求失败，状态码：{response.status_code}"

    def shutdown(self):
        self.task_queue.put(None)
        self.worker_thread.join()

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
                response_data = response.json()
                text_response = response_data['response'] if 'response' in response_data else 'No response available'
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

    def web_search_glm4(self, query):
        tools = [{
            "type": "web_search",
            "web_search": {
                "enable": True,  # 禁用：False，启用：True，默认为 True。
                "search_result": True  # 禁用：False，启用：True，默认为禁用
            }
        }]

        response = client.chat.completions.create(
            model="glm-4-air",
            messages=[
                {"role": "system", "content": "You are a helpful assistant.\n\nCurrent Date: 2024-07-11"},
                {"role": "user", "content": query}
            ],
            top_p=0.7,
            temperature=0.3,
            tools=tools
        )

        # 初始化结果字典
        result = {
            "model_response": None,
            "web_search_results": []
        }

        # 处理模型的回答
        if response.choices:
            model_response = response.choices[0].message.content
            result["model_response"] = model_response
            self.logger.info(f"模型回答: {model_response}")

            # 处理 web_search 结果（仅当存在时）
        if hasattr(response, 'web_search') and response.web_search:
            for search_result in response.web_search:
                search_response = {
                    "title": search_result['title'],
                    "content": search_result['content'],
                    "link": search_result.get('link', '')
                }
                result["web_search_results"].append(search_response)
                self.logger.info(f"网络搜索结果: {search_response}")
        else:
            self.logger.warning("没有找到 web_search 结果或 web_search 属性不存在")

        return result


    def function_calling(self, query):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        ]
        dashscope.api_key = self.Tyqwen_api_key
        messages = [{"role": "user", "content": query}]

        response = dashscope.Generation.call(
            model='qwen-max',
            top_p=0.7,
            temperature=1.0,
            messages=messages,
            enable_search=True,
            tools=tools,
            result_format='message',  # 设置输出为 'message' 格式
            # stream=True  # 开启流式输出
        )
        print(response)

    def get_answer_from_hunyuan_stream(self, prompt, top_p=0.8, temperature=0.6):
        client = OpenAI(
            api_key=os.environ.get("HUNYUAN_API_KEY"), # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1", # 混元 endpoint
        )
                # 自定义参数传参示例
        completion = client.chat.completions.create(
            model="hunyuan-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream = True,
            top_p = top_p,
            temperature = temperature,
            extra_body={
                "enable_enhancement": True
            },# <- 自定义参数
        )
        # 流式响应的标准处理方式：
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)


    def get_answer_from_internvl_stream(self, prompt, top_p=0.9, temperature=0.8):
        client = OpenAI(
            api_key="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MzMwODYzMCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0NDcyOTA5NCwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTM5NDA4Nzg5OTEiLCJvcGVuSWQiOm51bGwsInV1aWQiOiIyZjcyZWI4OS05NzkyLTRhYmMtODM0MC05NzU4NDIzNWVkODciLCJlbWFpbCI6IiIsImV4cCI6MTc2MDI4MTA5NH0.kFVEFfm9yB08ZDQtEn6cXAQSrKX5YFzU7C_LhLPKgAEsNPgmQjKyhizmHDYlPeHLQqhz1rAJeYQfKGUUsLwwrw",  # 环境变量中设置 token，不带 Bearer
            base_url="https://chat.intern-ai.org.cn/api/v1/",  # InternVL 的接口地址
        )

        # 发起流式请求
        completion = client.chat.completions.create(
            model="internvl2.5-latest",
            messages=prompt,
            stream=True,
            top_p=top_p,
            temperature=temperature,
        )

        # 处理流式输出
        for chunk in completion:
            # OpenAI 返回的 chunk 是 ChatCompletionChunk 类型，应该通过属性访问
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content'):
                    content = delta.content
                    # 处理流式输出的数据格式
                    # 检查是否是 [DONE] 标志
                    if content == "[DONE]":
                        yield '{"code": 200, "msg": "success", "data": {"text": "finish!"}}'
                        break  # 停止流式输出

                    # 返回每个chunk的内容，格式为 JSON
                    yield f'{{"code": 200, "msg": "success", "data": {{"text": "{content}"}}}}\n'

# config = Config(env='production')
# config.load_config("/home/ubuntu/work/kmcGPT/KMC/config/config.json")  # 指定配置文件的路径
# # 创建ElasticSearchHandler实例
# llm_builder = LargeModelAPIService(config)
# # 定义要发送给 deepseek 模型的 prompt
# prompt = [
#         {
#             "role": "user",
#             "content": "你好"
#         },
#         {
#             "role": "assistant",
#             "content": "你好，我是 internvl"
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Describe the image please"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://static.openxlab.org.cn/internvl/demo/visionpro.png"
#                     }
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://static.openxlab.org.cn/puyu/demo/000-2x.jpg"
#                     }
#                 }
#             ]
#         }
#     ]
# llm_builder.get_answer_from_internvl_stream(prompt)


