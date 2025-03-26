# -*- coding: utf-8 -*-
import sys
import urllib3
from flask import Flask, request, jsonify, Response, stream_with_context, current_app
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import threading
import queue
import logging
import requests
from logging.handlers import RotatingFileHandler
import re
import uuid
import jieba.posseg as pseg
from FlagEmbedding import FlagReranker
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)
CORS(app)
# 加载配置
# 使用环境变量指定环境并加载配置
config = Config(env='production')
config.load_config()  # 指定配置文件的路径
config.load_predefined_qa()
logger = config.logger
record_path = config.record_path
# 创建 FileManager 实例
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)


@app.route('/api/AI_generate_question', methods=['POST'])
def NL2test():
    try:
        # Get the question from the request payload
        data = request.get_json()
        question = data.get('question', '')
        difficulty_level = data.get('difficulty_level', '中等')  # 难度等级
        question_type = data.get('question_type', '单选')  # 题型
        question_count = data.get('question_count', 5)  # 题目数量
        llm = data.get('llm', 'deepseek')
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.6)

        if not question:
            return jsonify({'error': '参数不完整'}), 400

        prompt = PromptBuilder.generate_test_prompt(
            query=question,
            difficulty_level=difficulty_level,
            question_type=question_type,
            question_count=question_count
        )

        # 使用 Response 和 stream_with_context 来处理流式响应
        def generate():
            try:
                full_answer = ""
                if llm == 'deepseek':
                    ans_generator = large_model_service.get_answer_from_deepseek_stream(
                        prompt,
                        top_p=top_p,
                        temperature=temperature
                    )
                if llm == 'qwen':
                    ans_generator = large_model_service.get_answer_from_Tyqwen_stream(
                        prompt,
                        top_p=top_p,
                        temperature=temperature
                    )

                for chunk in ans_generator:
                    full_answer += chunk
                    yield chunk

            except Exception as e:
                # 在生成器内部捕获异常，但不要在这里使用 jsonify
                current_app.logger.error(f"Stream generation error: {str(e)}")
                yield str({'status': 'error', 'error': f'生成试题时发生错误: {str(e)}'})

        return Response(
            stream_with_context(generate()),
            content_type='text/plain'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'生成试题时发生错误: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5777, debug=False)
