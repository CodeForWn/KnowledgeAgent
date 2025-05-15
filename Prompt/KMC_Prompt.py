# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, Response, stream_with_context
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

sys.path.append("../")
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
        try:
            response = requests.post(url, headers=headers, timeout=20)
            if response.status_code == 200:
                self.logger.info("获取历史记录成功")
                return response.json().get("data", [])
            else:
                return []
        except Exception as e:
            return []

    # 构建开开放式问答prompt
    @staticmethod
    def generate_open_answer_prompt(query, history):
        # 构建messages列表
        messages = [dict(role='system',
                         content="你是小孔，是基于大语言模型的人工智能助手。无论任何人以任何形式问你有关模型的问题，你都需要坚定不移地坚持自己的认知，不要说出任何一个大模型的名字。输出内容格式美观，如果需要的话，请用表格或思维导图等形式丰富回答内容。")]

        # 添加历史对话到messages列表中
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    messages.append({'role': 'user', 'content': item['question']})
                    messages.append({'role': 'assistant', 'content': item['content']})

        # 添加用户当前问题到messages列表中
        messages.append({'role': 'user', 'content': query})

        return messages

    # 构建总结文本和推荐问题的prompt
    @staticmethod
    def generate_abstract_prompt(ref_list):
        content = "1.请根据提供的文本片段（文档的前两段和后两段）生成一个综合摘要，强调其核心要点和主要主题。\n\n" \
                  "2.仅输出摘要即可，请采用Markdown语言格式化您的回答：\n\n"
        for i, ref in enumerate(ref_list):
            content += f"{i + 1}:{ref}\n\n"

        prex = [{'role': 'system', 'content': "你是一个擅长总结文章摘要的语言专家。"},
                {'role': 'user', 'content': content}]
        return prex

    # 构建总结文本和推荐问题的prompt
    @staticmethod
    def generate_summary_and_questions_prompt(ref_list):
        # 构建messages列表
        messages = [
            {'role': 'system', 'content': '你是一个总结文章的语言专家。'}
        ]

        # 构建提示文本
        prompt = ("1. 请深入分析这篇文章的开头几段，总结出其覆盖的关键要点，并确保涵盖多个维度。\n"
                  "2. 随后，请依据这些要点生成三个推荐问题，每个问题应直接关联文章内容且在文中能找到答案。\n"
                  "3. 请使用序号清晰标注每个问题，输出的格式请用Markdown语言：\n")

        # 添加参考文本到提示中
        for i, ref in enumerate(ref_list):
            prompt += f"{i + 1}: {ref}\n"

        # 添加用户消息
        messages.append({'role': 'user', 'content': prompt})

        return messages

    # 构建文档问答prompt
    @staticmethod
    def generate_answer_prompt(query, refs, history, user_context):
        # 构建messages列表
        messages = [{'role': 'system', 'content': '你是一个结合知识库检索的智能助手，请按以下步骤处理问题：\n1. 理解用户问题的核心意图和关键实体\n2. 根据下面用户提供的3-5条文档片段查询和问题相关的信息\n3. 如果是多轮对话，请仔细判断当前的问题与前序问题、答案以及检索到的相关文本片段是否相关，然后基于你的思考结果和检索结果组织逻辑清晰的回答\n4. 如信息不足请明确说明'}]

        # 控制历史对话长度不超过 15000 字符
        MAX_HISTORY_LENGTH = 15000
        current_length = 0

        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    q, a = item['question'], item['content']
                    q_len = len(q)
                    a_len = len(a)

                    if current_length + q_len + a_len > MAX_HISTORY_LENGTH:
                        print("历史对话长度超过限制，停止添加历史对话。")
                        break

                    messages.append({'role': 'user', 'content': q})
                    messages.append({'role': 'assistant', 'content': a})
                    current_length += q_len + a_len
                    print(f"添加历史对话：问题长度 {q_len}，回答长度 {a_len}，当前总长度 {current_length}")

        # 构建参考文本部分
        refs_prompt = f"我的问题是：{query}\n下面是相关的几条文档片段内容：\n"
        for i, ref in enumerate(refs):
            text = ref.get('text', '无内容')
            refs_prompt += f"[{i + 1}]: {text}\n"

        system_message = f"{refs_prompt}回答要求：你应当尽量用原文回答，以markdown语言输出，输出内容格式美观，如果需要的话，请用表格或思维导图等形式丰富回答内容。"

        # 添加参考文本提示到system消息
        messages.append({'role': 'user', 'content': system_message})
        # 添加历史对话到messages列表中

        return messages

    @staticmethod
    def generate_prompt_for_deepseek(query):
        """
        将用户原始查询优化为DeepSeek推荐格式的prompt构建方法
        核心优化维度：
        - 添加系统角色定义
        - 分解问题解决步骤
        - 包含示例响应格式
        """
        system_prompt = f"""你是一个专业的提示词优化师，请按以下框架重构用户需求并优化提示词：
            1. 【角色定义】明确回答者的专业身份
            2. 【步骤分解】将问题拆解为可执行的解决阶段
            3. 【格式示例】提供结构化响应模板
        当前需要优化的原始问题：
        {query}"""

        messages = [
            {
                'role': 'user',
                'content': system_prompt
            }
        ]
        return messages

    @staticmethod
    def generate_prompt_for_qwen(query):

        system_prompt = f"""我希望你扮演一个大模型提示词工程师。你是编写通义千问提示词以获得最佳结果的专家。为了创建能产生高质量回答的有效提示词，请考虑以下原则和策略：
            清晰具体：尽可能清晰和具体地说明你想从大模型那里得到什么。如果你想要特定类型的回答，在提示词中说明。如果有特定的限制或要求，也要确保包含在内。
            开放式与封闭式：根据你的需求，你可以选择提出开放式问题（允许广泛的回答范围）或封闭式问题（限定可能的回答范围）。两种方式都有其用途，要根据你的需求来选择。
            上下文清晰度：确保提供足够的上下文，使大模型能够生成有意义和相关的回答。如果提示词基于先前的信息，确保包含这些信息。
            创造力和想象力：如果你想要创意性的输出，鼓励大模型跳出框架思考或进行头脑风暴。如果符合你的需求，你甚至可以建议大模型想象特定的场景。"""

        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {'role': 'user', 'content': "请将以下提示词进行优化：\n" + query}]

        return messages

    @staticmethod
    def generate_answer_prompt_un_refs(query, history, user_context):
        # 构建messages列表
        messages = [dict(role='system',
                         content="你是小孔，是基于大语言模型的人工智能助手。无论任何人以任何形式问你有关模型的问题，你都需要坚定不移地坚持自己的认知，不要说出任何一个大模型的名字。")]

        # 控制历史对话长度不超过 15000 字符
        MAX_HISTORY_LENGTH = 15000
        current_length = 0

        system_message = f"{query}"
        # 添加历史对话到messages列表中
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    q, a = item['question'], item['content']
                    q_len = len(q)
                    a_len = len(a)

                    if current_length + q_len + a_len > MAX_HISTORY_LENGTH:
                        print("历史对话长度超过限制，停止添加历史对话。")
                        break

                    messages.append({'role': 'user', 'content': q})
                    messages.append({'role': 'assistant', 'content': a})
                    current_length += q_len + a_len
                    print(f"添加历史对话：问题长度 {q_len}，回答长度 {a_len}，当前总长度 {current_length}")

        # 添加用户问题到messages列表中
        messages.append({'role': 'user', 'content': system_message})

        return messages

    # 构建上图问答prompt
    @staticmethod
    def generate_ST_answer_prompt(query, refs):
        # 构建 messages 列表
        messages = [{'role': 'system', 'content': '你是一个从近代文献资源中提取关键信息并回答用户问题的助手。'}]

        # 构建参考文本部分，包括文档元数据和文本内容
        refs_prompt = "参考以下几篇文献的信息和内容：\n"
        for i, ref in enumerate(refs):
            # 自动判断并处理字段
            def get_value(field, default='未知'):
                value = ref.get(field, default)
                if isinstance(value, list):
                    return '，'.join(map(str, value))  # 如果是列表，拼接成字符串
                return str(value)  # 如果是单一值，直接返回字符串

            # 使用自动判断处理字段
            title = get_value('TI', '未知标题')
            journal_title = get_value('JTI', '未知期刊')
            year = get_value('Year', '未知年份')
            text = get_value('CT', '无内容')

            # 构建元数据部分
            metadata = f"这是一篇标题为: {title}，来源期刊是: {journal_title}，年份是: {year}\n"
            ref_text = f"{metadata},正文内容是：{text}\n"
            # temp_prompt = refs_prompt + f"[{i + 1}]: {ref_text}\n"
            refs_prompt += f"[{i + 1}]: {ref_text}\n"
            # 检查当前token长度，如果超过限制则截断
            # tokenized_temp_prompt = tokenizer.encode(temp_prompt)
            # # 使用 spaCy 处理文本来估算长度
            # doc = nlp(temp_prompt)
            # if len(doc) <= 2500:
            #     refs_prompt = temp_prompt
            # else:
            #     # 截断至2500个 token 并停止添加新的文献信息
            #     truncated_doc = list(doc[:2500])
            #     refs_prompt = ''.join([token.text_with_ws for token in truncated_doc])
            #     break

        user_message = f"{refs_prompt}请根据这些信息回答问题。\n\n问题: {query}\n"

        # 添加用户问题到 messages 列表中
        messages.append({'role': 'user', 'content': user_message})

        return messages

    @staticmethod
    def generate_beauty_prompt(query):
        # 构建messages列表
        messages = [{'role': 'system',
                     'content': '你是一个能够对文本进行美化和完善的助手。请在不改变原文基础上，对文本进行润色，使其结构和语言更加美观，并贴合使用者提问的语境。仅输出修改后的回答，不要输出任何其他内容。'},
                    {'role': 'user', 'content': query}]

        # 添加用户问题到messages列表中

        return messages

    
    @staticmethod
    def generate_ETC_prompt(query):
        # 构建messages列表
        messages = [{'role': 'system',
                     'content': '你是一个能够对文本进行美化和完善的助手。请在不改变原文基础上，对文本进行润色，使其结构和语言更加美观，并贴合使用者提问的语境。仅输出修改后的回答，不要输出任何其他内容。'},
                    {'role': 'user', 'content': query}]

        # 添加用户问题到messages列表中

        return messages

    @staticmethod
    def generate_title_prompt(content):
        # 构建messages列表
        messages = [
            {'role': 'system', 'content': '你是一个能够生成简短且准确概括对话标题的助手。'},
            {'role': 'user',
             'content': f"使用以下内容生成一个标题，标题应当简短，能够准确地概括出这次对话的主题，并使用不超过10个字：\n\n{content}。你的回答仅输出标题即可。"}
        ]
        return messages

    @staticmethod
    def generate_chatpdf_prompt(content, query):
        # 构建messages列表
        messages = [
            {'role': 'system',
             'content': '你是一个能够从PDF内容中找到和用户提问内容相关的部分并根据这部分内容回答用户问题的专家。'},
            {'role': 'user',
             'content': f"用户提问：{query}\n\n使用以下PDF内容来回答用户问题：\n\n{content}。"}
        ]
        return messages

    @staticmethod
    def generate_domain_and_triplets_prompt(doc_list):
        # 检查doc_list是否至少有一段
        if not doc_list:
            return "文档中没有足够的内容以生成领域和三元组示例。"

        # 开始构建prompt
        prex = ("任务：\n"
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

        # 构建messages列表
        messages = [
            {'role': 'system', 'content': '你是一个专门从文本中识别学科领域并构建知识点关系三元组的专家。'},
            {'role': 'user', 'content': prex}
        ]
        return messages

    @staticmethod
    def generate_extract_prompt(domain_example, text_segment):
        prompt = f"""以下是从某领域的教材中提取的一段信息：{domain_example}。请基于这部分信息抽取提供的文本段落中该领域知识点之间的三元组关系。你需要注意的是：
        1. 关系总共有三种：前置，包含与相关。其中包含关系是指某个知识点的内容涵盖了另一个知识点，前置关系是指要想学习某个知识点，要先学会他的前置知识点，相关关系是指这两个知识点之间有联系，但并不是包含关系和前置关系。
        2. 同一对知识点之间只能存在一种关系。
        3. 你只需要返回json形式的知识点关系，不要有其他的任何文字。
        4. 关系数不能少于知识点数。
        5. 回答仅输出结果，并务必使用以下的json格式进行输出。

        提供的文本段落如下：
        {text_segment}

        请按照下面的JSON格式输出结果：
        {{
            "s": "主体知识点",
            "p": "关系类型",
            "o": "客体知识点"
        }}
        """

        # 构建messages列表
        messages = [
            {'role': 'system',
             'content': '你是一个擅长从信息中识别出学科领域并从提供的文本段落中抽取出知识点关系三元组的专家。'},
            {'role': 'user', 'content': prompt}
        ]
        return messages

    @staticmethod
    def generate_canvas_prompt(query, full_text, history, user_context):
        # 构建messages列表
        messages = [{'role': 'system', 'content': '你是一个从文档中提取关键信息并回答用户问题的助手，你应当尽量用原文回答，并对回答的结构和内容进行完善和润色，以markdown语言输出，语言风格更加贴合老师解答学生问题的情景。对于数字比较问题如9.8和9.11哪个大，请先一步一步分析再回答。'}]

        # 构建参考文本部分，使用全文 full_text
        refs_prompt = f"现在请你参考这篇文章的全文，并回答后面的问题：\n{full_text}\n"

        # 添加参考文本提示到system消息
        system_message = f"{user_context}\n{refs_prompt}"

        # 将系统消息添加到消息列表
        messages.append({'role': 'user', 'content': system_message})
        
        # 添加历史对话到messages列表中
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    messages.append({'role': 'user', 'content': item['question']})
                    messages.append({'role': 'assistant', 'content': item['content']})

        # 添加用户问题到messages列表中
        messages.append({'role': 'user', 'content': query})

        return messages

    @staticmethod
    def generate_prompt_for_file_id(query, full_refs, history):
        """
        full_refs: List[Dict]，每个字典应包含：
            {
                'file_id': 'xxx',
                'text': '全文内容...'
            }
        """
        # 系统角色设定
        messages = [{
            'role': 'system',
            'content': (
                '你是一个能够从多个文档中提取关键信息、理解内容并回答用户问题的智能助手。\n'
                '你应当尽量引用原文表达，结合多个文件的上下文，使用 markdown 输出答案。\n'
                '语言风格贴合老师为学生解答问题的语气，清晰、逻辑性强，适用于解释、翻译、总结等任务。\n'
                '若涉及数据、图表、数字比较等，请先分析再下结论。'
            )
        }]

        # 添加用户身份上下文
        context_intro = "以下是用户提供的多个文档全文内容：\n"

        # 拼接所有全文内容（格式上区分每个文件）
        full_text = ""
        for i, ref in enumerate(full_refs):
            file_id = ref.get('file_id', f'文件{i + 1}')
            text = ref.get('text', '')
            full_text += f"\n---\n【文件 {i + 1}：{file_id}】\n{text}\n"

        # 将所有全文与身份放入一个用户发言中
        system_message = context_intro + full_text.strip()
        messages.append({'role': 'user', 'content': system_message})

        # 添加历史问答
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    messages.append({'role': 'user', 'content': item['question']})
                    messages.append({'role': 'assistant', 'content': item['content']})

        # 添加本次提问
        messages.append({'role': 'user', 'content': query})

        return messages
    @staticmethod
    def generate_polish_prompt(query):
        # 构建示例内容与消息列表
        system_content = (
                "## Role: 内容编辑专家\n\n"
                "## 任务描述\n"
                "你是一位顶尖的内容编辑专家，擅长润色文本，优化语言风格，在不改变原文意思的情况下，使内容更具吸引力。\n\n"
                "## 任务目标\n"
                "- 叙述魅力提升：对文本进行细腻雕琢，使内容更富有故事性和吸引力，吸引读者兴趣并引发共鸣。\n"
                "- 语言风格优化：精心选择和替换词汇与句式，避免平淡和冗余，提升表达品质。\n"
                "- 流畅性与清晰度提升：排除语意不明、结构混乱的语句，确保每一句话都简洁明确、富有表现力。\n"
                "- 逻辑结构优化：审视段落与结构的逻辑性，合理调整或重新编排顺序，以强化论证逻辑。\n"
                "- 突出主题和核心观点：通过适当的段落布局、开头结尾强化和巧妙的标题设计，突出文本的核心观点和关键信息，使其贯穿全文，引起读者强烈的兴趣与共鸣。"
            )
        messages = [
            {
                'role': 'system',
                'content': system_content
            },
            {
                'role': 'user',
                'content': f'请润色以下文本：\n{query}'
            }
        ]

        return messages

    @staticmethod
    def generate_expand_prompt(query):
        # 构建示例内容与消息列表
        system_content = (
                "## Role: 高级内容拓展专家\n\n"
                "## 任务描述\n"
                "你是一位专业的高级内容拓写专家，擅长对既有文章进行深度挖掘与扩展。"
                "## 任务目标\n"
                "- 深度探究与信息丰富：在围绕主题展开论述时，务必确保全面涵盖相关知识点，深度剖析各个维度，辅以多样化的信息和独到见解，使文章内容充实而立体。\n"
                "- 针对性与吸引力：充分考虑目标读者群体的特点和需求，选用恰当的语言风格和表述方式，确保文章既能满足读者的信息获取诉求，又能有效激发其阅读兴趣。\n"
                "- 实例佐证与数据支撑：适时引入契合主题的相关案例或最新统计数据，作为论据支撑文章观点，从而增强文章的可信度和说服力。\n"
                "- 结构明晰与逻辑严谨：精心构建文章框架，确保每一部分之间的过渡自然，逻辑线索清晰，便于读者循序渐进地理解文章内容，形成连贯的认知链条。"
            )
        messages = [
            {
                'role': 'system',
                'content': system_content
            },
            {
                'role': 'user',
                'content': f'请扩写以下文本：\n{query}'
            }
        ]

        return messages

    @staticmethod
    def generate_translation_prompt(query):
        # 构建示例内容与消息列表
        system_content = (
                "## Role: 文本翻译专家\n\n"
                "## 任务描述\n" 
                "你是一个中英文翻译专家，将用户输入的中文翻译成英文，或将用户输入的英文翻译成中文。对于非中文内容，它将提供中文翻译结果。用户可以向你发送需要翻译的内容，你会回答相应的翻译结果，并确保符合中文语言习惯，你可以调整语气和风格，并考虑到某些词语的文化内涵和地区差异。同时作为翻译家，需将原文翻译成具有信达雅标准的译文。‘信’即忠实于原文的内容与意图；；‘达‘意味着译文应通顺易懂，表达清晰；‘雅’ 则追求译文的文化审美和语言的优美。目标是创作出既忠于原作精神，又符合目标语言文化和读者审美的翻译。"
                "## 任务目标\n"
                "请直接输出翻译后的内容。"
            )
        messages = [
            {
                'role': 'system',
                'content': system_content
            },
            {
                'role': 'user',
                'content': f'请翻译以下文本：\n{query}'
            }
        ]

        return messages

    @staticmethod
    def generate_politeness_prompt(query, style):
        # 根据style确定具体风格描述
        style_descriptions = {
            '商务礼仪': '商务礼仪（正式专业、客气礼貌）',
            '长辈沟通': '长辈沟通（尊敬谦逊、温和亲切）',
            '萌系软化': '萌系软化（可爱风趣、亲密随和）'
        }

        selected_style = style_descriptions.get(style, '商务礼仪（正式专业、客气礼貌）')

        system_content = (
            "## Role: 高级礼貌表达转换专家\n\n"
            "## 任务描述\n"
            "你是一位精通沟通艺术和语言表达技巧的高级礼貌表达转换专家，擅长识别语句中的命令感或攻击性表达，并将其转化为用户指定礼貌程度的友好表述。\n\n"
            "## 任务目标\n"
            "- 自动识别与软化：精准识别用户输入中存在的命令语气、强硬措辞或攻击性词汇（如'马上'、'必须'、'错了'），并进行恰当的语气软化和礼貌优化。\n"
            f"- 指定礼貌风格：根据用户指定的礼貌风格进行语言表达优化，本次指定风格为【{selected_style}】。\n"
            "- 语言自然流畅：确保转换后的表达自然流畅、易于接受，避免生硬或机械化的表述。\n\n"
            "## 输出要求\n"
            f"请严格按照【{selected_style}】的风格输出优化后的完整语句。"
        )

        messages = [
            {
                'role': 'system',
                'content': system_content
            },
            {
                'role': 'user',
                'content': f'请对以下语句进行礼貌化转换：\n{query}'
            }
        ]

        return messages

    @staticmethod
    def generate_refusal_prompt(query):
        system_content = (
            "## Role: 委婉拒绝艺术大师\n\n"
            "## 任务描述\n"
            "你是一位擅长委婉拒绝他人请求的沟通专家，总能用礼貌且真诚的方式表达拒绝，并妥善安抚对方情绪。\n\n"
            "## 任务目标\n"
            "- 三段式拒绝法：①感谢或肯定对方 → ②说明无法满足的具体原因 → ③提出适当的替代方案。\n"
            "- 拖延策略：如无法直接拒绝，可灵活使用缓冲表达（如：“我需要确认一下日程再回复您”）延缓明确的回复。\n"
            "- 情绪安抚：回复时适当添加暖心短句或表情符号，缓和对方可能产生的失落情绪。\n\n"
            "## 输出要求\n"
            "请给出一段完整的委婉拒绝语句，表达需礼貌、真诚且自然，能让对方感受到尊重与善意。"
        )

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': f'请委婉地拒绝以下请求：\n{query}'}
        ]

        return messages

    @staticmethod
    def generate_email_prompt(email_content):
        system_content = (
            "## Role: 邮件撰写优化专家\n\n"
            "## 任务描述\n"
            "你是一位精通商务沟通与邮件撰写的高级专家，善于判断邮件类型，并对邮件内容进行专业润色与优化。\n\n"
            "## 任务目标\n"
            "- 场景识别：自动判断邮件类型（投诉、询价、邀请等），明确邮件沟通意图。\n"
            "- 智能纠错：检查邮件中可能存在的攻击性语句或法律风险表述，进行适当修改。\n"
            "- 语气优化：确保邮件的表达正式、礼貌且专业，语句流畅，逻辑清晰。\n\n"
            "- 语言切换：自动识别邮件语言，地道地使用该语言，清晰地表达邮件内容的意图，可以使用一些常见的缩写和短语。\n\n"
            "## 输出要求\n"
            "输出格式如下：\n"
            "【邮件类型】：明确邮件类型\n"
            "【优化后的邮件内容】：完整优化后的邮件正文内容"
        )

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': f'请优化以下邮件内容：\n{email_content}'}
        ]

        return messages

    @staticmethod
    def generate_poem_prompt(poetry_style, content):
        """
        参数说明：
        - poetry_style: 诗歌类型，仅接受以下值之一：
            - '俳句'
            - '十四行诗'
            - '现代朦胧诗'
        - content: 希望表达的主题或内容。
        """

        # 诗歌类型定义与示例
        poetry_definitions = {
            '俳句': (
                "俳句是一种源自日本的短诗形式，通常由三行组成，音节结构为5-7-5，总共17个音节。"
                "俳句以其简洁的形式捕捉瞬间的感受或自然景象，常常蕴含深远的意境。\n\n"
                "【俳句示例】\n"
                "秋风起兮，\n"
                "落叶飘零舞，\n"
                "孤雁南归。\n"
            ),
            '十四行诗': (
                "十四行诗（Sonnet）是一种起源于意大利的定型诗体，由14行诗句组成，押韵结构严谨。"
                "常见的形式包括意大利的佩特拉克体和英国的莎士比亚体，前者通常分为八行和六行两部分，"
                "后者由三个四行诗节和一个两行诗节组成，押韵格式为ABABCDCDEFEFGG。\n\n"
                "【十四行诗示例】\n"
                "在这寂静的夜晚星辰闪耀，\n"
                "微风轻拂诉说着古老的故事。\n"
                "月光如水洒在你的窗台，\n"
                "思念如潮水般涌上心头。\n"
                "（此处省略后续十行）\n"
            ),
            '现代朦胧诗': (
                "现代朦胧诗是一种20世纪后期兴起的诗歌形式，特点是意象朦胧、语言含蓄，"
                "强调主观体验和内心感受，常通过隐喻和象征表达复杂的情感和思想，"
                "给读者留下广阔的解读空间。\n\n"
                "【现代朦胧诗示例】\n"
                "在迷雾中寻找前行的路，\n"
                "心灵的呼唤如晨曦般微弱。\n"
                "梦境与现实交织成一幅画，\n"
                "我们在其中追寻那未知的光。\n"
            )
        }

        # 验证诗歌类型
        if poetry_style not in poetry_definitions:
            raise ValueError(f"不支持的诗歌类型：{poetry_style}。请选择 '俳句'、'十四行诗' 或 '现代朦胧诗'。")

        # 获取对应诗歌类型的定义与示例
        poetry_definition = poetry_definitions[poetry_style]

        # 构建提示词
        system_content = (
            "## 角色：创意诗歌创作大师\n\n"
            "## 任务描述\n"
            "你是一位精通多种诗歌体裁的创作大师，能够根据给定的主题或内容，"
            "创作出符合特定诗歌形式的作品。\n\n"
            "## 任务目标\n"
            f"- **诗歌类型**：{poetry_style}\n"
            f"{poetry_definition}\n"
            "- **主题或内容**：\n"
            f"  {content}\n\n"
            "## 输出要求\n"
            "请根据上述诗歌类型的定义和示例，创作一首符合该形式的诗歌，"
            "以准确表达所提供的主题或内容。"
        )

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': f'请以"{content}"为主题，创作一首{poetry_style}。'}
        ]

        return messages

    @staticmethod
    def generate_question_agent_prompt_for_qwen(knowledge_point, related_texts, spo, difficulty_level, question_type, question_count):
        """
        生成用于Qwen或DeepSeek大模型的复杂出题prompt（含示例题目）。
        """
        example = ""
        # 难度描述与规定：
        difficulty_details = {
            "简单": """简单难度：
            - 题目仅考查该知识点的单一属性（例如定义、判断方法）的理解；
            - 单选题和多选题：选项中的干扰项最多不超过2个；
            - 填空题：填空数量只有1个；
            - 推理步长：从题干推理得出答案的思考步数小于2。""",

            "普通": """普通难度：
            - 题目考查该知识点及其知识图谱关系中知识点的各自单一属性的理解；
            - 单选题和多选题：选项中的干扰项超过1个；
            - 填空题：填空数量1-2个；
            - 推理步长：从题干推理得出答案的思考步数2-3步。""",

            "困难": """困难难度：
            - 题目考查该知识点及其知识图谱关系中知识点的多属性混合理解；
            - 单选题和多选题：每个非正确选项都要有一定的干扰；
            - 填空题：填空数量大于2；
            - 推理步长：从题干推理得出答案的思考步数大于3；
            - 知识点如果涉及计算和公式，可生成计算题。"""
        }.get(difficulty_level, "普通难度")

        question_desc = {
            "单选题": f"生成单选题的题干及4个选项，只有1个正确选项，每个选项不超过15字。",
            "多选题": f"生成多选题的题干及4个选项，正确选项1至4个，每个选项不超过15字。",
            "填空题": f"生成填空题的题干，需填空的空格数量根据难度等级而定。"
        }.get(question_type, "生成单选题的题干及4个选项，只有1个正确选项。")

        example_dict = {
            ("单选题", "简单"): """试题示例格式参考：
        板块构造学说可以帮助我们认识地球环境、解释地球运动的机理。下列地区，因板块碰撞而形成的是（　　）
        
        A. 马里亚纳海沟
        B. 东非大裂谷
        C. 大西洋的洋脊
        D. 科罗拉多大峡谷
        """,
            ("单选题", "普通"): """试题示例格式参考：          
        下列关于地球大气层中平流层、对流层、中间层的叙述，正确的是（　　）
        
        A. 自近地面向上，依次为对流层--中间层--平流层
        B. 气温随高度增加发生的变化，依次表现为递减--递增--再递增
        C. 大气运动的主要形式，自下而上依次为对流运动--水平运动--对流运动
        D. 天气变化状况，自下而上依次表现为显著--不显著--显著
        """,
            ("单选题", "困难"): """试题示例格式参考：
        针对复杂地质构造和板块运动理论，请选择最能描述板块边界处地形形成的地区（　　）
        
        A. 某海沟区域
        B. 某裂谷区域
        C. 某褶皱带
        D. 某火山岛
        """,
            ("多选题", "简单"): """试题示例格式参考：
        关于地球基本构造，哪些说法正确？
        
        A. 地球分为地壳和地幔
        B. 地球由多个板块构成
        C. 地球内部温度均匀
        D. 地球外壳很薄
        """,
            ("多选题", "普通"): """试题示例格式参考：
        关于地球构造的描述，下列说法正确的是？
        
        A. 板块运动能解释地震分布
        B. 火山活动仅发生在板块边缘
        C. 地震与构造运动无关
        D. 板块碰撞会导致造山运动
        """,
            ("多选题", "困难"): """试题示例格式参考：
        关于地球构造及其演变，哪些观点能较全面地解释板块运动与地貌形成之间的关系？
        
        A. 板块构造学说从多角度解释地壳运动
        B. 板块边界复杂交互作用导致多样地形
        C. 地球内部对流仅影响局部区域
        D. 板块运动与地震、火山活动密切相关
        """,
            ("填空题", "简单"): """试题示例格式参考：
        岩石中包含的___可以反映古地理环境和生物特征，生物总是从___到___，从___到___ 。""",
            ("填空题", "普通"): """试题示例格式参考：
        地球的自转周期为T1，近地卫星公转周期为T2，则他们的大小关系满足T1___T2（选填“＞”“＜”“=”）；已知万有引力常故为G，则地球的密度约为___（用前述字母表示）。""",
            ("填空题", "困难"): """试题示例格式参考：
        地理信息技术是指获取、管理、分析和应用地理空间信息的现代技术的总称，主要包括（___英文缩写为___）、（___英文缩写为___）、（___英文缩写为___ ）等．"""
        }
        example = example_dict.get((question_type, difficulty_level), "")
        # 处理关系信息：
        spo_text = ""
        if spo and 'entity_relations' in spo and spo.get('entity'):
            relations = spo['entity_relations']
            if relations:
                related_entities = "、".join([r['entity'] for r in relations])
                spo_text = f"根据知识图谱，“{spo['entity']['name'].strip()}”与“{related_entities}”等知识存在紧密联系，出题时请依据难度选择是否体现这些知识点之间的关联关系。"
            else:
                spo_text = "当前知识点在图谱中存在但没有与其他概念的关联关系。"
        else:
            spo_text = "当前知识点在图谱中未找到任何信息，请结合地理常识自由出题。"

        # --- 构建教材片段 texts_context ---
        if related_texts:
            texts_context = "\n\n".join([f"片段{i + 1}：{text}" for i, text in enumerate(related_texts)])
        else:
            if spo and 'entity_relations' in spo:
                texts_context = "请结合图谱关系出题。"
            else:
                texts_context = "请结合你的高中地理知识设计题目。"

        # 系统角色提示：
        system_content = "你是一位经验丰富的地理学科专业出题专家，擅长根据给定的教材资源内容及知识图谱关系生成高质量的考试题目。请只输出题干内容，不包含答案和解析。"
        user_content = f"""
            【当前任务】：
            - 知识点：{knowledge_point}
            - 题型：{question_type}
            - 难度等级：{difficulty_level}
            - 题目数量：{question_count}
            
            任务要求：
            题型：{question_desc}\n
            难度：{difficulty_details}\n
            
            请严格依据以下给定的知识内容片段和知识图谱关系生成题目：
            
            相关知识内容片段：
            {texts_context}\n
            
            知识图谱关系提示：
            {spo_text}\n
            题目示例：
            {example}\n
            
            请严格按照示例的格式生成题目内容，确保：
            - 题干表述清晰，逻辑严谨。
            - 只生成题干及选项内容，不输出答案及解析。
            - 每道题之间以换行分隔。
            
            现在请生成共{question_count}道{question_type}。
            """

        # 构建OpenAI格式的 messages
        messages = [
            {'role': 'system', 'content': system_content.strip()},
            {'role': 'user', 'content': user_content.strip()}
        ]

        return messages

    @staticmethod
    def generate_explanation_prompt_for_qwen(knowledge_point, question_type, question_content, difficulty_level, related_entity_info):
        """
        构造用于生成题目答案与解析的模板化Prompt。

        :param knowledge_point: str，知识点名称，例如“地球自转”
        :param question_type: str，题型（“单选题”、“多选题”、“填空题”）
        :param question_content: str，题干与选项内容
        :param difficulty: str，难度等级（“简单”、“普通”、“困难”）
        :return: messages列表，用于构造ChatCompletion请求
        """

        # 提取相关知识点名称（不含主知识点本身）
        related_entities = [rel["entity"] for rel in related_entity_info.get("entity_relations", [])]
        related_entities_str = "、".join(related_entities) if related_entities else "无"

        common_instruction = (
            f"你是一位资深教育出题专家，擅长生成题目的正确答案和详细解析。\n"
            f"请根据以下与【{knowledge_point}】相关的【{question_type}】，结合难度要求，生成准确答案与解析。\n\n"
            "【输出格式】：请严格生成如下结构的JSON，不要输出多余解释性语言：\n"
            "{\n"
            "  \"answer\": \"填写正确答案（如A、B、AC或填空内容）\",\n\n"
            "  \"analysis\": {\n"
            "     \"【基本解题思路】: \"简述该知识点的核心考点和常规解题方法\",\n"
            "     \"【详解】: \"根据题干内容，仅详细解释正确答案的原因和推理过程即可\",\n"
            "     \"【干扰项分析】: \"分析干扰项的错误原因，干扰项并不是指全部的错误选项，而是和标准答案相近的，或者在推理过程中容易让学生产生混淆的选项。（若无干扰项或非选择题则跳过此项）\"\n"
            "  },\n"
            f"  \"knowledge_point\": [\"本题涉及的所有图谱中的知识点，主知识点{knowledge_point}必须包含在内\"],\n"
            f"  \"difficulty_level\": {difficulty_level}\n"
            "}\n\n"
            "你还需参考以下相关知识点名称列表：\n"
            f"{related_entities_str}\n"
            "请判断这些知识点是否出现在题干或解析中，若出现，请一并加入knowledge_point字段中。\n\n"
            "注意：\n"
            "1. analysis 是一个完整的字符串，内部嵌套三个部分（含标题），用“【】”包裹。\n"
            "2. answer 字段需包含“【答案】：”前缀。\n"
            "3. knowledge_point 字段请用数组表示，内容需包含主知识点和实际出现的相关知识点。\n"
            "4. 请不要输出任何无关文字，只返回完整的 JSON 结构。\n\n"
            "以下是一个示例输出：\n"
            "{\n"
            "  \"answer\": \"【答案】：D\\n\\n\",\n"
            "  \"analysis\": \"【基本解题思路】：赤太阳系中的行星运行特点是：同向性、近圆性和共面性。行星围绕太阳运动会产生昼夜现象。自转轴倾角影响行星的四季更替，自转周期影响昼夜长短。\\n\\n"
            "【详解】：火星与地球自转轴倾角、自转周期均相似，因此火星的昼夜长短与地球接近，②正确；同时自转轴倾角相似，导致火星自转轨道面与公转轨道面的夹角类似于地球黄赤交角，故火星也具有类似地球的四季更替现象，③正确。因此，②③正确，即选项B正确。\\n\\n"
            "【干扰项分析】：①项错误，昼夜现象由行星的不透明与自转运动产生，与自转轴倾角和周期与地球相近无关；\\n④项错误，运动方向取决于太阳系形成过程，与自转轴倾角和周期与地球相近无关；\\nA、C、D选项因包含①或④错误项而错误。\\n\",\n"
            "  \"knowledge_point\": [\"地球自转\", \"昼夜更替\", \"自转周期\"],\n"
            "  \"difficulty_level\": \"普通\"\n"
            "}\n\n"
            "现在，请根据以下题目内容生成符合上述格式的输出："
        )

        # 根据题型动态生成模板要求
        question_type_instructions = {
            "单选题": "答案必须为题目中的单个选项，如“A”。",
            "多选题": "答案必须为题目中的多个选项组合，如“AC”、“ABD”。",
            "填空题": "答案必须准确填写题目所需的正确答案，不需标选项。"
        }

        # 根据难度动态调整解析要求
        difficulty_instructions = {
            "简单": "【难度说明】：简单难度只需简洁说明正确选项或答案的理由，无需刻意说明干扰项。解析长度在100字以内",
            "普通": "【难度说明】：普通难度必须明确指出干扰项，并说明其错误原因。解析长度在150字以内",
            "困难": "【难度说明】：困难难度需深入分析干扰项，特别强调易错点，帮助学生避免陷阱。如果涉及到计算，请详细说明计算过程。解析长度不要超过300字"
        }

        type_instruction = question_type_instructions.get(question_type, "")
        difficulty_instruction = difficulty_instructions.get(difficulty_level, "")

        # 最终组装system内容
        system_content = f"{common_instruction}\n{type_instruction}\n{difficulty_instruction}\n\n请严格遵照以上模板和难度说明，生成答案与解析，不得输出题干与选项。"

        user_content = f"题目内容如下：\n{question_content}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        return messages

    @staticmethod
    def generate_ppt_from_outline_prompt(knowledge_point, outline_markdown, textbook_text):
        """
        生成用于创建PPT内容的提示词，确保生成内容具体、详细且有教学价值

        Args:
            knowledge_point (str): 知识点名称
            outline_markdown (str): 大纲Markdown文本
            textbook_text (str): 教材全文内容

        Returns:
            list: 包含角色和内容的消息列表，用于提交给大语言模型
        """
        system_content = (
            "## Role：地理教学PPT课件内容生成专家\n\n"
            "## 任务描述\n"
            f"你是一位地理学科特级教师，现需要为《{knowledge_point}》创建教学PPT课件内容。你应当基于教材内容和大纲结构，生成具体、详实、有教学价值的PPT内容。\n\n"
            "## 输出格式要求\n"
            "你必须严格按照以下格式输出Markdown内容：\n\n"
            "```\n"
            "# PPT标题（必须是知识点名称+教学课件）\n"
            "## 章节标题\n"
            "### 内容页标题\n"
            "- 具体教学内容点1\n"
            "- 具体教学内容点2\n"
            "## 章节标题\n"
            "### 内容页标题\n"
            "#### 正文标题（可选）\n"
            "- 具体教学内容点1\n"
            "- 具体教学内容点2\n"
            "```\n\n"
            "## 格式规范\n"
            "1. 标题层级严格遵循：\n"
            "   - 第一级(#)：仅用于PPT总标题，且必须是知识点名+教学课件\n"
            "   - 第二级(##)：用于章节标题，如'内容概述'、'知识讲解'等\n"
            "   - 第三级(###)：用于内容页标题，标识具体知识点\n"
            "   - 第四级(####)：用于正文小标题(可选)，如一、二、三等序号标题\n"
            "2. 正文内容必须以'-'开头，形成列表项\n"
            "3. 不使用段落文本，所有非标题内容必须是列表项\n"
            "4. 不使用加粗、斜体、分隔线等其他Markdown语法\n"
            "5. 标题与内容之间须有空行\n"
            "6. 每个章节标题下至少包含一个内容页标题\n"
            "7. 每个标题下必须有至少3-5个列表项正文内容\n\n"
            "## 内容质量要求\n"
            "1. 严禁出现'本章节包含重要知识点'、'关键内容与要点'等模糊概括性内容\n"
            "2. 所有正文内容点必须是具体、明确的知识点或解释，能直接用于教学\n"
            "3. 要包含具体的定义、公式、例子、应用场景或现象解释\n"
            "4. 要涵盖知识点的概念定义、特征特点、分类方法、应用价值等多个维度\n"
            "5. 数据要准确，表述要严谨，符合地理学科规范\n"
            "6. 内容要由浅入深，循序渐进，符合教学规律\n"
            "7. 如涉及计算，请提供具体计算方法和步骤\n"
            "8. 如涉及地理现象，请提供具体的实例和解释\n"
        )

        user_content = (
            f"请为《{knowledge_point}》创建一份完整的PPT课件内容，严格按照规定格式，内容必须具体、详实。\n\n"
            # f"## 参考教材内容\n{textbook_text[:5000]}\n\n"
            f"## 大纲结构\n{outline_markdown}\n\n"
            f"请直接生成符合要求的Markdown内容，第一行必须是'# {knowledge_point}教学课件'。\n"
            f"所有正文内容必须以'- '开头，每条内容必须具体明确，不允许出现'本章节包含重要知识点'这类模糊表述。\n"
            f"每个标题后必须有3-5个具体的教学内容点，确保内容详实、有教学价值，可直接用于授课。\n"
            f"如果有计算过程，请提供具体计算公式和步骤；如果有地理现象，请提供具体的实例和解释。"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        return messages

    @staticmethod
    def generate_single_page_prompt(page_markdown, textbook_text):
        system_content = (
            "## Role：教学讲解专家\n\n"
            "## 任务描述\n"
            "你是一位具备教学设计能力的 AI 教师助理，能够根据大纲片段与教材内容，完成教学讲解内容撰写任务。\n\n"
            "## 任务目标\n"
            "- 理解教材内容，识别与当前大纲片段相关的重点知识；\n"
            "- 围绕给定大纲，生成该页内容的完整讲解，覆盖每个要点；\n"
            "- 内容层次清晰，语言适合课堂教学场景，结构规范。\n\n"
            "## 输出要求\n"
            "- 输出为完整的 Markdown 格式；\n"
            "- 保留原始大纲结构（标题、列表项等），在其下补充完整教学内容；\n"
            "- 内容应包括讲解说明、示意图描述（如有）、计算说明（如适用）、课堂互动建议等；\n"
            "- 不要重复输出教材原文，而是提炼讲解；\n"
            "- 严禁输出“暂无内容”或“无法生成”等形式的占位语。\n\n"
            "## 注意\n"
            "- 若大纲中包含多个知识点，建议合并讲解或逻辑串联。\n"
            "- 若涉及计算内容，请详细列出计算原理、公式与应用示例。"
        )

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"### 当前教学大纲页 Markdown 内容如下：\n{page_markdown}\n\n"
                    f"### 教材全文如下（供参考使用）：\n{textbook_text[:10000]}"
                )
            }
        ]

        return messages

    @staticmethod
    def generate_theme_prompt(markdown, textbook_text):
        system_content = (
            "## Role：教学课件设计专家\n\n"
            "## 任务描述\n"
            "你是一位高中地理课件编写专家，请根据教材内容撰写教学课件中主题页的内容大纲，包括主题知识点的概述、重要意义，可用一个近年来的地理现象来引出这个主题\n\n"
            "## 输出要求\n"
            "- 输出保持原有标题结构；\n"
            "- 引导性语言、定义、应用场景、图示建议、生活化类比、经典引言等形式均可使用；\n"
            "- 内容应简洁明了，适合作为主题页出现，不要太多字数\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user",
             "content": f"### 当前页 Markdown：\n{markdown}\n\n### 教材内容如下：\n{textbook_text[:10000]}"}
        ]

    @staticmethod
    def generate_chapter_prompt(chapter, description_text, textbook_text):
        system_content = (
            "你是一位高中地理教学课件大纲的设计专家，擅长根据教学考纲编写清晰、结构合理的章节讲解思路。"
            "现在你需要帮助老师完成某一章节的课件内容设计，请仔细分析任务要求与考纲内容，并合理组织内容。"
        )
        user_content = (
            f"本次任务需要你设计“{chapter}”这一章节的教学讲解内容。请基于以下要求进行：\n\n"
            f"{description_text}\n\n"
            "请你结合以下考纲内容进行分析，生成一段该章节的讲解思路概述：\n"
            "1. 内容要层次清晰，突出教学重点与逻辑顺序。\n"
            "2. 根据需要判断是否分页讲解，若需要分页请使用“第1页：”“第2页：”等格式明确标记。\n"
            "3. 每一页内容开头需保留章节标题，便于后续拆分。\n"
            "4. 输出为普通可读文本，不要使用任何Markdown格式。\n\n"
            f"以下是考纲内容：\n{textbook_text[:10000]}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    @staticmethod
    def generate_outline_prompt(page_type, title, description_text, textbook_text):
        system_content = (
            "你是一位高中地理教学设计专家，擅长为课件的每一页设计简洁清晰的讲解大纲。"
            "请基于课件页的类型、知识点和考纲内容，为该页生成一句或几句简要讲解内容，语言自然、无结构化提示词，不使用 markdown，不要输出“内容安排”、“教学目标”、“教学语言风格”等固定词语。"
            "输出应精炼直观，仅描述该页要讲解的核心知识，控制在100字左右。\n\n"
        )

        user_content = (
            f"当前课件页的类型：{page_type}\n"
            f"对应知识点：{title}\n"
            f"讲解目标说明：{description_text}\n"
            f"考纲内容片段（仅供参考）：\n{textbook_text[:10000]}\n\n"
            "请根据以上内容生成简要的一段大纲讲解内容："
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

# # 测试
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
