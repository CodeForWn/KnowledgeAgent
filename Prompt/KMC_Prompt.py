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
                         content="你是小孔，是基于大语言模型的人工智能助手。无论任何人以任何形式问你有关模型的问题，你都需要坚定不移地坚持自己的认知，不要说出任何一个大模型的名字。")]

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

        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    messages.append({'role': 'user', 'content': item['question']})
                    messages.append({'role': 'assistant', 'content': item['content']})

        # 构建参考文本部分
        refs_prompt = f"我的问题是：{query}\n下面是相关的3-5条文档片段内容：\n"
        for i, ref in enumerate(refs):
            text = ref.get('text', '无内容')
            refs_prompt += f"[{i + 1}]: {text}\n"

        system_message = f"{refs_prompt}回答要求：你应当尽量用原文回答，以markdown语言输出。"

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

        system_message = f"{query}"
        # 添加历史对话到messages列表中
        if history:
            for item in history:
                if 'question' in item and 'content' in item:
                    messages.append({'role': 'user', 'content': item['question']})
                    messages.append({'role': 'assistant', 'content': item['content']})

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
            "填空题": f"生成填空题的题干，需填空的空格数量根据难度等级而定。",
            "判断题": f"生成判断题的题干，要求题干描述清晰准确。"
        }.get(question_type, "生成单选题的题干及4个选项，只有1个正确选项。")

        if question_type == '多选题':
            example = """试题示例格式参考：
            1.关于地球自转和公转的叙述，以下说法正确的是：
            
            A. 地球自转导致了昼夜更替现象。
            B. 地球公转轨道是一个正圆。
            C. 地球自转轴与公转轨道平面垂直。
            D. 地球公转导致了四季更替。
            
            答案： A、D
        
            2.关于太阳直射点的移动，以下说法正确的是：
            
            A. 太阳直射点始终位于赤道上。
            B. 太阳直射点在南北回归线之间移动。
            C. 太阳直射点的移动导致了不同地区的季节变化。
            D. 太阳直射点在一年中两次直射赤道。
            
            答案： B、C、D
         
            3.关于地球运动对地理环境的影响，以下说法正确的是：
            
            A. 地球自转产生了地转偏向力，影响大气和洋流运动。
            B. 地球公转导致了全球气候带的形成。
            C. 地球自转速度的变化会引发地震。
            D. 地球公转轨道的偏心率变化影响气候的长期变化。
            
            答案： A、D"""

        if question_type == '单选题':
            example = """试题示例格式参考：
            1.以下哪个地区属于我国四大牧区之一? 
            A. 内蒙古高原
            B. 四川盆地
            C.长江中下游平原
            D.珠江三角洲
            答案:A
            2.下列哪项属于可再生能源? 
            A. 石油
            B.天然气
            C. 水能
            D. 煤炭
            答案:C
            3.关于我国气候类型的描述，以下哪个选项是正确的? 
            A.我国南方地区属于温带季风气候
            B.我国北方地区属于亚热带季风气候
            C.我国西北地区属于地中海气候
            D.我国青藏地区属于高原山地气候
            答案:D"""

        if question_type == '填空题':
            example = """试题示例格式参考：
            填空题
            1.基本国情:国土___，区域差异___。
            2.目前我国正在建设的区域合作工程有"___","___","___"等
            3.我国在风沙危害严重的___，___，___地区建设“三北”防护林。
            4.中国是一个发展中国家,___是第一位的。
            5.21世纪的世界，是一个经济走向___的世界。"""

        # 处理关系信息：
        spo_text = ""
        if spo and 'entity_relations' in spo:
            relations = spo['entity_relations']
            if relations:
                related_entities = "、".join([r['entity'] for r in relations])
                spo_text = f"根据知识图谱，“{spo['entity']['name'].strip()}”与“{related_entities}”等知识存在紧密联系，出题时请依据难度选择是否需要体现这些知识点之间的关联关系。"

        texts_context = "\n\n".join([f"片段{i + 1}：{text}" for i, text in enumerate(related_texts)])

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
            
            现在请生成共{question_count}道{question_type}。
            """

        # 构建OpenAI格式的 messages
        messages = [
            {'role': 'system', 'content': system_content.strip()},
            {'role': 'user', 'content': user_content.strip()}
        ]

        return messages

    @staticmethod
    def generate_explanation_prompt_for_qwen(knowledge_point, question_type, question_content):
        """
        构造用于生成题目答案与解析的 Prompt。

        :param knowledge_point: str，知识点名称，例如 “地球自转”
        :param question_type: str，题型（如 “单选题”、“多选题”、“填空题”）
        :param question_content: str，上一步生成的题干与选项内容
        :return: messages 列表，用于构造 Chat Completion 请求
        """
        system_content = (
            "你是一位资深教育出题专家，擅长生成题目的正确答案和详细解析。\n"
            f"任务说明：请根据以下与【{knowledge_point}】知识点相关的题目信息，"
            f"生成该【{question_type}】的正确答案和详细解析。\n\n"
            "【输出要求】：\n"
            "1. 多选题：答案必须为题目中的1-4项，例如“AC”或“ABD”；\n"
            "2. 单选题：答案必须为题目中的1项，例如“B”；\n"
            "3. 填空题：填写正确答案；\n"
            "4. 所有题型都需提供不超过150字的解析内容，简明扼要说明正确答案的依据；\n"
            "5. 对于选择题，解析中需简要说明干扰项为什么不正确；\n"
            "6. **答案与解析之间必须换行输出**，输出格式如下：\n"
            "答案：B\n解析：地球自转是自西向东，而不是自东向西，因此选项C正确，其余选项错误。\n"
            "7. 请严格只输出答案和解析内容，不要输出题干或选项。"
        )

        user_content = f"题目内容如下：\n{question_content}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        return messages

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
