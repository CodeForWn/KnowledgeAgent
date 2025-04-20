# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import traceback
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
import datetime
import threading
import queue
import urllib3
import logging
import requests
from logging.handlers import RotatingFileHandler
import sys
import re
import markdown
import uuid
import jieba.posseg as pseg
from FlagEmbedding import FlagReranker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder
from Neo4j.KMC_neo4j import KMCNeo4jHandler
from neo4j import GraphDatabase, basic_auth
from MongoDB.KMC_Mongo import KMCMongoDBHandler
from app import markdown_to_ppt
import types
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import os
from docx import Document
from urllib.parse import urlparse


app = Flask(__name__)
CORS(app)
# 加载配置
# 使用环境变量指定环境并加载配置
config = Config(env='production')
config.load_config()  # 指定配置文件的路径
config.load_predefined_qa()
logger = config.logger
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)
neo4j_handler = KMCNeo4jHandler(config)
file_manager = FileManager(config)
mongo_handler = KMCMongoDBHandler(config)
es_handler = ElasticSearchHandler(config)
rerank_model_path = config.rerank_model_path
reranker = FlagReranker(rerank_model_path, use_fp16=True)


@app.route("/api/question_agent", methods=["POST"])
def get_question_agent():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        difficulty_level = data.get('difficulty_level', '普通')  # 难度等级
        question_type = data.get('question_type', '单选题')  # 题型
        question_count = data.get('question_count', 3)  # 题目数量
        llm = data.get('llm', 'deepseek')
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.8)
        query = knowledge_point + "的地理定义和意义是什么？"

        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 调用 get_entity_details 方法获取该知识点的详细信息
        result = neo4j_handler.get_entity_details(knowledge_point)
        if not result or not result.get("resources"):
            logger.info(f"知识点 {knowledge_point} 不存在于图谱中或未关联资源，直接使用大模型生成题目")
            # 空资源情况下，仍构造 prompt
            prompt = prompt_builder.generate_question_agent_prompt_for_qwen(
                knowledge_point=knowledge_point,
                related_texts=[],  # 为空
                spo={},  # 空结构
                difficulty_level=difficulty_level,
                question_type=question_type,
                question_count=question_count
            )
            if llm.lower() == 'qwen':
                response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
                return Response(response_generator, content_type='text/plain; charset=utf-8')

            if llm.lower() == 'deepseek':
                response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
                return Response(response_generator, content_type='text/plain; charset=utf-8')

        logger.info(f"获取知识点 {knowledge_point} 的子图信息：{result}")
        resources = result.get("resources", [])

        # 返回 JSON 格式的结果
        # logger.info(f"候选子图为：{result}")
        # 创建一个空列表，用于存储所有资源处理后的 doc_list
        combined_doc_list = []
        index_name = f"temp_kb_{knowledge_point}_{int(time.time())}"
        for res in resources:
            docID = res.get("docID")
            # 从 MongoDB 查询该资源的详细信息（例如 file_path 等）
            resource_detail = mongo_handler.get_resource_by_docID(docID)
            logger.info(f"资源 docID={docID} 的详情信息：{resource_detail}")
            if not resource_detail:
                logger.warning(f"资源未找到：docID = {docID}")
                continue

            if resource_detail:
                # 提取文件路径和文件名（注意，文件名可以直接从 neo4j 的返回值中获取）
                file_path = resource_detail.get("file_path", "")
                file_name = resource_detail.get("file_name")
                subject = resource_detail.get("subject")
                resource_type = resource_detail.get("resource_type")
                metadata = resource_detail.get("metadata", {})
                diff_level = resource_detail.get("difficulty_level", "")
                ques_type = resource_detail.get("question_type", "")

                # 判断文件是否存在
                if not os.path.exists(file_path):
                    logger.error(f"文件 {file_path} 不存在，跳过资源 {file_name}")
                    continue

                # ✅ 特殊处理：题库资源，直接返回全文作为接口输出
                if resource_type == "试题" and diff_level == difficulty_level.strip() and ques_type == question_type.strip():
                    try:
                        if file_path.endswith(".docx"):
                            doc = Document(file_path)
                            paras = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
                            question_text = "\n".join(paras)
                            logger.info(f"命中试题资源，直接返回：{file_name}")
                            return Response(question_text, content_type='text/plain; charset=utf-8')
                        else:
                            logger.warning(f"试题资源暂不支持格式：{file_path}")
                            continue
                    except Exception as e:
                        logger.error(f"读取试题资源失败：{e}", exc_info=True)
                        continue
                # ✅ 其他资源：继续执行原有分段 + 索引 + 检索逻辑
                try:
                    # 使用文件处理模块对该文件进行分段处理
                    doc_list = file_manager.process_pdf_file(file_path, file_name)
                except Exception as e:
                    logger.error(f"处理文件 {file_name} 时发生异常: {e}", exc_info=True)
                    continue

                if doc_list:
                    success = es_handler.create_temp_index(index_name, doc_list, docID, file_name, file_path, subject, resource_type, metadata)
                    if success:
                        logger.info(f"临时索引 {index_name} 创建成功")
                        # 此处可以执行 ES 检索操作，例如：
                        bm25_hits = es_handler.agent_search_bm25(index_name, query, ref_num=10)

                        # 测试 Embed 检索：对索引中存储的文档进行向量检索
                        embed_hits = es_handler.agent_search_embed(index_name, query, ref_num=10)

                        combined_doc_list.extend(bm25_hits)
                        combined_doc_list.extend(embed_hits)
                    else:
                        logger.error("临时索引创建失败")
                        continue
                else:
                    logger.error(f"文件处理失败：{file_name}")
            else:
                logger.error(f"未在MongoDB中找到 docID: {docID} 的资源信息")

        es_handler.delete_index(index_name)
        logger.info(f"索引 {index_name} 已删除")
        if not combined_doc_list:
            return jsonify({"error": "没有成功检索到任何资源数据"}), 404
        # logger.info(f"{combined_doc_list}")

        # 确保文档包含 text 字段
        filtered_combined_doc_list = [
            doc for doc in combined_doc_list
            if '_source' in doc and 'text' in doc['_source'] and doc['_source']['text'].strip()
        ]

        if not filtered_combined_doc_list:
            logger.warning("经过过滤后，没有有效的文档片段。")
            return jsonify({"error": "没有有效的文档片段用于重排"}), 404
        # 将所有检索到的结果组合，并去除重复的text片段
        unique_docs = {
            doc['_source']['text']: doc for doc in filtered_combined_doc_list
        }.values()
        # 为每个文档与查询组合成对以计算重排得分
        ref_pairs = [[query, doc['_source']['text']] for doc in unique_docs]

        # 计算每对的重排得分并归一化
        scores = reranker.compute_score(ref_pairs, normalize=True)

        # 根据得分排序
        sorted_docs_with_scores = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

        # 日志记录前5名最高分及对应文档的text
        top5_info = [
            {"score": score, "text": doc['_source']['text'][:100]}  # 截取前100个字符，避免日志过长
            for doc, score in sorted_docs_with_scores[:5]
        ]
        logger.info(f"重排后得分最高的前5个结果：{json.dumps(top5_info, ensure_ascii=False, indent=2)}")
        # 选择所有得分大于0.3的片段
        threshold = 0.3
        final_docs = [doc for doc, score in sorted_docs_with_scores if score > threshold]
        final_scores = [score for doc, score in sorted_docs_with_scores if score > threshold]

        if not final_docs:
            logger.warning(f"重排后没有得分超过阈值 {threshold} 的文档片段。")
            return jsonify({"error": f"重排后无得分超过阈值 {threshold} 的文档片段"}), 404

        # 记录重排后的分值信息
        logger.info(f"重排后超过阈值 {threshold} 的分值：{final_scores}")

        # 更新 combined_doc_list 为最终的去重并筛选后的文档列表
        combined_doc_list = final_docs
        # 调用刚才定义的提示词方法，构建prompt：
        related_texts = [doc['_source']['text'] for doc in combined_doc_list]
        # 返回候选子图信息以及 ES 检索结果
        result_data = {
            "candidate_subgraph": result,
            "related_texts": related_texts
        }
        # 生成完整提示词：
        prompt = prompt_builder.generate_question_agent_prompt_for_qwen(
            knowledge_point=knowledge_point,
            related_texts=related_texts,
            spo=result,
            difficulty_level=difficulty_level,
            question_type=question_type,
            question_count=question_count
        )

        if llm.lower() == 'qwen':
            response_text = large_model_service.get_answer_from_Tyqwen(prompt)
            result = json.loads(response_text)  # 假设返回的是 JSON 字符串
            logger.info(f"大模型返回原始内容: {repr(response_text)}")

            return jsonify({
                "code": 200,
                "msg": "success",
                "data": result
            })

        if llm.lower() == 'deepseek':
            response_text = large_model_service.get_answer_from_deepseek(prompt)
            result = json.loads(response_text)  # 假设返回的是 JSON 字符串
            logger.info(f"大模型返回原始内容: {repr(response_text)}")

            return jsonify({
                "code": 200,
                "msg": "success",
                "data": result
            })

    except Exception as e:
        logger.error(f"Error in get_question_agent: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


def highlight_answer_with_html(raw_answer: str, color: str = "#3D8BFF") -> str:
    """
    将原始 answer 内容中【答案】：部分加粗、加颜色，并将换行符转为 <br>。
    保持与 highlight_analysis_with_html 相同风格。
    """
    pattern = r"(【答案】：)([\s\S]*)"  # ✅ 用 [\s\S]* 匹配多行内容，支持 \n
    match = re.match(pattern, raw_answer.strip(), flags=re.DOTALL)

    if match:
        title = match.group(1)
        content = match.group(2).replace("\n", "<br>")  # ✅ 保留换行
        html_title = f"<strong style='color:{color};'>{title}</strong>"
        return f"{html_title}{content}"
    else:
        # 如果不匹配【答案】：格式，也做换行替换
        return raw_answer.replace("\n", "<br>")


def highlight_analysis_with_html(raw_analysis: str, color: str = "#3D8BFF") -> str:
    """
    将原始 analysis 内容中【基本解题思路】、【详解】、【干扰项分析】三段内容：
    - 段标题加 <strong> 和颜色
    - 每段之间加 <br><br>
    """
    pattern = r"(【答案】：.*?)(?=【基本解题思路】：|【详解】：|【干扰项分析】：|$)" \
              r"|(?:(【基本解题思路】：.*?)(?=【详解】：|【干扰项分析】：|$))" \
              r"|(?:(【详解】：.*?)(?=【干扰项分析】：|$))" \
              r"|(?:(【干扰项分析】：.*))"
    matches = re.findall(pattern, raw_analysis, flags=re.DOTALL)

    # 处理并包上HTML样式
    html_parts = []
    for m in matches:
        for part in m:
            if part:
                # 提取标题和正文
                title_match = re.match(r"(【.*?】：)", part)
                if title_match:
                    title = title_match.group(1)
                    content = part.replace(title, "").strip()
                    html_title = f"<strong style=\'color:{color};\'>{title}</strong>"
                    html_parts.append(f"{html_title}{content}")

    return "<br>".join(html_parts)


@app.route("/api/question_explanation_agent", methods=["POST"])
def get_question_explanation_agent():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        question_content = data.get("question_content", "")
        difficulty_level = data.get('difficulty_level', '困难')  # 难度等级
        question_type = data.get('question_type', '单选题')  # 题型
        llm = data.get('llm', 'deepseek')
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.4)

        if not question_content:
            return jsonify({"error": "题干为空"}), 400

        result = neo4j_handler.get_entity_details(knowledge_point)

        # 生成完整提示词：
        prompt = prompt_builder.generate_explanation_prompt_for_qwen(
            knowledge_point=knowledge_point,
            question_type=question_type,
            difficulty_level=difficulty_level,
            question_content=question_content,
            related_entity_info=result
        )

        if llm.lower() == 'qwen':
            response_text = large_model_service.get_answer_from_Tyqwen(prompt)
            result = json.loads(response_text)  # 假设返回的是 JSON 字符串
            result['answer'] = highlight_answer_with_html(result['answer'])
            result['analysis'] = highlight_analysis_with_html(result['analysis'])
            return jsonify({
                "code": 200,
                "msg": "success",
                "data": result
            })

        if llm.lower() == 'deepseek':
            response_text = large_model_service.get_answer_from_deepseek(prompt, top_p, temperature)
            result = json.loads(response_text)  # 假设返回的是 JSON 字符串
            result['answer'] = highlight_answer_with_html(result['answer'])
            result['analysis'] = highlight_analysis_with_html(result['analysis'])
            return jsonify({
                "code": 200,
                "msg": "success",
                "data": result
            })


    except Exception as e:
        logger.error(f"Error in get_explanation_question_agent: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": str(e),
            "data": traceback.format_exc()
        }), 500


def extract_text_from_docx(docx_path):
    """提取 .docx 文件中的纯文本内容"""
    if not os.path.exists(docx_path):
        return f"[文件未找到] {docx_path}"
    try:
        doc = Document(docx_path)
        return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return f"[提取失败] {str(e)}"


def get_exercises_by_knowledge(knowledge_point):
    """查找某知识点相关的习题并提取内容"""
    docIDs = neo4j_handler.get_resource_docIDs(knowledge_point)
    exercises = []

    for docID in docIDs:
        res = mongo_handler.get_resource_by_docID(docID)
        if not res:
            logger.warning(f"docID {docID} 无法在 MongoDB 中找到对应数据")
            continue

        if res.get("resource_type") == "试题" and \
           res.get("difficulty_level") == "普通" and \
           res.get("question_type") == "单选题":

            content = extract_text_from_docx(res["file_path"])
            logger.info(f"习题 [{res['file_name']}] 内容提取结果前100字: {content[:100]}")
            exercises.append({
                "title": res.get("file_name", ""),
                "content": content
            })
        else:
            logger.info(f"docID {docID} 不满足试题条件，已跳过")

    return exercises

# 收集树中的所有知识点名称
def collect_all_nodes(tree):
    nodes = []
    def dfs(node, children):
        nodes.append(node)
        for child, sub in children.items():
            dfs(child, sub)
    for root, children in tree.items():
        dfs(root, children)
    return nodes

# 统计某一子图下节点数量
def count_subtree_nodes(subtree):
    count = 1
    for child, child_subtree in subtree.items():
        count += count_subtree_nodes(child_subtree)
    return count

# 按阈值拆分：收集节点（按深度优先）直到达到 max_nodes 数
def split_subtree_by_count(root, subtree, max_nodes):
    pages = []

    def dfs(node, children, acc, max_count):
        if len(acc) >= max_count:
            return False
        acc.append((node, children))
        for child, sub in children.items():
            if not dfs(child, sub, acc, max_count):
                return False
        return True

    flat_nodes = []
    dfs(root, subtree, flat_nodes, max_nodes)
    pages.append(flat_nodes)
    return pages

# ✅ 新的讲解页渲染方式：每章一个页，列出所有子知识点 + 自动讲解提示
def render_lecture_page(node_children_pairs, level):
    chapter_title, sub_tree = node_children_pairs[0]
    sub_points = list(sub_tree.keys())
    md = [f"{'#' * level} {chapter_title}", ""]
    md.append(f"{'#' * (level + 1)} 本章知识点概览\n")
    md.extend([f"- {p}" for p in sub_points] if sub_points else ["（暂无子知识点）", ""])
    md.append("<!-- 请大模型根据教材内容自动分析上述知识点：")
    md.append("1. 判断知识点之间的逻辑关系和串联讲解方式；")
    md.append("2. 若涉及计算，请详细介绍计算原理及应用步骤；")
    md.append("3. 根据内容合理拆分为多个页面，每页明确主标题和副标题。 -->\n")
    return md


# ✅ 不变：拍平子树（返回的是 [(node, subtree)]）
def flatten_subtree(node, subtree):
    result = [(node, subtree)]
    for child, sub in subtree.items():
        result.extend(flatten_subtree(child, sub))
    return result


# ✅ 修改后的分页逻辑：不再按照 max_nodes_per_page 拆分，而是每章一页
def render_paginated_outline_final(tree, level=2):
    md = []
    for root, children in tree.items():
        for first_level_node, sub_tree in children.items():
            md.extend(render_lecture_page([(first_level_node, sub_tree)], level))
    return md


def render_summary_page(knowledge_point):
    return [
        "小结",
        f"本节内容围绕{knowledge_point}展开，涵盖其相关原理与知识点。",
        "各子知识点之间的关系与逻辑顺序。\n",
        ""
    ]

def render_exercise_pages(knowledge_points, level=2):
    md, seen = [], set()
    idx = 1
    for point in knowledge_points:
        for ex in get_exercises_by_knowledge(point):
            key = (ex["title"], ex["content"][:30])
            if key in seen: continue
            seen.add(key)
            md.append(f"{'#' * level} 习题 {idx}")
            md.append(f"### {ex['title']}")
            md.extend(ex["content"].splitlines())
            md.append("")
            idx += 1
    return md


def convert_markdown_to_structured_json(markdown_str: str, allowed_components: list) -> list:
    result = []

    def include(label):
        return label in allowed_components

    # 1. 主题
    if include("主题"):
        theme_match = re.search(r"# (.*?)——(.*?)\n+([^#\n]+(?:\n[^#\n]+)*)", markdown_str)
        if theme_match:
            result.append({
                "label": "主题",
                "type": "main",
                "content": f"{theme_match.group(1)}——{theme_match.group(2)}\n\n{theme_match.group(3).strip()}"
            })

    # 2. 教学要求
    if include("教学要求"):
        teach_match = re.search(r"## 教学基本要求\n+([\s\S]+?)\n+# ", markdown_str)
        if not teach_match:
            teach_match = re.search(r"## 教学基本要求\n+([\s\S]+)", markdown_str)
        if teach_match:
            result.append({
                "label": "教学要求",
                "type": "main",
                "content": teach_match.group(1).strip()
            })

    # 知识讲解（修改后的版本）
    if include("知识讲解"):
        current_chapter, chapter_subpoints, in_chapter = None, [], False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            chapter_match = re.match(r"^# (.+)", line)
            if chapter_match:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if next_line.startswith("## 本章知识点概览"):
                    if current_chapter:
                        content = f"本章节围绕【{current_chapter}】展开讲解，涵盖以下知识点：{', '.join(chapter_subpoints)}。"
                        content += "\n\n请根据教材内容，自行判断知识点的逻辑关系及串联方式。若涉及计算，请详细说明计算应用与步骤。请合理划分为多个页面，每页明确主标题和副标题。"
                        result.append({"label": "知识讲解", "type": "main", "content": content, "children": []})
                    current_chapter, chapter_subpoints, in_chapter = chapter_match.group(1), [], True
                    i += 2
                    continue
            if in_chapter and line.startswith("- "):
                chapter_subpoints.append(line[2:].strip())
            i += 1
        if current_chapter:
            content = f"本章节围绕【{current_chapter}】展开讲解，涵盖以下知识点：{', '.join(chapter_subpoints)}。"
            content += "\n\n请根据教材内容，自行判断知识点的逻辑关系及串联方式。若涉及计算，请详细说明计算应用与步骤。请合理划分为多个页面，每页明确主标题和副标题。"
            result.append({"label": "知识讲解", "type": "main", "content": content, "children": []})

        # 最后一章内容添加（避免遗漏）
        if current_chapter:
            chapter_content = f"本章节围绕【{current_chapter}】展开讲解，涵盖以下知识点：{', '.join(chapter_subpoints)}。"
            chapter_content += "\n\n请根据教材内容，自行判断知识点的逻辑关系及串联方式。若涉及计算，请详细说明计算应用与步骤。请合理划分为多个页面，每页明确主标题和副标题。"

            result_sections.append({
                "label": "知识讲解",
                "type": "main",
                "content": chapter_content,
                "children": []
            })

        result.extend(result_sections)

    # 小结
    if include("小结"):
        summary_match = re.search(r"## 小结\n([\s\S]*?)(?:\n## |\Z)", markdown_str)
        if summary_match:
            result.append({
                "label": "小结",
                "type": "main",
                "content": summary_match.group(1).strip()
            })

    # 习题
    if include("习题"):
        exercise_match = re.search(r"(## 习题[\s\S]*)", markdown_str)
        if exercise_match:
            result.append({
                "label": "习题",
                "type": "main",
                "content": exercise_match.group(1).strip()
            })

    return result


@app.route("/api/generate_ppt_outline_old", methods=["POST"])
def get_ppt_outline():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        textbook_pdf_path = data.get("textbook_pdf_path", "")
        # 可选：用户指定输出哪些部分
        components = data.get("components", ["主题", "教学要求", "知识讲解", "小结", "习题"])
        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 获取主题描述 & 教学要求
        entity_details = neo4j_handler.get_entity_details(knowledge_point)
        entity_info = entity_details.get("entity", {})
        teaching_requirements = entity_info.get("teaching_requirements", "")

        # 构建完整知识树（保留中间节点）
        knowledge_tree = neo4j_handler.build_predecessor_tree(knowledge_point)
        logger.info(f"知识树结构如下：{json.dumps(knowledge_tree, ensure_ascii=False)}")

        # 构建 Markdown + 类型标记
        pages = []

        # 1️⃣ 构建 markdown 内容
        md = []
        if "主题" in components:
            md.append(f"# {knowledge_point}——探索{knowledge_point}的原理、影响及实际应用")
            md.append(f"\n“{knowledge_point}”的定义、基本特征及应用背景。")
            md.append("")  # 添加一个换行，确保每个知识点分段

        if "教学要求" in components and teaching_requirements:
            md.append("## 教学基本要求")
            md.append("<!-- 教学要求部分，包含考纲目标与学习提示 -->")
            for line in teaching_requirements.strip().splitlines():
                if line.strip():
                    md.append(f"- {line.strip()}")
            md.append("")  # 添加一个换行，确保每个知识点分段

        if "知识讲解" in components:
            md.extend(
                render_paginated_outline_final(knowledge_tree))
            md.append("")  # 添加一个换行，确保每个知识点分段

        if "小结" in components:
            md.extend(render_summary_page(knowledge_point))
            md.append("")  # 添加一个换行，确保每个知识点分段

        if "习题" in components:
            all_knowledge_points = collect_all_nodes(knowledge_tree)
            md.extend(render_exercise_pages(
                knowledge_points=all_knowledge_points,
                level=2
            ))
            md.append("")  # 添加一个换行，确保每个知识点分段

        markdown_str = "\n".join(md)
        textbook_content = mongo_handler.read_pdf(textbook_pdf_path)
        prompt = prompt_builder.generate_ppt_from_outline_prompt(knowledge_point, markdown_str, textbook_content)

        if llm.lower() == 'qwen':
            stream = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
        elif llm.lower() == 'deepseek':
            stream = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
        else:
            return jsonify({"error": f"不支持的模型类型: {llm}"}), 400

        return Response(stream, content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.exception("生成大纲失败")
        return jsonify({"error": str(e)}), 500

def split_content_to_pages(title, content, page_type, max_lines=30):
    pages = []

    # 以“第1页：”“第2页：”等为分隔点进行拆分
    pattern = r"(第\d+页：)"
    parts = re.split(pattern, content)
    grouped = []

    # 将分页标题和内容组合起来
    temp = ""
    for i in range(len(parts)):
        if re.match(pattern, parts[i]):
            if temp:
                grouped.append(temp.strip())
            temp = parts[i]
        else:
            temp += parts[i]
    if temp:
        grouped.append(temp.strip())

    # 构建分页结果
    for i, page_content in enumerate(grouped):
        if i == 0:
            page = {
                "label": page_type,
                "type": page_type,
                "title": title,
                "content": page_content,
                "children": []
            }
        else:
            page = {
                "label": "知识讲解",
                "type": "sub",
                "title": title,
                "content": page_content,
                "children": []
            }
        pages.append(page)

    return pages


def render_exercise_pages_grouped_by_kp(knowledge_points, level=2):
    seen = set()
    exercise_pages = []

    for point in knowledge_points:
        exercises = get_exercises_by_knowledge(point)
        if not exercises:
            continue

        lines = [f"习题页：{point}"]
        count = 1
        for ex in exercises:
            key = (ex["title"], ex["content"][:30])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"题目 {count}: {ex['title']}")
            lines.extend(ex["content"].splitlines())
            lines.append("")
            count += 1

        if count > 1:
            page_md = "\n".join(lines)
            exercise_pages.append({
                "label": "习题",
                "type": "main",
                "title": point,
                "content": page_md,
                "children": []
            })

    return exercise_pages

@app.route("/api/generate_ppt_outline", methods=["POST"])
def generate_ppt_outline():
    try:
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        textbook_pdf_path = data.get("textbook_pdf_path", "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/教学要点与单元实施（选择性必修）第一单元.docx")
        llm = data.get("llm", "qwen")
        top_p = data.get("top_p", 0.9)
        temperature = data.get("temperature", 0.7)
        components = data.get("components", ["主题", "教学要求", "知识讲解", "小结", "习题"])

        if not knowledge_point or not textbook_pdf_path:
            return jsonify({"error": "参数缺失：knowledge_point 或 textbook_pdf_path"}), 400

        # 教材全文
        textbook_content = mongo_handler.read_docx(textbook_pdf_path)
        logger.info(f"[教材加载成功] 教材长度: {len(textbook_content)} 字符")

        if not textbook_content:
            return jsonify({"error": "教材内容为空或读取失败"}), 400

        # 教学要求 & 知识结构
        entity_details = neo4j_handler.get_entity_details(knowledge_point)
        entity_info = entity_details.get("entity", {})
        teaching_requirements = entity_info.get("teaching_requirements", "")
        knowledge_tree = neo4j_handler.build_predecessor_tree(knowledge_point)

        # 构建 Markdown + 类型标记
        pages = []

        if "主题" in components:
            pages.append({
                "type": "主题",
                "title": knowledge_point,
                "description": f"本页任务：用一句话围绕知识点{knowledge_point}，结合考纲内容，输出该知识点的讲解思路。可以通过近年来和这个知识点相关的地理现象来引出这个知识点。"
            })

        if "教学要求" in components and teaching_requirements:
            lines = ["教学基本要求（来自考纲）："]
            lines += [line.strip() for line in teaching_requirements.strip().splitlines() if line.strip()]
            pages.append({
                "type": "教学要求",
                "description": "\n".join(lines)
            })

        # 知识讲解页
        if "知识讲解" in components:
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    subpoints = list(sub_tree.keys())

                    # 第一页：main（只讲核心概念）
                    description_main = (
                        f"请为知识点“{chapter}”设计本页课件的大纲讲解思路。\n\n"
                        f"本页是“{chapter}”这一一级知识点的主讲页面，目标是帮助学生理解其核心概念、基本特征、产生背景及地理意义。\n"
                        f"请结合考纲和教材内容设计一段讲解思路，包括内容安排顺序、重点提示、教学语言风格等。\n"
                        f"若内容较多，请建议分页，并使用“第1页：”等格式标出，并在每页开头保留章节标题{chapter}"
                    )
                    pages.append({
                        "type": "main",
                        "title": chapter,
                        "description": description_main
                    })

                    # 第二页：sub（统一讲解子知识点）
                    description_sub = (
                        f"请为知识点“{chapter}”下的子知识点设计本页课件的大纲讲解思路。\n\n"
                        f"以下是“{chapter}”下的子知识点：{', '.join(subpoints) if subpoints else '暂无子知识点'}。\n"
                        f"请结合考纲要求说明这些子知识点之间的教学顺序、内在联系、适合的引导方法及易错点提醒等。\n"
                        f"请生成清晰结构化的讲解思路。若内容较多，请建议分页，并使用“第1页：”等格式标出，并在每页开头保留章节标题“{chapter}”。"
                    )
                    pages.append({
                        "type": "sub",
                        "title": chapter,
                        "description": description_sub
                    })

        if "小结" in components:
            all_kps = []
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    all_kps.extend(list(sub_tree.keys()))

            pages.append({
                "type": "小结",
                "title": knowledge_point,
                "description": f"请为此章节设计一段小结页的大纲讲解思路。本页用于梳理{knowledge_point}涉及的各知识点之间的关系，涉及的子知识点包括：{', '.join(all_kps)}，此页用于强化理解，提示学习方法，帮助学生形成知识体系。请指出小结页应包含哪些要素、按照怎样的逻辑结构讲解，不要过于复杂，简明扼要即可。"
            })

        # 最终结构化输出
        structured_json = []

        for idx, page in enumerate(pages, 1):
            logger.info(f"[处理页] 第 {idx} 页，类型: {page['type']}")

            if page["type"] in ["主题", "main", "sub", "小结"]:
                messages = prompt_builder.generate_outline_prompt(
                    page_type=page["type"],
                    title=page["title"],
                    description_text=page["description"],
                    textbook_text=textbook_content
                )

                if llm == "qwen":
                    content = "".join(large_model_service.get_answer_from_Tyqwen_stream(messages, top_p, temperature))
                elif llm == "deepseek":
                    content = "".join(large_model_service.get_answer_from_deepseek_stream(messages, top_p, temperature))
                else:
                    return jsonify({"error": f"不支持的模型类型: {llm}"}), 400

                logger.info(f"[模型返回] 第 {idx} 页生成完毕，内容长度: {len(content)}")

                split_pages = split_content_to_pages(page.get("title", knowledge_point), content, page["type"], max_lines=30)

                if page["type"] == "main":
                    # 将第一页作为 main，其余作为 children
                    main_page = split_pages[0]
                    main_obj = {
                        "label": "知识讲解",
                        "type": "main",
                        "title": page["title"],
                        "content": main_page["content"],
                        "addAble": True,
                        "deleteAble": True,
                        "children": []
                    }
                    for sub_page in split_pages[1:]:
                        main_obj["children"].append({
                            "label": "知识讲解",
                            "type": "sub",
                            "content": sub_page["content"],
                            "addAble": True,
                            "deleteAble": True
                        })
                    structured_json.append(main_obj)

                elif page["type"] == "sub":
                    # 如果先单独处理了sub，合并到上面的 main 会更好，这里可跳过或暂存
                    continue

                elif page["type"] == "主题":
                    structured_json.append({
                        "label": "主题",
                        "type": "main",
                        "title": page["title"],
                        "content": split_pages[0]["content"],
                        "addAble": False,
                        "deleteAble": False,
                        "children": []
                    })

                elif page["type"] == "小结":
                    structured_json.append({
                        "label": "小结",
                        "type": "main",
                        "title": page["title"],
                        "content": split_pages[0]["content"],
                        "addAble": True,
                        "deleteAble": True,
                        "children": []
                    })

            else:
                logger.info(f"[跳过模型] 第 {idx} 页类型为 {page['type']}，直接填入 JSON")
                structured_json.append({
                    "label": page["type"],
                    "type": "main",
                    "title": page.get("title", ""),
                    "content": page["description"].strip(),
                    "addAble": True,
                    "deleteAble": True,
                    "children": []
                })

        # 习题部分（无需进入大模型）
        if "习题" in components:
            all_kps = collect_all_nodes(knowledge_tree)
            exercise_pages = render_exercise_pages_grouped_by_kp(all_kps, level=2)
            structured_json.extend(exercise_pages)

        # 保存结果到 JSON 文件
        save_dir = "/home/ubuntu/work/kmcGPT/temp/resource/测试结果"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{knowledge_point}_outline.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(structured_json, f, ensure_ascii=False, indent=2)

        logger.info(f"[保存成功] 大纲已保存到 {save_path}")
        response_data = {
            "code": 200,
            "msg": "success",
            "data": {
                "outline": structured_json,
            }
        }
        return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"接口异常：{str(e)}"}), 500


@app.route("/api/generate_ppt_outline_stream_old", methods=["POST"])
def generate_ppt_outline_stream_old():
    try:
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        textbook_pdf_path = data.get("textbook_pdf_path",
                                     "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/教学要点与单元实施（选择性必修）第一单元.docx")
        llm = data.get("llm", "qwen")
        top_p = data.get("top_p", 0.9)
        temperature = data.get("temperature", 0.7)
        components = data.get("components", ["主题", "教学要求", "知识讲解", "小结", "习题"])

        if not knowledge_point or not textbook_pdf_path:
            return jsonify({"error": "参数缺失：knowledge_point 或 textbook_pdf_path"}), 400

        # 教材全文
        textbook_content = mongo_handler.read_docx(textbook_pdf_path)
        logger.info(f"[教材加载成功] 教材长度: {len(textbook_content)} 字符")

        if not textbook_content:
            return jsonify({"error": "教材内容为空或读取失败"}), 400

        # 教学要求 & 知识结构
        entity_details = neo4j_handler.get_entity_details(knowledge_point)
        entity_info = entity_details.get("entity", {})
        teaching_requirements = entity_info.get("teaching_requirements", "")
        knowledge_tree = neo4j_handler.build_predecessor_tree(knowledge_point)

        # 构建 Markdown + 类型标记
        pages = []

        if "主题" in components:
            pages.append({
                "type": "主题",
                "title": knowledge_point,
                "description": f"本页任务：用一句话围绕知识点{knowledge_point}，结合考纲内容，输出该知识点的讲解思路。可以通过近年来和这个知识点相关的地理现象来引出这个知识点。"
            })

        if "教学要求" in components and teaching_requirements:
            lines = ["教学基本要求（来自考纲）："]
            lines += [line.strip() for line in teaching_requirements.strip().splitlines() if line.strip()]
            pages.append({
                "type": "教学要求",
                "description": "\n".join(lines)
            })

        # 知识讲解页
        if "知识讲解" in components:
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    subpoints = list(sub_tree.keys())

                    # 第一页：main（只讲核心概念）
                    description_main = (
                        f"请为知识点“{chapter}”设计本页课件的大纲讲解思路。\n\n"
                        f"本页是“{chapter}”这一一级知识点的主讲页面，目标是帮助学生理解其核心概念、基本特征、产生背景及地理意义。\n"
                        f"请结合考纲和教材内容设计一段讲解思路，包括内容安排顺序、重点提示、教学语言风格等。\n"
                        f"若内容较多，请建议分页，并使用“第1页：”等格式标出，并在每页开头保留章节标题{chapter}"
                    )
                    pages.append({
                        "type": "main",
                        "title": chapter,
                        "description": description_main
                    })

                    # 第二页：sub（统一讲解子知识点）
                    description_sub = (
                        f"请为知识点“{chapter}”下的子知识点设计本页课件的大纲讲解思路。\n\n"
                        f"以下是“{chapter}”下的子知识点：{', '.join(subpoints) if subpoints else '暂无子知识点'}。\n"
                        f"请结合考纲要求说明这些子知识点之间的教学顺序、内在联系、适合的引导方法及易错点提醒等。\n"
                        f"请生成清晰结构化的讲解思路。若内容较多，请建议分页，并使用“第1页：”等格式标出，并在每页开头保留章节标题“{chapter}”。"
                    )
                    pages.append({
                        "type": "sub",
                        "title": chapter,
                        "description": description_sub
                    })

        if "小结" in components:
            all_kps = []
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    all_kps.extend(list(sub_tree.keys()))

            pages.append({
                "type": "小结",
                "title": knowledge_point,
                "description": f"请为此章节设计一段小结页的大纲讲解思路。本页用于梳理{knowledge_point}涉及的各知识点之间的关系，涉及的子知识点包括：{', '.join(all_kps)}，此页用于强化理解，提示学习方法，帮助学生形成知识体系。请指出小结页应包含哪些要素、按照怎样的逻辑结构讲解，不要过于复杂，简明扼要即可。"
            })
        # ... 参数提取与教材加载保持不变

        def generate_pages_stream():
            for idx, page in enumerate(pages, 1):
                try:
                    logger.info(f"[处理页] 第 {idx} 页，类型: {page['type']}")
                    structured_obj = None

                    if page["type"] in ["主题", "main", "sub", "小结"]:
                        messages = prompt_builder.generate_outline_prompt(
                            page_type=page["type"],
                            title=page["title"],
                            description_text=page["description"],
                            textbook_text=textbook_content
                        )
                        if llm == "qwen":
                            content = "".join(large_model_service.get_answer_from_Tyqwen_stream(messages, top_p, temperature))
                        elif llm == "deepseek":
                            content = "".join(large_model_service.get_answer_from_deepseek_stream(messages, top_p, temperature))
                        else:
                            yield json.dumps({"error": f"不支持的模型类型: {llm}"}) + "\n"
                            continue

                        split_pages = split_content_to_pages(page.get("title", knowledge_point), content, page["type"], max_lines=30)

                        if page["type"] == "main":
                            main_page = split_pages[0]
                            # ✅ 清洗掉“第X页：”开头的标记
                            cleaned_main_content = re.sub(r"^第\d+页[:：]\s*", "", main_page["content"])

                            main_obj = {
                                "label": "知识讲解",
                                "type": "main",
                                "title": page["title"],
                                "content": cleaned_main_content,
                                "addAble": True,
                                "deleteAble": True,
                                "children": []
                            }
                            for sub_page in split_pages[1:]:
                                cleaned_sub_content = re.sub(r"^第\d+页[:：]\s*", "", sub_page["content"])
                                main_obj["children"].append({
                                    "label": "知识讲解",
                                    "type": "sub",
                                    "content": cleaned_sub_content,
                                    "addAble": True,
                                    "deleteAble": True
                                })
                            structured_obj = main_obj

                        elif page["type"] == "主题":
                            cleaned = re.sub(r"^第\d+页[:：]\s*", "", split_pages[0]["content"])
                            structured_obj = {
                                "label": "主题",
                                "type": "main",
                                "title": page["title"],
                                "content": cleaned,
                                "addAble": False,
                                "deleteAble": False,
                                "children": []
                            }

                        elif page["type"] == "小结":
                            cleaned = re.sub(r"^第\d+页[:：]\s*", "", split_pages[0]["content"])
                            structured_obj = {
                                "label": "小结",
                                "type": "main",
                                "title": page["title"],
                                "content": cleaned,
                                "addAble": True,
                                "deleteAble": True,
                                "children": []
                            }

                    else:
                        # 非模型页直接构建
                        structured_obj = {
                            "label": page["type"],
                            "type": "main",
                            "title": page.get("title", ""),
                            "content": page["description"].strip(),
                            "addAble": True,
                            "deleteAble": True,
                            "children": []
                        }

                    if structured_obj:
                        wrapped_obj = {
                            "code": 200,
                            "msg": "success",
                            "data": structured_obj
                        }
                        json_str = json.dumps(wrapped_obj, ensure_ascii=False)
                        logger.info(f"[已生成] 第 {idx} 页输出：{json_str[:100]}...")
                        yield json_str + "\n"

                except Exception as page_err:
                    logger.warning(f"页码 {idx} 生成异常: {str(page_err)}")
                    yield json.dumps({"error": f"页码 {idx} 处理异常: {str(page_err)}"}) + "\n"

            # 最后处理习题页
            if "习题" in components:
                all_kps = collect_all_nodes(knowledge_tree)
                exercise_pages = render_exercise_pages_grouped_by_kp(all_kps, level=2)
                for ex_idx, ex_page in enumerate(exercise_pages, 1):
                    ex_json_str = json.dumps({
                        "code": 200,
                        "msg": "success",
                        "data": ex_page
                    }, ensure_ascii=False)
                    logger.info(f"[习题页] 第 {ex_idx} 题输出：{ex_json_str[:100]}...")
                    yield ex_json_str + "\n"

        return Response(stream_with_context(generate_pages_stream()), content_type='application/json')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"接口异常：{str(e)}"}), 500


# 页面生成并发处理
def generate_pages_stream(context):
    pages = context["pages"]
    components = context["components"]
    knowledge_tree = context["knowledge_tree"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(process_page, idx, page, context): idx
            for idx, page in enumerate(pages)
        }

        ordered_futures = sorted(future_to_index.items(), key=lambda item: item[1])

        for future, idx in ordered_futures:
            try:
                json_str = future.result()
                yield json_str + "\n"
            except Exception as e:
                logger.warning(f"[并发异常] 第 {idx + 1} 页出错：{e}")
                yield json.dumps({"error": f"第 {idx + 1} 页异常: {str(e)}"}, ensure_ascii=False) + "\n"

    if "习题" in components:
        all_kps = collect_all_nodes(knowledge_tree)
        exercise_pages = render_exercise_pages_grouped_by_kp(all_kps, level=2)
        for ex_idx, ex_page in enumerate(exercise_pages, 1):
            ex_json_str = json.dumps({
                "code": 200,
                "msg": "success",
                "data": ex_page
            }, ensure_ascii=False)
            logger.info(f"[习题页] 第 {ex_idx} 题输出：{ex_json_str[:100]}...")
            yield ex_json_str + "\n"


def process_page(idx, page, context):
    textbook_content = context["textbook_content"]
    knowledge_point = context["knowledge_point"]
    top_p = context["top_p"]
    temperature = context["temperature"]
    llm = context["llm"]

    logger.info(f"[并发处理] 第 {idx + 1} 页，类型: {page['type']}")
    structured_obj = None

    try:
        if page["type"] in ["主题", "main", "小结"]:
            messages = prompt_builder.generate_outline_prompt(
                page_type=page["type"],
                title=page.get("title", ""),
                description_text=page["description"],
                textbook_text=textbook_content
            )
            if llm == "qwen":
                content = "".join(large_model_service._get_answer_from_Tyqwen_stream(messages, top_p, temperature))
            elif llm == "deepseek":
                content = "".join(large_model_service.get_answer_from_deepseek_stream(messages, top_p, temperature))
            else:
                return json.dumps({"error": f"不支持的模型类型: {llm}"})

            split_pages = split_content_to_pages(page.get("title", ""), content, page["type"], max_lines=30)
            if not split_pages:
                raise ValueError("模型返回内容为空或无法拆分页")

            cleaned_main = re.sub(r"^第\d+页[:：]?\s*", "", split_pages[0]["content"])
            main_obj = {
                "label": "知识讲解" if page["type"] == "main" else page["type"],
                "type": "main",
                "title": page["title"],
                "content": cleaned_main,
                "addAble": True,
                "deleteAble": True,
                "children": []
            }

            for sub_page in split_pages[1:]:
                cleaned_sub = re.sub(r"^第\d+页[:：]?\s*", "", sub_page["content"])
                main_obj["children"].append({
                    "label": "知识讲解",
                    "type": "sub",
                    "content": cleaned_sub,
                    "addAble": True,
                    "deleteAble": True
                })

            structured_obj = main_obj

        elif page["type"] == "教学要求":
            structured_obj = {
                "label": "教学要求",
                "type": "main",
                "title": "教学要求",
                "content": page["description"].strip(),
                "addAble": False,
                "deleteAble": False,
                "children": []
            }

        if structured_obj:
            wrapped_obj = {
                "code": 200,
                "msg": "success",
                "data": structured_obj
            }
        else:
            logger.warning(f"[空内容] 第 {idx + 1} 页未生成内容，类型: {page['type']}")
            wrapped_obj = {
                "code": 500,
                "msg": f"第 {idx + 1} 页内容为空",
                "data": None
            }

        return json.dumps(wrapped_obj, ensure_ascii=False)

    except Exception as e:
        logger.exception(f"[异常] 第 {idx + 1} 页处理失败：{e}")
        return json.dumps({
            "code": 500,
            "msg": f"第 {idx + 1} 页处理失败：{str(e)}",
            "data": None
        }, ensure_ascii=False)

@app.route("/api/generate_ppt_outline_stream", methods=["POST"])
def generate_ppt_outline_stream():
    try:
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        textbook_pdf_path = data.get("textbook_pdf_path",
                                     "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/教学要点与单元实施（选择性必修）第一单元.docx")
        llm = data.get("llm", "qwen")
        top_p = data.get("top_p", 0.9)
        temperature = data.get("temperature", 0.7)
        components = data.get("components", ["主题", "教学要求", "知识讲解", "小结", "习题"])

        if not knowledge_point or not textbook_pdf_path:
            return jsonify({"error": "参数缺失：knowledge_point 或 textbook_pdf_path"}), 400

        textbook_content = mongo_handler.read_docx(textbook_pdf_path)
        logger.info(f"[教材加载成功] 教材长度: {len(textbook_content)} 字符")

        if not textbook_content:
            return jsonify({"error": "教材内容为空或读取失败"}), 400

        entity_details = neo4j_handler.get_entity_details(knowledge_point)
        entity_info = entity_details.get("entity", {})
        teaching_requirements = entity_info.get("teaching_requirements", "")
        knowledge_tree = neo4j_handler.build_predecessor_tree(knowledge_point)

        pages = []

        if "主题" in components:
            pages.append({
                "type": "主题",
                "title": knowledge_point,
                "description": f"本页任务：用一句话围绕知识点{knowledge_point}，结合考纲内容，输出该知识点的讲解思路。可以通过近年来和这个知识点相关的地理现象来引出这个知识点。"
            })

        if "教学要求" in components and teaching_requirements:
            lines = ["教学基本要求（来自考纲）："]
            lines += [line.strip() for line in teaching_requirements.strip().splitlines() if line.strip()]
            pages.append({
                "type": "教学要求",
                "description": "\n".join(lines)
            })

        if "知识讲解" in components:
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    subpoints = list(sub_tree.keys())

                    pages.append({
                        "type": "main",
                        "title": chapter,
                        "description": (
                            f"请为知识点“{chapter}”设计本页课件的大纲讲解思路。\n"
                            f"你需要覆盖此知识点及其相关的子知识点：{', '.join(subpoints) if subpoints else '暂无子知识点'}。\n"
                            f"请根据教材内容与考纲要求进行完整讲解，如内容较多，请合理分页，建议使用“第1页：”、“第2页：”等格式标出。"
                        )
                    })

        if "小结" in components:
            all_kps = []
            for _, children in knowledge_tree.items():
                for chapter, sub_tree in children.items():
                    all_kps.extend(list(sub_tree.keys()))
            pages.append({
                "type": "小结",
                "title": knowledge_point,
                "description": f"小结{knowledge_point}涉及的各知识点关系，包括：{', '.join(all_kps)}，用于强化理解、提示方法、构建知识体系。"
            })

        # 构建上下文传参
        context = {
            "pages": pages,
            "components": components,
            "knowledge_tree": knowledge_tree,
            "textbook_content": textbook_content,
            "knowledge_point": knowledge_point,
            "top_p": top_p,
            "temperature": temperature,
            "llm": llm
        }

        return Response(stream_with_context(generate_pages_stream(context)), content_type='application/json')

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"接口异常：{str(e)}"}), 500


@app.route("/api/generate_ppt_from_outline", methods=["POST"])
def generate_ppt():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        markdown = data.get("markdown", "")
        textbook_pdf_path = data.get("textbook_pdf_path", "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/第一单元地球自转部分.pdf")
        llm = data.get("llm", "deepseek")
        top_p = data.get("top_p", 0.9)
        temperature = data.get("temperature", 0.7)

        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 读取教材内容
        textbook_content = mongo_handler.read_pdf(textbook_pdf_path)

        # 构造提示词
        prompt = prompt_builder.generate_ppt_from_outline_prompt(
            knowledge_point,
            markdown,
            textbook_content
        )

        # 根据模型类型调用不同的流式接口
        if llm.lower() == 'qwen':
            response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

        elif llm.lower() == 'deepseek':
            response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')
        else:
            return jsonify({"error": f"不支持的模型类型: {llm}"}), 400

    except Exception as e:
        return jsonify({"error": f"接口异常: {str(e)}"}), 500


def node_to_markdown(node: dict, level: int) -> str:
    """
    递归将一个节点转换为 Markdown 文本。
    level 表示标题的层级（1 -> "#", 2 -> "##", 3 -> "###", ...）。
    """
    md = ""
    header_prefix = "#" * level

    # 如果是 PPT 标题（第一层且 label 为“主题”），直接用内容的第一行作为标题
    if level == 1 and node.get("label", "") == "主题":
        # 将内容第一行作为标题
        lines = node.get("content", "").splitlines()
        if lines:
            md += f"{header_prefix} {lines[0]}\n"
            # 如果标题后还有其他内容，作为正文段落输出
            if len(lines) > 1:
                md += "\n".join(lines[1:]) + "\n"
        else:
            md += f"{header_prefix} {node.get('label', '主题')}\n"
    else:
        # 非 PPT 标题部分：先输出标题（用节点的 label）
        if node.get("label"):
            md += f"{header_prefix} {node['label']}\n"
        # 再输出内容
        if node.get("content"):
            # 如果内容中已包含换行符，也确保每行都以换行结尾
            for line in node["content"].splitlines():
                md += line.rstrip() + "\n"
    # 如果存在子节点，则递归转换，层级加 1
    if node.get("children"):
        for child in node["children"]:
            md += node_to_markdown(child, level + 1)
    return md


def json_to_markdown(json_data: list) -> str:
    """
    将传入的 JSON 数据列表转换为 Markdown 文本。
    如果第一个节点为“主题”，则将其作为 PPT 标题（一级标题），其余节点作为章节（二级标题）。
    """
    md = ""
    for idx, node in enumerate(json_data):
        # 如果第一个节点且 label 为“主题”，则使用一级标题
        if idx == 0 and node.get("label", "") == "主题":
            md += node_to_markdown(node, 1)
        else:
            md += node_to_markdown(node, 2)
    return md

def get_filename_from_url(url):
    parsed = urlparse(url)
    return os.path.basename(parsed.path)


def clean_markdown(md_text: str) -> str:
    """
    清除 Markdown 中的特殊符号，例如 ** 加粗、--- 分割线等
    """
    # 删除加粗符号 **内容**
    md_text = re.sub(r'\*\*(.*?)\*\*', r'\1', md_text)
    # 删除分割线 ---
    md_text = re.sub(r'\n?-{3,}\n?', '\n', md_text)
    # 删除 Markdown 注释 <!-- ... -->
    md_text = re.sub(r'<!--.*?-->', '', md_text, flags=re.DOTALL)
    # 可根据需要继续添加其他规则，如 * 列表等
    return md_text.strip()

@app.route("/api/generate_ppt_from_outline_and_render", methods=["POST"])
def generate_and_render_ppt():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json(force=True)
        knowledge_point = data.get("knowledge_point", "")
        outline_json = data.get("outline_json", [])
        textbook_pdf_path = data.get("textbook_pdf_path", "")
        llm = data.get("llm", "qwen")
        top_p = data.get("top_p", 0.9)
        temperature = data.get("temperature", 0.4)

        if not knowledge_point:
            return jsonify({"error": "knowledge_point 参数为空"}), 400

        # 读取教材内容
        textbook_content = mongo_handler.read_pdf(textbook_pdf_path)

        # 将大纲 JSON 转换为标准格式 Markdown
        outline_markdown = json_to_markdown(outline_json)
        # logger.info(f"转换的markdown：{outline_markdown}")

        # 构造提示词
        prompt = prompt_builder.generate_ppt_from_outline_prompt(
            knowledge_point,
            outline_markdown,
            textbook_content
        )

        # 根据模型类型调用不同的流式接口
        if llm.lower() == 'qwen':
            response_text = large_model_service.get_answer_from_Tyqwen(prompt)
            # return Response(response_generator, content_type='text/plain; charset=utf-8')

        elif llm.lower() == 'deepseek':
            response_text = large_model_service.get_answer_from_deepseek(prompt)
            # return Response(response_generator, content_type='text/plain; charset=utf-8')
        else:
            return jsonify({"error": f"不支持的模型类型: {llm}"}), 400

        filled_markdown = response_text
        # 清洗 markdown 中的特殊格式
        cleaned_markdown = clean_markdown(filled_markdown)

        # ==== 保存 markdown 内容为文件 ====
        save_dir = "/home/ubuntu/work/kmcGPT/temp/resource/测试结果/ppt内容结果"
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        save_path = os.path.join(save_dir, f"{knowledge_point}_ppt.md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(cleaned_markdown)
        logger.info(f"markdown内容已保存到：{save_path}")

        # 调用渲染 PPT 的方法，生成 PPT 并获取下载链接
        ppt_url = markdown_to_ppt.render_markdown_to_ppt(title=knowledge_point, markdown_text=filled_markdown)

        # 3. 下载 PPT 文件到本地（临时目录）
        TEMP_DIR = "/home/ubuntu/work/kmcGPT/temp/resource/"  # 或者你指定其他用于临时文件的路径
        PUBLIC_URL_PREFIX = "http://119.45.164.254/resource/"  # 替换为你的公网访问路径前缀
        filename = get_filename_from_url(ppt_url)
        local_ppt_path = os.path.join(TEMP_DIR, filename)
        # 注意：download_file 是你定义的下载方法（所属对象根据实际情况调整，比如 self.download_file 或者其他实例）
        file_manager.download_file(ppt_url, local_ppt_path)

        # 4. 转换下载到本地的 PPT 文件为 PDF
        # 如果你的 PPT 文件后缀为 .pptx，请确认 LibreOffice 转换参数适用于 PPTX，如有需要可以修改转换命令
        pdf_local_path = file_manager.convert_docx_to_pdf(local_ppt_path, TEMP_DIR)
        if pdf_local_path is None:
            raise Exception("PPT 转 PDF 转换失败")
        # 替换路径，生成公网访问路径
        pdf_public_url = pdf_local_path.replace(TEMP_DIR, PUBLIC_URL_PREFIX)

        # 5. 构建返回结果，ppt_url 保持在线下载链接，pdf_url 是服务器上 PDF 的存储路径
        response_data = {
            "code": 200,
            "msg": "success",
            "data": {
                "ppt_url": ppt_url,
                "pdf_url": pdf_public_url
            }
        }
        logger.info(f"生成的 PPT 在线下载链接: {ppt_url}")
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"接口异常: {str(e)}"}), 500

# 获取知识图谱数据
@app.route("/api/graph/full_export", methods=["GET"])
def export_graph():
    """
    获取整张图谱，返回适配 G6 的数据格式
    """
    result = neo4j_handler.export_full_graph_for_g6()
    return jsonify(result)

# 资源库筛选查询接口
@app.route("/api/resource/filter", methods=["POST"])
def filter_data():
    try:
        data = request.get_json(force=True)
        folder_id = data.get("folder_id")


        # 根据 folder_id 判断选择资源库或题库查询
        if folder_id != "1911604997812920321":
            # 进入资源库查询
            filters = {
                "kb_id": data.get("kb_id", ""),
                "file_name": data.get("file_name", []),
                "status": data.get("status", []),
                "folder_id": folder_id,
                "knowledge_point" : data.get("knowledge_point", []),
            }

            # 调用资源库查询方法
            results = mongo_handler.filter_documents(filters)

        elif folder_id == "1911604997812920321":
            # 进入题库查询
            filters = {
                "kb_id": data.get("kb_id", ""),
                "type": data.get("type", []),
                "diff_level": data.get("diff_level", []),
                "status": data.get("status", []),
                "knowledge_point" : data.get("knowledge_point", []),
            }

            # 调用题库查询方法
            results = mongo_handler.filter_questions(filters)

        else:
            return jsonify({"code": 400, "msg": "无效的 folder_id", "data": {}})

        return jsonify({
            "code": 200,
            "msg": "success",
            "data": [dict(r, _id=str(r["_id"])) for r in results]
        })

    except Exception as e:
        return jsonify({"code": 500, "msg": f"查询失败：{str(e)}", "data": {}})

# 题库筛选查询接口
@app.route("/api/question/filter", methods=["POST"])
def filter_questions():
    data = request.get_json(force=True)
    filters = {
        "kb_id": data.get("kb_id", ""),
        "type": data.get("type", []),
        "diff_level": data.get("diff_level", []),
        "status": data.get("status", "")
    }
    results = mongo_handler.filter_questions(filters)
    return jsonify({
        "code": 200,
        "msg": "success",
        "data": [dict(r, _id=str(r["_id"])) for r in results]
    })

# 资源库更新接口
@app.route("/api/resource", methods=["PUT"])
def update_resource():
    try:
        data = request.get_json(force=True)
        doc_id = data.get("docID")
        update_fields = {}

        # 仅支持修改以下字段
        for field in ["file_name", "resource_type", "status", "file_path", "knowledge_point", "folder_id"]:
            if field in data:
                update_fields[field] = data[field]

        if not doc_id or not update_fields:
            return jsonify({"code": 400, "msg": "参数缺失或无更新字段", "data": {}})

        result = mongo_handler.update_document_by_id(doc_id, update_fields)
        if result and result.modified_count > 0:
            return jsonify({"code": 200, "msg": "更新成功", "data": {"modified": result.modified_count}})
        else:
            return jsonify({"code": 404, "msg": "未找到对应资源或内容未修改", "data": {}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"更新失败：{str(e)}", "data": {}})

# 试题库更新接口
@app.route("/api/question", methods=["PUT"])
def update_question():
    try:
        data = request.get_json(force=True)
        doc_id = data.get("docID")
        update_fields = {}

        # 仅允许更新以下字段
        for field in ["question", "answer", "analysis", "status", "type", "diff_level", "knowledge_point"]:
            if field in data:
                update_fields[field] = data[field]

        if not doc_id or not update_fields:
            return jsonify({"code": 400, "msg": "参数缺失或无更新字段", "data": {}})

        result = mongo_handler.update_question_by_id(doc_id, update_fields)
        if result and result.modified_count > 0:
            return jsonify({"code": 200, "msg": "更新成功", "data": {"modified": result.modified_count}})
        else:
            return jsonify({"code": 404, "msg": "未找到对应试题或内容未修改", "data": {}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"更新失败：{str(e)}", "data": {}})


# 资源库新增接口
@app.route("/api/resource", methods=["POST"])
def add_resource():
    try:
        data = request.get_json(force=True)
        # 生成 docID
        doc_id = mongo_handler.generate_docid()

        # 设置默认值并填充字段
        document = {
            "docID": doc_id,
            "file_name": data.get("file_name", ""),
            "resource_type": data.get("resource_type", ""),
            "status": data.get("status", ""),
            "file_path": data.get("file_path", ""),
            "kb_id": data.get("kb_id", ""),
            "created_at": datetime.datetime.utcnow(),
            "subject": data.get("subject", ""),
            "folder_id": data.get("folder_id", ""),
            "knowledge_point": data.get("knowledge_point", [])
        }

        # 插入到数据库
        inserted_id = mongo_handler.insert_document("geo_documents", document)
        return jsonify({"code": 200, "msg": "新增成功", "data": {"docID": doc_id, "inserted_id": str(inserted_id)}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"新增失败：{str(e)}", "data": {}})

# 题库新增接口
@app.route("/api/question", methods=["POST"])
def add_question():
    try:
        data = request.get_json(force=True)
        # 生成 docID
        doc_id = mongo_handler.generate_docid()

        # 设置默认值并填充字段
        question = {
            "docID": doc_id,
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "analysis": data.get("analysis", ""),
            "type": data.get("type", ""),
            "diff_level": data.get("diff_level", "普通"),
            "status": data.get("status", 'on'),
            "created_at": datetime.datetime.utcnow(),
            "resource_type": "试题",
            "subject": data.get("subject", ""),
            "kb_id": data.get("kb_id", ""),
            "folder_id": data.get("folder_id", "1911604997812920321"),
            "knowledge_point": data.get("knowledge_point", []),
        }

        # 插入到数据库
        inserted_id = mongo_handler.insert_question("edu_question", question)
        return jsonify({"code": 200, "msg": "新增成功", "data": {"docID": doc_id, "inserted_id": str(inserted_id)}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"新增失败：{str(e)}", "data": {}})

# 资源库删除接口
@app.route("/api/resource", methods=["DELETE"])
def delete_resource():
    try:
        data = request.get_json(force=True)
        doc_id = data.get("docID")
        if not doc_id:
            return jsonify({"code": 400, "msg": "docID 是必填字段", "data": {}})

        # 删除对应文档
        result = mongo_handler.delete_document_by_id(doc_id, "geo_documents")
        if result and result.deleted_count > 0:
            return jsonify({"code": 200, "msg": "删除成功", "data": {"deleted": result.deleted_count}})
        else:
            return jsonify({"code": 404, "msg": "未找到对应资源", "data": {}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"删除失败：{str(e)}", "data": {}})

# 题库删除接口
@app.route("/api/question", methods=["DELETE"])
def delete_question():
    try:
        data = request.get_json(force=True)
        doc_id = data.get("docID")
        if not doc_id:
            return jsonify({"code": 400, "msg": "docID 是必填字段", "data": {}})

        # 删除对应题目
        result = mongo_handler.delete_question_by_id(doc_id, "edu_question")
        if result and result.deleted_count > 0:
            return jsonify({"code": 200, "msg": "删除成功", "data": {"deleted": result.deleted_count}})
        else:
            return jsonify({"code": 404, "msg": "未找到对应试题", "data": {}})
    except Exception as e:
        return jsonify({"code": 500, "msg": f"删除失败：{str(e)}", "data": {}})


@app.route("/api/neo4j/bind_resource_to_entities", methods=["POST"])
def bind_resource_to_entities():
    try:
        data = request.get_json()
        docID = data.get("docID")
        entity_names = data.get("entity_names", [])
        file_name = data.get("file_name", "")
        resource_type = data.get("resource_type", "课件")

        if not docID or not entity_names:
            return jsonify({"code": 400, "msg": "参数 docID 或 entity_names 缺失", "data": {}}), 400

        neo4j_handler.bind_resource_to_entities(
            docID=docID,
            entity_names=entity_names,
            file_name=file_name,
            resource_type=resource_type
        )

        return jsonify({
            "code": 200,
            "msg": "资源成功绑定到知识点",
            "data": {
                "docID": docID,
                "entity_names": entity_names
            }
        })

    except Exception as e:
        logger.error(f"绑定资源到知识点失败: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": "内部错误",
            "data": {
                "error": str(e)
            }
        }), 500


@app.route("/api/neo4j/fuzzy_entity_search", methods=["GET"])
def fuzzy_entity_search():
    try:
        kb_id = request.args.get("kb_id", "")
        query = request.args.get("query", "")
        if not kb_id or not query:
            return jsonify({"code": 400, "msg": "参数 kb_id 或 query 缺失", "data": []})

        matched_entities = neo4j_handler.fuzzy_search_entities(kb_id, query)

        return jsonify({
            "code": 200,
            "msg": "success",
            "data": matched_entities
        })

    except Exception as e:
        logger.error(f"模糊查询知识点失败: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": "服务器内部错误",
            "data": []
        })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7777, threaded=True)