# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import traceback
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
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
import os
from docx import Document

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
                            from docx import Document
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
            return jsonify({
                "code": 200,
                "msg": "success",
                "data": result
            })

        if llm.lower() == 'deepseek':
            response_text = large_model_service.get_answer_from_Tyqwen(prompt)
            result = json.loads(response_text)  # 假设返回的是 JSON 字符串
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

# 渲染一页讲解
def render_lecture_page(node_children_pairs, level):
    md = []
    for node, children in node_children_pairs:
        md.append(f"{'#' * level} {node}")
        md.append(f"<!-- 知识点“{node}”讲解页 -->")
        md.append(f"- **知识点定义与讲解**：【请填写】对“{node}”的定义、概念讲解与背景说明。")
        md.append("")  # 添加一个换行，确保每个知识点分段
    return md

def flatten_subtree(node, subtree):
    """将一个子树拍平成 (node, children) 列表，深度优先"""
    result = [(node, subtree)]
    for child, sub in subtree.items():
        result.extend(flatten_subtree(child, sub))
    return result

def render_paginated_outline_final(tree, max_nodes_per_page, level=2):
    md = []
    for root, children in tree.items():
        # ✅ 不再介绍根节点，只讲解它的一级子节点及其下属子图
        for first_level_node, sub_tree in children.items():
            flat_nodes = flatten_subtree(first_level_node, sub_tree)
            total = len(flat_nodes)
            for i in range(0, total, max_nodes_per_page):
                page_nodes = flat_nodes[i:i + max_nodes_per_page]
                md.append(f"# {first_level_node}")
                md.append("")  # 添加一个换行，确保每个知识点分段
                md.extend(render_lecture_page(page_nodes, level))
    return md

def render_summary_page(knowledge_point):
    return [
        "## 小结",
        f"- 本节内容围绕 **{knowledge_point}** 展开，涵盖其相关原理与知识点。",
        "- 【请填写】各子知识点之间的关系与逻辑顺序。\n",
        ""
    ]

def render_exercise_pages(knowledge_points, level=2):
    md = []
    exercise_id = 1
    seen_questions = set()
    for point in knowledge_points:
        exercises = get_exercises_by_knowledge(point)
        for ex in exercises:
            content_key = (ex["title"], ex["content"][:30])
            if content_key in seen_questions:
                continue
            seen_questions.add(content_key)
            md.append(f"{'#' * level} 习题 {exercise_id}")
            md.append(f"### {ex['title']}")
            md.extend(ex["content"].splitlines())
            md.append("")  # 添加一个换行，确保每个知识点分段
            exercise_id += 1
    return md

def render_outline_with_exercises(tree, level=2):
    """渲染知识结构 + 习题页 Markdown，返回每一行组成的 list[str]"""
    md = []

    def dfs(node, children, lvl):
        # 知识点介绍页
        md.append(f"{'#' * lvl} {node}")
        md.append(f"<!-- 知识点“{node}”讲解页 -->")
        md.append(f"- **知识点定义与讲解**：【请填写】“{node}”的定义、概念讲解与背景说明。\n")

        # 知识点对应习题页
        exs = get_exercises_by_knowledge(node)
        logger.info(f"知识点 [{node}] 找到 {len(exs)} 条试题")
        if exs:
            md.append(f"{'#' * lvl} {node} - 随堂练习")
            for idx, ex in enumerate(exs, 1):
                md.append(f"### {idx}. {ex['title']}")
                content = ex.get("content", "")
                # ⚠️ 强制转为纯字符串行
                lines = content.splitlines() if isinstance(content, str) else [str(content)]
                for line in lines:
                    md.append(str(line))

        # 子节点递归
        for child, sub in children.items():
            dfs(child, sub, lvl + 1)

    for root, children in tree.items():
        dfs(root, children, level)

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

    # 知识讲解
    if include("知识讲解"):
        lines = markdown_str.splitlines()
        current_main = None
        current_children = []
        result_sections = []

        for i, line in enumerate(lines):
            # 匹配一级章节标题（比如 # 地转偏向）
            if re.match(r"^# (.+)", line):
                # 如果之前有主章节，保存
                if current_main:
                    result_sections.append({
                        "label": "知识讲解",
                        "type": "main",
                        "content": current_main,
                        "children": current_children
                    })
                    current_children = []

            # 匹配知识点讲解行（子章节）
            if "**知识点定义与讲解**" in line:
                match = re.search(r"对“(.+?)”的定义.*?说明。?", line)
                if match:
                    point_name = match.group(1)
                    point_content = line.strip()
                    if not current_main:
                        current_main = point_content
                    else:
                        current_children.append({
                            "label": "知识讲解",
                            "type": "sub",
                            "content": point_content
                        })

        # 最后一组也要加进去
        if current_main:
            result_sections.append({
                "label": "知识讲解",
                "type": "main",
                "content": current_main,
                "children": current_children
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


@app.route("/api/generate_ppt_outline", methods=["POST"])
def get_ppt_outline():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
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

        # 1️⃣ 构建 markdown 内容
        md = []
        if "主题" in components:
            md.append(f"# {knowledge_point}——探索{knowledge_point}的原理、影响及实际应用")
            md.append(f"\n【请填写】对“{knowledge_point}”的定义、基本特征及应用背景。")
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
                render_paginated_outline_final(knowledge_tree, max_nodes_per_page=data.get("max_nodes_per_page", 6)))
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
        # logger.info("生成的Markdown：\\n" + markdown_str)

        # 2️⃣ 将 Markdown 转换为结构化 JSON（使用新版 strict 方法）
        structured_json = convert_markdown_to_structured_json(markdown_str, components)

        return jsonify(structured_json)

    except Exception as e:
        logger.exception("生成 PPT 大纲失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_ppt_from_outline", methods=["POST"])
def generate_ppt():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        markdown = data.get("markdown", "")
        textbook_pdf_path = data.get("textbook_pdf_path", "")
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


@app.route("/api/generate_ppt_from_outline_and_render", methods=["POST"])
def generate_and_render_ppt():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        markdown = data.get("markdown", "")
        textbook_pdf_path = data.get("textbook_pdf_path", "")
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
            response_generator = large_model_service.get_answer_from_Tyqwen(prompt)
            # return Response(response_generator, content_type='text/plain; charset=utf-8')

        elif llm.lower() == 'deepseek':
            response_generator = large_model_service.get_answer_from_deepseek(prompt)
            # return Response(response_generator, content_type='text/plain; charset=utf-8')
        else:
            return jsonify({"error": f"不支持的模型类型: {llm}"}), 400

        # 4. 渲染为PPT
        ppt_url = markdown_to_ppt.render_markdown_to_ppt(response_generator, title=knowledge_point)
        return jsonify({
            "status": "success",
            "ppt_url": ppt_url,
            "markdown_preview": markdown_text[:200] + "..."  # 可选
        })

    except Exception as e:
        return jsonify({"error": f"接口异常: {str(e)}"}), 500


@app.route("/api/graph/full_export", methods=["GET"])
def export_graph():
    """
    获取整张图谱，返回适配 G6 的数据格式
    """
    result = neo4j_handler.export_full_graph_for_g6()
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7777)