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
            response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

        if llm.lower() == 'deepseek':
            response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.error(f"Error in get_question_agent: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/api/question_explanation_agent", methods=["POST"])
def get_question_explanation_agent():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        knowledge_point = data.get("knowledge_point", "")
        question_content = data.get("question_content", "")
        question_type = data.get('question_type', '单选题')  # 题型
        llm = data.get('llm', 'deepseek')
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0.4)

        if not question_content:
            return jsonify({"error": "题干为空"}), 400

        # 生成完整提示词：
        prompt = prompt_builder.generate_explanation_prompt_for_qwen(
            knowledge_point=knowledge_point,
            question_type=question_type,
            question_content=question_content
        )

        if llm.lower() == 'qwen':
            response_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

        if llm.lower() == 'deepseek':
            response_generator = large_model_service.get_answer_from_deepseek_stream(prompt, top_p, temperature)
            return Response(response_generator, content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.error(f"Error in get_explanation_question_agent: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7777)