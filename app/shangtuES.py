# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import threading
import queue
import urllib3
import logging
import time
import requests
from logging.handlers import RotatingFileHandler
import sys
import re
import uuid
import jieba.posseg as pseg
from FlagEmbedding import FlagReranker
sys.path.append("/work/kmc/kmcGPT/KMC/")
from app.SecurityUtility import SecurityUtility
from config.KMC_config import Config
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager
from LLM.KMC_LLM import LargeModelAPIService
from Prompt.KMC_Prompt import PromptBuilder
import types
import traceback
import time
import json
import requests
import traceback
from flask import jsonify, Response, stream_with_context
from werkzeug.serving import run_simple
import json
import time
import requests
import traceback
from flask import jsonify, request, Response, stream_with_context
from werkzeug.exceptions import BadRequest


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
backend_notify_api = config.external_api_backend_notify
# 创建 FileManager 实例
file_manager = FileManager(config)
# 创建ElasticSearchHandler实例
es_handler = ElasticSearchHandler(config)
large_model_service = LargeModelAPIService(config)
prompt_builder = PromptBuilder(config)
model_path = "/work/kmc/kmcGPT/model/bge-reranker-base"
# 初始化重排模型
reranker = FlagReranker(model_path, use_fp16=True)
# 定义调用SnoopIE模型的接口地址
api_url = "http://chat.cheniison.cn/api/chat"
# 创建队列
file_queue = queue.Queue()
index_lock = threading.Lock()
logger.info('服务启动中。。。')


def _thread_index_func(isFirst):
    while True:
        try:
            _index_func(isFirst)
        except Exception as e:
            print("索引处理失败:", e)


def _index_func(isFirst):
    global file_queue
    try:
        file_data = file_queue.get(timeout=5)
        logger.info("获取到队列语料")
        _process_file_data(file_data)
        logger.info("队列语料处理完毕")
        return True
    except queue.Empty as e:
        if isFirst:
            files = pull_file_data()
            for file_data in files:
                _push(file_data)
            if len(files) == 0:
                # get failed files from server
                pass
        return False
    except Exception as e:
        logger.error("索引功能异常: {}".format(e))


def pull_file_data():
    # 模拟从服务器获取文件数据
    return []


def _push(file_data):
    global file_queue
    file_queue.put(file_data)


@app.route('/api/ST_OCR', methods=['POST'])
def ocr_indexing():
    try:
        data = request.json

        # 检查数据中是否包含docs字段
        if 'docs' not in data:
            raise KeyError("'docs'字段不存在")

        documents = data['docs']
        # 索引名称
        index_name = 'st_ocr'

        # 索引mapping
        mappings = {
            "properties": {
                "AB": {"type": "text", "analyzer": "standard"},
                "CT": {"type": "text", "analyzer": "standard"},
                "Id": {"type": "keyword"},
                "Issue_F": {"type": "keyword"},
                "KW": {"type": "keyword"},
                "JTI": {"type": "keyword"},
                "Pid": {"type": "keyword"},
                "Piid": {"type": "keyword"},
                "TI": {"type": "text", "analyzer": "standard"},
                "Year": {"type": "integer"},
                "AB_embed": {"type": "dense_vector", "dims": 1024},
                "CT_embed": {"type": "dense_vector", "dims": 1024}
            }
        }

        # 检查索引是否存在，如果不存在则创建索引
        if not es_handler.index_exists(index_name):
            logger.info(f"创建索引 {index_name}")
            es_handler.es.indices.create(index=index_name, mappings=mappings)

        for document in documents:
            doc_id = document.get('Id')
            doc_abstract = document.get('AB', None)
            doc_content = document.get('CT', '')

            # 如果摘要为空，生成摘要
            if not doc_abstract or not any(doc_abstract):
                logger.info("摘要为空，生成摘要")
                abstract_prompt = prompt_builder.generate_abstract_prompt(doc_content)
                doc_abstract = large_model_service.get_answer_from_Tyqwen(abstract_prompt)  # 使用大模型生成摘要
                logger.info(f"生成摘要成功，文档ID: {doc_id}, 摘要: {doc_abstract}")
                document['AB'] = [doc_abstract]

            # 计算嵌入
            ab_embed = es_handler.cal_passage_embed(doc_abstract)
            ct_embed = es_handler.cal_passage_embed(doc_content)

            # 添加嵌入到文档中
            document['AB_embed'] = ab_embed
            document['CT_embed'] = ct_embed

            es_handler.es.index(index=index_name, id=doc_id, document=document)
            logger.info(f"插入文档 {doc_id} 到索引 {index_name}")

        logger.info(f"索引 {index_name} 创建并插入数据成功")
        return jsonify({'status': 'success'}), 200

    except KeyError as e:
        logger.error(f"索引创建或插入数据失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

    except Exception as e:
        logger.error(f"索引创建或插入数据失败: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/ST_chatpdf', methods=['POST'])
def ST_chatpdf():
    try:
        # 读取请求参数
        data = request.json
        query = data.get('query')
        file_id_ = data.get('file_id')
        file_id = SecurityUtility.decrypt(file_id_)
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        if not query or not file_id:
            return jsonify({'error': '参数不完整'}), 400

        # 从 Elasticsearch 中获取全文内容
        all_content = ""
        try:
            all_content = es_handler.get_full_text_by_Id(file_id.strip())

        except Exception as e:
            logger.error(f"检索文本内容时出错 {file_id}: {e}")

        if not all_content:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 生成 prompt
        prompt_messages = prompt_builder.generate_chatpdf_prompt(all_content, query)
        logger.info(f"Prompt: {prompt_messages}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

                logger.info(f"问题：{query}")
                logger.info(f"生成的答案：{full_answer}")

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        else:
            return jsonify({'error': '未知的大模型服务'}), 400

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ST_Get_Answer_by_file_id', methods=['POST'])
def ST_get_answer_by_file_id():
    try:
        # 读取请求参数
        data = request.json
        query = data.get('query')
        file_ids = data.get('file_ids')
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.8)
        temperature = data.get('temperature', 0)

        if not query or not file_ids:
            return jsonify({'error': '参数不完整'}), 400

        # 从 Elasticsearch 中获取全文内容
        all_content = ""
        max_chars = 5000
        current_chars_count = 0

        all_content = es_handler.get_full_text_by_Id(
            [SecurityUtility.decrypt(file_id.strip()) for file_id in file_ids])

        if not all_content:
            def generate():
                full_answer = "您的问题没有在文献资料中找到答案，正在使用预训练知识库为您解答："
                prompt = [{'role': 'user', 'content': query}]
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt, top_p=top_p, temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'matches': [], 'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')

        # 生成 prompt
        prompt_messages = prompt_builder.generate_ST_answer_prompt(query, all_content)
        logger.info(f"Prompt: {prompt_messages}")

        if llm == 'qwen':
            def generate():
                full_answer = ""
                ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p,
                                                                                  temperature=temperature)
                for chunk in ans_generator:
                    full_answer += chunk
                    data_stream = json.dumps({'answer': full_answer}, ensure_ascii=False)
                    yield data_stream + '\n'

                logger.info(f"问题：{query}")
                logger.info(f"生成的答案：{full_answer}")

            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')
        else:
            return jsonify({'error': '未知的大模型服务'}), 400
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return jsonify({'error': str(e)}), 500



@app.errorhandler(BadRequest)
def handle_bad_request(e):
    logger.error(f"Bad Request: {e.description}")
    logger.error(f"Request data: {request.data}")
    logger.error(f"Request headers: {request.headers}")
    return jsonify(error=str(e)), 400



@app.route('/ST_Get_Answer', methods=['POST'])
def ST_Get_Answer():
    try:
        start_time = time.time()
        # 读取请求参数
        data = request.json
        query = data.get('query')
        ref_num = data.get('ref_num', 5)
        llm = data.get('llm', 'qwen').lower()
        top_p = data.get('top_p', 0.7)
        temperature = data.get('temperature', 0.7)
        logger.info(f"Received query: {query}")
        logger.info(f"Request parameters: ref_num={ref_num}, llm={llm}, top_p={top_p}, temperature={temperature}")
        if not query:
            raise BadRequest('参数不完整')

        # 执行混合检索
        bm25_start_time = time.time()
        bm25_refs = es_handler.ST_search_bm25(query, ref_num)
        bm25_end_time = time.time()
        logger.info(f"BM25检索用时: {bm25_end_time - bm25_start_time:.2f} seconds")

        embed_start_time = time.time()
        embed_refs = es_handler.ST_search_embed(query, ref_num)
        embed_end_time = time.time()
        logger.info(f"嵌入检索用时: {embed_end_time - embed_start_time:.2f} seconds")

        after_search_time = time.time()
        logger.info(f"BM25和嵌入检索总用时: {after_search_time - bm25_start_time:.2f} seconds")

        merge_start_time = time.time()
        combined_refs = bm25_refs + embed_refs
        unique_refs = {ref['Id']: ref for ref in combined_refs}.values()
        all_refs = list(unique_refs)
        merge_end_time = time.time()
        logger.info(f"合并结果用时: {merge_end_time - merge_start_time:.2f} seconds")
        logger.info(f"合并后all_refs长度: {len(all_refs)}")

        find_refs_time = time.time()
        logger.info(f"总检索用时: {find_refs_time - start_time:.2f} seconds")

        def generate_stream_response(prompt, matches=None):
            full_answer = ""
            buffer = ""
            headers = {
                'Content-Type': 'application/json'
            }
            payload = {
                'model': '/modelscope/hub/qwen/Qwen-14B-Chat',
                'messages': prompt,
                'max_tokens': 1500,
                'temperature': 0.7,
                'stream': True,
                'stop_token_ids': [151645, 151643],
                "top_k": 40,
                "top_p": 0.8,
                'min_tokens': 10,
                'truncate_prompt_tokens': 1000
                
            }

            # 将 payload 写入文件
            with open('vllm_input.txt', 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            logger.info(f"Payload has been written to vllm_input.txt")

            try:
                with requests.post('http://106.14.20.122/37-50004/v1/chat/completions',
                                   headers=headers,
                                   json=payload,
                                   stream=True) as response:
                    response.raise_for_status()
                    model_output_start_time = time.time()
                    logger.info(f"模型开始输出用时: {model_output_start_time - start_time:.2f} seconds")

                    with open('vllm_output.txt', 'w', encoding='utf-8') as output_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                chunk_str = chunk.decode('utf-8')
                                buffer += chunk_str
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    if line.strip() == "data: [DONE]":
                                        logger.info("模型输出完成")
                                        return
                                    if line.startswith("data: "):
                                        try:
                                            data = json.loads(line[6:])
                                            content = data['choices'][0]['delta'].get('content', '')
                                            if content:
                                                full_answer += content
                                                output_file.write(content)
                                                output_file.flush()
                                                yield json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False).encode('utf-8') + b'\n'
                                        except json.JSONDecodeError as e:
                                            logger.error(f"JSON解析错误: {e} - 行内容: {line}")
                                        except KeyError as e:
                                            logger.error(f"键错误: {e} - 数据结构: {data}")

                        if buffer:
                            logger.warning(f"缓冲区中有未处理的数据: {buffer}")

            except requests.RequestException as e:
                logger.error(f"请求失败: {e}")
                if e.response is not None:
                    logger.error(f"响应内容: {e.response.text}")
                    logger.error(f"HTTP状态码: {e.response.status_code}")
                    logger.error(f"请求头: {headers}")
                    logger.error(f"请求体: {payload}")
                logger.error(f"本地模型服务失败: {e}, 切换到 qwen_api")
                for chunk in generate():  # generate 是 qwen_api 的流式生成器
                    yield chunk  # 保持流式输出    
                # yield json.dumps({'error': str(e), 'details': e.response.text if e.response else None},
                                 # ensure_ascii=False).encode('utf-8') + b'\n'
            
        def generate():
            full_answer = ""
            ans_generator = large_model_service.get_answer_from_Tyqwen_stream(prompt_messages, top_p=top_p, temperature=temperature)
            

            model_output_start_time = time.time()

            logger.info(f"当前正在使用API~~")
            logger.info(f"模型开始输出用时: {model_output_start_time - start_time:.2f} seconds")

            for chunk in ans_generator:
                full_answer += chunk
                data_stream = json.dumps({'matches': matches, 'answer': full_answer}, ensure_ascii=False)
                yield data_stream + '\n'


        if not all_refs:
            prompt = [{'role': 'system', 'content': "你是一个近代历史文献研究专家"},
                      {'role': 'user', 'content': query}]
            return Response(stream_with_context(generate_stream_response(prompt)),
                            content_type='application/json; charset=utf-8')

        # 使用重排模型进行重排并归一化得分
        rerank_start_time = time.time()
        ref_pairs = [[query, str(ref['CT'])] for ref in all_refs]
        scores = reranker.compute_score(ref_pairs, normalize=True)
        rerank_end_time = time.time()
        logger.info(f"重排计算用时: {rerank_end_time - rerank_start_time:.2f} seconds")

        # 排序结果
        sort_start_time = time.time()
        sorted_refs = sorted(zip(all_refs, scores), key=lambda x: x[1], reverse=True)
        sort_end_time = time.time()
        logger.info(f"排序用时: {sort_end_time - sort_start_time:.2f} seconds")
        top_list = sorted_refs[:3]
        top_scores = [score for _, score in top_list]
        top_refs = [ref for ref, _ in top_list]
        logger.info(f"重排后最高分：{top_scores}")

        # 生成Prompt
        prompt_start_time = time.time()
        prompt_messages = PromptBuilder.generate_ST_answer_prompt(query, top_refs)
        logger.info(f"生成Prompt: {prompt_messages}")
        prompt_end_time = time.time()
        logger.info(f"生成Prompt用时: {prompt_end_time - prompt_start_time:.2f} seconds")

        # 构建返回的匹配结果
        matches_start_time = time.time()
        matches = [{
            'Pid': ref['Pid'],
            'TI': ref.get('TI', '无标题'),
            'score': ref['score'],
            'rerank_score': score,
            'Id': ref['Id'],
            'Id_': SecurityUtility.encrypt(ref['Id'])
        } for ref, score in top_list]
        matches_end_time = time.time()
        logger.info(f"构建匹配结果用时: {matches_end_time - matches_start_time:.2f} seconds")

        if llm == 'qwen':
            return Response(stream_with_context(generate_stream_response(prompt_messages, matches)),
                                content_type='application/json; charset=utf-8')
        if llm == 'qwen_api':
            return Response(stream_with_context(generate()), content_type='application/json; charset=utf-8')
        else:
            raise BadRequest('未知的大模型服务')

    except BadRequest as e:
        logger.error(f"BadRequest in ST_Get_Answer: {e}")
        logger.error(f"Request data: {request.data}")
        logger.error(f"Request headers: {request.headers}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in ST_Get_Answer: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '服务器内部错误'}), 500




def gpt_4o(prompt):
    data = {
        "model": "gpt-4o",
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt},
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }
    json_data = json.dumps(data)
    response = requests.post(api_url, headers=headers, data=json_data)
    if response.status_code == 200:
        response_data = response.json()
        result_data = response_data['choices'][0]['message']['content']
        result_data = json.loads(json.dumps(result_data, ensure_ascii=False))
        logger.info(f"GPT-4o response: {result_data}")
        return result_data
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None


def escape_solr_query(value):
    special_chars = r'+-&&||!(){}[]^"~*?:\\/'
    escaped_value = ''
    for char in value:
        if char in special_chars:
            escaped_value += '\\' + char
        else:
            escaped_value += char
    return escaped_value


def process_solr_query(solr_query):
    def add_or_all(match):
        field = match.group(1)
        value = match.group(2)
        if value.startswith('(') and value.endswith(')'):
            value = value[1:-1]
        value = value.strip('"\'')
        value_escaped = escape_solr_query(value)
        logger.info(f"Matched field: {field}, value: {value}")
        return f'({field}:"{value_escaped}" OR All:"{value_escaped}")'

    logger.info(f"Before processing: {solr_query}")
    solr_query = solr_query.replace('\\"', '"')
    processed_query = re.sub(r'(TI|JTI|AU):(\(.*?\)|".*?")', add_or_all, solr_query)
    processed_query = processed_query.replace("'", '"')  # Ensure all quotes are double quotes
    logger.info(f"After processing: {processed_query}")
    return processed_query


def generate_all_query(query):
    words = pseg.cut(query)
    nouns = [word for word, flag in words if flag.startswith('n')]
    if not nouns:
        return None
    all_query = ' AND '.join([f'All:"{escape_solr_query(noun)}"' for noun in nouns])
    logger.info(f"Generated All query: {all_query}")
    return all_query


def is_valid_solr_query(solr_query):
    return bool(re.search(r'\b(TI|JTI|AU):', solr_query))


def natural_language_to_solr_query(natural_language_query):
    # 构建GPT-4的请求消息，包含上下文示例
    messages = [
        {'role': 'system',
         'content': "你是一个能将自然语言转换为solr查询字段的专家，你只需要从一个问题中提取关键词，最后形成一个solr的检索语句，不要回答或解释问题。关键词分为4类：标题关键词（对应solr字段为TI"
                    "）、文献来源关键词（对应solr字段为JTI）、年份关键词（对应solr资源为Year）、作者关键词（对应solr字段为AU）。"
                    "对于相同类型的关键词用OR连接，对于不同类的关键词请用AND连接，（例如TI:(机器学习 OR Machine Learning) AND Year:[2023 TO *] "
                    "）。请你注意要区分好人名的用法，有时候人名是标题关键词（蒋介石在西安的谈话显示出他怎样的态度，这里的蒋介石需要在TI"
                    "中检索），有时候人名是作者关键词（请找到三十年代蒋介石发表的文章）。请你注意要区分好历史时期的用法，有时候历史时期是标题关键词（第一次世界大战使用的武器有哪些，这里的第一次世界大战需要在TI"
                    "中检索），有时候历史时期是年份关键词（在第一次世界大战期间，美国对德国的态度是怎样的？。这里的第一次世界大战需要在Year中检索）。一般来说一个关键词只能使用一次，不要在TI、AU "
                    "、Year中重复使用。"},
        {'role': 'assistant', 'content': "好的"},
        {'role': 'user', 'content': "请说明中法战争期间《申报》的舆论导向有何变化？并说明这种变化的原因。"},
        {'role': 'assistant',
         'content': "JTI:\"申报\" AND Year:[1881 TO 1886] AND TI:('中法战争' OR 'Sino-French War')"},
        {'role': 'user',
         'content': "1945年8月6日和9日，美国先后向日本广岛和长崎各投下一颗原子弹，各大报纸争相报道，延安《解放日报》也不例外。请总结一下这一时期《解放日报》对原子弹的报道有哪些变化？这些变化与毛泽东有何关联？"},
        {'role': 'assistant',
         'content': "TI:('广岛' OR '长崎' OR '原子弹' OR '毛泽东') AND JTI:\"解放日报\" AND Year:1945"},
        {'role': 'user', 'content': "请介绍一下鲁迅在四十年代代发表在野草期刊上的文章有哪些？"},
        {'role': 'assistant', 'content': "JTI:\"野草\" AND Year:[1940 TO 1949] AND AU:\"鲁迅\""},
        {'role': 'user', 'content': "最新关于GPT方面的文章有哪些？请说明最近机器学习的方向在哪里？"},
        {'role': 'assistant', 'content': "TI:('GPT' OR '机器学习') AND Year:[2023 TO *]"},
        {'role': 'user', 'content': "知识图谱在信息安全领域有哪些突破和进步？"},
        {'role': 'assistant',
         'content': "TI:('知识图谱' OR 'Knowledge Graph') AND TI:('信息安全' OR '信息保障' OR 'Cybersecurity') AND Year:[2023 TO *]"},
        {'role': 'user', 'content': "第二次世界大战期间，美国对日本的态度有哪些变化？分为几个阶段？"},
        {'role': 'assistant',
         'content': "TI:('美国' OR 'United States') AND TI:('日本' OR 'Japan') AND Year:[1939 TO 1945]"},
        {'role': 'user', 'content': natural_language_query}
    ]
    # 将消息内容转换为字符串
    prompt = json.dumps(messages, ensure_ascii=False)

    # 调用gpt_4函数并返回结果
    solr_query = gpt_4o(prompt)
    if solr_query:
        if is_valid_solr_query(solr_query):
            # 对结果进行处理，增加OR All字段
            processed_solr_query = process_solr_query(solr_query)
            logger.info(f"Processed solr_query: {processed_solr_query}")
            return processed_solr_query
        else:
            logger.warning("转换失败，自动分词")
            all_query = generate_all_query(natural_language_query)
            logger.info(f"分词结果: {all_query}")
            if all_query:
                return all_query
            else:
                return "对不起，无法生成有效的Solr查询语句。"
    else:
        return "对不起，生成Solr查询语句失败。"


@app.route('/convert_query', methods=['POST'])
def convert_query():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    solr_query = natural_language_to_solr_query(query)
    if solr_query:
        logger.info(f"转换结果: {solr_query}")
        return jsonify({'solr_query': solr_query})
    else:
        return jsonify({'error': 'Failed to generate solr query'}), 500


if __name__ == '__main__':
    threads = [threading.Thread(target=_thread_index_func, args=(i == 0,)) for i in range(4)]
    for t in threads:
        t.start()

    app.run(host='0.0.0.0', port=5555, debug=False)
