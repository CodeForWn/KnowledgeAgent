# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch, helpers
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
import queue
import threading
import spacy
import logging
from logging.handlers import RotatingFileHandler
import sys
import datetime
sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config

# 全局锁对象
index_lock = threading.Lock()


class ElasticSearchHandler:
    def __init__(self, config):
        # 从配置中提取Elasticsearch相关配置
        self.es_hosts = config.elasticsearch_hosts
        self.es_username = config.elasticsearch_basic_auth_username
        self.es_password = config.elasticsearch_basic_auth_password
        self.backend_notify_api = config.external_api_backend_notify
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModel.from_pretrained(config.model_path)
        self.model.eval()
        self.config = config

        # 使用全局logger实例
        self.logger = self.config.logger

        # 连接到Elasticsearch
        self.es = Elasticsearch(
            hosts=self.es_hosts,
            basic_auth=(self.es_username, self.es_password),
            verify_certs=False
        )
        # 检测是否成功连接到ES
        if self.es.ping():
            self.logger.info("成功连接到Elasticsearch")
        else:
            self.logger.error("无法连接到Elasticsearch")

    def notify_backend(self, file_id, result, failure_reason=None):
        """通知后端接口处理结果"""
        url = self.backend_notify_api  # 更新后的后端接口URL
        headers = {'token': file_id}
        payload = {
            'id': file_id,
            'result': result
        }
        if failure_reason:
            payload['failureReason'] = failure_reason

        response = requests.post(url, json=payload, headers=headers)
        self.logger.info("后端接口返回状态码：%s", response.status_code)
        return response.status_code

    def index_exists(self, index_name):
        # 查询索引是否存在
        return self.es.indices.exists(index=index_name)

    def cal_passage_embed(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy()[0].tolist()

    def cal_query_embed(self, query):
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self.cal_passage_embed(instruction + query)

    def create_answers_index(self):
        try:
            # 定义索引的映射
            mappings = {
                "mappings": {
                    "properties": {
                        "file_id": {"type": "keyword"},  # file_id字段
                        "sum_rec": {"type": "text"}  # sum_rec字段，总结后的文本
                    }
                }
            }

            # 创建索引
            if not self.es.indices.exists(index="answers_index"):
                self.es.indices.create(index="answers_index", body=mappings)
                self.logger.info("索引 'answers_index' 已成功创建")
            else:
                self.logger.info("索引 'answers_index' 已经存在")
        except Exception as e:
            self.logger.error(f"创建索引 'answers_index' 失败: {e}")

    def create_index(self, index_name, doc_list, user_id, assistant_id, file_id, file_name, tenant_id, download_path,
                     tag, createTime):
        with index_lock:
            try:
                mappings = {
                    "properties": {
                        "text": {"type": "text", "analyzer": "standard"},
                        "embed": {"type": "dense_vector", "dims": 1024},
                        "user_id": {"type": "keyword"},
                        "assistant_id": {"type": "keyword"},
                        "tenant_id": {"type": "keyword"},
                        "file_id": {"type": "keyword"},
                        "file_name": {"type": "keyword"},
                        "download_path": {"type": "keyword"},
                        "tag": {"type": "keyword"},
                        "createTime": {
                            "type": "date",
                            "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                        }
                    }
                }
                # 构建查询条件
                query = {"query": {"term": {"file_id": file_id}}}
                if self.index_exists(index_name):
                    self.logger.info("助手已存在，检查文件片段是否存在。。。")
                    if self.check_file_id_exists(index_name, file_id):
                        self.logger.info(f"文件{file_id}片段已存在，删除旧文档...")
                        self.delete_by_query(index_name, query_body=query)
                else:
                    self.logger.info("助手不存在，创建助手。。。")
                    self.es.indices.create(index=index_name, body={'mappings': mappings})

                # 此时无论那种情况都有助手，并没有相关文件的片段，插入文档
                for item in doc_list:
                    embed = self.cal_passage_embed(item['text'])

                    # # 生成总结和问题推荐
                    # ref_list = [item['text'] for item in doc_list]
                    # prompt = prompt_builder.generate_summary_and_questions_prompt(ref_list)
                    # summary = large_model_service.get_answer_from_Tyqwen(prompt)

                    document = {
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "file_id": file_id,
                        "file_name": file_name,
                        "tenant_id": tenant_id,
                        "download_path": download_path,
                        "tag": tag,
                        "createTime": createTime,
                        "page": item['page'],
                        "text": item['text'],
                        "original_text": item['original_text'],
                        "embed": embed
                    }
                    self.es.index(index=index_name, document=document)

                self.logger.info(f"助手{index_name} 创建并插入文件{file_id}成功")
                return True
            except Exception as e:
                self.logger.error(f"创建助手{index_name} 或插入文件{file_id}失败: {e}")
                return False

    def check_file_id_exists(self, index_name, file_id):
        query = {
            "query": {
                "term": {
                    "file_id": {
                        "value": file_id
                    }
                }
            }
        }
        try:
            result = self.es.search(index=index_name, body=query, size=0)
            return result['hits']['total']['value'] > 0
        except Exception as e:
            self.logger.error(f"检查文件ID失败: {e}")
            return False

    def _search_(self, index_pattern, query_body, size=10):
        try:
            result = self.es.search(index=index_pattern, body=query_body, size=size)
            return result
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return None

    def delete_index(self, index_name):
        try:
            # 删除索引
            if self.index_exists(index_name):
                self.es.indices.delete(index=index_name)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting index: {e}")
            return False

    def delete_by_query(self, index, query_body):
        try:
            response = self.es.delete_by_query(index=index, body=query_body)
            return response
        except Exception as e:
            self.logger.error(f"删除操作失败: {str(e)}")
            return None

    def delete_summary_answers(self, file_id):
        try:
            # 构建删除请求的查询
            query = {
                "query": {
                    "term": {"file_id": file_id}
                }
            }
            # 执行删除操作
            self.es.delete_by_query(index="answers_index", body=query)
            self.logger.info(f"成功删除文件ID {file_id} 的相关答案")
            return True
        except Exception as e:
            self.logger.info(f"删除文件ID {file_id} 的相关答案失败: {e}")
            return False

    def index(self, index_name, document):
        try:
            self.es.index(index=index_name, document=document)
            self.logger.info(f"文档已成功插入到索引 {index_name}")
        except Exception as e:
            self.logger.error(f"文档插入失败: {e}")

    def search(self, assistant_id, query_body, ref_num=10):
        try:
            # 在所有符合条件的ES索引中查询
            index_pattern = assistant_id
            result = self.es.search(index=index_pattern, body=query_body, size=ref_num)

            if 'hits' in result and 'hits' in result['hits']:
                # 命中结果
                hits = result['hits']['hits']
                self.logger.info(f"Found {len(hits)} hits")
                refs = [{
                    'text': hit['_source']['text'],
                    'original_text': hit['_source']['original_text'],
                    'page': hit['_source']['page'],
                    'file_id': hit['_source']['file_id'],
                    'file_name': hit['_source']['file_name'],
                    'score': hit['_score'],
                    'download_path': hit['_source']['download_path']
                } for hit in hits]

                return refs

        except Exception as e:
            # 如果未找到相关文本片段
            self.logger.error(f"Error during search: {e}")
            return []

    def ST_search(self, assistant_id, query_body, ref_num=10):
        try:
            # 在所有符合条件的ES索引中查询
            result = self.es.search(index=assistant_id, body=query_body, size=ref_num)

            if 'hits' in result and 'hits' in result['hits']:
                # 命中结果
                hits = result['hits']['hits']
                refs = [{
                    'text': hit['_source']['text'],
                    'file_id': hit['_source']['file_id'],
                    'file_name': hit['_source']['file_name'],
                    'score': hit['_score'],
                } for hit in hits]
                return refs

        except Exception as e:
            # 如果未找到相关文本片段
            self.logger.error(f"Error during search: {e}")
            return []

    def ST_file_search(self, query_body, ref_num=10):
        try:
            # 在所有符合条件的ES索引中查询
            index_name = 'document_metadata'
            result = self.es.search(index=index_name, body=query_body, size=ref_num)
            # self.logger.info(f"Search result: {result}")

            if 'hits' in result and 'hits' in result['hits']:
                # 命中结果
                hits = result['hits']['hits']
                self.logger.info(f"Found {len(hits)} hits")
                file_ids = [hit['_source']['file_id'] for hit in hits]

                return file_ids

        except Exception as e:
            # 如果未找到相关文本片段
            self.logger.error(f"Error during search: {e}")
            return []

    def search_bm25(self, assistant_id, query, ref_num=10, file_id_list=None):
        # 使用BM25方法搜索特定索引
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "text": query
                        }
                    },
                    "should": []
                }
            }
        }
        if file_id_list:
            query_body['query']['bool']['should'] = [{"term": {"file_id": file_id}} for file_id in file_id_list]
            query_body['query']['bool']['minimum_should_match'] = 1  # 至少匹配一个

        # self.logger.info(f"Query Body: {query_body}")
        return self.search(assistant_id, query_body, ref_num)

    def search_embed(self, assistant_id, query, ref_num=10, file_id_list=None):
        # 使用Embed方法搜索特定索引
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "script_score": {
                            "query": {
                                "match_all": {}  # 在整个索引中搜索
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                                "params": {"query_vector": query_embed}
                            }
                        }
                    },
                    "should": []
                }
            }
        }
        if file_id_list:
            query_body['query']['bool']['should'] = [{"term": {"file_id": file_id}} for file_id in file_id_list]
            query_body['query']['bool']['minimum_should_match'] = 1  # 至少匹配一个

        # self.logger.info(f"Query Body: {query_body}")
        return self.search(assistant_id, query_body, ref_num)

    def delete_summary_answers(self, file_id):
        try:
            # 构建删除请求的查询
            query = {
                "query": {
                    "term": {"file_id": file_id}
                }
            }
            # 执行删除操作
            self.es.delete_by_query(index="answers_index", body=query)
            self.logger.info(f"成功删除文件ID {file_id} 的相关答案")
            return True
        except Exception as e:
            self.logger.info(f"删除文件ID {file_id} 的相关答案失败: {e}")
            return False

    def ST_search_bm25(self, assistant_id, query, ref_num=10, file_id_list=None):
        # 使用BM25方法搜索特定索引
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "text": query
                        }
                    },
                    "should": []
                }
            }
        }
        if file_id_list:
            query_body['query']['bool']['should'] = [{"term": {"file_id": file_id}} for file_id in file_id_list]
            query_body['query']['bool']['minimum_should_match'] = 1  # 至少匹配一个

        # self.logger.info(f"Query Body: {query_body}")
        return self.ST_search(assistant_id, query_body, ref_num)

    def ST_search_embed(self, assistant_id, query, ref_num=10, file_id_list=None):
        # 使用Embed方法搜索特定索引
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "script_score": {
                            "query": {
                                "match_all": {}  # 在整个索引中搜索
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                                "params": {"query_vector": query_embed}
                            }
                        }
                    },
                    "should": []
                }
            }
        }
        if file_id_list:
            query_body['query']['bool']['should'] = [{"term": {"file_id": file_id}} for file_id in file_id_list]
            query_body['query']['bool']['minimum_should_match'] = 1  # 至少匹配一个

        # self.logger.info(f"Query Body: {query_body}")
        return self.ST_search(assistant_id, query_body, ref_num)

    def ST_search_abstract_bm25(self, query, ref_num=10):
        # 使用BM25方法搜索'document_metadata'索引中的abstract字段
        query_body = {
            "query": {
                "match": {
                    "abstract": query
                }
            }
        }
        return self.ST_file_search(query_body, ref_num)

    def ST_search_abstract_embed(self, query, ref_num=10):
        # 使用Embed方法搜索'document_metadata'索引中的abstract字段
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}  # 在整个索引中搜索
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'abstract_embed') + 1.0",
                        "params": {"query_vector": query_embed}
                    }
                }
            }
        }
        return self.ST_file_search(query_body, ref_num)
# # 加载配置
# # 使用环境变量指定环境并加载配置
# config = Config(env='development')
# config.load_config('config\\config.json')  # 指定配置文件的路径
#
# # 创建ElasticSearchHandler实例
# es_handler = ElasticSearchHandler(config)
