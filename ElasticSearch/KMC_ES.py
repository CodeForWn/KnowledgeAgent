# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import re
import shutil
import pickle
import time
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import jieba.posseg as pseg
import tempfile
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config

# 全局锁对象
index_lock = threading.Lock()


class ElasticSearchHandler:
    def __init__(self, config):
        try:
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
        except elasticsearch.exceptions.ConnectionError as e:
            self.logger.error(f"Elasticsearch连接错误: {e}")
        except Exception as e:
            self.logger.error(f"连接过程中出现其他错误: {e}")

    def notify_backend(self, file_id, result, sync_url, failure_reason=None):
        """通知后端接口处理结果"""
        self.logger.info(f"[ES Handler] 开始回调后端通知 - file_id: {file_id}, result: {result}, sync_url: {sync_url}")
        
        if not sync_url:
            self.logger.warning(f"[ES Handler] 回调失败：sync_url为空 - file_id: {file_id}")
            return None
            
        headers = {'token': file_id}
        payload = {
            'id': file_id,
            'result': result
        }
        if failure_reason:
            payload['failureReason'] = failure_reason
            self.logger.info(f"[ES Handler] 回调包含失败原因 - file_id: {file_id}, failure_reason: {failure_reason}")

        try:
            self.logger.info(f"[ES Handler] 发送回调请求 - file_id: {file_id}, payload: {payload}, headers: {headers}")
            response = requests.post(sync_url, json=payload, headers=headers, timeout=30)
            self.logger.info(f"[ES Handler] 回调成功 - file_id: {file_id}, 状态码: {response.status_code}, 响应内容: {response.text}")
            
            if response.status_code == 200:
                self.logger.info(f"[ES Handler] 后端接口回调成功 - file_id: {file_id}")
            else:
                self.logger.warning(f"[ES Handler] 后端接口回调异常 - file_id: {file_id}, 状态码: {response.status_code}")
                
            return response.status_code
        except requests.exceptions.Timeout:
            self.logger.error(f"[ES Handler] 回调超时 - file_id: {file_id}, sync_url: {sync_url}")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"[ES Handler] 回调连接失败 - file_id: {file_id}, sync_url: {sync_url}")
            return None
        except Exception as e:
            self.logger.error(f"[ES Handler] 回调发生异常 - file_id: {file_id}, 错误: {str(e)}")
            return None

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
        instruction = "为这个句子生成表示以用于检索相关文档片段："
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

    def ST_search_bm25(self, query, ref_num=10):
        # 使用BM25方法搜索特定索引
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["CT"]  # 全文字段
                        }
                    },
                    "should": []
                }
            },
            "size": ref_num
        }

        return self.ST_Search(query_body)

    def ST_search_embed(self, query, ref_num=10):
        # 使用Embed方法搜索特定索引
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'CT_embed') + 1.0",
                                "params": {"query_vector": query_embed}
                            }
                        }
                    },
                    "should": []
                }
            },
            "size": ref_num
        }

        return self.ST_Search(query_body)


    def _process_field(self, field):
        # 处理字段，确保返回字符串
        if isinstance(field, list):
            return '，'.join(map(str, field))  # 将列表中的元素拼接成字符串
        return str(field)  # 如果是单一值，直接转换为字符串


    def ST_Search(self, query_body):
        # 执行Elasticsearch查询
        response = self.es.search(index='st_ocr', body=query_body)
        if response['hits']['total']['value'] > 0:
            hits = response['hits']['hits']
            return [{
                'AB': self._process_field(hit['_source'].get('AB', '无摘要')),
                'AU': self._process_field(hit['_source'].get('AU', '未知作者')),
                'CT': self._process_field(hit['_source'].get('CT', '无内容')),
                'Id': hit['_source'].get('Id', ''),
                'Issue_F': self._process_field(hit['_source'].get('Issue_F', '')),
                'JTI': self._process_field(hit['_source'].get('JTI', '未知期刊')),
                'KW': self._process_field(hit['_source'].get('KW', '')),
                'Pid': hit['_source'].get('Pid', ''),
                'Piid': hit['_source'].get('Piid', ''),
                'TI': self._process_field(hit['_source'].get('TI', '无标题')),
                'Year': str(hit['_source'].get('Year', '未知年份')),
                'score': hit['_score']
            } for hit in hits]
        else:
            return []

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

    def get_full_text_by_Id(self, ids):
        # 如果 ids 是单个字符串，将其转换为列表以统一处理
        if isinstance(ids, str):
            ids = [ids]

        # 构建查询，支持多个 Id 的查询
        query = {
            "query": {
                "terms": {
                    "Id": ids
                }
            }
        }

        return self.ST_Search(query)

    def get_full_text_by_file_id(self, assistant_id, file_id):
        """
        获取ES中指定assistant_id索引内file_id字段值等于指定file_id的所有文档片段，
        并将每一条数据的text字段按顺序拼接起来形成完整的文章内容。

        参数：
        assistant_id (str): 索引名。
        file_id (str): 需要检索的file_id。

        返回：
        str: 拼接后的完整文章内容。
        """
        try:
            # 构建查询条件
            query_body = {
                "query": {
                    "term": {
                        "file_id": {
                            "value": file_id
                        }
                    }
                },
                "sort": [
                    {"page": {"order": "asc"}}  # 根据page字段升序排序
                ]
            }

            # 查询ES索引
            result = self.es.search(index=assistant_id, body=query_body, size=100)  # 假设文档片段不会超过10000条

            # 检查查询结果
            if 'hits' in result and 'hits' in result['hits']:
                hits = result['hits']['hits']

                # 提取并拼接text字段内容
                full_text = "".join(hit['_source']['text'] for hit in hits)
                preview_text = full_text[:10]  # 截取前10个字符
                self.logger.info(f"成功获取索引 {assistant_id} 中 file_id 为 {file_id} 的完整文章内容，前10个字为：{preview_text}")
                return full_text

            else:
                self.logger.warning(f"索引 {assistant_id} 中未找到 file_id 为 {file_id} 的文档片段")
                return ""

        except Exception as e:
            self.logger.error(f"获取索引 {assistant_id} 中 file_id 为 {file_id} 的完整文章内容失败: {e}")
            return ""


    def create_temp_index(self, index_name, doc_list, docID, file_name, file_path, subject, resource_type, metadata):
        """
        创建临时索引，将 doc_list 中的每条文本段向量化后连同文件元数据存入ES索引中。
        参数：
            - index_name: 临时索引名称
            - doc_list: 每个元素格式为 {'page': 页码, 'text': 分段文本, 'original_text': 原始文本}
            - 其它参数：文件元数据（docID, file_name, file_path, subject, resource_type, metadata, created_at）
        """
        with index_lock:
            try:
                mappings = {
                    "mappings": {
                        "properties": {
                            "page": {"type": "integer"},
                            "text": {"type": "text", "analyzer": "standard"},
                            "embed": {"type": "dense_vector", "dims": 1024},
                            "docID": {"type": "keyword"},
                            "file_name": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "subject": {"type": "keyword"},
                            "resource_type": {"type": "keyword"},
                            "metadata": {"type": "object"},
                        }
                    }
                }

                if not self.es.indices.exists(index=index_name):
                    self.logger.info("临时索引 %s 不存在，正在创建...", index_name)
                    self.es.indices.create(index=index_name, body=mappings)
                    self.logger.info("临时索引 %s 创建成功", index_name)
                else:
                    self.logger.info("临时索引 %s 已存在，直接向其中添加数据", index_name)

                # 构造需要插入ES的文档
                actions = []
                for item in doc_list:
                    # 对文本段进行向量化（假设 cal_passage_embed 方法返回1024维向量）
                    embed = self.cal_passage_embed(item['text'])
                    document = {
                        "page": item.get("page"),
                        "text": item.get("text"),
                        "embed": embed,
                        "docID": docID,
                        "file_name": file_name,
                        "file_path": file_path,
                        "subject": subject,
                        "resource_type": resource_type,
                        "metadata": metadata
                    }
                    actions.append({
                        "_index": index_name,
                        "_source": document
                    })

                # helpers.bulk 返回 (成功数量, 错误列表)
                success_count, errors = helpers.bulk(self.es, actions, raise_on_error=False)
                if errors:
                    self.logger.error("以下文档索引失败: %s", errors)
                    return False
                self.logger.info("共索引了 %d 条文档到临时索引 %s 中。", success_count, index_name)
                return True
            except Exception as e:
                self.logger.error(f"创建临时索引 {index_name} 或插入文档失败: {e}", exc_info=True)
                return False

    def agent_search(self, index_name, query_body, ref_num=10):
        """
        统一封装搜索方法，执行 ES 搜索并返回前 ref_num 条结果
        """
        try:
            query_body["_source"] = {"excludes": ["embed"]}
            res = self.es.search(index=index_name, body=query_body)
            hits = res['hits']['hits'][:ref_num]
            self.logger.info("在索引 %s 中，搜索到 %d 条结果", index_name, len(hits))
            return hits
        except Exception as e:
            self.logger.error("ES搜索时出错: %s", e)
            return []

    def agent_search_bm25(self, index_name, query, ref_num=10, file_id_list=None):
        """
        使用 BM25 方法在指定索引中搜索文本，返回匹配的文档片段。
        参数：
          - index_name：要检索的索引名称（对于临时索引可以直接传入临时索引名）
          - query：用户输入的查询文本
          - ref_num：返回文档数量
          - file_id_list：可选，限制搜索范围为特定的文件ID列表
        """
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

        # 这里直接调用内部 search 方法进行检索
        return self.agent_search(index_name, query_body, ref_num)


    def agent_search_embed(self, index_name, query, ref_num=10, file_id_list=None):
        """
        使用 Embed 方法（基于向量相似度）在指定索引中搜索文本分段。
        参数同上：
          - index_name：检索的索引名称
          - query：用户输入的查询文本
          - ref_num：返回文档数量
          - file_id_list：可选，限制搜索范围
        """
        # 计算查询文本的向量，假设 cal_query_embed 方法已实现，返回1024维向量
        query_embed = self.cal_query_embed(query)
        query_body = {
            "query": {
                "bool": {
                    "must": {
                        "script_score": {
                            "query": {"match_all": {}},
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
            query_body['query']['bool']['minimum_should_match'] = 1

        return self.agent_search(index_name, query_body, ref_num)
# if __name__ == "__main__":
#     # 加载配置
#     # 使用环境变量指定环境并加载配置
#     config = Config(env='production')
#     config.load_config()  # 指定配置文件的路径
#
#     # 创建ElasticSearchHandler实例
#     es_handler = ElasticSearchHandler(config)
#
#     # 模拟一个 doc_list，实际可由 file_manager.process_pdf_file 得到
#     doc_list = [
#         {"page": 1, "text": "示例文本内容1"},
#         {"page": 2, "text": "示例文本内容2"}
#     ]
#
#     # 假设已有以下文件元数据
#     docID = "0033-8784-3716"
#     file_name = "第一单元地球自转部分.pdf"
#     file_path = "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/第一单元地球自转部分.pdf"
#     subject = "高中地理"
#     resource_type = "教材"
#     metadata = {"version": "v1.0"}
#
#     # 生成一个临时索引名称，例如：
#     index_name = f"temp_kb_{docID}_{int(time.time())}"
#
#     success = es_handler.create_temp_index(index_name, doc_list, docID, file_name, file_path, subject, resource_type, metadata)
#     if success:
#         print("临时索引创建成功")
#     else:
#         print("临时索引创建失败")
#         sys.exit(1)
#
#     # 模拟使用索引（例如可以检索、测试等，此处仅等待几秒钟）
#     time.sleep(2)
#     query = "地方时是什么？"
#     # 测试 BM25 检索：对索引中存储的文档进行 BM25 搜索
#     bm25_hits = es_handler.agent_search_bm25(index_name, query, ref_num=10)
#     print("BM25 搜索结果:")
#     print(json.dumps(bm25_hits, indent=2, ensure_ascii=False, default=str))
#
#     # 测试 Embed 检索：对索引中存储的文档进行向量检索
#     embed_hits = es_handler.agent_search_embed(index_name, query, ref_num=10)
#     print("Embed 搜索结果:")
#     print(json.dumps(embed_hits, indent=2, ensure_ascii=False, default=str))
#
#     # 删除临时索引
#     es_handler.delete_index(index_name)
#     print(f"索引 {index_name} 已删除")