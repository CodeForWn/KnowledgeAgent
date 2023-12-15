from elasticsearch import Elasticsearch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
import requests
import numpy as np
import jieba
import jieba.posseg as pseg
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import nltk
import traceback

# 用户通过创建 DocHandler_Es 类的实例（例如 d = DocHandler_Es()）来初始化一个 DocHandler_Es 对象。
# 使用 handler 方法，将文档/文档库的路径作为参数传递给 DocHandler_Es 对象（例如 doc_base = d.handler('./file_db/1.pkl')）。handler 方法会调用内部的
# __create_es 方法来构建一个 Elasticsearch 数据库，并返回一个 Docbase_Es 对象（doc_base）。
# 一旦 Docbase_Es 对象被创建，它可以被用于执行不同类型的查询操作。用户可以使用 getRef 方法来获取相关文本，还可以使用 getAllFile 方法来获取所有文件列表。这些方法会与已构建的 Elasticsearch
# 数据库进行交互，执行查询并返回相应的结果。

print(">>>>BGE MODEL")
model_path = "/root/es/bge-large-zh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()
print(">>>>BGE MODEL DONE")


def cal_passage_embed(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.inference_mode():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()[0].tolist()


def cal_query_embed(query):
    instruction = "为这个句子生成表示以用于检索相关文章："
    return cal_passage_embed(instruction + query)


class Docbase_Es():
    def __init__(self, es, index_name, filename_list, assistant_name) -> None:
        self.es = es
        self.index_name = index_name
        self.filename_list = filename_list
        self.assistant_name = assistant_name

    def __search(self, query, ref_num):
        result = self.es.search(index=self.index_name, body=query)
        # print(result)
        topk = ref_num
        hits = result['hits']['hits'][:topk]
        refs = [{'text': hit['_source']['text'], 'page': hit['_source']['page'], 'file_id': hit['_source']['file_id'],
                 'file_name': hit['_source']['file_name']} for hit in hits]
        return refs

    def __generate_query(self, ftype, query, func, assistant_name, username, doc_type):
        if ftype == 'file':
            if func == 'embed':
                query_embed = cal_query_embed(query)
                my_query = {
                    "query": {
                        "bool": {
                            "must": [{"term": {"assistant_name": assistant_name}},
                                     {"term": {"username": username}},
                                     {"term": {"type": doc_type}},
                                     {"script_score": {
                                         "query": {"match_all": {}},
                                         "script": {
                                             "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                                             "params": {"query_vector": query_embed}
                                         }
                                     }}]
                        }
                    }
                }
            else:
                my_query = {
                    "query": {
                        "bool": {
                            "must": [{"term": {"assistant_name": assistant_name}},
                                     {"term": {"username": username}},
                                     {"term": {"type": doc_type}},
                                     {"match": {"text": query}}
                                     ]
                        }
                    }
                }
        else:
            # ftype == 'db'
            if func == 'embed':
                query_embed = cal_query_embed(query)
                my_query = {
                    "query": {
                        "bool": {
                            "must": [{"term": {"assistant_name": assistant_name}},
                                     {"term": {"username": username}},
                                     {"term": {"type": doc_type}},
                                     {"script_score": {
                                         "query": {"match_all": {}},
                                         "script": {
                                             "source": "cosineSimilarity(params.query_vector, 'embed') + 1.0",
                                             "params": {"query_vector": query_embed}
                                         }
                                     }}]
                        }
                    }
                }
            else:
                my_query = {
                    "query": {
                        "bool": {
                            "must": [{"term": {"assistant_name": assistant_name}},
                                     {"term": {"username": username}},
                                     {"term": {"type": doc_type}},
                                     {"match": {"text": query}}
                                     ]
                        }
                    }
                }
        return my_query

    def getRef(self, gtype, query, ref_num, func='bm25', assistant_name='', username='', doc_type=''):
        query = self.__generate_query(gtype, query, func, assistant_name, username, doc_type)
        refs = self.__search(query, ref_num)
        return refs

    def getAllFile(self):
        return self.filename_list

    def getAssistantFiles(self):
        # 获取特定助手的全部文件列表
        assistant_files = [file for file in self.filename_list if file['assistant_name'] == self.assistant_name]
        return assistant_files


class DocHandler_Es():
    # 对于单文档或者文档库进行es数据库的构建，并返回一个doc_base
    # 参数说明：db_path: 文档/文档库路径
    #         prefix：默认文档库，单文档为file

    def __init__(self):
        # 当前助手名称
        self.current_assistant = None

    def set_current_assistant(self, assistant_name):
        # 设置当前助手名称
        self.current_assistant = assistant_name

    def handler(self, db_path, prefix='db') -> Docbase_Es:
        data = pd.read_pickle(db_path)
        es, index_name, filename_list = self.__create_es(data, prefix)
        return Docbase_Es(es, index_name, filename_list, assistant_name=self.current_assistant)

    # 建立es索引
    def __create_es(self, data, prefix):
        filename_list = []
        ids = data['db_id']
        index_name = f'{prefix}_{ids}'
        es = Elasticsearch(hosts=['http://127.0.0.1:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
        es.indices.create(index=index_name, ignore=400, body={
            "mappings": {
                "properties": {
                    "embed": {
                        "type": "dense_vector",
                        "dims": 1024
                    },
                    "username": {
                        "type": "keyword"  # 使用keyword类型，确保精确匹配
                    },
                    "type": {
                        "type": "keyword"  # 使用keyword类型，确保精确匹配
                    },
                    "assistant_name": {
                        "type": "keyword"  # 使用keyword类型，确保精确匹配
                    }
                }
            }
        })
        for item in data['file_list']:
            filename = item['name']
            id = item['id']
            assistant_name = item.get('assistant_name', '')  # 获取助手名称
            username = item.get('username', '')  # 获取用户名
            doc_type = item.get('type', '')  # 获取文档类型
            for tuple in item['file']:
                document = {
                    "file_name": filename,
                    "file_id": id,
                    "page": tuple['page'],
                    "text": tuple['text'],
                    "embed": tuple['embed'],
                    "assistant_name": assistant_name,
                    "username": username,
                    "type": doc_type
                }
                es.index(index=index_name, document=document)
            filename_list.append({'id': id, 'name': filename})
        return es, index_name, filename_list


if __name__ == '__main__':
    d = DocHandler_Es()
    d.set_current_assistant('新员工适应向导')
    doc_base = d.handler('./file_db/1.pkl')
    print("done")
    print(doc_base.getRef('db',
                          '处理步骤重启iBMC，查看告警是否清除。具体操作详见服务器iBMC用户指南。\n是 => 处理完毕。\n否 => 2\n\n更换PCIe卡，查看告警是否清除。是 => 处理完毕\n否 => 3\n\n请联系技术支持处理。',
                          3,
                          func='bm25',
                          assistant_name='新员工适应向导',
                          username='admin',
                          doc_type='public'))
