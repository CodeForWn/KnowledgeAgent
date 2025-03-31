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
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from neo4j import GraphDatabase, basic_auth


class KMCNeo4jHandler:
    def __init__(self, config):
        try:
            # 从配置中提取Neo4j相关配置
            self.neo4j_uri = config.neo4j_uri
            self.neo4j_username = config.neo4j_username
            self.neo4j_password = config.neo4j_password
            self.config = config
            self.logger = self.config.logger

            # 建立Neo4j连接
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record.get("test") == 1:
                    self.logger.info("成功连接到 Neo4j {}".format(self.neo4j_uri))
                else:
                    self.logger.error("Neo4j连接测试失败")

        except Exception as e:
            self.logger.error("连接Neo4j时出现错误: {}".format(e))

    def close(self):
        """
        关闭 Neo4j 连接
        """
        try:
            self.driver.close()
            self.logger.info("Neo4j 连接已关闭")
        except Exception as e:
            self.logger.error("关闭 Neo4j 连接时出现错误: {}".format(e))

    @staticmethod
    def load_json(filepath):
        """
        从指定路径加载 JSON 数据
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error("加载 JSON 文件时出错: {}".format(e))
            return None

    def import_graph(self, data):
        """
        导入图数据到 Neo4j 数据库。
        JSON 数据格式需包含：
            - keywords: 实体名称列表
            - graph: 三元组列表，每个三元组包含 subject, predicate, object
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._import_graph, data)
            self.logger.info("数据已成功导入 Neo4j")
            return True
        except Exception as e:
            self.logger.error("导入数据到 Neo4j 时出错: {}".format(e))
            return False

    @staticmethod
    def _import_graph(tx, data):
        keywords = data.get("keywords", [])
        relations = data.get("graph", [])
        # 创建实体节点
        for keyword in keywords:
            tx.run("MERGE (e:Entity {name: $name})", name=keyword)
        # 创建关系（三元组）
        for relation in relations:
            tx.run(
                "MATCH (sub:Entity {name: $subject}) "
                "MATCH (obj:Entity {name: $object}) "
                "MERGE (sub)-[:RELATION {type: $predicate}]->(obj)",
                subject=relation.get("subject"),
                predicate=relation.get("predicate"),
                object=relation.get("object")
            )

    def add_entity(self, name, properties=None):
        """
        添加实体节点，支持附加属性。
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._add_entity, name, properties)
            self.logger.info("实体节点 '{}' 添加成功".format(name))
            return True
        except Exception as e:
            self.logger.error("添加实体节点 '{}' 时出错: {}".format(name, e))
            return False

    @staticmethod
    def _add_entity(tx, name, properties):
        if properties is None:
            properties = {}
        query = "MERGE (e:Entity {name: $name}) SET e += $properties RETURN e"
        return tx.run(query, name=name, properties=properties)

    def update_entity(self, name, properties):
        """
        更新实体节点的属性
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._update_entity, name, properties)
            self.logger.info("实体节点 '{}' 更新成功".format(name))
            return True
        except Exception as e:
            self.logger.error("更新实体节点 '{}' 时出错: {}".format(name, e))
            return False

    @staticmethod
    def _update_entity(tx, name, properties):
        query = "MATCH (e:Entity {name: $name}) SET e += $properties RETURN e"
        return tx.run(query, name=name, properties=properties)

    def delete_entity(self, name):
        """
        删除实体节点及其所有关联关系
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._delete_entity, name)
            self.logger.info("实体节点 '{}' 删除成功".format(name))
            return True
        except Exception as e:
            self.logger.error("删除实体节点 '{}' 时出错: {}".format(name, e))
            return False

    @staticmethod
    def _delete_entity(tx, name):
        query = "MATCH (e:Entity {name: $name}) DETACH DELETE e"
        return tx.run(query, name=name)

    def add_relationship(self, subject, object, relation_type, properties=None):
        """
        添加实体之间的关系，支持附加属性。
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._add_relationship, subject, object, relation_type, properties)
            self.logger.info("关系 '{}' 在 '{}' 与 '{}' 之间添加成功".format(relation_type, subject, object))
            return True
        except Exception as e:
            self.logger.error("添加关系 '{}' 时出错: {}".format(relation_type, e))
            return False

    @staticmethod
    def _add_relationship(tx, subject, object, relation_type, properties):
        if properties is None:
            properties = {}
        query = (
            "MATCH (sub:Entity {name: $subject}) "
            "MATCH (obj:Entity {name: $object}) "
            "MERGE (sub)-[r:RELATION {type: $relation_type}]->(obj) "
            "SET r += $properties RETURN r"
        )
        return tx.run(query, subject=subject, object=object, relation_type=relation_type, properties=properties)

    def update_relationship(self, subject, object, relation_type, properties):
        """
        更新实体之间关系的属性
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._update_relationship, subject, object, relation_type, properties)
            self.logger.info("关系 '{}' 在 '{}' 与 '{}' 之间更新成功".format(relation_type, subject, object))
            return True
        except Exception as e:
            self.logger.error("更新关系 '{}' 时出错: {}".format(relation_type, e))
            return False

    @staticmethod
    def _update_relationship(tx, subject, object, relation_type, properties):
        query = (
            "MATCH (sub:Entity {name: $subject})-[r:RELATION {type: $relation_type}]->(obj:Entity {name: $object}) "
            "SET r += $properties RETURN r"
        )
        return tx.run(query, subject=subject, object=object, relation_type=relation_type, properties=properties)

    def delete_relationship(self, subject, object, relation_type):
        """
        删除实体之间的指定关系
        """
        try:
            with neo4j_lock, self.driver.session() as session:
                session.write_transaction(self._delete_relationship, subject, object, relation_type)
            self.logger.info("关系 '{}' 在 '{}' 与 '{}' 之间删除成功".format(relation_type, subject, object))
            return True
        except Exception as e:
            self.logger.error("删除关系 '{}' 时出错: {}".format(relation_type, e))
            return False

    @staticmethod
    def _delete_relationship(tx, subject, object, relation_type):
        query = (
            "MATCH (sub:Entity {name: $subject})-[r:RELATION {type: $relation_type}]->(obj:Entity {name: $object}) "
            "DELETE r"
        )
        return tx.run(query, subject=subject, object=object, relation_type=relation_type)

    def run_query(self, query, **parameters):
        """
        执行自定义 Cypher 查询并返回结果
        """
        try:
            with self.driver.session() as session:
                result = session.read_transaction(lambda tx: list(tx.run(query, **parameters)))
            self.logger.info("查询执行成功")
            return result
        except Exception as e:
            self.logger.error("执行查询时出错: {}".format(e))
            return None

    def get_entity_details(self, knowledge_point_name):
        query = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH (e)-[:`RELATION`]-(related:Entity)
        OPTIONAL MATCH (e)-[:`相关`]->(res:Resource)
        RETURN properties(e) as entity,
               collect(DISTINCT {relationship:'RELATION', entity: related.name}) as entity_relations,
               collect(DISTINCT {
                   docID: res.docID, 
                   file_name: res.file_name, 
                   resource_type: res.resource_type
               }) as resources
        """

        with self.driver.session() as session:
            result = session.run(query, name=knowledge_point_name).single()

            if result is None:
                return {}

            entity_props = result["entity"]
            entity_relations = result["entity_relations"]
            resources = result["resources"]

            return {
                "entity": entity_props,
                "entity_relations": [er for er in entity_relations if er["entity"] is not None],
                "resources": [res for res in resources if res["docID"] is not None]
            }

    def get_predecessor_tree(self, entity_name):
        query = """
        MATCH (parent:Entity {name: $entity_name})-[:RELATION {type: "前置"}]->(child:Entity)
        OPTIONAL MATCH path = (child)-[:RELATION*]->(leaf:Entity)
        WHERE ALL(r IN relationships(path) WHERE r.type = "前置")
          AND NOT (leaf)-[:RELATION {type: "前置"}]->()
        RETURN child.name AS child_name,
               collect(DISTINCT {name: leaf.name}) AS leaves
        """
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name).data()

            knowledge_tree = {}
            for record in result:
                knowledge_tree[record['child_name']] = {
                    "description": f"请在此处补充对“{record['child_name']}”的定义与讲解。",
                    "leaves": record.get("leaves", [])
                }

            return knowledge_tree

    def get_related_exercises(self, entity_name):
        query = """
        MATCH (entity:Entity {name: $entity_name})-[:RELATED_TO]->(res:Resource)
        WHERE res.resource_type = '试题'
          AND res.metadata.difficulty IS NOT NULL
          AND res.metadata.question_type IS NOT NULL
        RETURN res
        """
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name).data()

            exercises = []
            for record in result:
                resource = record['res']
                exercises.append({
                    "title": resource.get("title", ""),
                    "type": resource["metadata"].get("question_type", ""),
                    "difficulty": resource["metadata"].get("difficulty", ""),
                    "content": self.get_resource_content(resource["docID"])  # 假设已有方法
                })

            return exercises

    def get_resource_docIDs(self, entity_name):
        query = """
        MATCH (entity:Entity {name: $entity_name})-[:相关]->(res:Resource)
        RETURN DISTINCT res.docID AS docID
        """
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name).data()
            docIDs = [record['docID'] for record in result if record['docID'] is not None]
        return docIDs

    def get_one_hop_predecessors(self, entity_name):
        query = """
        MATCH (parent:Entity {name: $entity_name})-[:RELATION {type: "前置"}]->(child:Entity)
        RETURN child.name AS child_name
        """
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name).data()
            return [record["child_name"] for record in result]

    def build_predecessor_tree(self, root_entity, max_depth=5):
        def dfs(entity_name, depth):
            if depth > max_depth:
                return {}
            children = self.get_one_hop_predecessors(entity_name)
            return {child: dfs(child, depth + 1) for child in children}

        return {root_entity: dfs(root_entity, 1)}

# if __name__ == "__main__":

#     config = Config()
#     config.load_config()  # 如果 Config 中没有此方法，可以删除这行
#     neo4j_handler = KMCNeo4jHandler(config)
#     knowledge_point_name = "地方时"
#     data = neo4j_handler.get_entity_details(knowledge_point_name)
#     print(data)

    # print(f"实体节点: {data['entity']}")
    #
    # print("\n相关实体关系:")
    # for relation in data["entity_relations"]:
    #     print(f"{data['entity']} -[{relation['relationship']}]-> {relation['entity']}")
    #
    # print("\n相关资源:")
    # for res in data["resources"]:
    #     print(f"资源文件名: {res['file_name']}, docID: {res['docID']}, 类型: {res['resource_type']}")

    # neo4j_handler.close()

