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
from MongoDB.KMC_Mongo import KMCMongoDBHandler



class KMCNeo4jHandler:
    def __init__(self, config):
        try:
            # 从配置中提取Neo4j相关配置
            self.neo4j_uri = config.neo4j_uri
            self.neo4j_username = config.neo4j_username
            self.neo4j_password = config.neo4j_password
            self.config = config
            self.logger = self.config.logger
            self.neo4j_lock = threading.Lock()

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
            with self.neo4j_lock, self.driver.session() as session:
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
                   resource_type: res.resource_type,
                   folder_id: res.folder_id
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
        MATCH (parent:Entity {name: $entity_name})-[:RELATION {type: "前置于"}]->(child:Entity)
        OPTIONAL MATCH path = (child)-[:RELATION*]->(leaf:Entity)
        WHERE ALL(r IN relationships(path) WHERE r.type = "前置于")
          AND NOT (leaf)-[:RELATION {type: "前置于"}]->()
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
        MATCH (parent:Entity {name: $entity_name})-[:RELATION {type: "前置于"}]->(child:Entity)
        RETURN child.name AS child_name
        """
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name).data()
            return [record["child_name"] for record in result]

    def get_related_entities(self, entity_name):
        """
        获取通过 前置/包含/相关 关系连接的一跳 Entity 节点
        """
        query = """
        MATCH (parent:Entity {name: $entity_name})-[r:RELATION|相关]->(child:Entity)
        WHERE r.type IN ["前置于", "包含", "相关"]
        RETURN DISTINCT child.name AS child_name
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

    def build_relation_subgraph(self, root_entity, max_depth=3):
        visited = set()

        def dfs(entity_name, depth):
            if depth > max_depth:
                return {}
            if entity_name in visited:
                return {}  # 防止回环
            visited.add(entity_name)

            related_entities = self.get_related_entities(entity_name)
            subtree = {child: dfs(child, depth + 1) for child in related_entities}

            visited.remove(entity_name)
            return subtree

        return {root_entity: dfs(root_entity, 1)}

    def export_full_graph_for_g6(self):
        """
        导出图谱数据（对 ID 做简化映射，适配 G6 显示）
        """
        try:
            with self.driver.session() as session:
                # 查询所有节点并构建 ID 映射
                node_result = session.run(
                    "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props")
                nodes = []
                node_id_map = {}  # elementId -> shortId
                node_index = 0

                for record in node_result:
                    eid = record["id"]
                    short_id = f"n{node_index}"
                    node_id_map[eid] = short_id

                    labels = record["labels"]
                    props = record["props"]

                    node = {
                        "id": short_id,
                        "label": props.get("name", labels[0] if labels else "Entity"),
                        "type": labels[0] if labels else "Entity",
                        "rawProps": props
                    }
                    nodes.append(node)
                    node_index += 1

                # 查询所有边
                rel_result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN elementId(r) AS id, elementId(a) AS source, elementId(b) AS target, type(r) AS type, properties(r) AS props
                """)
                edges = []
                edge_index = 0

                for record in rel_result:
                    rel_id = f"r{edge_index}"
                    source_id = node_id_map.get(record["source"], record["source"])
                    target_id = node_id_map.get(record["target"], record["target"])

                    rel_type = record["type"]
                    rel_props = record["props"]

                    edge = {
                        "id": rel_id,
                        "source": source_id,
                        "target": target_id,
                        "label": rel_props.get("type", rel_type),
                        "type": rel_type,
                        "rawProps": rel_props
                    }
                    edges.append(edge)
                    edge_index += 1

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            self.logger.error("导出图谱数据失败: {}".format(e))
            return {"nodes": [], "edges": [], "error": str(e)}

    def update_entity_name(self, old_name, new_name):
        query = "MATCH (e:Entity {name: $old_name}) SET e.name = $new_name"
        with self.driver.session() as session:
            session.run(query, old_name=old_name, new_name=new_name)

    def fill_missing_entity_fields(self, default_values: dict):
        """
        为所有 Entity 节点补充缺失字段（不覆盖已有值），并记录日志。
        """
        try:
            with self.driver.session() as session:
                # 查询所有 Entity 节点及其属性
                result = session.run("MATCH (e:Entity) RETURN e.name AS name, properties(e) AS props")
                entities = result.data()

                self.logger.info(f"共找到 {len(entities)} 个 Entity 节点")

                field_counts = {k: 0 for k in default_values.keys()}
                missing_entities = {k: [] for k in default_values.keys()}

                for entity in entities:
                    name = entity["name"]
                    props = entity["props"]
                    for field, default in default_values.items():
                        if field not in props or props[field] is None:
                            field_counts[field] += 1
                            missing_entities[field].append(name)

                # 打印缺失统计
                for field, count in field_counts.items():
                    self.logger.info(f"字段 {field} 缺失节点数量: {count}")
                    if count > 0:
                        preview_names = ", ".join(missing_entities[field][:5])
                        self.logger.info(f"缺失字段 {field} 的前几个节点: {preview_names}")

                # 补齐字段
                query = """
                MATCH (e:Entity)
                SET
                    e.unit = COALESCE(e.unit, $unit),
                    e.kb_id = COALESCE(e.kb_id, $kb_id),
                    e.root_name = COALESCE(e.root_name, $root_name),
                    e.difficulty = COALESCE(e.difficulty, $difficulty),
                    e.type = COALESCE(e.type, $type),
                    e.teaching_requirements = COALESCE(e.teaching_requirements, $teaching_requirements)
                RETURN count(e) AS updated_count
                """
                result = session.run(query, **default_values)
                updated_count = result.single()["updated_count"]
                self.logger.info(f"补齐操作执行完成，共更新 {updated_count} 个知识点字段（仅补缺）")

                return field_counts

        except Exception as e:
            self.logger.error(f"补齐字段失败：{e}")
            return {}

    def update_all_entity_root_name(self, new_root_name):
        """
        将所有 Entity 节点的 root_name 字段统一修改为指定值。
        """
        query = "MATCH (e:Entity) SET e.root_name = $new_root_name RETURN count(e) AS updated_count"
        with self.driver.session() as session:
            result = session.run(query, new_root_name=new_root_name)
            updated = result.single()["updated_count"]
            self.logger.info(f"所有 Entity 节点的 root_name 字段已更新为：{new_root_name}，共计 {updated} 个节点")
            return updated

    def clear_entity_fields(self, name, fields: list):
        """
        将指定名称的知识点的部分字段设为空。
        """
        set_clauses = ", ".join([f"e.{field} = NULL" for field in fields])
        query = f"""
        MATCH (e:Entity {{name: $name}})
        SET {set_clauses}
        RETURN properties(e) AS updated
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            updated_props = result.single()["updated"]
            self.logger.info(f"已将节点“{name}”的字段 {fields} 清空。当前属性为：{updated_props}")
            return updated_props

    def bind_resource_to_entities(self, docID: str, entity_names: list, file_name: str = "",
                                  resource_type: str = "", folder_id: str = "", kb_id: str = "1911603842693210113"):
        """
        将某一资源绑定到多个知识点（Entity 节点），建立“相关”关系。
        - docID: Resource 节点的唯一标识
        - entity_names: 要绑定的知识点名称列表
        - file_name/resource_type: 可选，补充 Resource 节点属性
        """

        bound_entities = []  # ✅ 记录成功绑定的知识点
        skipped_entities = []  # ✅ 记录不存在的知识点

        with self.driver.session() as session:
            for name in entity_names:
                # ✅ 1. 先检查图谱中是否存在该知识点Entity节点
                result = session.run("""
                                MATCH (e:Entity {name: $entity_name})
                                RETURN e
                            """, entity_name=name).single()

                if result:
                    # ✅ 2. 如果Entity存在，则执行绑定
                    session.run("""
                        MERGE (e:Entity {name: $entity_name})
                        MERGE (r:Resource {docID: $docID})
                        SET r.file_name = $file_name,
                            r.resource_type = $resource_type,
                            r.folder_id = $folder_id,
                            r.kb_id = $kb_id
                        MERGE (e)-[:相关]->(r)
                    """, entity_name=name, docID=docID, file_name=file_name, resource_type=resource_type, folder_id=folder_id, kb_id=kb_id)

                    bound_entities.append(name)  # 记录成功绑定的节点
                else:
                    skipped_entities.append(name)

        # ✅ 3. 绑定完成后，统一打印日志
        self.logger.info(f"资源 docID={docID} 成功绑定的知识点有：{bound_entities}")
        if skipped_entities:
            self.logger.warning(f"资源 docID={docID} 以下知识点不存在，跳过绑定：{skipped_entities}")

    def delete_resource_and_relations(self, docID: str):
        """
        删除资源节点及其所有与知识点的“相关”关系。
        """
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (r:Resource {docID: $docID})
                    OPTIONAL MATCH (e:Entity)-[rel:相关]->(r)
                    DELETE rel, r
                """, docID=docID)
            self.logger.info(f"已删除资源节点 docID={docID} 及其相关关系")
        except Exception as e:
            self.logger.error(f"删除资源节点 docID={docID} 失败: {e}")

    def fuzzy_search_entities(self, kb_id: str, query: str, limit: int = 50):
        if query.strip() == "":
            # query 为空字符串时，返回该kb_id下的所有知识点
            cypher = """
            MATCH (e:Entity)
            WHERE e.kb_id = $kb_id
            RETURN e.name AS name
            """
            parameters = {
                "kb_id": kb_id
            }
        else:
            # 正常模糊查询
            cypher = """
            MATCH (e:Entity)
            WHERE e.kb_id = $kb_id AND toLower(e.name) CONTAINS toLower($query)
            RETURN e.name AS name
            LIMIT $limit
            """
            parameters = {
                "kb_id": kb_id,
                "query": query,
                "limit": limit
            }

        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [record["name"] for record in result]

    def update_kb_id_for_entities(self, old_kb_id: str, new_kb_id: str):
        """
        将所有 Entity 节点中 kb_id = old_kb_id 的节点，更新为 new_kb_id。
        """
        query = """
        MATCH (e:Entity {kb_id: $old_kb_id})
        SET e.kb_id = $new_kb_id
        RETURN count(e) AS updated_count
        """
        with self.driver.session() as session:
            result = session.run(query, old_kb_id=old_kb_id, new_kb_id=new_kb_id)
            count = result.single()["updated_count"]
            self.logger.info(f"已将 kb_id = {old_kb_id} 的 Entity 节点共 {count} 个，更新为：{new_kb_id}")
            return count

    def import_triples_from_json(self, filepath: str) -> bool:
        """
        从 JSON 文件加载三元组并导入到 Neo4j。
        JSON 文件应为列表，每项包含 subject, predicate, object。
        实体节点均使用标签 Entity，关系边上的 type 属性存放 predicate。
        返回 True 表示导入成功，False 表示失败。
        """
        # 加载原始三元组列表
        triples = self.load_json(filepath)
        if not isinstance(triples, list):
            self.logger.error(f"无效的三元组文件格式: {filepath}")
            return False

        # 收集所有实体名称
        entities = set()
        for t in triples:
            subj = t.get('subject')
            obj  = t.get('object')
            if not subj or not obj:
                self.logger.warning(f"跳过不完整的三元组: {t}")
                continue
            entities.add(subj)
            entities.add(obj)

        # 构造 import_graph 所需的数据结构
        data = {
            "keywords": list(entities),
            "graph": [
                {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]}
                for t in triples
                if t.get("subject") and t.get("object")
            ]
        }

        # 调用已有的 import_graph 方法执行导入
        success = self.import_graph(data)
        if success:
            self.logger.info(f"已从 {filepath} 成功导入 {len(data['graph'])} 条三元组")
        else:
            self.logger.error(f"从 {filepath} 导入三元组失败")
        return success

    def fetch_main_tree(self, root_name="高中地理"):
        query = """
            MATCH path = (root:Entity {name: $root})-[:RELATION*1..]->(child:Entity)
            WHERE ALL(r IN relationships(path) WHERE r.type = "前置于" OR r.type = "包含")
            WITH collect(nodes(path)) AS all_nodes, collect(relationships(path)) AS all_rels
            UNWIND all_nodes AS nodes_list
            UNWIND nodes_list AS node
            UNWIND all_rels AS rels_list
            UNWIND rels_list AS rel
            RETURN 
                collect(DISTINCT node.name) AS node_names,
                collect(DISTINCT {parent: startNode(rel).name, child: endNode(rel).name}) AS edges
            """
        with self.driver.session() as session:
            result = session.run(query, root=root_name).single()
            if result:
                node_names = set(result["node_names"])
                edges = [(edge["parent"], edge["child"]) for edge in result["edges"]]
                return node_names, edges
            else:
                return set(), []

    # 查询所有知识点节点
    def fetch_all_knowledge_points(self):
        query = "MATCH (n:Entity) RETURN n.name AS name"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["name"] for record in result]

    # 查询所有前置于、包含的关系
    def fetch_all_valid_edges(self):
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WHERE r.type = '前置于' OR r.type = '包含'
        RETURN a.name AS parent, b.name AS child
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [(record["parent"], record["child"]) for record in result]

    def group_resources_by_type(self):
        """
        查询所有Resource节点，按照resource_type分类，列出docID和file_name
        """
        try:
            query = """
            MATCH (r:Resource)
            RETURN r.resource_type AS resource_type, r.docID AS docID, r.file_name AS file_name
            """
            grouped = {}

            with self.driver.session() as session:
                result = session.run(query)
                for record in result:
                    resource_type = record["resource_type"] if record["resource_type"] else "未知类型"
                    docID = record["docID"]
                    file_name = record["file_name"]

                    if resource_type not in grouped:
                        grouped[resource_type] = []
                    grouped[resource_type].append({
                        "docID": docID,
                        "file_name": file_name
                    })

            return grouped
        except Exception as e:
            self.logger.error(f"查询Resource节点失败: {e}")
            return {}

    def update_resources(self):
        mongo_handler = KMCMongoDBHandler(config)
        with self.driver.session() as session:
            # 查询所有Resource节点
            result = session.run("""
                MATCH (r:Resource)
                RETURN id(r) AS node_id, r.docID AS docID
            """)

            update_count = 0
            delete_count = 0

            for record in result:
                node_id = record["node_id"]
                docID = record["docID"]

                if not docID:
                    continue  # 跳过没有docID的节点

                # 查询MongoDB中是否存在资源
                resource_info = mongo_handler.get_folder_id_by_docID(docID)

                if resource_info:
                    folder_id = resource_info.get("folder_id", None)
                    if folder_id:
                        # 更新Neo4j节点，加上folder_id字段
                        session.run("""
                            MATCH (r:Resource)
                            WHERE id(r) = $node_id
                            SET r.folder_id = $folder_id
                        """, node_id=node_id, folder_id=folder_id)
                        update_count += 1
                    else:
                        print(f"[警告] docID {docID} 找到了资源但没有folder_id，跳过更新。")
                else:
                    # MongoDB不存在，删除Resource节点及其与Entity的关系
                    session.run("""
                        MATCH (r:Resource)
                        WHERE id(r) = $node_id
                        OPTIONAL MATCH (r)-[rel]-()
                        DELETE rel, r
                    """, node_id=node_id)
                    delete_count += 1

            print(f"✅ 更新了 {update_count} 个 Resource 节点的 folder_id")
            print(f"✅ 删除了 {delete_count} 个无效 Resource 节点")

    def update_file_names(self):
        mongo_handler = KMCMongoDBHandler(config)
        with self.driver.session() as session:
            # 查询所有 folder_id = 1911604997812920321 的 Resource节点
            result = session.run("""
                MATCH (r:Resource)
                WHERE r.folder_id = '1911604997812920321'
                RETURN id(r) AS node_id, r.docID AS docID
            """)

            update_count = 0
            missing_in_mongo = 0

            for record in result:
                node_id = record["node_id"]
                docID = record["docID"]

                if not docID:
                    continue  # 跳过没有docID的节点

                # 去 MongoDB 的 edu_question 集合中查询
                resource_info = mongo_handler.db["edu_question"].find_one({"docID": docID})

                if resource_info:
                    question_text = resource_info.get("question", None)
                    if question_text:
                        # 更新 Neo4j 中 file_name 字段
                        session.run("""
                            MATCH (r:Resource)
                            WHERE id(r) = $node_id
                            SET r.file_name = $new_file_name
                        """, node_id=node_id, new_file_name=question_text[:500])  # 限制一下长度，防止太长
                        update_count += 1
                    else:
                        print(f"[警告] docID {docID} 在Mongo中找到了，但question字段为空，跳过")
                else:
                    print(f"[警告] docID {docID} 不存在于Mongo的edu_question集合中")
                    missing_in_mongo += 1

            print(f"✅ 成功更新了 {update_count} 个 Resource 节点的 file_name")
            print(f"⚠️ 有 {missing_in_mongo} 个 docID 在MongoDB中未找到或无question字段")

if __name__ == "__main__":
    config = Config()
    config.load_config()  # 如果 Config 中没有此方法，可以删除这行
    neo4j_handler = KMCNeo4jHandler(config)
    neo4j_handler.update_resources()
    mongo_handler = KMCMongoDBHandler(config)
    neo4j_handler.update_file_names()
    neo4j_handler.close()
    mongo_handler.close()

    grouped_resources = neo4j_handler.group_resources_by_type()
    # 保存到 /home/ubuntu/work/kmcGPT/KMC/Neo4j/resource_summary.json
    save_path = "/home/ubuntu/work/kmcGPT/KMC/Neo4j/resource_summary.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(grouped_resources, f, ensure_ascii=False, indent=2)
        print(f"资源分类结果已成功保存到：{save_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")



#     def check_and_clean_orphan_resources():
#         try:
#             with neo4j_handler.driver.session() as session:
#                 # 查询所有Resource节点及其docID
#                 query = "MATCH (r:Resource) RETURN r.docID AS docID"
#                 result = session.run(query)
#                 resource_docids = [record["docID"] for record in result if record["docID"]]
#
#             total = len(resource_docids)
#             deleted_count = 0
#             skipped_count = 0
#
#             print(f"共检测到 {total} 个 Resource 节点，开始检查...")
#
#             for idx, docID in enumerate(resource_docids, 1):
#                 if not docID:
#                     continue
#
#                 # 查询MongoDB中是否存在该docID（优先资源库，再查题库）
#                 filters = {
#                     "kb_id": "1911603842693210113",
#                     "docID": docID
#                 }
#                 documents = mongo_handler.filter_documents(filters)
#                 questions = mongo_handler.filter_questions(filters)
#                 exists_in_mongo = documents or questions
#
#                 if exists_in_mongo:
#                     skipped_count += 1
#                     continue
#
#                 # ✅ 每次自己开session删除，避免Session closed问题
#                 with neo4j_handler.driver.session() as session:
#                     delete_query = """
#                             MATCH (r:Resource {docID: $docID})
#                             OPTIONAL MATCH (r)<-[rel:相关]-(e:Entity)
#                             DELETE rel, r
#                             """
#                     session.run(delete_query, docID=docID)
#                     deleted_count += 1
#
#                 if idx % 50 == 0:
#                     print(f"已处理 {idx}/{total}，已删除 {deleted_count} 个孤立资源")
#
#             print(f"检查完成，总资源数: {total}，跳过: {skipped_count}，删除: {deleted_count}")
#
#         except Exception as e:
#             print(f"清理过程中发生错误: {e}")
#
#     check_and_clean_orphan_resources()
#     neo4j_handler.import_triples_from_json("/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/选必一所有知识点.json")
# #     knowledge_point_name = "地方时"
# #     data = neo4j_handler.get_entity_details(knowledge_point_name)
# #     print(data)
#
#     # print(f"实体节点: {data['entity']}")
#     #
#     # print("\n相关实体关系:")
#     # for relation in data["entity_relations"]:
#     #     print(f"{data['entity']} -[{relation['relationship']}]-> {relation['entity']}")
#     #
#     # print("\n相关资源:")
#     # for res in data["resources"]:
#     #     print(f"资源文件名: {res['file_name']}, docID: {res['docID']}, 类型: {res['resource_type']}")
#     # neo4j_handler.update_entity_name("地球运动", "高中地理")
#     default_values = {
#         "unit": "",
#         "kb_id": "1911603842693210113",
#         "root_name": "高中地理知识图谱",
#         "difficulty": "",
#         "type": "概念型",
#         "teaching_requirements": ""
#     }
#
    # neo4j_handler.update_kb_id_for_entities("高中地理", "1911603842693210113")
    # neo4j_handler.fill_missing_entity_fields(default_values)
    # neo4j_handler.update_all_entity_root_name("选必一")
    # neo4j_handler.clear_entity_fields("高中地理", ["root_name", "unit", "type"])

