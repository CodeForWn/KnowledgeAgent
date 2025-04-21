import json
from neo4j import GraphDatabase

# Neo4j连接信息（修改为你自己的服务器IP、用户名和密码）
uri = "bolt://localhost:7687"
username = "neo4j"
password = "gzc19980214"

# 连接Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

#
# # 加载JSON数据
# def load_json(filepath):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         return json.load(file)
#
#
# # 将数据导入Neo4j
# def import_graph(tx, data):
#     keywords = data["keywords"]
#     relations = data["graph"]
#
#     # 创建实体节点
#     for keyword in keywords:
#         tx.run("""
#             MERGE (e:Entity {name: $name})
#             """, name=keyword)
#
#     # 创建关系（三元组）
#     for relation in relations:
#         tx.run("""
#             MATCH (sub:Entity {name: $subject})
#             MATCH (obj:Entity {name: $object})
#             MERGE (sub)-[:RELATION {type: $predicate}]->(obj)
#             """, subject=relation["subject"], predicate=relation["predicate"], object=relation["object"])
#
#
# if __name__ == "__main__":
#     data = load_json("/home/ubuntu/work/kmcGPT/KMC/neo4j/unit1.json")
#
#     with driver.session() as session:
#         session.execute_write(import_graph, data)
#
#     driver.close()
#     print("数据已成功导入Neo4j。")

# 2. 读取三元组
# with open("/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/选必一所有知识点.json", "r", encoding="utf-8") as f:
#     triples = json.load(f)
#
# def delete_triples(tx, subject, predicate, obj):
#     tx.run(
#         """
#         MATCH (s:Entity {name:$subject})-[r:RELATION {type:$predicate}]->(o:Entity {name:$object})
#         DELETE r
#         """,
#         subject=subject, predicate=predicate, object=obj
#     )
#
# with driver.session() as sess:
#     # 分批删除
#     for t in triples:
#         subj = t.get("subject")
#         pred = t.get("predicate")
#         obj  = t.get("object")
#         if subj and obj:
#             sess.write_transaction(delete_triples, subj, pred, obj)
#
#     # （可选）删除所有孤立的 Entity 节点
#     sess.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e")
#
# driver.close()
# print("已撤销所有对应的三元组及孤立节点。")