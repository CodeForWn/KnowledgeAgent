import json
from neo4j import GraphDatabase
import re
# Neo4j连接信息（修改为你自己的服务器IP、用户名和密码）
uri = "bolt://localhost:7687"
username = "neo4j"
password = "-eZ7mQ_tqjVGHjz"

# 连接Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

# 固定属性值
default_props = {
    "kb_id": "1927242001741434881",
    "unit": "",
    "root_name": "100395",
    "difficulty": "",
    "type": "概念型",
    "teaching_requirements": ""
}

# 加载三元组数据
with open("/home/ubuntu/work/kmcGPT/KMC/Neo4j/编译原理精简版.json", "r", encoding="utf-8") as f:
    triplets = json.load(f)

def create_entity_and_relation(tx, s, p, o):
    # 创建节点及关系（MERGE 避免重复）
    tx.run("""
        MERGE (a:Entity {name: $s})
        SET a += $props
        MERGE (b:Entity {name: $o})
        SET b += $props
        MERGE (a)-[r:RELATION {type: $p}]->(b)
        """, s=s, o=o, p=p, props=default_props)

def batch_import(triplets):
    with driver.session() as session:
        for s, p, o in triplets:
            session.write_transaction(create_entity_and_relation, s, p, o)
    print(f"导入完成，共写入 {len(triplets)} 条三元组。")

if __name__ == "__main__":
    batch_import(triplets)
    driver.close()
# # 批量写入
# with driver.session() as session:
#     for item in triplets:
#         session.write_transaction(create_entity_and_relation,
#                                   item["subject"], item["predicate"], item["object"])
#
# driver.close()
# print("✅ 图谱导入完成")

# # 新节点属性
# root_entity = {
#     "name": "医学微生物学",
#     "kb_id": "1924751678557442049",
#     "difficulty": "",
#     "teaching_requirements": ""
# }
#
# # 被包含的节点
# child_entity_name = "细菌学"
# relation_type = "包含"
#
#
# def create_root_and_link(tx, root, child, rel_type):
#     tx.run("""
#         MERGE (r:Entity {name: $root_name})
#         SET r.kb_id = $kb_id,
#             r.difficulty = $difficulty,
#             r.teaching_requirements = $teaching_requirements
#         MERGE (c:Entity {name: $child_name})
#         MERGE (r)-[:RELATION {type: $rel_type}]->(c)
#     """, root_name=root["name"], kb_id=root["kb_id"],
#          difficulty=root["difficulty"],
#          teaching_requirements=root["teaching_requirements"],
#          child_name=child, rel_type=rel_type)
#
# # 写入数据库
# with driver.session() as session:
#     session.write_transaction(create_root_and_link, root_entity, child_entity_name, relation_type)
#
# driver.close()
# print("✅ 根节点“医学微生物学”及其包含关系已创建完成")
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
# 解析Word内容（你发的内容我这里直接贴成字符串了）
# word_content = """
# S:高中地理P:包含O:地球自转
# S:高中地理P:包含O:地球公转
# S:高中地理P:包含O:岩石圈
# S:高中地理P:包含O:地表形态
# S:高中地理P:包含O:天气系统
# S:高中地理P:包含O:大气环流
# S:高中地理P:包含O:陆地水
# S:高中地理P:包含O:海洋水
# S:高中地理P:包含O:自然环境
# S:地球自转P:包含O:自转方向
# S:地球自转P:包含O:自转周期
# S:地球自转P:包含O:自转速度
# S:地球自转P:前置于O:昼夜更替
# S:地球自转P:前置于O:地方时
# S:地球自转P:前置于O:地转偏向现象
# S:自转周期P:包含O:太阳日
# S:自转周期P:包含O:恒星日
# S:自转速度P:包含O:角速度
# S:自转速度P:包含O:线速度
# S:昼半球P:前置于O:昼夜更替
# S:夜半球P:前置于O:昼夜更替
# S:昼夜更替P:相关O:晨昏线
# S:昼半球P:相关O:昼弧
# S:夜半球P:相关O:夜弧
# S:晨昏线P:相关O:昼弧
# S:晨昏线P:相关O:夜弧
# S:地方时P:前置于O:时区
# S:地方时P:相关O:区时
# S:地方时P:前置于O:日界线
# S:时区P:前置于O:区时
# S:时区P:前置于O:日界线
# S:区时P:相关O:北京时间
# S:日界线P:包含O:国际日界线
# S:日界线P:包含O:自然日界线
# S:地转偏向力P:前置于O:地转偏向现象
# S:地转偏向力P:相关O:洋流
# S:地转偏向力P:相关O:大气环流
# S:地转偏向力P:相关O:季风
# S:地转偏向力P:相关O:信风
# S:地球公转P:包含O:公转方向
# S:地球公转P:包含O:公转周期
# S:地球公转P:包含O:公转速度
# S:地球公转P:前置于O:黄赤交角
# S:地球公转P:前置于O:正午太阳高度
# S:地球公转P:前置于O:五带
# S:地球公转P:前置于O:四季
# S:公转速度P:相关O:近日点
# S:公转速度P:相关O:远日点
# S:公转周期P:包含O:恒星年
# S:公转周期P:包含O:回归年
# S:黄道P:前置于O:黄赤交角
# S:黄赤交角P:前置于O:太阳直射点
# S:太阳直射点P:前置于O:春分日
# S:太阳直射点P:前置于O:夏至日
# S:太阳直射点P:前置于O:秋分日
# S:太阳直射点P:前置于O:冬至日
# S:太阳直射点P:前置于O:北回归线
# S:太阳直射点P:前置于O:南回归线
# S:太阳直射点P:前置于O:正午太阳高度
# S:太阳直射点P:前置于O:昼夜长短变化
# S:太阳高度角P:包含O:正午太阳高度
# S:正午太阳高度P:相关O:五带
# S:昼夜长短变化P:相关O:五带
# S:正午太阳高度P:相关O:四季
# S:昼夜长短变化P:相关O:四季
# S:昼夜长短变化P:包含O:昼长夜短
# S:昼夜长短变化P:包含O:昼短夜长
# S:昼夜长短变化P:包含O:昼夜等长
# S:昼夜长短变化P:前置于O:极昼
# S:昼夜长短变化P:前置于O:极夜
# S:五带P:包含O:热带
# S:五带P:包含O:北温带
# S:五带P:包含O:北寒带
# S:五带P:包含O:南温带
# S:五带P:包含O:南寒带
# S:四季P:包含O:春季
# S:四季P:包含O:夏季
# S:四季P:包含O:秋季
# S:四季P:包含O:冬季
# S:四季P:相关O:二十四节气
# S:岩石圈P:包含O:岩浆岩
# S:岩石圈P:包含O:沉积岩
# S:岩石圈P:包含O:变质岩
# S:岩浆岩P:前置于O:沉积岩
# S:岩浆岩P:前置于O:变质岩
# S:岩浆岩P:包含O:侵入岩
# S:岩浆岩P:包含O:喷出岩
# S:沉积岩P:前置于O:变质岩
# S:变质岩P:前置于O:沉积岩
# S:岩浆P:前置于O:岩浆岩
# S:岩浆岩P:前置于O:岩浆
# S:沉积岩P:前置于O:岩浆
# S:变质岩P:前置于O:岩浆
# S:内力作用P:前置于O:地表形态
# S:外力作用P:前置于O:地表形态
# S:地表形态P:相关O:人类活动
# S:内力作用P:包含O:变质作用
# S:内力作用P:包含O:岩浆活动
# S:内力作用P:包含O:地壳运动
# S:内力作用P:包含O:地震
# S:地壳运动P:包含O:垂直运动
# S:地壳运动P:包含O:水平运动
# S:地壳运动P:相关O:板块构造学说
# S:垂直运动P:前置于O:地质构造
# S:水平运动P:前置于O:地质构造
# S:垂直运动P:前置于O:断层
# S:水平运动P:前置于O:断层
# S:水平运动P:前置于O:褶皱
# S:地质构造P:包含O:断层
# S:地质构造P:包含O:断层
# S:外力作用P:包含O:风化作用
# S:外力作用P:包含O:侵蚀作用
# S:外力作用P:包含O:搬运作用
# S:外力作用P:包含O:沉积作用
# S:人类活动P:包含O:农业生产
# S:人类活动P:包含O:工程建设
# S:人类活动P:包含O:城市发展
# S:天气系统P:包含O:锋
# S:天气系统P:包含O:气旋
# S:天气系统P:包含O:反气旋
# S:锋P:包含O:冷锋
# S:锋P:包含O:暖锋
# S:锋P:包含O:准静止锋
# S:冷锋P:相关O:寒潮
# S:准静止锋P:相关O:梅雨
# S:气旋P:包含O:热带气旋
# S:气旋P:包含O:温带气旋
# S:气旋P:包含O:极地气旋
# S:气旋P:前置于O:阴雨天气
# S:反气旋P:包含O:冷性反气旋
# S:反气旋P:包含O:暖性反气旋
# S:反气旋P:前置于O:晴好天气
# S:冷性反气旋P:相关O:蒙古西伯利亚寒潮源地
# S:暖性反气旋P:相关O:长江中下游盛夏伏旱
# S:大气环流P:包含O:三圈环流（近地面）
# S:大气环流P:包含O:季风环流
# S:三圈环流（近地面）P:包含O:气压带
# S:三圈环流（近地面）P:包含O:风带
# S:气压带P:包含O:低气压带
# S:气压带P:包含O:高气压带
# S:上升气流P:前置于O:低气压带
# S:下沉气流P:前置于O:高气压带
# S:风带P:包含O:西风带
# S:风带P:包含O:信风带
# S:季风环流P:前置于O:季风气候
# S:海陆热力性质差异P:前置于O:季风气候
# S:气压带风带季节性移动P:前置于O:季风气候
# S:陆地水P:包含O:河流水
# S:陆地水P:包含O:湖泊水
# S:陆地水P:包含O:沼泽水
# S:陆地水P:包含O:冰川水
# S:陆地水P:包含O:地下水
# S:大气降水P:前置于O:陆地水
# S:河流水P:前置于O:湖泊水
# S:湖泊水P:前置于O:河流水
# S:湖泊水P:包含O:淡水湖
# S:湖泊水P:包含O:咸水湖
# S:河流水P:前置于O:沼泽水
# S:沼泽水P:前置于O:河流水
# S:冰川水P:前置于O:河流水
# S:河流水P:前置于O:地下水
# S:地下水P:前置于O:河流水
# S:地下水P:包含O:潜水
# S:地下水P:包含O:承层水
# S:海洋水P:包含O:洋流
# S:海洋水P:相关O:海—气相互作用
# S:海洋水P:相关O:海—气相互作用
# S:大气降水P:前置于O:海洋水
# S:洋流P:包含O:风海流
# S:洋流P:包含O:密度流
# S:洋流P:包含O:补偿流
# S:洋流P:包含O:暖流
# S:洋流P:包含O:寒流
# S:洋流P:前置于O:全球热量平衡
# S:洋流P:前置于O:海洋生物分布
# S:洋流P:相关O:海洋航行
# S:洋流P:相关O:污染物扩散
# S:补偿流P:包含O:垂直补偿
# S:补偿流P:包含O:水平补偿
# S:海—气相互作用P:包含O:水分交换
# S:海—气相互作用P:包含O:热量交换
# S:海—气相互作用P:相关O:沃克环流
# S:沃克环流P:包含O:厄尔尼诺现象
# S:沃克环流P:包含O:拉尼娜现象
# S:自然环境P:包含O:自然环境的整体性
# S:自然环境P:包含O:自然环境的差异性
# S:自然环境P:相关O:自然带
# S:自然环境的整体性P:包含O:内在联系性
# S:自然环境的整体性P:包含O:自动调节和平衡
# S:自然环境的整体性P:包含O:统一演化
# S:自然带P:包含O:热带雨林带
# S:自然带P:包含O:热带季雨林带
# S:自然带P:包含O:热带稀树草原带
# S:自然带P:包含O:热带荒漠带
# S:自然带P:包含O:亚热带常绿阔叶林带
# S:自然带P:包含O:亚热带常绿硬叶林带
# S:自然带P:包含O:温带落叶阔叶林带
# S:自然带P:包含O:温带草原带
# S:自然带P:包含O:温带荒漠带
# S:自然带P:包含O:亚寒带针叶林带
# S:自然带P:包含O:极地苔原、冰原带
# S:自然带P:包含O:高山植被
# S:自然环境的差异性P:包含O:地带性差异
# S:自然环境的差异性P:包含O:非地带性差异
# S:地带性差异P:包含O:纬度地带性分异
# S:地带性差异P:包含O:从沿海到内陆的地带性分异
# S:地带性差异P:包含O:垂直地带性分异
# S:热量P:前置于O:纬度地带性分异
# S:水分P:前置于O:从沿海到内陆的地带性分异
# S:海拔高度P:前置于O:垂直地带性分异
# S:地形起伏P:前置于O:非地带性差异
# S:局部环流和洋流P:前置于O:非地带性差异
# S:局部热量异常P:前置于O:非地带性差异
# S:人类活动P:前置于O:非地带性差异
# S:自然环境的整体性P:相关O:自然带
# S:自然环境的差异性P:相关O:自然带
# """
#
# # 解析SPO
# pattern = r"S:(.*?)P:(.*?)O:(.*)"
# spo_list = re.findall(pattern, word_content)
#
# # 开始操作
# with driver.session() as session:
#     # 第一步:删除所有 Entity → Entity 的关系
#     session.run("""
#         MATCH (a:Entity)-[r]->(b:Entity)
#         DELETE r
#     """)
#     print("✅ 已删除所有 Entity 间的关系")
#
#     # 第二步:按 Word文件新增关系
#     for s, p, o in spo_list:
#         cypher = f"""
#         MERGE (s:Entity {{name: "{s}"}})
#         MERGE (o:Entity {{name: "{o}"}})
#         MERGE (s)-[:RELATION {{type: "{p}"}}]->(o)
#         """
#         session.run(cypher)
#
#     print(f"✅ 已根据文档成功插入 {len(spo_list)} 条关系")
#
# driver.close()


























