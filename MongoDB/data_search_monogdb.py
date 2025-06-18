import os
import zipfile
import datetime
import sys
import random
import re
from pymongo import MongoClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from MongoDB.KMC_Mongo import KMCMongoDBHandler
from ElasticSearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager

if __name__ == '__main__':
    # 加载配置
    config = Config()
    config.load_config()
    mongo_handler = KMCMongoDBHandler(config)
    
    # 建议只需一次，确保唯一索引
    mongo_handler.ensure_unique_index("data_search", key="dataset_id")

    # 示例数据（已标准化字段名）
    example_data = {
        # "dataset_id" 可省略，会自动生成
        "name": "CN-DBpedia",
        "dataset_describe": {
            "content": "CN-DBpedia is a never-ending Chinese knowledge extraction system...",
            "type": [
                "knowledge base",
                "knowledge extraction system",
                "Chinese encyclopedia"
            ],
            "domain": [
                "knowledge graph",
                "semantic web",
                "natural language processing",
                "Chinese language processing"
            ],
            "fields": [
                "knowledge extraction",
                "ontology reuse",
                "facts extraction",
                "knowledge base construction",
                "semantic data"
            ]
        },
        "paper_refs": [
            {
                "title": "CN-DBpedia: A Never-Ending Chinese Knowledge Extraction System",
                "authors": [
                        {"name": "Bo Xu", "institution": "Fudan University"},
                        {"name": "Yong Xu", "institution": "Fudan University"},
                        {"name": "Jiaqing Liang", "institution": "Fudan University"},
                        {"name": "Chenhao Xie", "institution": "Fudan University"},
                        {"name": "Bin Liang", "institution": "Fudan University"},
                        {"name": "Wanyun Cui", "institution": "Fudan University"},
                        {"name": "Yanghua Xiao", "institution": "Fudan University"}
                ],
                "venue": "IEA/AIE",
                "year": 2017,
                "url": "https://link.springer.com/chapter/10.1007/978-3-319-60045-1_44",
                "is_fellow": False
            }
        ],
        "dataset_link": "http://kw.fudan.edu.cn/cndbpedia/",
        "platform": "Fudan University Knowledge Works Lab"
    }

    # 插入数据
    try:
        inserted_id = mongo_handler.insert_dataset("data_search", example_data)
        if inserted_id:
            print(f"数据插入成功，_id: {inserted_id}")
        else:
            print("数据插入失败")
    except Exception as e:
        print(f"插入过程中出现异常: {e}")
