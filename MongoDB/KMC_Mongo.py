import datetime
import logging
from pymongo import MongoClient, errors
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config


class KMCMongoDBHandler:
    def __init__(self, config):
        try:
            # 从配置中提取MongoDB相关配置
            self.mongodb_host = config.mongodb_host
            self.mongodb_port = config.mongodb_port
            self.mongodb_database = config.mongodb_database
            self.mongodb_username = getattr(config, "mongodb_username", None)
            self.mongodb_password = getattr(config, "mongodb_password", None)
            self.config = config
            self.logger = self.config.logger

            # 建立MongoDB连接
            if self.mongodb_username and self.mongodb_password:
                self.client = MongoClient(
                    host=self.mongodb_host,
                    port=self.mongodb_port,
                    username=self.mongodb_username,
                    password=self.mongodb_password
                )
            else:
                self.client = MongoClient(host=self.mongodb_host, port=self.mongodb_port)
            self.db = self.client[self.mongodb_database]
            self.logger.info("成功连接到 MongoDB {}:{}".format(self.mongodb_host, self.mongodb_port))
        except errors.ConnectionFailure as e:
            self.logger.error("MongoDB连接失败: {}".format(e))
        except Exception as e:
            self.logger.error("连接MongoDB时出现错误: {}".format(e))

    def insert_document(self, collection_name, document):
        """向指定集合中插入单个文档"""
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            self.logger.info("向集合 {} 插入文档成功，ID: {}".format(collection_name, result.inserted_id))
            return result.inserted_id
        except Exception as e:
            self.logger.error("向集合 {} 插入文档失败: {}".format(collection_name, e))
            return None

    def close(self):
        """关闭MongoDB连接"""
        try:
            self.client.close()
            self.logger.info("关闭MongoDB连接成功")
        except Exception as e:
            self.logger.error("关闭MongoDB连接时出错: {}".format(e))


if __name__ == '__main__':
    # 示例用法
    config = Config()  # 假设 Config 类中包含mongodb_host、mongodb_port、mongodb_database、logger等配置项
    config.load_config()
    mongo_handler = KMCMongoDBHandler(config)

