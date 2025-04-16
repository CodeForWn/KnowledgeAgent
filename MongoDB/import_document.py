import os
import zipfile
import datetime
import sys
import os
import random
from pymongo import MongoClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from MongoDB.KMC_Mongo import KMCMongoDBHandler


def generate_docid():
    """
    生成一个唯一的 docID，格式为三个4位数字拼接，如 "1234-5678-9012"
    """
    return "{}-{}-{}".format(
        str(random.randint(0, 9999)).zfill(4),
        str(random.randint(0, 9999)).zfill(4),
        str(random.randint(0, 9999)).zfill(4)
    )


def extract_zip(zip_path, extract_to):
    """
    解压 zip 文件到指定目录
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to
    except Exception as e:
        print(f"解压 {zip_path} 失败: {e}")
        return None


def process_file(file_path, mongo_handler, collection_name, resource_type=""):
    """
    处理单个文件：
      - 如果是 zip 文件，则先解压后递归处理解压后的目录；
      - 如果是普通文件，则将文件信息存入 MongoDB，并增加学科字段（固定为“高中地理”）。
    """
    if not os.path.exists(file_path):
        print("文件不存在:", file_path)
        return

    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        if file_name.lower().endswith('.zip'):
            # 解压 zip 文件到一个同级目录中，目录名为 zip 文件名（去除扩展名）
            extract_folder = os.path.join(os.path.dirname(file_path), os.path.splitext(file_name)[0])
            if not os.path.exists(extract_folder):
                os.makedirs(extract_folder)
            extracted = extract_zip(file_path, extract_folder)
            if extracted:
                # 解压后的文件夹中可能包含多个文件，调用 process_directory 处理
                process_directory(extract_folder, mongo_handler, collection_name)
        else:
            # 普通文件，构造文档并插入到 MongoDB
            document = {
                "docID": generate_docid(),
                "file_name": file_name,
                "file_path": file_path,
                "subject": "高中地理",
                "metadata": {"version": "v1.0"},
                "difficulty_level": "困难",
                "question_type": "填空题",
                "created_at": datetime.datetime.utcnow()
            }
            inserted_id = mongo_handler.insert_document(collection_name, document)
            print("Inserted document ID:", inserted_id)
    else:
        print("指定路径不是一个文件:", file_path)


def process_directory(directory_path, mongo_handler, collection_name, resource_type=""):
    """
    遍历指定目录下的所有文件：
      - 如果是 zip 文件，则先解压后递归处理解压后的目录
      - 如果是普通文件，则将文件信息存入 MongoDB，并增加学科字段（固定为“高中地理”）
      - 同时过滤掉以 "._" 开头的文件，避免乱码问题
    """
    for file in os.listdir(directory_path):
        # 过滤掉以 "._" 开头的隐藏或辅助文件
        if file.startswith("._"):
            continue
        full_path = os.path.join(directory_path, file)
        # 如果是文件
        if os.path.isfile(full_path):
            if file.lower().endswith('.zip'):
                # 解压 zip 文件到一个同级目录中，目录名为 zip 文件名（去除扩展名）
                extract_folder = os.path.join(directory_path, os.path.splitext(file)[0])
                if not os.path.exists(extract_folder):
                    os.makedirs(extract_folder)
                extracted = extract_zip(full_path, extract_folder)
                if extracted:
                    # 递归处理解压后的文件夹
                    process_directory(extract_folder, mongo_handler, collection_name)
            else:
                # 普通文件，构造文档并插入到 MongoDB
                document = {
                    "docID": generate_docid(),
                    "file_name": file,
                    "file_path": full_path,
                    "subject": "高中地理",
                    "resource_type": resource_type,
                    "metadata": {"version": "v1.0"},
                    "created_at": datetime.datetime.utcnow()
                }
                inserted_id = mongo_handler.insert_document(collection_name, document)
                print("Inserted document ID:", inserted_id)
        # 如果是目录，也递归处理
        elif os.path.isdir(full_path):
            process_directory(full_path, mongo_handler, collection_name)


def complete_all_resources(mongo_handler, collection_name):
    """
    遍历数据库中所有资源文档，确保以下字段存在：
    - kb_id（默认空字符串）
    - resource_type（默认空字符串）
    - status（统一设为"on"）
    """
    collection = mongo_handler.db[collection_name]
    cursor = collection.find({})

    total = 0
    for doc in cursor:
        update_fields = {}

        if 'kb_id' not in doc:
            update_fields['kb_id'] = ''
        if 'resource_type' not in doc:
            update_fields['resource_type'] = ''
        if doc.get('status') != 'on':
            update_fields['status'] = 'on'
        if 'user_id' not in doc:
            update_fields['user_id'] = ''

        if update_fields:
            collection.update_one({'_id': doc['_id']}, {'$set': update_fields})
            total += 1

    print(f"已更新 {total} 条资源记录，确保字段完整")


if __name__ == '__main__':
    # 加载配置并创建 MongoDB 连接对象
    config = Config()
    config.load_config()  # 确保配置加载成功，包括 mongodb_host、mongodb_port 等
    mongo_handler = KMCMongoDBHandler(config)

    collection_name = "geo_documents"  # 集合名称，根据实际情况设置
    collection_name_ques = "edu_question"

    # 指定要处理的文件夹路径，例如：
    # folder_path = "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理"
    # process_directory(folder_path, mongo_handler, collection_name)
    # file_path = "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 练习/其他练习/地方时填空题（困难）.docx"
    # process_file(file_path, mongo_handler, collection_name)
    # 第二步：补全数据库中已有的资源字段
    complete_all_resources(mongo_handler, collection_name)

    # 插入一条空题目文档，用于建立字段结构
    # mongo_handler.db[collection_name_ques].insert_one({
    #     "docID": generate_docid(),
    #     "question": "",
    #     "answer": "",
    #     "analysis": "",
    #     "type": "",
    #     "diff_level": "",
    #     "created_at": datetime.datetime.utcnow(),
    #     "user_id": "",
    #     "status": "on",
    #     "resource_type": "试题",
    #     "subject": "高中地理",
    #     "kb_id": ""
    # })
    #
    # print(f"✅ edu_questions 集合已初始化")

    mongo_handler.close()
