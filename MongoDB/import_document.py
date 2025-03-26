import os
import zipfile
import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from MongoDB.KMC_Mongo import KMCMongoDBHandler

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

def process_directory(directory_path, mongo_handler, collection_name):
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
                    "file_name": file,
                    "file_path": full_path,
                    "subject": "高中地理",
                    "created_at": datetime.datetime.utcnow()
                }
                inserted_id = mongo_handler.insert_document(collection_name, document)
                print("Inserted document ID:", inserted_id)
        # 如果是目录，也递归处理
        elif os.path.isdir(full_path):
            process_directory(full_path, mongo_handler, collection_name)


if __name__ == '__main__':
    # 加载配置并创建 MongoDB 连接对象
    config = Config()
    config.load_config()  # 确保配置加载成功，包括 mongodb_host、mongodb_port 等
    mongo_handler = KMCMongoDBHandler(config)
    collection_name = "geo_documents"  # 集合名称，根据实际情况设置
    # 指定要处理的文件夹路径，例如：
    folder_path = "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理"
    process_directory(folder_path, mongo_handler, collection_name)
    mongo_handler.close()
