# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
from MongoDB.KMC_Mongo import KMCMongoDBHandler
from Elasticsearch.KMC_ES import ElasticSearchHandler
from File_manager.KMC_FileHandler import FileManager

# 初始化配置与组件
config = Config(env="production")
config.load_config()
mongo_handler = KMCMongoDBHandler(config)
es_handler = ElasticSearchHandler(config)
query_processor = QueryProcessor(config)

# ES 索引名
index_name = "geo_documents_vector"

# 如果有必要，可创建向量索引结构（可选）
es_handler.create_index_if_not_exist(index_name)  # 你可以手动添加这个方法来映射 dense_vector

# 获取全部文档
documents = mongo_handler.filter_documents({}, collection_name="geo_documents")

error_files = []

print(f"共检测到 {len(documents)} 个文档，开始处理...\n")

for doc in tqdm(documents, desc="构建向量索引中"):
    try:
        docID = doc.get("docID")
        file_path = doc.get("file_path", "")
        file_name = doc.get("file_name", "")
        subject = doc.get("subject", "")
        resource_type = doc.get("resource_type", "")
        metadata = doc.get("metadata", {})
        kb_id = doc.get("kb_id", "")
        folder_id = doc.get("folder_id", "")

        if not os.path.exists(file_path):
            config.logger.warning(f"[跳过] 文件不存在：{file_path}")
            error_files.append((docID, "文件不存在"))
            continue

        # 读取全文
        content = mongo_handler.read_file_content(file_path)
        if "出错" in content or not content.strip():
            config.logger.warning(f"[跳过] 内容为空或读取失败：{file_path}")
            error_files.append((docID, "文件读取失败"))
            continue

        # 分段（按段落分）
        segments = [seg.strip() for seg in content.split("\n") if len(seg.strip()) >= 30]

        for idx, seg in enumerate(segments):
            try:
                embed = query_processor.cal_passage_embed(seg)
                es_handler.insert_geo_vector_document(
                    index_name=index_name,
                    docID=docID,
                    file_name=file_name,
                    file_path=file_path,
                    subject=subject,
                    resource_type=resource_type,
                    metadata=metadata,
                    kb_id=kb_id,
                    folder_id=folder_id,
                    page=idx,
                    text=seg,
                    embed=embed
                )
            except Exception as inner_e:
                config.logger.warning(f"[跳过段落] docID={docID}, 段落={idx}, 错误={inner_e}")
                continue

    except Exception as outer_e:
        config.logger.error(f"[跳过文档] docID={doc.get('docID')}，错误：{outer_e}")
        error_files.append((doc.get("docID"), str(outer_e)))
        continue

# 打印错误报告
print("\n📄 构建完成，以下文档处理失败：")
for docID, reason in error_files:
    print(f"- docID: {docID}，原因: {reason}")

print(f"\n✅ 共处理完成 {len(documents) - len(error_files)} 个文档，跳过 {len(error_files)} 个文档。")
