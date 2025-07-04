# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
from collections import defaultdict
import os
import sys
import jieba.posseg as pseg
import requests
import pandas as pd
from werkzeug.utils import secure_filename
# 确保使用UTF-8编码
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append("/work/kmc/kmcGPT/KMC/")
from config.KMC_config import Config
from File_manager.KMC_FileHandler import FileManager
from Prompt.KMC_Prompt import PromptBuilder
from LLM.KMC_LLM import LargeModelAPIService
import csv
import re
import time
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

config = Config(env='production')
config.load_config()  # 指定配置文件的路径
logger = config.logger

ALLOWED_EXTENSIONS = {'pdf', 'docx'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 设置文件上传的目标文件夹
UPLOAD_FOLDER = "/work/kmc/kmcPython/KmcGPT/kmctemp/workspace"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建 FileManager 实例
file_manager = FileManager(config)
prompt_builder = PromptBuilder(config)
large_model_service = LargeModelAPIService(config)
# 定义调用SnoopIE模型的接口地址
api_url = "http://chat.cheniison.cn/api/chat"


def save_data_to_file(data, file_path):
    """将字典数据保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_data_from_file(file_path):
    """从文件加载字典数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # 如果文件不存在，返回空字典


knowledge_entities_dict = {}  # 全局字典，用于存储PDF路径与知识点实体列表的映射
# 在Flask应用对象定义之后立即加载数据`
knowledge_entities_dict = load_data_from_file("/work/kmc/kmcPython/KmcGPT/kmctemp/knowledge_entities_dict.json")


@app.route('/')
def index():
    # 使用knowledge_entities_dict中的数据
    print("已加载字典文件")
    return jsonify(knowledge_entities_dict)


def save_entities_to_csv(knowledge_entities, pdf_path):
    """
    将知识点实体列表保存为CSV文件。
    """
    # 定义CSV文件的保存目录
    csv_dir = r"E:\工作\同济\同济大学自动化专业相关数据包 20240119"
    # 从PDF路径获取文件名（无扩展名）
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # 生成CSV文件名
    csv_file_name = f"{base_name}_entities.csv"
    # 完整的CSV文件路径
    csv_path = os.path.join(csv_dir, csv_file_name)
    required_fields = {"本体", "实体"}  # 定义必需的字段集合

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["本体", "实体"])
        csv_writer.writeheader()
        for item in knowledge_entities:
            # 检查当前条目是否包含所有必需的字段
            if all(field in item for field in required_fields):
                csv_writer.writerow(item)
            else:
                print(f"缺少对应字段: {item}")

    return csv_path


def save_relationships_to_csv(data_rows, columns, pdf_path):
    """
    将知识点之间的关系数据保存为CSV文件。

    :param data_rows: 数据行，每行是一个包含数据的列表。
    :param columns: CSV文件的列名。
    :pdf_path:pdf的路径。
    """
    csv_dir = "/work/kmc/kmcPython/KmcGPT/kmctemp/"
    # 从PDF路径获取文件名（无扩展名）
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # 生成关系CSV文件名
    csv_file_name = f"{base_name}_relationships.csv"
    # 完整的关系CSV文件路径
    relation_csv_path = os.path.join(csv_dir, csv_file_name)
    with open(relation_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入列名
        writer.writerow(columns)
        # 写入数据
        writer.writerows(data_rows)

    return relation_csv_path


def normalize_quotes(text):
    """
    尝试更全面地清理文本中的非标准引号和可能的问题字符。
    """
    # 替换中文双引号
    text = text.replace('“', '"').replace('”', '"')
    # 替换中文单引号
    text = text.replace("‘", '"').replace("’", '"')
    # 试图捕获和替换其他潜在的非标准字符，比如特殊的空格、制表符等
    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    return text


def get_knowledge_entities(pdf_path):
    """
    调用SnoopIE模型，获取知识点的实体列表。
    """
    # 调用SnoopIE模型
    text_json_path = file_manager.extract_text_from_pdf(pdf_path)
    with open(text_json_path, 'r', encoding='utf-8') as json_file:
        pdf_text_by_page = json.load(json_file)

    all_knowledge_entities = []
    seen_entities = set()  # 用于去重的集合

    # 遍历JSON文件中的每一页
    for page, content in pdf_text_by_page.items():
        # 将页面内容按行分割
        if content.strip():  # 确保页面内容不为空
            data = {
                "messages": [
                    {"role": "user", "content": "在下面的文本中抽取出与大学物理学相关的专业实体，并以用一个逗号分隔的列表"
                                                "的形式返回，如：牛顿第二定律，质点，力学。\n列表中内容是你抽取出来的专业实体名称,"
                                                "请直接返回列表内容。\n 待抽取文本:" + content}
                ]
            }
            # 发送POST请求到接口
            # 设置请求头
            headers = {
                "Content-Type": "application/json"
            }

            # 将数据转换为JSON格式
            json_data = json.dumps(data)

            # 发送POST请求
            response = requests.post(api_url, headers=headers, data=json_data)
            # 检查响应状态码
            if response.status_code == 200:
                # 获取响应数据
                response_data = response.json()
                # 提取模型回答的content字段
                model_answer = response_data['choices'][0]['message']['content']
                model_answer = normalize_quotes(model_answer)
                print(f"模型回答: {model_answer}")
            else:
                print(f"请求失败，状态码: {response.status_code}")

            # 使用正则表达式来分割中文逗号和英文逗号
            entities = re.split('，|,', model_answer)
            for entity in entities:
                entity = entity.strip()
                if entity and entity not in seen_entities:  # 确保实体不为空
                    # 为每个实体创建一个包含“本体”和“实体”字段的字典
                    seen_entities.add(entity)
                    all_knowledge_entities.append({"本体": "知识点", "实体": entity.strip()})

    # 在处理完所有页面后返回所有知识点实体
    return all_knowledge_entities


def get_entities_relationship(knowledge_entities, batch_size=10):
    # 构造一个包含所有知识点的文本
    context = """
        {
      "relations": [
        {
          "from": "质点",
          "to": "运动力学",
          "type": "前置"
        },
        {
          "from": "力学",
          "to": "牛顿第二定律",
          "type": "前置"
        },
        {
          "from": "运动学",
          "to": "运动状态",
          "type": "相关"
        },
        {
          "from": "位移",
          "to": "直线运动",
          "type": "包括"
        }
      ]
    }
        """
    data_rows = []
    processed_pairs = set()
    for i in range(0, len(knowledge_entities), batch_size):
        batch_entities = knowledge_entities[i:i + batch_size]
        knowledge_points_text = '请对以下知识点进行处理：\n' + "\n".join([f"- {item['实体']}" for item in batch_entities])
        # 构造请求文本，询问知识点之间的关系
        prompt = f"""以下是大学物理教材的部分知识点，请找到知识点之间的关系。你需要注意的是：\n1.关系总共有三种：前置，包含与相关。其中包含关系是指某个知识点的内容涵盖了另一个知识点，前置关系是指要想学习某个知识点，要先学会他的前置知识点，相关关系是指这两个知识点之间有联系，但并不是包含关系和前置关系。\n2.同一对知识点之间只能存在一种关系\n
                3.你只需要返回json形式的知识点关系，不要有其他的任何文字\n
                4.关系数不能少于知识点数\n
                5.请务必使用以下的json格式：\n{context}\n\n{knowledge_points_text}"""

        data = {
            "messages": [
                {"role": "user", "content":  prompt},
            ]
        }
        # 发送POST请求到接口
        # 设置请求头
        headers = {
            "Content-Type": "application/json"
        }

        # 将数据转换为JSON格式
        json_data = json.dumps(data)

        # 发送POST请求
        response = requests.post(api_url, headers=headers, data=json_data)
        # 检查响应状态码
        if response.status_code == 200:
            # 获取响应数据
            response_data = response.json()
            # 提取模型回答的content字段
            relationship_data = response_data['choices'][0]['message']['content']
            relationship_data = normalize_quotes(relationship_data)
            print(f"模型回答: {relationship_data}")
        else:
            print(f"请求失败，状态码: {response.status_code}")

        open_brace_position = relationship_data.find('{')
        close_brace_position = relationship_data.rfind('}')

        if open_brace_position != -1 and close_brace_position != -1:
            # 返回从第一个"{"到最后一个"}"之间的内容
            relationship_data = relationship_data[open_brace_position:close_brace_position + 1]
        else:
            print("LLM此次返回内容不符合基本格式，请重试")

        relationship_data = json.loads(relationship_data)
        for relation in relationship_data['relations']:
            a = relation['from'].strip()
            b = relation['to'].strip()
            k_type = relation['type'].strip()

            from_to_pair = (a, b)
            to_from_pair = (b, a)

            # 直接添加原始关系的数据行
            if from_to_pair not in processed_pairs:
                data_rows.append(['知识点', a, k_type, '知识点', b])
                processed_pairs.add(from_to_pair)

            # 如果关系类型为“相关”，添加反向关系的数据行
            if k_type == '相关' and to_from_pair not in processed_pairs:
                data_rows.append(['知识点', b, k_type, '知识点', a])
                processed_pairs.add(to_from_pair)

    columns = ['主题本体', '主体', '关系', '客体本体', '客体']
    return columns, data_rows


def convert_csv_to_xlsx(csv_path, xlsx_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    # 保存为XLSX文件
    df.to_excel(xlsx_path, index=False, engine='openpyxl')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # 如果用户没有选择文件，浏览器可能会发送一个空的文件名
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # 使用secure_filename清理文件名
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 获取文件扩展名
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() == '.docx':
            # 如果文件是.docx格式，转换为.pdf
            print("检测到DOCX文件，转换为PDF...")
            pdf_filename = filename.replace('.docx', '.pdf')
            pdf_save_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            # 假设convert_docx_to_pdf已经实现并正确保存了PDF文件
            file_manager.convert_docx_to_pdf(save_path, pdf_save_path)
            save_path = pdf_save_path  # 更新保存路径为PDF文件路径
        else:
            file.save(save_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'storage_path': save_path}, 200)
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    PROCESSED_FILE_DIR = "/work/kmc/kmcPython/KmcGPT/kmctemp/"
    file_path = os.path.join(PROCESSED_FILE_DIR, filename)
    return send_file(file_path, as_attachment=True)


@app.route('/extract-entities', methods=['POST'])
def extract_entities():
    """
    接收PDF文件路径，返回抽取的知识点实体CSV文件路径。
    """
    data = request.get_json()
    pdf_path = data.get('storage_path')
    try:
        # 处理PDF文件并返回知识点实体列表
        knowledge_entities = get_knowledge_entities(pdf_path)
        if knowledge_entities:
            # 使用PDF文件的名称或路径作为键，保存知识点实体列表到全局字典中
            knowledge_entities_dict[pdf_path] = knowledge_entities
            csv_path = save_entities_to_csv(knowledge_entities, pdf_path)
            # 将CSV转换为XLSX
            xlsx_path = csv_path.replace('.csv', '.xlsx')
            convert_csv_to_xlsx(csv_path, xlsx_path)

            # 构造XLSX文件的下载URL
            xlsx_filename = os.path.basename(xlsx_path)
            download_url = f'http://localhost:5678/download/{xlsx_filename}'

            return jsonify({'message': 'Entities extracted successfully', 'downloadUrl': download_url})
        else:
            return jsonify({'error': 'PDF中没有抓取到任何实体'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/extract-relationships', methods=['POST'])
def extract_relationships():
    # global knowledge_entities_dict  # 指明我们要使用的是全局变量
    # # 重新加载字典文件，确保我们使用的是最新的数据
    # knowledge_entities_dict = load_data_from_file(r"E:\工作\同济\同济大学自动化专业相关数据包 20240119\knowledge_entities_dict.json")
    data = request.get_json()
    pdf_path = data.get('storage_path')
    # 从全局字典中使用PDF路径作为键获取对应的知识点实体列表
    if pdf_path in knowledge_entities_dict:
        try:
            knowledge_entities = knowledge_entities_dict[pdf_path]
            # 处理知识点之间关系的代码...
            columns, data_rows = get_entities_relationship(knowledge_entities)
            # 创建DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            xlsx_dir = "/work/kmc/kmcPython/KmcGPT/kmctemp/"
            # 从PDF路径获取文件名（无扩展名）
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            # 生成关系CSV文件名
            xlsx_file_name = f"{base_name}_relationships.xlsx"
            # 完整的关系CSV文件路径
            relation_xlsx_path = os.path.join(xlsx_dir, xlsx_file_name)
            # 保存到Excel文件
            df.to_excel(relation_xlsx_path, index=False)
            download_url = f'http://localhost:5678/download/{xlsx_file_name}'
            print("生成实体关系表完毕")
            return jsonify({'message': 'Relationships extracted successfully', 'downloadUrl': download_url})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': '没有找到这个pdf的实体表'}), 404


def generate_file_id_with_timestamp(url):
    base_name = url.split('/')[-1].split('?')[0]  # 获取基础文件名，去除查询参数
    timestamp = int(time.time())  # 获取当前时间戳
    return f"{timestamp}_{base_name}"


def extract_and_parse_model_outputs(output):
    # 寻找 JSON 对象的开始和结束位置，并正确处理它们
    try:
        results = []
        # 尝试找到所有 JSON 对象
        json_objects = re.findall(r'\{.*?\}', output, re.DOTALL)
        for json_str in json_objects:
            results.append(json.loads(json_str))
        return results
    except json.JSONDecodeError as e:
        logger.error(f"解析JSON时出错：{e}")
        return None
    except ValueError as e:
        logger.error(f"从输出中提取JSON时出错：{e}")
        return None


def clean_and_merge_json_results(results):
    all_nodes = []
    all_links = []
    node_map = {}  # 存储节点名称到ID的映射

    # 处理每一个解析结果
    for result in results:
        for item in result:
            source_name = item['s']
            target_name = item['o']
            relation_name = item['p']

            # 为源节点分配ID
            if source_name not in node_map:
                node_map[source_name] = len(all_nodes) + 1
                all_nodes.append({"id": node_map[source_name], "name": source_name})

            # 为目标节点分配ID
            if target_name not in node_map:
                node_map[target_name] = len(all_nodes) + 1
                all_nodes.append({"id": node_map[target_name], "name": target_name})

            # 添加关系链接
            link = {
                "source": node_map[source_name],
                "target": node_map[target_name],
                "name": relation_name
            }
            # 确保不重复添加相同的链接
            if link not in all_links:
                all_links.append(link)

    # 新增逻辑：过滤source和target相同但name不同的情况，只保留"包含"
    all_links = filter_links_by_preference(all_links)

    # 返回结构化的节点和链接数据
    return json.dumps({"nodes": all_nodes, "links": all_links}, indent=2)


def filter_links_by_preference(links):
    unique_links = {}

    for link in links:
        source = link['source']
        target = link['target']
        name = link['name']

        identifier = (source, target)

        if identifier in unique_links:
            if name == "包含":
                unique_links[identifier] = link
        else:
            unique_links[identifier] = link

    return list(unique_links.values())


@app.route('/extract-all', methods=['POST'])
def process_file():
    data = request.get_json()
    logger.info(f"接收到数据: {data}")
    download_urls = data.get('downloadUrls')

    if not download_urls or not isinstance(download_urls, list):  # 检查download_urls是否为空或不是列表
        return jsonify({'error': '无效的downloadUrls，需要一个非空列表'}), 400

    results = []
    errors = []

    for download_url in tqdm(download_urls, desc="下载和处理文件中。。。"):
        file_id = generate_file_id_with_timestamp(download_url)

        # 下载文件
        file_path = file_manager.download_pdf(download_url, file_id)
        if not file_path:
            logger.error(f"下载文件失败 {file_id}")
            continue  # 处理下一个文件而不是停止整个请求

        # 处理PDF文件
        doc_list = file_manager.process_canvas_file(file_path)
        if doc_list is None:
            logger.error(f"文件处理失败 {file_id}")
            errors.append(f"文件处理失败: {file_path}")
            continue

        # 可以在这里添加进一步的处理，比如调用其他方法分析文本和抽取知识点
        example_prompt = prompt_builder.generate_domain_and_triplets_prompt(doc_list)
        examples = large_model_service.get_answer_from_Tyqwen(example_prompt)

        for segment in tqdm(doc_list, desc=f"抽取教材{file_id}段落中。。。 "):
            text = segment['text']
            extract_prompt = prompt_builder.generate_extract_prompt(examples, text)
            output = large_model_service.get_answer_from_Tyqwen(extract_prompt)
            logger.info(f"抽取结果：{output}")
            parsed_data = extract_and_parse_model_outputs(output)
            if parsed_data is None:
                errors.append(f"无法解析模型输出为有效的 JSON: {output}")
                continue

            logger.info(f"解析结果：{parsed_data}")
            results.append(parsed_data)

    # 处理并合并结果
    final_json = clean_and_merge_json_results(results)

    # 返回处理结果
    logger.info(f"最终结果：{final_json}")
    return jsonify({'data': final_json}), 200


def test_extract_json_from_output():
    # 示例使用
    output = '''
    Here is some text before the JSON.
    {
      "nodes": [
        {"id": 1, "name": "Node1"},
        {"id": 2, "name": "Node2"}
      ],
      "links": [
        {"source": 1, "target": 2, "name": "Link1"}
      ]
    }
    Here is some text after the JSON.
    '''
    extracted_data = extract_json_from_output(output)
    if extracted_data:
        print(json.dumps(extracted_data, indent=2))
    else:
        print("未找到有效的JSON数据")


def test_clean_and_merge_json_results():
    # 正常情况，不同的节点和链接
    results = [
        '{"nodes": [{"id": 1, "name": "Node1"}], "links": [{"source": 1, "target": 2, "name": "Link1"}]}',
        '{"nodes": [{"id": 2, "name": "Node2"}], "links": [{"source": 2, "target": 3, "name": "Link2"}]}'
    ]
    merged_json = clean_and_merge_json_results(results)
    assert len(json.loads(merged_json)['nodes']) == 2, "测试失败：节点数量不正确。"
    assert len(json.loads(merged_json)['links']) == 2, "测试失败：链接数量不正确。"

    # 包含重复节点和链接的情况
    results_with_duplicates = [
        '{"nodes": [{"id": 1, "name": "Node1"}], "links": [{"source": 1, "target": 2, "name": "Link1"}]}',
        '{"nodes": [{"id": 1, "name": "Node1"}], "links": [{"source": 1, "target": 2, "name": "Link1"}]}'
    ]
    merged_json_duplicates = clean_and_merge_json_results(results_with_duplicates)
    assert len(json.loads(merged_json_duplicates)['nodes']) == 1, "测试失败：应当合并重复的节点。"
    assert len(json.loads(merged_json_duplicates)['links']) == 1, "测试失败：应当合并重复的链接。"

    print("所有测试通过！")


def run_tests():
    test_extract_json_from_output()
    test_clean_and_merge_json_results()
    print("所有测试完成。")


if __name__ == '__main__':
    # 在应用启动前运行测试
    # run_tests()
    app.run(host='0.0.0.0', port=5678, debug=False)
