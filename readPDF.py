import re
from pdf2markdown import *
from nltk.tokenize import word_tokenize
import pickle
import shutil
import os


# 计算汉字字符数量
def count_chinese_chars(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return len(pattern.findall(text))


def readPDF(file_path, assistant_name, username, file_type):
    # 定义保存文件的路径
    path_1 = './test'
    file_save = path_1 + '.txt'
    print('start read pdf...')
    # os.system(f"pdftotext {file_path} {file_save}")
    # 调用 pdf2markdown 模块中的 PDF 类来解析 PDF 文件，将其转换为文本，并保存为 .txt 文件
    PDF(file_path, file_save).parsePDF()
    # 初始化一个变量，用于标识文本是否包含中文内容
    ISZH = False
    # 使用 with 语句打开保存的文本文件，使用 UTF-8 编码
    with open(file_save, encoding='utf-8') as f:
        doc = f.read()
        cnt = count_chinese_chars(doc)
        if cnt > 20:
            ISZH = True
    if ISZH:
        # 将文本按换页符 '\f' 分割成一个页面的列表，存储在 doc 中
        doc = doc.split('\f')
    # 定义一个空列表，用于存储最终的文本数据
    doc_list = []

    if ISZH:
        for idx, txt in enumerate(doc):
            # 去掉文本中的回车、换行符
            txt = txt.replace('\n', '').replace('\r', '')

            # 如果文本长度小于等于20，则跳过
            if len(txt) <= 20:
                continue

            # 如果文本中包含句号大于20，则跳过
            if txt.count('.') > 20:
                continue

            # 将文本分割成小段，每段长度为 400，存储在 text_split 中
            text_split = []
            for i in range(0, len(txt), 350):
                text_split.append(txt[i:i+400])

            # 将分割后的文本段落存储为字典，包括页码和文本内容，然后添加到 doc_list 中
            for i in text_split:
                # 添加元数据到文本段
                metadata = {
                    'assistant_name': assistant_name,
                    'username': username,
                    'type': file_type
                }
                doc_list.append({'page': idx+1, 'text': i, 'metadata': metadata})
    else:  # 如果文本不包含中文内容，执行以下操作：
        for idx, txt in enumerate(doc):
            txt = txt.replace('\n', '').replace('\r', '')
            # 对文本进行分词，得到单词列表
            txt = word_tokenize(txt)
            if len(txt) <= 20:
                continue

            # 将文本分割成小段，每段长度为 300，存储在 text_split 中
            text_split = []
            for i in range(0, len(txt), 250):
                text_split.append(txt[i:i+300])
            # 将分割后的文本段落存储为字典，包括页码和文本内容，然后添加到 doc_list 中
            for i in text_split:
                # 添加元数据到文本段
                metadata = {
                    'assistant_name': assistant_name,
                    'username': username,
                    'type': file_type
                }
                doc_list.append({'page': idx+1, 'text': " ".join(i), 'metadata': metadata})

    print('read pdf done.')
    return doc_list
