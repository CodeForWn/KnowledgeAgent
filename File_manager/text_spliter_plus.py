import spacy
import re
import PyPDF2
import pypdf
import pikepdf  # 假设这是处理PDF的自定义类
from matplotlib import pyplot as plt
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import matplotlib
from prettytable import PrettyTable
matplotlib.use('TkAgg')  # 或尝试其他后端，如 'Agg', 'Qt5Agg' 等

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# print(spacy.util.get_package_path("zh_core_web_sm"))


def spacy_chinese_text_splitter(text, max_length=400):
    # 加载spaCy中文模型
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(text)

    chunks = []
    current_chunk = ""
    sentence_delimiters = re.compile(r'[。！？!?]')  # 匹配中文和英文的句号、感叹号、问号

    for sent in doc.sents:
        sentence = sent.text.strip()
        # 检查句子长度和当前块长度之和是否超过最大长度
        if len(sentence) + len(current_chunk) > max_length:
            # 如果当前块不为空，保存当前块，并开始一个新块
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 如果句子长度本身超过最大长度，需要进一步分割
            if len(sentence) > max_length:
                # 使用正则表达式在句子分隔符处分割
                sub_sentences = sentence_delimiters.split(sentence)
                sub_chunk = ""
                for sub_sentence in sub_sentences:
                    # 检查分割后的句子是否为空，以避免添加空字符串
                    if sub_sentence:
                        # 如果子句子与分隔符一起的长度小于最大长度，添加分隔符
                        if len(sub_chunk + sub_sentence) + 1 <= max_length:
                            sub_chunk += sub_sentence + "。"  # 假设句子以句号结束
                        else:
                            # 如果子块不为空，保存子块
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = sub_sentence + "。"  # 开始新的子块
                # 检查并添加最后的子块
                if sub_chunk:
                    chunks.append(sub_chunk)
            else:
                # 如果整个句子长度小于最大长度，直接开始新块
                current_chunk = sentence
        else:
            # 如果没有超过最大长度，继续累积句子到当前块
            current_chunk += sentence

    # 保存最后的块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_text_by_page(pdf_path):
    """从PDF文件中提取每一页的文本"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return [page.page_content for page in pages]


def visualize_text_splitter_results(pdf_path):
    texts = extract_text_by_page(pdf_path)
    all_chunks = []

    # 创建一个表格实例
    table = PrettyTable()
    table.field_names = ["编号", "长度", "前十个字", "后十个字"]

    for text in texts:
        chunks = spacy_chinese_text_splitter(text)
        all_chunks.extend(chunks)

    # 填充表格数据
    for i, chunk in enumerate(all_chunks):
        start_text = chunk[:10].replace('\n', '')  # 取前10个字符，并去除换行符
        end_text = chunk[-10:].replace('\n', '')  # 取后10个字符，并去除换行符
        length = len(chunk)
        table.add_row([i + 1, length, start_text, end_text])

    # 打印表格
    print(table)


if __name__ == '__main__':
    pdf_path = r"E:\工作\BkmGPT语料\同济\admin\public\智慧校园答疑\同济大学网站群申请步骤.pdf"
    visualize_text_splitter_results(pdf_path)