import logging
import subprocess
import requests
import spacy
from langchain.document_loaders import PyPDFLoader
import jieba.posseg as pseg
import mimetypes
import magic
from fpdf import FPDF
import re
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.KMC_config import Config
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上两级目录
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

class FileManager:
    def __init__(self, config):
        self.file_storage_path = config.file_storage_path
        self.spacy_model = spacy.load(config.spacy_model)
        self.backend_notify_api = config.external_api_backend_notify
        self.stopwords = config.stopwords
        # 使用全局logger实例
        self.config = config

        # 使用全局logger实例
        self.logger = self.config.logger

    def notify_backend(self, file_id, result, failure_reason=None):
        """通知后端接口处理结果"""
        url = self.backend_notify_api  # 更新后的后端接口URL
        headers = {'token': file_id}
        payload = {
            'id': file_id,
            'result': result
        }
        if failure_reason:
            payload['failureReason'] = failure_reason

        response = requests.post(url, json=payload, headers=headers)
        print("后端接口返回状态码：%s", response.status_code)
        return response.status_code

    def download_pdf(self, download_url, file_id):
        try:
            headers = {'token': file_id}
            response = requests.get(download_url, headers=headers)

            if response.status_code == 200:
                file_path = os.path.join(self.file_storage_path, f"{file_id}")
                with open(file_path, "wb") as pdf_file:
                    pdf_file.write(response.content)
                self.logger.info(f"FILE downloaded successfully: {file_path}")
                return file_path
            else:
                self.logger.error(f"下载PDF失败: {download_url}, Status code: {response.status_code}")
                return None

        except Exception as e:
            self.logger.exception(f"下载文件失败: {download_url}, Error: {e}")
            return None

    def convert_txt_to_pdf(self, txt_path, output_path):
        self.logger.info(f"开始将文本文件转换为PDF: {txt_path}")
        pdf = FPDF()
        pdf.add_page()
        # 拼接路径，确保字体文件路径正确
        font_path = os.path.join(grandparent_dir, 'model', 'font', '微软雅黑.ttf')
        # 添加字体，确保字体文件在正确的路径
        pdf.add_font('YaHei', '', font_path, uni=True)
        pdf.set_font('YaHei', '', 12)  # 设置字体为微软雅黑

        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                for line in file:
                    pdf.cell(200, 10, txt=line, ln=True)

            output_pdf_path = os.path.join(output_path, os.path.splitext(os.path.basename(txt_path))[0] + '.pdf')
            pdf.output(output_pdf_path)
            self.logger.info(f"文本文件转换为PDF完成: {output_pdf_path}")
            return output_pdf_path
        except Exception as e:
            self.logger.error(f"转换文本文件到PDF失败: {e}")
            return None

    def convert_docx_to_pdf(self, input_path, output_path):
        base_name = os.path.basename(input_path)
        converted_pdf_path = os.path.join(output_path, os.path.splitext(base_name)[0] + '.pdf')
        try:
            # 确保输出目录存在
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            cmd = [
                '/usr/bin/libreoffice',
                '--headless',  # 无界面模式
                '--convert-to', 'pdf:writer_pdf_Export',
                '--outdir', output_path,
                input_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.logger.info(f"转换完成：{converted_pdf_path}")
                return converted_pdf_path
            else:
                self.logger.error(f"转换失败，LibreOffice 错误：{result.stderr.decode()}")
                converted_pdf_path = None
                return converted_pdf_path

        except Exception as e:
            self.logger.error(f"转换失败：{e}")
            return None

    def process_pdf_file(self, pdf_path, file_name, output_path=None):
        # 获取文件扩展名
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()

        # 检查并设置输出路径
        if output_path is None:
            output_path = os.path.dirname(pdf_path)

        # 支持的文件扩展名检查
        supported_extensions = ['.docx', '.doc', '.ppt', '.pptx', '.xls', '.xlsx', '.txt']
        # 如果文件是.docx格式，首先转换为.pdf
        if file_extension in supported_extensions:
            if file_extension in ['.docx', '.doc', '.ppt', '.pptx', '.xls', '.xlsx', '.txt']:
                self.logger.info(f"检测到{file_extension}文件，转换为PDF...")
                converted_pdf_path = self.convert_docx_to_pdf(pdf_path, output_path)
                if converted_pdf_path is not None:
                    pdf_path = converted_pdf_path
                else:
                    self.logger.error(f"{file_extension}转PDF失败，无法继续处理。")
                    return None
            elif file_extension == '.txt':
                converted_pdf_path = self.convert_txt_to_pdf(pdf_path, output_path)
                if converted_pdf_path is not None:
                    pdf_path = converted_pdf_path
                else:
                    self.logger.error(f"{file_extension}转PDF失败，无法继续处理。")
                    return None

        elif file_extension != '.pdf':
            self.logger.error("不支持的文件格式")
            return None

        """处理单个PDF文件"""
        doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            self.logger.info("加载 %d 页，文件源：%s", len(pages), pdf_path)
            self.logger.info("按页处理开始。。。")
            stopwords = self.load_stopwords()
            document_texts = set()
            filtered_texts = set()  # 存储处理后的文本
            for page_index, page in enumerate(pages, start=1):
                page_content = page.page_content.replace('\n', '').replace('\r', '')
                split_text = self.spacy_chinese_text_splitter(page_content, max_length=450)
                for text in split_text:
                    # self.logger.info(text)
                    if text not in document_texts:
                        document_texts.add(text)
                        words = pseg.cut(text)
                        filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                        if filtered_text not in filtered_texts:
                            filtered_texts.add(filtered_text)
                            doc_list.append({'page': page_index, 'text': filtered_text, 'original_text': text})
        except Exception as e:
            self.logger.error("PDF文件 %s 处理过程出现错误: %s", pdf_path, str(e))
        finally:
            try:
                os.remove(pdf_path)
                self.logger.info("成功删除文件：%s", pdf_path)
            except OSError as e:
                self.logger.error("删除文件失败：%s，错误：%s", pdf_path, str(e))
        return doc_list

    def process_canvas_file(self, file_path, output_path=None):
        # 使用 magic 库检测文件类型
        file_mime_type = magic.from_file(file_path, mime=True)

        # 检查并设置输出路径
        if output_path is None:
            output_path = os.path.dirname(file_path)

        # 根据MIME类型决定如何处理文件
        if file_mime_type == 'application/pdf':
            pdf_path = file_path  # PDF 文件不需要转换
        elif file_mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            self.logger.info("检测到DOCX文件，转换为PDF...")
            pdf_path = self.convert_docx_to_pdf(file_path, output_path)
            if pdf_path is None:
                self.logger.error("DOCX转PDF失败，无法继续处理。")
                return
        else:
            self.logger.error(f"不支持的文件类型: {file_mime_type}")
            return

        """处理PDF文件的流程"""
        doc_list = []  # 初始化空列表以确保在出错时也能返回列表类型
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            self.logger.info("加载 %d 页，文件源：%s", len(pages), pdf_path)
            self.logger.info("按页处理开始。。。")
            stopwords = self.load_stopwords()
            document_texts = set()
            filtered_texts = set()  # 存储处理后的文本
            for page_index, page in enumerate(pages, start=1):
                split_text = self.spacy_chinese_text_splitter(page.page_content, max_length=2000)
                for text in split_text:
                    if text not in document_texts:
                        document_texts.add(text)
                        words = pseg.cut(text)
                        filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                        if filtered_text not in filtered_texts:
                            filtered_texts.add(filtered_text)
                            doc_list.append({'page': page_index, 'text': filtered_text, 'original_text': text})
        except Exception as e:
            self.logger.error("PDF文件 %s 处理过程出现错误: %s", pdf_path, str(e))
            print(f"PDF文件{pdf_path}处理过程出现错误: {str(e)}")
        finally:
            try:
                os.remove(pdf_path)
                self.logger.info("成功删除文件：%s", pdf_path)
            except OSError as e:
                self.logger.error("删除文件失败：%s，错误：%s", pdf_path, str(e))

        return doc_list

    def filter_text(self, text):
        stopwords = self.load_stopwords()
        words = pseg.cut(text)
        return ''.join(word for word, flag in words if word not in stopwords)

    def load_stopwords(self):
        with open(self.stopwords, "r", encoding="utf-8") as file:
            return set(line.strip() for line in file)

    # 调用小模型来进行文本分割
    def spacy_chinese_text_splitter(self, text, max_length=400):
        # 加载spaCy中文模型
        nlp = self.spacy_model
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

    def process_Canvas_file(self, pdf_path, file_name, output_path=None):
        # 获取文件扩展名
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()

        # 检查并设置输出路径
        if output_path is None:
            output_path = os.path.dirname(pdf_path)

        # 支持的文件扩展名检查
        supported_extensions = ['.docx', '.doc', '.ppt', '.pptx', '.xls', '.xlsx', '.txt']
        if file_extension in supported_extensions:
            try:
                self.logger.info(f"检测到 {file_extension} 文件，开始转换为 PDF...")
                if file_extension in ['.docx', '.doc', '.ppt', '.pptx', '.xls', '.xlsx']:
                    pdf_path = self.convert_docx_to_pdf(pdf_path, output_path)
                elif file_extension == '.txt':
                    pdf_path = self.convert_txt_to_pdf(pdf_path, output_path)
            except Exception as e:
                self.logger.error(f"{file_extension} 转 PDF 失败: {e}")
                return ""

        elif file_extension != '.pdf':
            self.logger.error(f"不支持的文件格式: {file_extension}")
            return ""

        # PDF 文件处理
        full_text = ""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            full_text = "\n".join(doc.page_content for doc in documents)
            self.logger.info("成功提取全文内容。")
        except Exception as e:
            self.logger.error(f"PDF 文件处理错误: {e}")
        finally:
            try:
                os.remove(pdf_path)
                self.logger.info(f"成功删除文件: {pdf_path}")
            except OSError as e:
                self.logger.error(f"删除文件失败: {e}")
        return full_text

# # 加载配置
# config = Config()
# config.load_config('config\\config.json')
#
# # 创建 FileManager 实例
# file_manager = FileManager(config)
#
# # 指定要处理的PDF文件路径
# pdf_file_path = r"E:\工作\BkmGPT语料\同济\admin\public\临时\canvas教程.pdf"
#
# # 调用 process_pdf_file 方法处理文件
# doc_list = file_manager.process_pdf_file(pdf_file_path)
#
# # 打印处理结果
# for doc in doc_list:
#     print(doc)
