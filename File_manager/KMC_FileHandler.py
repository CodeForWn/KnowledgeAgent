import logging
import subprocess
import requests
import spacy
import pymupdf4llm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pymupdf4llm import LlamaMarkdownReader
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
import json
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
        # Jina API配置
        self.jina_api_url = 'https://api.jina.ai/v1/segment'
        self.jina_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_7b3f3497e3ea4cd1b1c66dd34a12c7bc5x-kQCO8fPMoia_ecyKDrAlIF872"
        }

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

    def download_file(self, url, local_path):
        """
        从指定的 URL 下载文件并保存到 local_path
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果下载失败则抛异常
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path

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

    def jina_text_splitter(self, text, max_length=1000):
        """
        使用Jina API进行文本分段

        参数:
        - text: 需要分段的文本
        - max_length: 最大分段长度，默认1000

        返回:
        - 分段后的文本列表
        """
        try:
            # 如果文本为空或太短，直接返回
            if not text or len(text.strip()) < 10:
                return [text] if text.strip() else []

            data = {
                "content": text,
                "return_tokens": True,  # 返回token信息（虽然我们主要用chunks）
                "return_chunks": True,  # 返回分段结果
                "max_chunk_length": max_length
            }

            response = requests.post(
                self.jina_api_url,
                headers=self.jina_headers,
                data=json.dumps(data)
            )

            if response.status_code == 200:
                result = response.json()

                # 提取chunks字段中的文本
                chunks = result.get('chunks', [])

                if not chunks:
                    self.logger.warning("Jina API返回的chunks为空，回退到spacy方法")
                    return self.spacy_chinese_text_splitter(text, max_length)

                # 过滤空的chunk并清理文本
                text_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, str):
                        # chunk直接是字符串
                        chunk_text = chunk.strip()
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        # chunk是字典，包含text字段
                        chunk_text = chunk['text'].strip()
                    else:
                        # chunk是字符串（根据你的示例，chunks数组直接包含字符串）
                        chunk_text = str(chunk).strip()

                    if chunk_text and len(chunk_text) > 5:  # 过滤太短的片段
                        text_chunks.append(chunk_text)

                self.logger.info(f"Jina API分段成功，原文长度: {len(text)}, 分段数: {len(text_chunks)}")

                # 如果没有有效的分段，回退到spacy方法
                if not text_chunks:
                    self.logger.warning("Jina API分段结果为空，回退到spacy方法")
                    return self.spacy_chinese_text_splitter(text, max_length)

                # 记录一些分段示例用于调试
                if len(text_chunks) > 0:
                    self.logger.info(f"Jina分段示例 - 第一段前50字符: {text_chunks[0][:50]}...")
                    if len(text_chunks) > 1:
                        self.logger.info(f"Jina分段示例 - 第二段前50字符: {text_chunks[1][:50]}...")

                return text_chunks

            else:
                self.logger.error(f"Jina API调用失败，状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                # API调用失败，回退到原来的spacy分段方法
                return self.spacy_chinese_text_splitter(text, max_length)

        except requests.exceptions.Timeout:
            self.logger.error("Jina API请求超时，回退到spacy方法")
            return self.spacy_chinese_text_splitter(text, max_length)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Jina API网络请求异常: {e}")
            return self.spacy_chinese_text_splitter(text, max_length)
        except json.JSONDecodeError as e:
            self.logger.error(f"Jina API响应JSON解析失败: {e}")
            self.logger.error(f"原始响应: {response.text if 'response' in locals() else 'N/A'}")
            return self.spacy_chinese_text_splitter(text, max_length)
        except KeyError as e:
            self.logger.error(f"Jina API响应缺少预期字段: {e}")
            return self.spacy_chinese_text_splitter(text, max_length)
        except Exception as e:
            self.logger.error(f"Jina分段器未知异常: {e}")
            return self.spacy_chinese_text_splitter(text, max_length)


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
            # Step 1: 提取为 Llama 文档
            llama_reader = LlamaMarkdownReader()
            llama_docs = llama_reader.load_data(pdf_path)

            self.logger.info("PDF 加载完成，共 %d 页，文件源：%s", len(llama_docs), pdf_path)

            # Step 2: 合并为 markdown 格式内容
            markdown_text = ""
            for i, doc in enumerate(llama_docs):
                page = doc.metadata.get("page", i + 1)
                markdown_text += f"\n\n## Page {page}\n{doc.text.strip()}\n"

            markdown_text = self.clean_header_bold(markdown_text)

            # Step 3: 使用结构化 Markdown 分段方法
            doc_list = self.process_markdown_with_headers(markdown_text)

            self.logger.info(f"共处理出 {len(doc_list)} 个结构片段，保留标题路径")

        except Exception as e:
            self.logger.error("PDF文件 %s 处理过程出现错误: %s", pdf_path, str(e))

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
                    # self.logger.info(f"分段片段：{current_chunk[:100]}")  # <--- 日志
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
                                    # self.logger.info(f"分段片段：{current_chunk[:100]}")  # <--- 日志
                                sub_chunk = sub_sentence + "。"  # 开始新的子块
                    # 检查并添加最后的子块
                    if sub_chunk:
                        chunks.append(sub_chunk)
                        # self.logger.info(f"分段片段：{current_chunk[:100]}")  # <--- 日志
                else:
                    # 如果整个句子长度小于最大长度，直接开始新块
                    current_chunk = sentence
            else:
                # 如果没有超过最大长度，继续累积句子到当前块
                current_chunk += sentence

        # 保存最后的块
        if current_chunk:
            chunks.append(current_chunk)
            # self.logger.info(f"分段片段：{current_chunk[:100]}")  # <--- 日志
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

    def test_jina_api(self):
        """
        测试Jina API是否正常工作
        """
        test_text = "这是一个测试文本。我们用它来验证Jina API是否正常工作。它包含了中文内容，可以测试中文分段效果。"

        try:
            result = self.jina_text_splitter(test_text, max_length=50)
            self.logger.info(f"Jina API测试成功，分段数量: {len(result)}")
            for i, chunk in enumerate(result):
                self.logger.info(f"分段 {i + 1}: {chunk}")
            return True
        except Exception as e:
            self.logger.error(f"Jina API测试失败: {e}")
            return False

    def process_pdf_with_llm_direct(self, pdf_path, max_length=550):
        """
        使用 PyMuPDF4LLM 直接处理 PDF，返回分段后的 doc_list（不生成 md 文件）
        """
        doc_list = []
        try:
            # Step 1: 提取为 LlamaIndex 格式文档
            llama_reader = LlamaMarkdownReader()
            llama_docs = llama_reader.load_data(pdf_path)

            # Step 2: 合并所有页为纯文本
            full_text = "\n\n".join(doc.text.strip() for doc in llama_docs)

            # Step 3: 停用词 + 去重 + 分段
            stopwords = self.load_stopwords()
            document_texts = set()
            filtered_texts = set()

            split_texts = self.spacy_chinese_text_splitter(full_text, max_length=max_length)

            for i, text in enumerate(split_texts, start=1):
                if text not in document_texts:
                    document_texts.add(text)
                    words = pseg.cut(text)
                    filtered_text = ''.join(word for word, flag in words if word not in stopwords)
                    if filtered_text not in filtered_texts:
                        filtered_texts.add(filtered_text)
                        doc_list.append({'page': i, 'text': filtered_text, 'original_text': text})

            self.logger.info(f"[PyMuPDF4LLM] 共生成 {len(doc_list)} 个片段")

        except Exception as e:
            self.logger.error(f"PDF处理失败: {e}")

        return doc_list

    def process_markdown_with_headers(self, markdown_text):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(markdown_text)

        doc_list = []
        for i, doc in enumerate(md_header_splits, start=1):
            doc_list.append({
                'page': i,
                'text': doc.page_content,
                'original_text': doc.page_content
            })

        return doc_list

    def clean_header_bold(self, md_text):
        import re
        pattern = r"^(#{1,6})\s+\*\*(.*?)\*\*"
        return re.sub(pattern, r"\1 \2", md_text, flags=re.MULTILINE)


# if __name__ == "__main__":
#     llama_reader = pymupdf4llm.LlamaMarkdownReader()
#     llama_docs = llama_reader.load_data("/home/ubuntu/work/kmcGPT/KMC/File_manager/高中 地理 选修1(2)_7-13.pdf", write_images=True)
#     # 输出为 Markdown 文件
#     with open("geo_output.md", "w", encoding="utf-8") as f:
#         for i, doc in enumerate(llama_docs):
#             f.write(f"## Page {i + 1}\n\n")
#             f.write(doc.text)
#             f.write("\n\n")
#     # 加载配置
#     config = Config(env='production')
#     config.load_config()  # 指定配置文件的路径
#
#     # 创建 FileManager 实例
#     file_manager = FileManager(config)
#
#     # # 测试API
#     # if file_manager.test_jina_api():
#     #     print("Jina API工作正常")
#     # else:
#     #     print("Jina API有问题，将使用备用方法")
#
#     # 正常使用
#     doc_list = file_manager.process_pdf_file("/home/ubuntu/work/kmcGPT/KMC/File_manager/建设工程监理规范GBT50319-2013.pdf", "建设工程监理规范.pdf")