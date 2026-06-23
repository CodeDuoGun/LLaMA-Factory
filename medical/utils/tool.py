import time
import base64
import os
import shutil
import pandas as pd
import docx 
import markdown
from medical.utils.log import logger
import uuid
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    MHTMLLoader,
    UnstructuredMarkdownLoader,
)
from concurrent.futures import ProcessPoolExecutor
import PyPDF2
from langchain_core.documents import Document
from selectolax.parser import HTMLParser  # 超快的HTML解析
import requests
import fitz
from pdf2image import convert_from_bytes
from medical.config import config
from medical.llm_v2.doubao import DoubaoAIClient
import pytz

def generate_message_id():
    return str(uuid.uuid4())

def get_time(timestamp: int) -> str:
    """微博时间戳转可读时间"""
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def get_weekday(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    week_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return week_map[date.weekday()]

def get_cur_timezone_time():
    tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(tz)
    weekday = get_weekday(now.strftime("%Y-%m-%d"))
    return now, weekday    

def parse_chinese_number(num_str: str) -> int:
    """简单转换中文热度数值"""
    if not num_str:
        return 0
    num_str = str(num_str)
    if "万" in num_str:
        return int(float(num_str.replace("万", "")) * 1e4)
    elif "亿" in num_str:
        return int(float(num_str.replace("亿", "")) * 1e8)
    else:
        return int(float(num_str))


def perf_counter_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end = time.perf_counter()  # 记录结束时间
        logger.info(f"{func.__name__} cost time: {end - start:.6f} s")
        return result
    return wrapper


def delete_files_with_prefix(folder: str, prefix: str):
    folder_path = Path(folder)
    for f in folder_path.glob(f"{prefix}*"):
        if f.is_file():
            f.unlink()

def remove_special_punctuation(text, special_punctuation='**'):
    try:
        # 使用正则表达式替换这些特殊标点符号为空字符串
        cleaned_text = re.sub(r'\s*\*\*\s*', '', text)
        # cleaned_text = re.sub(special_punctuation, '', text)
    except Exception:
        logger.error(f"normalize failed for {text} cause {traceback.format_exc()}")
        return text
    return cleaned_text


def split_multi_q(questions, qa_data:dict, answer:str=""):
    """把多个q按照换行符分割成一个个q"""
    res = questions.strip("\n").split('\n') if questions else []
    for q in res:
        q = q.strip(" ")
        if not q:
            continue
        qa_data[q] = answer
    # 逐行放入list中
    return qa_data

def gen_qa_by_file(file_path:str="app/data/qa_out.csv"):
    df = pd.read_excel(file_path, engine='openpyxl')
    qa_res = {}
    # i = 0
    for _, row in df.iterrows():
        # i+=1
        doc = row.to_dict()
        # if i > 250:
        #     break
        qa_res = split_multi_q(doc["new_question"], qa_res, doc["答案"])
    # 临时入库脚本
    logger.debug(f"q num: {len(qa_res)}")
    return qa_res


def read_prompt_from_file(file_path):
    """从文件中读取提示词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    except FileNotFoundError:
        logger.error(f"文件 {file_path} 未找到。")
        return
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        return


import numpy as np

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def pd_read_xls(file_path:str, sheet_name:str=""):
    if not sheet_name:
        xls = pd.ExcelFile(file_path)
        print(xls.sheet_names) 
        # TDOO 后面有需要再补全
    else:
        df =pd.read_excel(file_path, sheet_name=sheet_name)
        data_records = df.to_dict(orient="records")
        logger.debug(data_records[:5])
        return data_records
        # 1. 转成 records 格式（list[dict]，每行一个 dict）
        

def is_file_path(path):
    return Path(path).exists()

def image_to_base64(img_url:str):
    image_base = b""
    if is_file_path(img_url):
        with open(img_url, "rb") as img_file:
            image_base = base64.b64encode(img_file.read()).decode("utf-8")
            return image_base
            # image_bases.append(image_base)
    return image_base


def reset_folder(folder_path: str):
    logger.info(f"reset folder {folder_path}")
    # 如果目录存在，直接删除整个目录树
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
    # 重新创建一个空目录
    os.makedirs(folder_path, exist_ok=True)


def fast_pdf_loader(f):
    docs = []
    try:
        llm = DoubaoAIClient(base_url=config.ARK_API_URL, api_key=config.ARK_API_KEY)
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": f, "page": i}))
            else:
                image_bases = pdf_images_to_base64(f)
                message_content = []
                for image_base in image_bases:
                    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base}"}})
                # 添加问题的文本内容
                message_content.append({"type": "text", "text": "帮我提取这些图片中的内容"})
                messages= [
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ]
                response = llm.image_describe(messages, stream=False)
                docs.append(Document(page_content=response.choices[0].message.content, metadata={"source": f, "page": i}))


    except Exception as e:
        logger.error(f"PDF解析失败 {f}: {e}")
    return docs

def fast_docx_loader(f):
    docs = []
    try:
        doc = docx.Document(f)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if text:
            docs.append(Document(page_content=text, metadata={"source": f}))
    except Exception as e:
        logger.error(f"DOCX解析失败 {f}: {e}")
    return docs

def fast_txt_loader(f):
    try:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": f})] if text.strip() else []
    except Exception as e:
        logger.error(f"TXT解析失败 {f}: {e}")
        return []

def fast_csv_loader(f):
    docs = []
    try:
        df = pd.read_csv(f, dtype=str, encoding="utf-8", errors="ignore")
        for i, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=row_text, metadata={"source": f, "row": i}))
    except Exception as e:
        logger.error(f"CSV解析失败 {f}: {e}")
    return docs

def fast_html_loader(f):
    docs = []
    try:
        html = Path(f).read_text(encoding="utf-8", errors="ignore")
        tree = HTMLParser(html)
        text = tree.body.text(separator="\n", strip=True) if tree.body else ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": f}))
    except Exception as e:
        logger.error(f"HTML解析失败 {f}: {e}")
    return docs

def fast_md_loader(f):
    docs = []
    try:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        # 解析 Markdown -> 去掉格式标签
        plain_text = re.sub(r"\s+", " ", markdown.markdown(text))
        if plain_text.strip():
            docs.append(Document(page_content=plain_text, metadata={"source": f}))
    except Exception as e:
        logger.error(f"MD解析失败 {f}: {e}")
    return docs


@perf_counter_timer
def load_all_files(data_path: str):
    all_files = [str(p) for p in Path(data_path).rglob("*") if p.is_file()]

    files_by_ext = defaultdict(list)
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        files_by_ext[ext].append(f)

    docs = []

    loader_map = {
        ".pdf": lambda f: fast_pdf_loader(f),
        ".docx": lambda f: fast_docx_loader(f),
        ".txt": lambda f: fast_txt_loader(f),
        ".csv": lambda f: fast_csv_loader(f),
        ".html": lambda f: fast_html_loader(f),
        ".mhtml": lambda f: fast_html_loader(f),  # MHTML 可以复用 HTML 简化解析
        ".md": lambda f: fast_md_loader(f),
    }

    for ext, file_list in files_by_ext.items():
        if ext in loader_map:
            for f in file_list:
                try:
                    docs.extend(loader_map[ext](f))
                except Exception as e:
                    logger.error(f"加载文件失败 {f}: {e}")

    return docs


@perf_counter_timer
def parse_html_to_documents(data_path: str, prefix:str=""):
    docs = []
    html_files = list(Path(data_path).rglob("*.html"))
    mhtml_files = list(Path(data_path).rglob("*.mhtml"))

    for f in html_files + mhtml_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                html = fp.read()
            tree = HTMLParser(html)
            text = tree.body.text(separator="\n", strip=True) if tree.body else ""

            # 🔑 只保留中文、英文、数字和常见符号
            filtered = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:，。！？；：'\"]+", " ", text)

            if filtered.strip():
                if "if the page does not redirect automatically ..." in filtered:
                    continue
                docs.append(Document(
                    page_content=filtered,
                    metadata={"source": str(f)}
                ))
        except Exception as e:
            logger.error(f"解析失败 {f}: {e}")

    return docs


def load_html_file(f):
    try:
        return UnstructuredHTMLLoader(str(f)).load()
    except Exception as e:
        logger.error(f"加载 HTML 失败 {f}: {e}")
        return []

def load_mhtml_file(f):
    try:
        return MHTMLLoader(str(f)).load()
    except Exception as e:
        logger.error(f"加载 MHTML 失败 {f}: {e}")
        return []

@perf_counter_timer
def load_html_mhtml_parallel(data_path: str, max_workers=2):
    html_files = list(Path(data_path).rglob("*.html"))
    mhtml_files = list(Path(data_path).rglob("*.mhtml"))
    logger.info(f"total html links {len(html_files)+len(mhtml_files)}")

    docs = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for r in exe.map(load_html_file, html_files):
            docs.extend(r)
        # for r in exe.map(load_mhtml_file, mhtml_files):
        #     docs.extend(r)

    return docs


def dump_file(file_name:str, url:str):
    """
    先del
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../app/model/internet
    app_dir = os.path.abspath(os.path.join(current_dir, "..")) 
    save_dir = os.path.join(app_dir, "data/knowledge-base-path")
    output_file = f"{save_dir}/{file_name}"
    
    if not url.startswith("http"):
        return output_file
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"文件{url}已保存到 {output_file}")
        else:
            logger.warning(f"下载失败，状态码：{response.status_code}")
    return output_file


def pdf_images_to_base64(url: str,need_dump: bool = False, dpi:int=200) -> list:
    """
    从 PDF (http 链接) 提取所有图片/扫描件，转成 Base64
    :param url: PDF 的 http 地址
    :return: List[str] -> 每页图片的 Base64 编码
    """
    # url = "https://images.sihuiyiliao.com/backend/images_20250924_014c27e0-7ff6-e787-571f-f16c2df40dde.pdf"
    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        basefile_name = url.split("/")[-1].split(".pdf")[0]
    else:
        with open(url, "rb") as f:
            pdf_bytes = f.read()
        basefile_name = os.path.basename(url).split(".pdf")[0]
    
    if need_dump:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)

        # 3. 保存到本地
        #TODO: rm
        output_dir = f"app/data/pdf_images"
        os.makedirs(output_dir, exist_ok=True)
        filenames = []
        for i, page in enumerate(pages):
            filename = os.path.join(output_dir, f"{basefile_name}_page_{i+1}.png")
            page.save(filename, "PNG")
            filenames.append(filename)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    base64_images = []
    for page_index in range(len(doc)):
        page = doc[page_index]

        pix = page.get_pixmap(dpi=dpi)  # 可以调节 dpi 提高清晰度
        img_bytes = pix.tobytes("png")

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        base64_images.append(img_base64)

    doc.close()
    return base64_images



def extract_json_only(text: str) -> Union[dict, list]:
    """
    从模型输出中提取第一个合法 JSON 对象或数组
    支持对象格式 { ... } 和数组格式 [ ... ]
    优先匹配数组格式，其次匹配对象格式
    """
    try:
        logger.debug(text)
        return json.loads(text)
    except json.JSONDecodeError:
        # 优先尝试匹配数组格式 [ ... ]
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 如果数组格式不匹配，尝试匹配对象格式 { ... }
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            raise ValueError("未找到合法 JSON")

        return json.loads(match.group())


def get_phone_city(phone_num:str):
    url = f"https://cx.shouji.360.cn/phonearea.php?number={phone_num}"
    city = "未知"
    try:
        resp = requests.get(url).json().get("data", {})
        if not resp:
            return city
        result = resp.get("province", "") + resp.get("city", "")
        return result
            
    except Exception:
        logger.error(f"get phone {phone_num} error")
        return city
