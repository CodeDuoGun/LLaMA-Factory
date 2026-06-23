import requests
import time
from app.utils.log import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from medical.utils.tool import perf_counter_timer, normalize_vector
from typing import List
from langchain.embeddings.base import Embeddings
import requests  # 假设你的接口是 http 请求

class DoubaoEmbeddingsV2(Embeddings):
    def __init__(self, api_url: str, api_key: str = None, model:str="doubao-embedding-large-text-240915", max_workers:int=5):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers  # 最大并发数

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """单次批量请求"""
        json_data = {
            "model": self.model,
            "input": texts
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        with requests.post(self.api_url, headers=headers, json=json_data, timeout=60) as resp:
            if resp.status_code == 200:
                return [item["embedding"] for item in resp.json()["data"]]
            logger.error(f"Doubao embedding error: {resp.status_code}, {resp.text}")
            return [[] for _ in texts]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """并发分批处理 embedding"""
        batch_size = 16  # 每次请求的最大 chunk 数，可根据豆包 API 限制调整
        results = []

        # 拆分成小批次
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self._embed_batch, batch): batch for batch in batches}

            for future in as_completed(future_to_batch):
                batch_result = future.result()
                results.extend(batch_result)

        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class DoubaoEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str = None, model:str="doubao-embedding-large-text-240915",batch_size: int = 100):
        self.api_url = api_url
        self.api_key = api_key
        self.model=model
        self.batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """调用豆包接口获取一批向量"""
        json_data = {
            "model": self.model,
            "input": texts
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        with requests.post(self.api_url, headers=headers, json=json_data, timeout=60) as resp:
            if resp.status_code == 200:
                return [item["embedding"] for item in resp.json()["data"]]
            logger.error(f"Doubao embedding {texts} error: {resp.status_code}, {resp.text}")
            return [[] for _ in texts]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """自动分批调用，保证每批不超过 batch_size"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(self._embed_batch(batch))
            time.sleep(0.5)
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

def get_embedding(text, model, dims:int=1024):
    """bge 向量检索 文本向量化"""
    if not text:
        return [0.0] * dims
    embedding = model.encode(text).tolist()
    return embedding

@perf_counter_timer
def get_bgem3_embedding(bgem3model, sentence):
    return [1.0] * 1024 
    embedding = bgem3model.encode([sentence])
    return embedding.tolist()[0]

@perf_counter_timer
def get_bge_code_embedding(query:str, model):
    try:
        sentences = [
            query
        ]
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # embeddings = model.encode(sentences, convert_to_tensor=False, device=device)
        # embeddings = embeddings.cpu().numpy().tolist() 
        embeddings = model.encode(sentences, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"embedding {query} failed for {e}")
        return [1.0] * 1536
    return embeddings.tolist()[0]


def get_multi_process_bgem3_embedding(bgem3model, sentence):
# from sentence_transformers import SentenceTransformer
    with ThreadPoolExecutor(8) as executor:
        future = executor.submit(get_bgem3_embedding, bgem3model,sentence)
        result = future.result()
        return result



@perf_counter_timer
def get_doubao_embedding(text, dims:int=0):
    """文本向量化"""
    json_data = {
        # "model": "doubao-embedding-vision-241215",
        "model": "doubao-embedding-large-text-240915", # dims=4096
        "input": [text],
        }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 71412db7-81ba-41a1-85cd-7f8b7c4c8018'
    }

    with requests.post('https://ark.cn-beijing.volces.com/api/v3/embeddings', headers=headers, json=json_data, timeout=60) as response:
        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]

            # embedding = normalize_vector(embedding)
            return embedding
        logger.error(f"Error: {response.status_code}, {response.text}")
        return []

# res1 = get_doubao_embedding("四惠南区中医门诊如何退号")
# res2 = get_doubao_embedding("四惠南区中医门诊如何退号")
# print(len(res1), res1[:10])
# print(len(res2), res2[:10])