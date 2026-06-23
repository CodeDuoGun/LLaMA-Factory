# ES Prescription RAG Design

## Index

Default index name: `medical_prescription_rag_v1`

Generate mapping:

```bash
python3 medical/rag/es/es_processor.py --dims 1024 --print-mapping
```

If your Elasticsearch cluster has the IK plugin, use:

```bash
python3 medical/rag/es/es_processor.py --dims 1024 --analyzer ik_max_word --print-mapping
```

Core fields:

- `chunk_type`: `template_prescription`, `disease_syndrome_strategy`, `case_prescription`
- `diagnosis_result`: 证名，keyword 精确过滤
- `diagnosis_result_text`: 证名，text BM25
- `syndrome_result`: 证候，keyword 精确过滤
- `syndrome_result_text`: 证候，text BM25
- `template_id`, `template_name`, `template_name_text`
- `sex`, `age`, `age_bucket`
- `clinical_symptoms_text`: 临床症状全文，专门用于同病同证后的症状相似度排序
- `symptom_tags`, `exam_tags`
- `drug_names`, `added_drugs`, `removed_drugs`
- `text`: 拼接后的 BM25 检索文本
- `metadata`: 结构化原始信息
- `embedding`: dense_vector，支持 kNN 向量检索

## Data Shape For Same-Disease Same-Syndrome Retrieval

为了支持“先找同病同证，再按临床症状找最相似病历和处方”，建议把 ES 数据拆成三类文档，并保证 `case_prescription` 文档字段完整：

```json
{
  "chunk_type": "case_prescription",
  "diagnosis_result": "粉刺",
  "diagnosis_result_text": "粉刺",
  "syndrome_result": "湿毒蕴肤证",
  "syndrome_result_text": "湿毒蕴肤证",
  "clinical_symptoms_text": "面部红斑、丘疹、脓疱，舌红苔白厚",
  "symptom_tags": ["红斑", "丘疹", "脓疱"],
  "exam_tags": ["舌红", "苔白厚"],
  "template_id": "template_001",
  "template_name": "痤疮方",
  "template_name_text": "痤疮方",
  "drug_names": ["黄芩片", "丹参"],
  "added_drugs": ["丹参"],
  "removed_drugs": [],
  "text": "诊断结果：粉刺\n证候结果：湿毒蕴肤证\n患者症状：面部红斑、丘疹、脓疱...",
  "metadata": {
    "inquiry_id": 859019,
    "prescription_order_id": 1618930000,
    "clinical_info": {},
    "drugs": [{"drug_name": "黄芩片", "dose": 8, "unit": "g"}]
  }
}
```

关键调整：

- `diagnosis_result`、`syndrome_result` 用 `keyword`，用于精确过滤同病同证。
- `clinical_symptoms_text` 用 `text`，只放症状、体征、检查摘要，不要混入处方和模板名，避免症状排序被药名污染。
- `text` 保留完整拼接文本，作为旧索引兼容和兜底检索字段。
- `template_id/template_name` 必须写入每条病历处方，前端会从 TopK 相似病历聚合最相似模板方。
- `metadata.drugs` 保留剂量和单位，用于剂量命中率、历史范围比例等评测。

## Chunk Types

### `template_prescription`

一条模板方一个文档，适合检索模板组成。

### `disease_syndrome_strategy`

一个诊断结果聚合策略一个文档，包含常用模板方、核心药物、常见加减药。

### `case_prescription`

一张真实内服处方一个文档，适合相似病例检索和 leave-one-out 评测。

## CRUD

```python
from elasticsearch import Elasticsearch
from medical.rag.es.es_processor import (
    create_index,
    build_case_prescription_doc,
    upsert_doc,
    get_doc,
    update_doc,
    delete_doc,
    bulk_upsert,
    iter_docs_from_analysis,
)

client = Elasticsearch("http://localhost:9200")
index_name = "medical_prescription_rag_v1"

create_index(client, index_name, dims=1024, text_analyzer="ik_max_word", recreate=True)

# Single upsert
doc = build_case_prescription_doc(row, embedding=[0.0] * 1024)
upsert_doc(client, index_name, doc)

# Bulk upsert
docs = iter_docs_from_analysis()
bulk_upsert(client, index_name, docs)

# Read / update / delete
get_doc(client, index_name, doc["_id"])
update_doc(client, index_name, doc["_id"], {"risk_tags": ["肝肾功能异常"]})
delete_doc(client, index_name, doc["_id"])
```

## Search

### BM25

```python
from medical.rag.es.es_processor import bm25_search

hits = bm25_search(
    client,
    index_name,
    query_text="面部红斑丘疹脓疱 舌红苔白厚",
    diagnosis_result="粉刺",
    syndrome_result="湿毒蕴肤证",
    chunk_types=["case_prescription"],
    size=20,
)
```

### Vector

```python
from medical.rag.es.es_processor import vector_search

hits = vector_search(
    client,
    index_name,
    query_vector=query_embedding,
    diagnosis_result="粉刺",
    syndrome_result="湿毒蕴肤证",
    chunk_types=["case_prescription"],
    k=20,
)
```

### Hybrid

Hybrid search runs BM25 and vector search, merges by reciprocal rank fusion, then reranks using structured metadata.

```python
from medical.rag.es.es_processor import hybrid_search

hits = hybrid_search(
    client,
    index_name,
    query_text="面部红斑丘疹脓疱 舌红苔白厚",
    query_vector=query_embedding,
    diagnosis_result="粉刺",
    syndrome_result="湿毒蕴肤证",
    chunk_types=["case_prescription", "disease_syndrome_strategy", "template_prescription"],
    top_k=20,
)
```

### Same Disease + Same Syndrome + Similar Symptoms

主推荐流程：先硬过滤同病同证病历，再用临床症状排序，最后从 TopK 病历聚合最相似模板方。

```python
from elasticsearch import Elasticsearch
from medical.rag.es.es_processor import retrieve_prescription_by_disease_syndrome_symptoms

client = Elasticsearch("http://localhost:9200")

result = retrieve_prescription_by_disease_syndrome_symptoms(
    client,
    index_name="medical_prescription_rag_v1",
    diagnosis_result="粉刺",
    syndrome_result="湿毒蕴肤证",
    clinical_symptoms="面部红斑丘疹脓疱，舌红苔白厚",
    top_k=10,
)

case_hits = result["case_hits"]
template_candidates = result["template_candidates"]
```

## Ranking Strategy

Recommended retrieval flow:

1. Hard filter by `chunk_type=case_prescription`.
2. Hard filter by exact `diagnosis_result` and `syndrome_result`.
3. Rank remaining cases by BM25 over `clinical_symptoms_text`, with `text`, `symptom_tags`, `exam_tags` as fallback signals.
4. Return TopK similar historical cases and their `metadata.drugs` prescriptions.
5. Aggregate `template_id/template_name` from TopK cases by case count and ES score to get the most similar template formula.
6. Use sex and age only as soft rerank features if same-disease same-syndrome cases are still too many.

Do not hard filter by sex or age unless enough cases remain. Use them as rerank features.

## Evaluation

Use historical cases as queries and expected similar case/template as labels.

```python
from medical.rag.es.es_processor import evaluate_retrieval, retrieve_prescription_by_disease_syndrome_symptoms

gold_cases = [
    {
        "query_id": "859019:1618930000",
        "clinical_symptoms": "面部红斑丘疹脓疱 舌红苔白厚",
        "diagnosis_result": "粉刺",
        "syndrome_result": "湿毒蕴肤证",
        "relevant_ids": ["case_prescription:859019:1618930000"],
        "expected_template_name": "痤疮",
    }
]

def search_fn(case, top_k):
    result = retrieve_prescription_by_disease_syndrome_symptoms(
        client,
        index_name=index_name,
        diagnosis_result=case["diagnosis_result"],
        syndrome_result=case["syndrome_result"],
        clinical_symptoms=case["clinical_symptoms"],
        top_k=top_k,
    )
    return result["case_hits"]

metrics = evaluate_retrieval(gold_cases, search_fn, top_k=10)
print(metrics)
```

Metrics:

- `hit_rate_at_k`: relevant case appears in Top-K
- `mrr_at_k`: mean reciprocal rank
- `template_top1_accuracy`: top result template equals expected template
