from __future__ import annotations

import argparse
import json
import time
import uuid
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import aiohttp

pd = None
Elasticsearch = None
helpers = None
BulkIndexError = Exception
Index = None
Q = None
Search = None
get_embedding = get_doubao_embedding = get_bgem3_embedding = get_bge_code_embedding = None

try:
    from medical.utils.log import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

def perf_counter_timer(func):
    return func


def normalize_vector(vector):
    return vector

try:
    from medical.config import config
except Exception:
    class _Config:
        ES_HOST = "localhost"
        ES_PORT = "9200"
        ES_USER = "elastic"
        ES_AUTH = ""
        embedding_model = ""
        env_version = "alpha"
        USE_BGE_CODE = False

    config = _Config()

try:
    from medical.constants import EMBED_TYPE
except Exception:
    class EMBED_TYPE:
        DOUBAO = "doubao"
        BGECODE = "bge_code"


DEFAULT_INDEX_NAME = "alpha_medical_prescription_rag_v1"
DEFAULT_ANALYSIS_FILE = Path("medical/data/wuweiping_prescription_strategy_analysis.json")
DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")
DEFAULT_WUWEIPING_JSONL_FILE = Path("medical/processed_data/wuweiping_es_prescription.jsonl")


def _require_dependency(dependency: Any, name: str) -> Any:
    if dependency is None:
        raise RuntimeError(f"Missing optional dependency required for this operation: {name}")
    return dependency


def _load_pandas() -> Any:
    global pd
    if pd is None:
        import pandas as pandas_module

        pd = pandas_module
    return pd


def _load_elasticsearch() -> Any:
    global Elasticsearch, helpers, BulkIndexError
    if Elasticsearch is None or helpers is None:
        from elasticsearch import Elasticsearch as ElasticsearchClass
        from elasticsearch import helpers as helpers_module
        from elasticsearch.helpers import BulkIndexError as BulkIndexErrorClass

        Elasticsearch = ElasticsearchClass
        helpers = helpers_module
        BulkIndexError = BulkIndexErrorClass
    return Elasticsearch


def _load_elasticsearch_dsl() -> None:
    global Index, Q, Search
    if Search is None or Q is None or Index is None:
        from elasticsearch_dsl import Index as IndexClass
        from elasticsearch_dsl import Q as QFunc
        from elasticsearch_dsl import Search as SearchClass

        Index = IndexClass
        Q = QFunc
        Search = SearchClass


def _load_embedding_tools() -> None:
    global get_embedding, get_doubao_embedding, get_bgem3_embedding, get_bge_code_embedding
    if get_doubao_embedding is None:
        from medical.model.embedding.tool import (
            get_bge_code_embedding as bge_code_embedding_func,
            get_bgem3_embedding as bgem3_embedding_func,
            get_doubao_embedding as doubao_embedding_func,
            get_embedding as embedding_func,
        )

        get_embedding = embedding_func
        get_doubao_embedding = doubao_embedding_func
        get_bgem3_embedding = bgem3_embedding_func
        get_bge_code_embedding = bge_code_embedding_func


def build_index_body(dims: int, text_analyzer: str = "standard") -> dict[str, Any]:
    return {
        "settings": {
            "index": {"number_of_shards": 1, "number_of_replicas": 0},
            "analysis": {
                "normalizer": {"lowercase_normalizer": {"type": "custom", "filter": ["lowercase"]}}
            },
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "chunk_id": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "source": {"type": "keyword"},
                "diagnosis_result": {"type": "keyword"},
                "diagnosis_result_text": {"type": "text", "analyzer": text_analyzer},
                "syndrome_result": {"type": "keyword"},
                "syndrome_result_text": {"type": "text", "analyzer": text_analyzer},
                "sex": {"type": "keyword"},
                "age": {"type": "integer"},
                "age_bucket": {"type": "keyword"},
                "template_id": {"type": "keyword"},
                "template_name": {"type": "keyword"},
                "template_name_text": {"type": "text", "analyzer": text_analyzer},
                "clinical_symptoms_text": {"type": "text", "analyzer": text_analyzer},
                "prescription_text": {"type": "text", "analyzer": text_analyzer},
                "template_prescription_text": {"type": "text", "analyzer": text_analyzer},
                "text": {"type": "text", "analyzer": text_analyzer},
                "metadata": {"type": "object", "enabled": True},
                "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            },
        },
    }


def age_bucket(age: Any) -> str:
    try:
        value = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if value < 0:
        return "unknown"
    if value < 10:
        return "0-9"
    if value >= 80:
        return "80+"
    start = value // 10 * 10
    return f"{start}-{start + 9}"


def compact_drug_text(drugs: Iterable[dict[str, Any]], dose_key: str = "dose") -> str:
    parts = []
    for drug in drugs or []:
        name = drug.get("drug_name")
        if not name:
            continue
        dose = drug.get(dose_key)
        unit = drug.get("unit") or ""
        parts.append(f"{name}{dose or ''}{unit}")
    return "、".join(parts)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _stable_doc_part(value: Any) -> str:
    text = _clean_text(value)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text).strip("_") or "unknown"


def _format_template_prescription(template_detail: dict[str, Any] | None, template_name: str = "") -> str:
    template_detail = template_detail or {}
    drugs = template_detail.get("drugs") or []
    drug_text = compact_drug_text(drugs, dose_key="quantity")
    if template_name and drug_text:
        return f"模板方：{template_name}\n组成：{drug_text}"
    if template_name:
        return f"模板方：{template_name}"
    if drug_text:
        return f"模板方组成：{drug_text}"
    return ""


def _normalize_dose(value: Any) -> Any:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def _parse_prescription_drug_detail(part: str, usage_type: str = "") -> dict[str, Any] | None:
    text = re.sub(r"\s+", "", _clean_text(part))
    text = text.strip(" 。,，、；;")
    if not text:
        return None
    match = re.match(r"^(?P<name>.+?)(?P<dose>\d+(?:\.\d+)?)(?P<unit>g|克|mg|ml|片|粒|袋|支|瓶|盒|丸)?$", text, flags=re.IGNORECASE)
    if match:
        return {
            "drug_name": match.group("name"),
            "dose": _normalize_dose(match.group("dose")),
            "unit": match.group("unit") or "",
            "usage_type": usage_type,
        }
    return {"drug_name": text, "dose": "", "unit": "", "usage_type": usage_type}


def build_wuweiping_embedding_text(row: dict[str, Any]) -> str:
    """
    向量检索中，只存储诊断结果和临床表现，不存储处方和模板方内容。
    """
    diagnosis_result = _clean_text(row.get("diagnosis_sickness"))
    syndrome_result = _clean_text(row.get("diagnosis_disease"))
    clinical_symptoms = _clean_text(row.get("clinical_symptoms_text") or row.get("clinical_symptoms"))
    return "\n".join(
        part
        for part in [
            f"{diagnosis_result}",
            f"{syndrome_result}",
            f"患者症状：{clinical_symptoms}",
        ]
        if part and not part.endswith("：")
    )


def build_wuweiping_prescription_doc(row: dict[str, Any], embedding: list[float] | None = None) -> dict[str, Any]:
    diagnosis_result = _clean_text(row.get("diagnosis_sickness"))
    syndrome_result = _clean_text(row.get("diagnosis_disease"))
    clinical_symptoms = _clean_text(row.get("clinical_symptoms_text") or row.get("clinical_symptoms"))
    prescriptions = [_clean_text(item) for item in row.get("ps") or [] if _clean_text(item)]
    prescription_text = "\n".join(prescriptions)
    template_detail = row.get("matched_template_detail") or {}
    template_name = _clean_text(row.get("matched_template_name"))
    template_prescription_text = _format_template_prescription(template_detail, template_name)

    doc_id = f"case_prescription:{_stable_doc_part(row.get('id'))}:{_stable_doc_part(row.get('order_sn'))}"
    text = build_wuweiping_embedding_text(row)
    source = {
        "chunk_type": "case_prescription",
        "source": "wuweiping_es_prescription",
        "diagnosis_result": diagnosis_result,
        "diagnosis_result_text": diagnosis_result,
        "syndrome_result": syndrome_result,
        "syndrome_result_text": syndrome_result,
        "sex": _clean_text(row.get("patient_sex")),
        "age": int(row.get("patient_age")) if str(row.get("patient_age") or "").isdigit() else None,
        "age_bucket": age_bucket(row.get("patient_age")),
        "template_id": _clean_text(row.get("matched_template_id")),
        "template_name": template_name,
        "template_name_text": template_name,
        "clinical_symptoms_text": clinical_symptoms,
        "prescription_text": prescription_text,
        "template_prescription_text": template_prescription_text,
        "text": text,
        "metadata": {
            "record": row,
            "inquiry_id": row.get("id"),
            "order_sn": row.get("order_sn"),
            "patient_id": row.get("patient_id"),
            "matched_template_score": row.get("matched_template_score"),
        },
    }
    if source["age"] is None:
        source.pop("age")
    if embedding is not None:
        source["embedding"] = embedding
    return _source_doc(doc_id, source)


def _source_doc(doc_id: str, source: dict[str, Any]) -> dict[str, Any]:
    source["chunk_id"] = doc_id
    return {"_id": doc_id, "_source": source}


def build_case_prescription_doc(row: dict[str, Any], embedding: list[float] | None = None) -> dict[str, Any]:
    clinical_info = row.get("clinical_info") or {}
    diagnosis_result = row.get("diagnosis_result") or row.get("diagnosis_name") or ""
    syndrome_result = row.get("diagnosis_syndrome") or ""
    drugs = row.get("drugs") or []
    clinical_symptoms = clinical_info.get("doctor_symptom_summary") or clinical_info.get("patient_symptoms") or ""
    doc_id = f"case_prescription:{row.get('inquiry_id')}:{row.get('prescription_order_id')}"
    text_parts = [
        f"诊断结果：{diagnosis_result}",
        f"证候结果：{syndrome_result}",
        f"患者症状：{clinical_symptoms}",
        f"现病史：{clinical_info.get('present_history') or ''}",
        f"检查所见：{clinical_info.get('exam_findings') or ''}",
        f"最相似模板方：{row.get('matched_template_name') or ''}",
        f"最终处方：{compact_drug_text(drugs)}",
        f"加药：{'、'.join(row.get('added_drugs') or [])}",
        f"减药：{'、'.join(row.get('removed_drugs') or [])}",
    ]
    source = {
        "chunk_type": "case_prescription",
        "source": "wuweiping_record",
        "diagnosis_result": diagnosis_result,
        "diagnosis_result_text": diagnosis_result,
        "syndrome_result": syndrome_result,
        "syndrome_result_text": syndrome_result,
        "template_id": str(row.get("matched_template_id") or ""),
        "template_name": row.get("matched_template_name") or "",
        "template_name_text": row.get("matched_template_name") or "",
        "symptom_tags": row.get("symptom_tags") or [],
        "clinical_symptoms_text": clinical_symptoms,
        "exam_tags": row.get("exam_tags") or [],
        "drug_names": [drug.get("drug_name") for drug in drugs if drug.get("drug_name")],
        "added_drugs": row.get("added_drugs") or [],
        "removed_drugs": row.get("removed_drugs") or [],
        "risk_tags": row.get("risk_tags") or [],
        "text": "\n".join(part for part in text_parts if part),
        "metadata": {
            "inquiry_id": row.get("inquiry_id"),
            "prescription_order_id": row.get("prescription_order_id"),
            "match_ratio": row.get("match_ratio"),
            "clinical_info": clinical_info,
            "drugs": drugs,
            "common_drugs": row.get("common_drugs") or [],
            "added_drugs": row.get("added_drugs") or [],
            "removed_drugs": row.get("removed_drugs") or [],
            "dose_changes": row.get("dose_changes") or [],
        },
    }
    if embedding is not None:
        source["embedding"] = embedding
    return _source_doc(doc_id, source)


def build_template_doc(template: dict[str, Any], embedding: list[float] | None = None) -> dict[str, Any]:
    template_id = template.get("template_id")
    template_name = template.get("template_name") or ""
    drugs = template.get("drugs") or []
    source = {
        "chunk_type": "template_prescription",
        "source": "wuweiping_template",
        "template_id": str(template_id or ""),
        "template_name": template_name,
        "template_name_text": template_name,
        "text": f"模板方：{template_name}\n组成：{compact_drug_text(drugs, dose_key='quantity')}",
        "metadata": {"template": template},
    }
    if embedding is not None:
        source["embedding"] = embedding
    return _source_doc(f"template_prescription:{template_id}", source)


def build_disease_strategy_doc(summary: dict[str, Any], embedding: list[float] | None = None) -> dict[str, Any]:
    diagnosis_result = summary.get("diagnosis_name") or ""
    top_templates = summary.get("top_templates") or []
    core_drugs = summary.get("core_drugs") or []
    source = {
        "chunk_type": "disease_syndrome_strategy",
        "source": "wuweiping_analysis",
        "diagnosis_result": diagnosis_result,
        "diagnosis_result_text": diagnosis_result,
        "template_name": "",
        "template_name_text": "、".join(item.get("template_name", "") for item in top_templates),
        "drug_names": [item.get("drug_name") for item in core_drugs if item.get("drug_name")],
        "text": "\n".join(
            [
                f"诊断结果：{diagnosis_result}",
                f"病历数：{summary.get('record_count')}，处方数：{summary.get('prescription_count')}",
                "常用模板方：" + "、".join(item.get("template_name", "") for item in top_templates[:10]),
                "核心药物：" + "、".join(item.get("drug_name", "") for item in core_drugs[:20]),
                "常见加药：" + "、".join(item.get("drug_name", "") for item in (summary.get("top_added_drugs") or [])[:20]),
                "常见减药：" + "、".join(item.get("drug_name", "") for item in (summary.get("top_removed_drugs") or [])[:20]),
            ]
        ),
        "metadata": summary,
    }
    if embedding is not None:
        source["embedding"] = embedding
    return _source_doc(f"disease_strategy:{diagnosis_result or 'unknown'}", source)


def _es_error_type(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("type") or "")
        if isinstance(error, str):
            return error
    return ""


def _es_status(exc: Exception) -> int | None:
    meta = getattr(exc, "meta", None)
    status = getattr(meta, "status", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def create_index(client: Any, index_name: str, dims: int, text_analyzer: str = "standard", recreate: bool = False) -> None:
    if recreate:
        try:
            client.indices.delete(index=index_name)
        except Exception as exc:
            if _es_status(exc) != 404 and _es_error_type(exc) != "index_not_found_exception":
                raise
    try:
        client.indices.create(index=index_name, body=build_index_body(dims, text_analyzer=text_analyzer))
    except Exception as exc:
        if not recreate and _es_error_type(exc) == "resource_already_exists_exception":
            return
        raise


def upsert_doc(client: Any, index_name: str, doc: dict[str, Any]) -> Any:
    return client.index(index=index_name, id=doc["_id"], document=doc["_source"])


def get_doc(client: Any, index_name: str, doc_id: str) -> Any:
    return client.get(index=index_name, id=doc_id)


def update_doc(client: Any, index_name: str, doc_id: str, partial: dict[str, Any]) -> Any:
    return client.update(index=index_name, id=doc_id, doc=partial)


def delete_doc(client: Any, index_name: str, doc_id: str) -> Any:
    return client.delete(index=index_name, id=doc_id)


def bulk_upsert(client: Any, index_name: str, docs: Iterable[dict[str, Any]]) -> Any:
    bulk_helper = _require_dependency(helpers, "elasticsearch.helpers")
    actions = [{"_op_type": "index", "_index": index_name, "_id": doc["_id"], "_source": doc["_source"]} for doc in docs]
    return bulk_helper.bulk(client, actions)


def structured_filter(
    diagnosis_result: str | None = None,
    syndrome_result: str | None = None,
    sex: str | None = None,
    age_bucket_value: str | None = None,
    chunk_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    filters = []
    if diagnosis_result is not None:
        filters.append({"term": {"diagnosis_result": diagnosis_result}})
    if syndrome_result is not None:
        filters.append({"term": {"syndrome_result": syndrome_result}})
    if sex is not None:
        filters.append({"term": {"sex": sex}})
    if age_bucket_value is not None:
        filters.append({"term": {"age_bucket": age_bucket_value}})
    if chunk_types:
        filters.append({"terms": {"chunk_type": chunk_types}})
    return filters


def bm25_query(
    query_text: str,
    diagnosis_result: str | None = None,
    syndrome_result: str | None = None,
    chunk_types: list[str] | None = None,
    size: int = 20,
) -> dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": structured_filter(diagnosis_result, syndrome_result, chunk_types=chunk_types),
                "should": [
                    {"match": {"text": {"query": query_text, "boost": 3}}},
                    {"match": {"diagnosis_result_text": {"query": query_text, "boost": 2}}},
                    {"match": {"syndrome_result_text": {"query": query_text, "boost": 2}}},
                    {"match": {"template_name_text": {"query": query_text, "boost": 1.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }


def vector_query(
    query_vector: list[float],
    diagnosis_result: str | None = None,
    syndrome_result: str | None = None,
    chunk_types: list[str] | None = None,
    k: int = 20,
    num_candidates: int = 100,
) -> dict[str, Any]:
    filters = structured_filter(diagnosis_result, syndrome_result, chunk_types=chunk_types)
    knn: dict[str, Any] = {"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": num_candidates}
    if filters:
        knn["filter"] = filters
    return {"knn": knn, "size": k}


def same_disease_syndrome_vector_query(
    query_vector: list[float],
    diagnosis_result: str,
    syndrome_result: str,
    k: int = 5,
    num_candidates: int = 100,
) -> dict[str, Any]:
    filters = structured_filter(diagnosis_result, syndrome_result)
    filters.append({"term": {"chunk_type": "case_prescription"}})
    return {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(num_candidates, k),
            "filter": filters,
        },
        "size": k,
    }


def bm25_search(client: Any, index_name: str, query_text: str, **kwargs: Any) -> list[dict[str, Any]]:
    response = client.search(index=index_name, body=bm25_query(query_text, **kwargs))
    return response.get("hits", {}).get("hits", [])


def same_disease_syndrome_case_query(
    diagnosis_result: str,
    syndrome_result: str,
    clinical_symptoms: str,
    size: int = 10,
) -> dict[str, Any]:
    filters = structured_filter(diagnosis_result, syndrome_result)
    filters.append({"term": {"chunk_type": "case_prescription"}})
    should = [
        {"match": {"clinical_symptoms_text": {"query": clinical_symptoms, "boost": 4}}},
        {"match": {"text": {"query": clinical_symptoms, "boost": 2}}},
        {"match": {"symptom_tags": {"query": clinical_symptoms, "boost": 1.5}}},
        {"match": {"exam_tags": {"query": clinical_symptoms, "boost": 1}}},
    ]
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": filters,
                "should": should,
                "minimum_should_match": 1 if clinical_symptoms else 0,
            }
        },
    }


def search_same_disease_syndrome_cases(
    client: Any,
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    clinical_symptoms: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    response = client.search(
        index=index_name,
        body=same_disease_syndrome_case_query(diagnosis_result, syndrome_result, clinical_symptoms, size=top_k),
    )
    return response.get("hits", {}).get("hits", [])


def count_same_disease_syndrome_cases(
    client: Any,
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
) -> int:
    response = client.count(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "filter": structured_filter(diagnosis_result, syndrome_result, chunk_types=["case_prescription"])
                }
            }
        },
    )
    return int(response.get("count", 0))


def vector_search_same_disease_syndrome_cases(
    client: Any,
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    query_vector: list[float],
    top_k: int = 5,
    num_candidates: int = 100,
) -> list[dict[str, Any]]:
    response = client.search(
        index=index_name,
        body=same_disease_syndrome_vector_query(
            query_vector=query_vector,
            diagnosis_result=diagnosis_result,
            syndrome_result=syndrome_result,
            k=top_k,
            num_candidates=num_candidates,
        ),
    )
    return response.get("hits", {}).get("hits", [])


def aggregate_template_candidates(case_hits: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for hit in case_hits:
        source = hit.get("_source", {})
        template_id = str(source.get("template_id") or source.get("template_name") or "")
        template_name = source.get("template_name") or template_id
        if not template_id and not template_name:
            continue
        item = templates.setdefault(
            template_id,
            {
                "template_id": template_id,
                "template_name": template_name,
                "case_count": 0,
                "score": 0.0,
                "drug_names": [],
                "case_ids": [],
            },
        )
        item["case_count"] += 1
        item["score"] += float(hit.get("_score", hit.get("rerank_score", hit.get("rrf_score", 0))) or 0)
        item["case_ids"].append(hit.get("_id", ""))
        for drug_name in source.get("drug_names") or []:
            if drug_name and drug_name not in item["drug_names"]:
                item["drug_names"].append(drug_name)

    ranked = sorted(templates.values(), key=lambda item: (item["case_count"], item["score"]), reverse=True)
    for item in ranked:
        item["score"] = round(item["score"], 6)
    return ranked[:limit]


def retrieve_prescription_by_disease_syndrome_symptoms(
    client: Any,
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    clinical_symptoms: str,
    top_k: int = 10,
    template_limit: int = 5,
) -> dict[str, Any]:
    case_hits = search_same_disease_syndrome_cases(
        client,
        index_name,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        clinical_symptoms=clinical_symptoms,
        top_k=top_k,
    )
    template_candidates = aggregate_template_candidates(case_hits, limit=template_limit)
    return {"case_hits": case_hits, "template_candidates": template_candidates}


def retrieve_wuweiping_prescription_by_vector(
    client: Any,
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    query_vector: list[float],
    top_k: int = 5,
    num_candidates: int = 100,
    template_limit: int = 5,
) -> dict[str, Any]:
    same_count = count_same_disease_syndrome_cases(client, index_name, diagnosis_result, syndrome_result)
    case_hits = vector_search_same_disease_syndrome_cases(
        client,
        index_name,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        query_vector=query_vector,
        top_k=top_k,
        num_candidates=max(num_candidates, same_count, top_k),
    )
    template_candidates = aggregate_template_candidates(case_hits, limit=template_limit)
    return {"same_disease_syndrome_count": same_count, "case_hits": case_hits, "template_candidates": template_candidates}


def embed_prescription_query(text: str, embedding_model_type: str = "doubao", embed_model: Any = None) -> list[float]:
    _load_embedding_tools()
    if embedding_model_type in {"bge", "bge_large_zh", "beg_large_zh"}:
        return get_embedding(text, embed_model)
    if embedding_model_type == "bge_code":
        return get_bge_code_embedding(text, embed_model)
    if embedding_model_type == "bgem3":
        return get_bgem3_embedding(embed_model, text)
    return get_doubao_embedding(text)


def vector_search(client: Any, index_name: str, query_vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
    response = client.search(index=index_name, body=vector_query(query_vector, **kwargs))
    return response.get("hits", {}).get("hits", [])


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    rank_constant: int = 60,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict[str, Any]] = {}
    for list_index, ranked_list in enumerate(ranked_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        for rank, hit in enumerate(ranked_list, start=1):
            doc_id = hit["_id"]
            scores[doc_id] += weight / (rank_constant + rank)
            docs.setdefault(doc_id, hit)

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        hit = dict(docs[doc_id])
        hit["rrf_score"] = score
        fused.append(hit)
    return fused[:limit] if limit else fused


def rerank_with_metadata(
    hits: list[dict[str, Any]],
    diagnosis_result: str = "",
    syndrome_result: str = "",
    symptom_tags: list[str] | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    symptom_tags = symptom_tags or []
    reranked = []
    for hit in hits:
        source = hit.get("_source", {})
        score = hit.get("rrf_score", hit.get("_score", 0.0))
        if diagnosis_result and source.get("diagnosis_result") == diagnosis_result:
            score += 0.3
        if syndrome_result and source.get("syndrome_result") == syndrome_result:
            score += 0.25
        source_symptoms = set(source.get("symptom_tags") or [])
        if symptom_tags and source_symptoms:
            score += 0.2 * (len(set(symptom_tags) & source_symptoms) / len(set(symptom_tags) | source_symptoms))
        item = dict(hit)
        item["rerank_score"] = score
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def hybrid_search(
    client: Any,
    index_name: str,
    query_text: str,
    query_vector: list[float],
    diagnosis_result: str | None = None,
    syndrome_result: str | None = None,
    chunk_types: list[str] | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    bm25_hits = bm25_search(
        client,
        index_name,
        query_text,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        chunk_types=chunk_types,
        size=top_k,
    )
    vector_hits = vector_search(
        client,
        index_name,
        query_vector,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        chunk_types=chunk_types,
        k=top_k,
    )
    fused = reciprocal_rank_fusion([bm25_hits, vector_hits], weights=[1.0, 1.2], limit=top_k * 2)
    return rerank_with_metadata(fused, diagnosis_result or "", syndrome_result or "", top_k=top_k)


def _drug_name_set(value: Any) -> set[str]:
    if not value:
        return set()
    names = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("drug_name") or item.get("name")
        else:
            name = item
        if name:
            names.append(str(name))
    return set(names)


def _source_from_hit(hit: dict[str, Any] | None) -> dict[str, Any]:
    return (hit or {}).get("_source", {}) if hit else {}


def _drugs_from_case(case: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    for key in keys:
        value = case.get(key)
        if value:
            return value
    metadata = case.get("metadata") or {}
    for key in keys:
        value = metadata.get(key)
        if value:
            return value
    return []


def _drugs_from_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = source.get("metadata") or {}
    drugs = metadata.get("drugs") or source.get("drugs") or []
    if drugs:
        return drugs
    return [{"drug_name": name} for name in source.get("drug_names") or []]


def _dose_map(drugs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    mapping = {}
    for drug in drugs or []:
        name = drug.get("drug_name") or drug.get("name")
        if name and drug.get("dose") is not None:
            mapping[str(name)] = drug.get("dose")
    return mapping


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _f1(expected: set[str], predicted: set[str]) -> float | None:
    if not expected and not predicted:
        return None
    if not expected or not predicted:
        return 0.0
    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0


def evaluate_retrieval(
    gold_cases: list[dict[str, Any]],
    search_fn: Callable[[dict[str, Any], int], list[dict[str, Any]]],
    top_k: int = 10,
) -> dict[str, Any]:
    hit_count = 0
    reciprocal_rank_sum = 0.0
    template_top1_count = 0
    template_top3_count = 0
    final_drug_jaccards = []
    core_drug_recalls = []
    added_drug_f1s = []
    removed_drug_f1s = []
    dose_hits = []
    dose_range_hits = []
    hallucinated_drug_count = 0
    high_risk_matches = []
    details = []

    for case in gold_cases:
        hits = search_fn(case, top_k)
        top_source = _source_from_hit(hits[0] if hits else None)
        relevant_ids = set(case.get("relevant_ids") or [])
        rank = 0
        for index, hit in enumerate(hits[:top_k], start=1):
            if hit.get("_id") in relevant_ids:
                rank = index
                break
        if rank:
            hit_count += 1
            reciprocal_rank_sum += 1 / rank

        expected_template = case.get("expected_template_name")
        top_template = top_source.get("template_name") if hits else None
        top3_templates = [hit.get("_source", {}).get("template_name") for hit in hits[:3]]
        if expected_template and top_template == expected_template:
            template_top1_count += 1
        if expected_template and expected_template in top3_templates:
            template_top3_count += 1

        expected_drugs = _drug_name_set(_drugs_from_case(case, ("expected_drugs", "drugs", "gold_drugs")))
        predicted_drugs = _drug_name_set(_drugs_from_source(top_source))
        if expected_drugs or predicted_drugs:
            final_drug_jaccards.append(len(expected_drugs & predicted_drugs) / len(expected_drugs | predicted_drugs))
            hallucinated_drug_count += len(predicted_drugs - expected_drugs)

        expected_core = _drug_name_set(case.get("expected_core_drugs") or case.get("core_drugs") or [])
        if expected_core:
            core_drug_recalls.append(len(expected_core & predicted_drugs) / len(expected_core))

        expected_added = _drug_name_set(case.get("expected_added_drugs") or case.get("added_drugs") or [])
        predicted_added = _drug_name_set(top_source.get("added_drugs") or (top_source.get("metadata") or {}).get("added_drugs") or [])
        added_f1 = _f1(expected_added, predicted_added)
        if added_f1 is not None:
            added_drug_f1s.append(added_f1)

        expected_removed = _drug_name_set(case.get("expected_removed_drugs") or case.get("removed_drugs") or [])
        predicted_removed = _drug_name_set(top_source.get("removed_drugs") or (top_source.get("metadata") or {}).get("removed_drugs") or [])
        removed_f1 = _f1(expected_removed, predicted_removed)
        if removed_f1 is not None:
            removed_drug_f1s.append(removed_f1)

        expected_doses = _dose_map(_drugs_from_case(case, ("expected_drugs", "drugs", "gold_drugs")))
        predicted_doses = _dose_map(_drugs_from_source(top_source))
        for drug_name, expected_dose in expected_doses.items():
            dose_hits.append(1.0 if predicted_doses.get(drug_name) == expected_dose else 0.0)

        dose_ranges = case.get("dose_ranges") or {}
        for drug_name, dose_range in dose_ranges.items():
            predicted_dose = _as_float(predicted_doses.get(drug_name))
            low = _as_float(dose_range.get("min") if isinstance(dose_range, dict) else None)
            high = _as_float(dose_range.get("max") if isinstance(dose_range, dict) else None)
            if predicted_dose is not None and low is not None and high is not None:
                dose_range_hits.append(1.0 if low <= predicted_dose <= high else 0.0)
            else:
                dose_range_hits.append(0.0)

        expected_risks = set(case.get("expected_risk_tags") or case.get("risk_tags") or [])
        predicted_risks = set(top_source.get("risk_tags") or (top_source.get("metadata") or {}).get("risk_tags") or [])
        high_risk_matches.append(1.0 if expected_risks == predicted_risks else 0.0)

        details.append(
            {
                "query_id": case.get("query_id"),
                "hit_rank": rank or None,
                "expected_template_name": expected_template,
                "top_template_name": top_template,
                "top3_template_names": top3_templates,
            }
        )

    total = len(gold_cases)
    return {
        "case_count": total,
        "hit_rate_at_k": round(hit_count / total, 4) if total else 0,
        "mrr_at_k": round(reciprocal_rank_sum / total, 4) if total else 0,
        "template_top1_accuracy": round(template_top1_count / total, 4) if total else 0,
        "template_top1_hit_rate": round(template_top1_count / total, 4) if total else 0,
        "template_top3_recall": round(template_top3_count / total, 4) if total else 0,
        "final_drug_jaccard": _avg(final_drug_jaccards),
        "core_drug_recall": _avg(core_drug_recalls),
        "added_drug_f1": _avg(added_drug_f1s),
        "removed_drug_f1": _avg(removed_drug_f1s),
        "dose_hit_rate": _avg(dose_hits),
        "dose_in_history_range_rate": _avg(dose_range_hits),
        "hallucinated_drug_count": hallucinated_drug_count,
        "high_risk_rule_accuracy": _avg(high_risk_matches),
        "details": details,
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
            yield data


def load_templates(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return data.get("templates", [])
    return data


def iter_docs_from_analysis(
    analysis_path: Path = DEFAULT_ANALYSIS_FILE,
    template_path: Path = DEFAULT_TEMPLATE_FILE,
) -> Iterable[dict[str, Any]]:
    analysis = load_json(analysis_path)
    for template in load_templates(template_path):
        yield build_template_doc(template)
    for summary in analysis.get("disease_summary", []):
        yield build_disease_strategy_doc(summary)
    for row in analysis.get("template_match_details", []):
        yield build_case_prescription_doc(row)


def iter_wuweiping_prescription_docs(path: Path = DEFAULT_WUWEIPING_JSONL_FILE) -> Iterable[dict[str, Any]]:
    for row in iter_jsonl(path):
        yield build_wuweiping_prescription_doc(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query Elasticsearch prescription RAG index.")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--dims", type=int, default=1024)
    parser.add_argument("--analyzer", default="standard", help="Use ik_max_word if your ES cluster has IK installed.")
    parser.add_argument("--print-mapping", action="store_true")
    return parser.parse_args()


class ElasticsearchHandler:
    def __init__(self):#, model_name="BAAI/bge-large-zh-v1.5"):
        """
        :param index_name: str, 索引名称
        """
        es_class = _load_elasticsearch()
        _load_elasticsearch_dsl()
        host = getattr(config, "ES_HOST", "localhost")
        port = getattr(config, "ES_PORT", "9200")
        user = getattr(config, "ES_USER", "elastic")
        password = getattr(config, "ES_AUTH", "")
        self.es = es_class(f'http://{host}:{port}', basic_auth=(user, password), request_timeout=80)
        # self.es = Elasticsearch(f'http://localhost:9200', basic_auth=('elastic', 'gungun'))
        self.embedding_model = None
        if getattr(config, "embedding_model", "") == "bge":
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    def create_index(self, index_name, mappings=None, need_del: bool=False):
        """
        创建索引
        """
        if need_del:
            try:
                self.delete_index(index_name)
                # self.es.indices.delete(index="es_doctor_info")
            except Exception:
                pass

        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mappings)
            logger.info(f"索引 {index_name} 创建成功")
        else:
            logger.info(f"索引 {index_name} 已存在")

    def create_prescription_rag_index(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        dims: int = 1024,
        text_analyzer: str = "standard",
        recreate: bool = False,
    ) -> None:
        create_index(self.es, index_name, dims=dims, text_analyzer=text_analyzer, recreate=recreate)

    def upsert_prescription_doc(self, index_name: str, doc: dict[str, Any]) -> Any:
        return upsert_doc(self.es, index_name, doc)

    def bulk_upsert_prescription_docs(self, index_name: str, docs: Iterable[dict[str, Any]]) -> Any:
        return bulk_upsert(self.es, index_name, docs)

    def bm25_prescription_search(self, index_name: str, query_text: str, **kwargs: Any) -> list[dict[str, Any]]:
        return bm25_search(self.es, index_name, query_text, **kwargs)

    def vector_prescription_search(self, index_name: str, query_vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
        return vector_search(self.es, index_name, query_vector, **kwargs)

    def hybrid_prescription_search(
        self,
        index_name: str,
        query_text: str,
        query_vector: list[float],
        diagnosis_result: str | None = None,
        syndrome_result: str | None = None,
        chunk_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        return hybrid_search(
            self.es,
            index_name,
            query_text,
            query_vector,
            diagnosis_result=diagnosis_result,
            syndrome_result=syndrome_result,
            chunk_types=chunk_types,
            top_k=top_k,
        )

    def retrieve_prescription_by_disease_syndrome_symptoms(
        self,
        index_name: str,
        diagnosis_result: str,
        syndrome_result: str,
        clinical_symptoms: str,
        top_k: int = 10,
        template_limit: int = 5,
    ) -> dict[str, Any]:
        return retrieve_prescription_by_disease_syndrome_symptoms(
            self.es,
            index_name=index_name,
            diagnosis_result=diagnosis_result,
            syndrome_result=syndrome_result,
            clinical_symptoms=clinical_symptoms,
            top_k=top_k,
            template_limit=template_limit,
        )

    def retrieve_wuweiping_prescription_by_vector(
        self,
        index_name: str,
        diagnosis_result: str,
        syndrome_result: str,
        query_vector: list[float],
        top_k: int = 5,
        num_candidates: int = 100,
        template_limit: int = 5,
    ) -> dict[str, Any]:
        return retrieve_wuweiping_prescription_by_vector(
            self.es,
            index_name=index_name,
            diagnosis_result=diagnosis_result,
            syndrome_result=syndrome_result,
            query_vector=query_vector,
            top_k=top_k,
            num_candidates=num_candidates,
            template_limit=template_limit,
        )


    def create_index_with_mapping(self, index_name, embedding_model:str="doubao"):
        """
        创建包含全文和向量字段的索引映射
        """
        # print(self.es.ping())
        try:
            self.delete_index(index_name)
        except Exception:
            pass
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},  # 存储文本内容，用于全文检索
                    "metadata": {"type": "object"},  # 存储其他元数据，如页码、来源等
                    "embedding": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1024 if embedding_model!="doubao" else 4096,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                        "index": True,
                        "similarity": "cosine"
                    },
                    "bge_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1024,            # 维度数，依据你所用的嵌入模型（这里以 OpenAI 为例）
                        "index": True,
                        "similarity": "cosine"
                    },
                    "bge_code_vector": {
                        "type": "dense_vector",  # 向量字段，用于向量检索
                        "dims": 1536,            
                        "index": True,
                        "similarity": "cosine"
                    }

                }
            }
        }
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mapping)
            logger.debug(f"索引 {index_name} 创建成功")
        else:
            logger.debug(f"索引 {index_name} 已存在")

    def index_documents(self, index_name, documents, embedding_model_type:str="doubao", bge_model=None):
        """
        将切分后的文档块向量化，并存入 Elasticsearch
        """
        for doc in documents:
            # 生成唯一 id
            doc_id = str(uuid.uuid4())
            # 生成向量，调用 embed_query（也可以用 embed_document，效果相似）
            # if bge_model:
            #     bgem3_vector = get_bgem3_embedding(bge_model, doc.page_content)
            # else:
            #     bgem3_vector = [0.0] * 1024
            if embedding_model_type == "bge_code":
                bge_code_vector = get_bge_code_embedding(doc.page_content, bge_model)
            else:
                bge_code_vector = [0.0] * 1536 #  TODO: 需要额外处理 都是0会计算报错，

            vector = get_doubao_embedding(doc.page_content)
            
            # 构建文档结构（包含全文、元数据和向量）
            # logger.debug(doc.metadata)
            # import pdb
            # pdb.set_trace()
            doc_body = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": vector,
                "bge_code_vector":bge_code_vector
            }
            # 入库
            self.es.index(index=index_name, id=doc_id, body=doc_body)
        logger.debug("文档入库完成")
    

    def update_document(self,index_name, doc_id, doc):
        """
        更新索引中的文档
        :param index_name: str, 索引名称
        :param doc_id: str, 文档 ID
        :param doc: dict, 更新的文档内容
        """
        self.es.update(index=index_name, id=doc_id, body={"doc": doc})
        logger.info(f"文档 {doc_id} 更新成功")

    def delete_document(self, index_name, doc_id):
        """
        删除索引中的文档
        :param index_name: str, 索引名称
        :param doc_id: str, 文档 ID
        """
        self.es.delete(index=index_name, id=doc_id)
        logger.info(f"文档 {doc_id} 删除成功")
    
    def search_keyword(self, index_name, query, top_k=5):
        """
        使用关键词检索：通过 match 查询对文档内容进行检索，
        返回与 query 匹配度较高的 top_k 个文档
        """
        es_query = {
            "size": top_k,
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        results = self.es.search(index=index_name, body=es_query)
        # return results
        # TODO： 看下返回结果
        hits = results["hits"]["hits"]
        retrieved_docs = []
        for hit in hits:
            doc = hit["_source"]
            doc_text = f"标题：{doc.get('title', '')}\n内容：{doc.get('content', '')}"
            retrieved_docs.append(doc_text)
        return retrieved_docs
    
    
    @perf_counter_timer 
    def search_by_vector(self, index_name, query, top_k=5,doctor_location:str="", embedding_model_type:str="doubao", embed_model=None):
        """"""
        try:
            if embedding_model_type=="beg_large_zh":
                query_vector = get_embedding(query, embed_model)
            elif embedding_model_type == "bge_code":
                query_vector = get_bge_code_embedding(query, embed_model) 
            else:
                query_vector = get_doubao_embedding(query)
            if index_name==f"{config.env_version}_doctor_info":
                script_query = Q(
                'script_score',
                query=Q('match_all'),
                script={
                    'source': "cosineSimilarity(params.query_vector, 'goodvector') + 1.0",
                    'params': {'query_vector': query_vector}
                    },
                )
            elif index_name==f"{config.env_version}_primary_disease":
                script_query = Q(
                'script_score',
                query=Q('match_all'),
                script={
                    'source': "cosineSimilarity(params.query_vector, 'primary_disease_vector') + 1.0",
                    'params': {'query_vector': query_vector}
                    },
                )
            elif index_name==f"{config.env_version}_doctor":
                # 优先根据医生所属医院检索
                cosine_script = {
                        'source': "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        }
                if doctor_location:
                    script_query = Q(
                    'script_score',
                    query=Q(
                        'bool',
                        filter=[
                            Q('match', 所在区域=doctor_location)  # 替换为您需要的区域
                        ]
                    ),
                    script=cosine_script,
                    )
                else:
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script=cosine_script
                    )
            elif index_name==f"{config.env_version}_qa":
                if embedding_model_type == "doubao":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                elif embedding_model_type == "bgem3":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
            elif index_name==f"{config.env_version}_primary_disease":
                if embedding_model_type == "doubao":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                elif embedding_model_type == "bgem3":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                        },
                    )
                
            else:
                if embedding_model_type=="bge_large_zh":
                    script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'bge_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                    }
                )
                elif embedding_model_type=="bge_code":
                    script_query = Q(
                        'script_score',
                        query=Q('match_all'),
                        script={
                            'source': "cosineSimilarity(params.query_vector, 'bge_code_vector') + 1.0",
                            'params': {'query_vector': query_vector}
                        }
                    )
                else:
                    script_query = Q(
                        'script_score',
                        query=Q('match_all'),
                        script={
                            'source': "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            'params': {'query_vector': query_vector}
                        }
                    )
            # 构建搜索对象并执行查询
            search = Search(using=self.es, index=index_name).query(script_query)
            response = search.execute()["hits"]["hits"]
        except Exception as e:
            logger.error(traceback.format_exc())
            response = None
        return response
    
    @perf_counter_timer 
    def search_by_keyword(self, index: str, 
                          keyword: str,
                          fields: list=["擅长病种（新媒体推病种+擅长）", "姓名","所在区域"], 
                          size: int = 5, 
                          doctor_name:str="", 
                          doctor_location:str="",
                          ):
        """
        :param index: 要查询的索引名
        :param fields: 需要匹配的字段列表，如 ["title", "description"]
        :param keyword: 搜索关键词
        :param size: 返回结果条数
        """
        # 构造 multi_match 查询
        if index == f"{config.env_version}_doctor_info":
            fields = ["擅长病种（新媒体推病种+擅长）"]
        elif index == f"{config.env_version}_doctor":
            fields = ["擅长", "所在区域", "姓名"] 
        elif index == f"{config.env_version}_secondary_disease":
            fields = ["secondary_disease"] 
        elif index == f"{config.env_version}_primary_disease":
            fields = ["primary_disease"] 

        if not doctor_name and not doctor_location:
            if index == f"{config.env_version}_doctor":
                query_body = {
                    "query": {
                        "multi_match": {
                            "query": keyword,
                            # "type": "phrase",
                            "fields": fields,
                            # "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }
            else:
                query_body = {
                    "query": {
                        "multi_match": {
                            "query": keyword,
                            "type": "phrase",
                            "fields": fields,
                            # "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }

        elif doctor_name and doctor_location:
            query_body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "bool": {
                                "must": [
                                    {
                                        "match": {
                                            "姓名": doctor_name  # 示例过滤条件，根据实际需求修改
                                        }
                                    },
                                    {
                                        "match": {
                                            "所在区域": doctor_location,  # 示例过滤条件，根据实际需求修改
                                        }
                                    }]
                            }
                        }
                    }
                },
                "size": size
            }
        elif doctor_location:
            print("location search")
            query_body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "match": {
                                "所在区域": doctor_location,  # 示例过滤条件，根据实际需求修改
                            }
                        }
                    }
                },
                "size": size
            }
        elif doctor_name:
            query_body = {
                "query": {
                    "bool": {
                        "should": {
                            "multi_match": {
                                "query": keyword,
                                "fields": fields,
                                "type": "best_fields",  # 可选: best_fields、most_fields、cross_fields、phrase、phrase_prefix
                                "operator": "or"
                            }
                        },
                        "filter": {
                            "match": {
                                "姓名": doctor_name,
                            }
                        }
                    }
                },
                "size": size
            }

        response = self.es.search(index=index, body=query_body)
        hits = response["hits"]["hits"]
        return hits

    def search_by_keyword_bm25(self, index: str, keyword: str,
                          field: str = "姓名", k: int = 5):
        """
        基于 BM25 的模糊匹配，优先匹配医生姓名
        """
        body = {
            "size": k,
            "query": {
                "match": {
                    field: {
                        "query": keyword,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        res = self.es.search(index=index, body=body)
        if res and res["hits"]["hits"] and keyword!=res["hits"]["hits"][0]["_source"]["姓名"]:
            return []
        return res["hits"]["hits"]

    def search_hybrid(self,
                      index: str,
                      doctor_name: str,
                      condition: str,
                      k: int = 5,
                      alpha: float = 0.3,
                      name_boost: float = 3.0):
        """
        混合检索：对医生姓名做 BM25，对症状描述做向量检索，
        最终得分 = alpha * 向量相似度 + (1-alpha) * BM25 得分
        """
        q_vec = get_doubao_embedding(condition)

        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "姓名": {
                                            "query": doctor_name,
                                            "fuzziness": "AUTO",
                                            "boost": name_boost
                                        }
                                    }
                                },
                                {
                                "multi_match": {
                                    "query": condition,
                                    "fields": ["擅长^1.5"],  # 关键词检索
                                    # "fuzziness": "AUTO"
                                }
                                },
                            ]
                        }
                    },
                    "script": {
                        "source": (
                            "params.alpha * cosineSimilarity(params.q, 'goodvector') "
                            "+ (1 - params.alpha) * _score"
                        ),
                        "params": {
                            "q": q_vec,
                            "alpha": alpha
                        }
                    }
                }
            }
        }
        res = self.es.search(index=index, body=body)
        return res["hits"]["hits"]

    
    def get_history_from_es(self, conversation_id, deviceId:str="", index_name:str="chat_history"):
        # 构造ES查询，假设文档中存有 uid、source 和 deviceId 字段
        s = Search(using=self.es, index=index_name)
        s = s.filter("term", conversation_id=conversation_id).filter("term", deviceId=deviceId)
        response = s.execute()
        
        # 如果查询到数据，则获取第一个文档的 history 字段，否则返回空列表
        if response.hits.total.value > 0:
            doc = response.hits[0]
            historyx = doc.history if hasattr(doc, "history") else []
        else:
            historyx = []
        
        # 保持历史记录数不超过 5 对（即最多 10 个记录）
        while len(historyx) > 5 * 2:
            historyx.pop(0)
            historyx.pop(0)
        
        # 过滤掉 content 为空的记录
        try:
            historyx = [item for item in historyx if len(item.get("content", "")) > 0]
        except Exception:
            pass
        
        return historyx
        
    
    def add_chat_history(self, conversation_id, history,index_name:str="chat_history"):
        """
        存入一组对话历史到 es 
        @param index_name: ES 索引名称
        @param conversation_id: 对话ID
        @param history: 对话历史，格式为 list，例如:
                        [{"role":"user", "content": "ghg"}, {"role":"assistant", "content": "fhghliG"}]
        @return: ES 响应结果
        """
        # 构造文档数据
        # TODO: @txueduo 每次写入删除旧的history
        document = {
            "conversation_id": conversation_id,
            "history": history,
        }
        # 将文档存入指定索引
        response = self.es.index(index=index_name, document=document)
        return response
    
    def semantic_search(self, index_name, query_sentence, top_k:int=5):
        """
        根据一句话的语义进行查询，比如 '帮我找擅长看胃病的医生',"王全胜医生的出诊时间是什么时候啊"
        """
        query = {
            "query": {
                "multi_match": {
                    "query": query_sentence,
                    # 针对多个字段进行查询，可根据数据情况增加或调整字段
                    "fields": [
                        "擅长病种（新媒体推病种+擅长）",
                        "治疗特色",
                        "所在区域",
                        "姓名",
                        "出诊时间",
                        "挂号费",
                        "职称/职务"
                    ],
                    "fuzziness": "AUTO"  # 自动模糊匹配，可捕捉一些拼写或语义上的相似性
                }
            }
        }
        response = self.es.search(index=index_name, body=query)#[:top_k]
        return response["hits"]["hits"]
    
    # TODO: @txueduo 修改下面保证所有格式兼容
    def bulk_excel_insert(self, xlsx_file, index_name:str="alpha_doctor_info",embed_model:str="doubao"):
        pd = _load_pandas()
        _load_embedding_tools()
        xlsx = pd.ExcelFile(xlsx_file, engine='openpyxl')
        # 获取所有工作表名称
        sheet_names = xlsx.sheet_names
        logger.debug(f"所有工作表:{sheet_names[1:]}")
        for sheet in sheet_names[1:]:
            df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=1, engine='openpyxl')
            # 删除所有列均为 NaN 的行
            df = df.dropna(subset=['姓名'])
            df = df.drop(['出生年份', '序号'], axis=1)
            df = df.fillna("")
            # 重命名 某列
            df = df.rename(columns={'所在\n区域': '所在区域'})
            df["职称/职务"] = df['职称/职务'].str.replace('\n', '、', regex=False)
            df["擅长病种（新媒体推病种+擅长）"] = df['擅长病种（新媒体推病种+擅长）'].str.replace(',', '，', regex=False)
            print(df['职称/职务'])
            # 清洗id列
            def clean_number(s):
                # 使用正则表达式提取所有数字
                numbers = re.findall(r'\d+', str(s))
                # 如果没有找到数字，返回 None 或其他默认值
                if not numbers:
                    return -999
                # 将第一个数字转换为整数
                return int(numbers[0])

            df["ID"] = df["ID"].apply(clean_number)
            actions = []
            if sheet in ("肿瘤"):
                df = df.dropna(axis=1, how='all')
                df = df.drop("Unnamed: 23", axis=1)
            # 遍历每一行数据
            for _, row in df.iterrows():
                # 将每行数据转换为字典
                doc = row.to_dict()
                # 对需要转换为数值类型的字段进行转换，防止出现 NaN
                doc["年龄"] = 0 if doc.get("年龄") == '——' else doc.get("年龄", 0)
                doc["挂号费"] = doc.get("挂号费", "")
                doc["处方单价"] = doc.get("处方单价", "")

                # 假设我们利用“擅长病种（新媒体推病种+擅长）”字段来生成向量，也可以选择其他字段或拼接多个字段
                text_for_embedding = doc['姓名'] + doc.get("擅长病种（新媒体推病种+擅长）", "")# + doc['治疗特色']
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    doc["goodvector"] = get_doubao_embedding(text_for_embedding)
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    doc["goodvector"] = get_embedding(embed_model, text_for_embedding)

                action = {
                    "_index": index_name,
                    "_source": doc
                }
                actions.append(action)

            logger.warning(f"sheet_name: {sheet}, actions: {len(actions)}")
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"{sheet} BulkIndexError: {e}")
                for error in e.errors:
                    pass
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk {sheet} for {traceback.format_exc()}")
        logger.debug("XLSX 数据导入完成")

    def bulk_insert_by_file(self, file, index_name:str="alpha_doctor", embed_model:str="doubao", sheet_name:str=""):
        pd = _load_pandas()
        _load_embedding_tools()
        if index_name == f"{config.env_version}_doctor":
            df = pd.read_excel(file, engine='openpyxl')
            # 清洗列表
            actions = []
            for _, row in df.iterrows():
                # 将每行数据转换为字典
                doc = row.to_dict()
                new_doc = {}
                new_doc["所在区域"] = doc["地区"]
                new_doc["序号"] = doc["序号"]
                new_doc["姓名"] = doc["姓名"]
                new_doc["简介"] = doc["简介"]
                new_doc["ID"] = doc["ID"]
                new_doc["擅长"] = doc["擅长"]
                new_doc["执业医院"] = doc["执业医院"]
                new_doc["出诊地点"] = doc["出诊地点"]
                text_for_embedding = doc["姓名"]+ "擅长：" + doc["擅长"]
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    new_doc["goodvector"] = get_doubao_embedding(text_for_embedding)
                    new_doc["vector"] = get_doubao_embedding(new_doc["出诊地点"]+text_for_embedding)
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    new_doc["goodvector"] = get_embedding(embed_model, text_for_embedding)
                action = {
                    "_index": index_name,
                    "_source": new_doc
                }
                actions.append(action)
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"BulkIndexError: {e}")
                for error in e.errors:
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk for {traceback.format_exc()}")
        else:
            xlsx = pd.ExcelFile(file, engine='openpyxl')
            actions = []
            # 获取所有工作表名称
            sheet_names = xlsx.sheet_names
            logger.debug(f"所有工作表:{sheet_names}")
            df = pd.read_excel(xlsx, sheet_name=sheet_name, engine='openpyxl')
            for _, row in df.iterrows():
                doc = row.to_dict()
                if sheet_name == "一级病种": 
                    action = {
                        "_index": index_name,
                        "_source": {"doctors": doc["专家姓名"], "primary_disease": doc["疾病种类"], "primary_disease_vector": get_doubao_embedding(doc["疾病种类"])}
                    }
                else:
                    action = {
                        "_index": index_name,
                        "_source": {"doctors": doc["专家姓名"], "secondary_disease": doc["疾病种类"], "kw_secondary_disease": doc["疾病种类"]}
                    }

                actions.append(action)
            try: 
                helpers.bulk(self.es, actions)
            except BulkIndexError as e:
                logger.error(f"BulkIndexError: {e}")
                for error in e.errors:
                    pass
                    logger.error(f"Failed documents:{error}")
                    
            except Exception:
                logger.error(f"error to bulk for {traceback.format_exc()}")


    def batch_qa(self, qa_data, actions:list, index_name:str="alpha_qa",embed_type:str="doubao", embed_model=None):
        """组成五个一组的qa并行处理"""
        with ThreadPoolExecutor(max_workers=len(qa_data)) as executor:
            # 提交任务到线程池
            if embed_type == "bge_code":
                bge_code_futures = [executor.submit(get_bge_code_embedding, q, embed_model) for q in qa_data.keys()]
            futures = [executor.submit(get_doubao_embedding, q) for q in qa_data.keys()]
            answers = [executor.submit(get_doubao_embedding, qa_data[q]) for q in qa_data.keys()]


            # vector
            results = [future.result() for future in as_completed(futures)]
            answers_results = [future.result() for future in as_completed(answers)]
            if config.USE_BGE_CODE:
                bge_results = [future.result() for future in as_completed(bge_code_futures)]
        for index,q in enumerate(list(qa_data.keys())):
            data_map = {
                "kw_question": q,
                "question": q, 
                "answer": qa_data[q],
                "answer_vector": answers_results[index],
                "q_vector": results[index],
                # "q_bge_code_vector": bge_results[index] if config.USE_BGE_CODE else [0.0] * 1536
            }
            action = {
                "_index": index_name,
                "_source": data_map
            }
            actions.append(action)

        yield actions
            
     

    def bulk_insert_qa(self, qa_data, index_name:str="alpha_qa", embed_type:str="doubao", embed_model=None):
        """
        qa_data: {"q1": a1, "q2": a2...}
        """
        actions = []
        batch_size = 10
        res = {}
        total = len(qa_data)
        try: 
            for q, a in qa_data.items():
                res[q] = a
                if len(res) % batch_size == 0:
                    total  -= len(res) 
                    for actions in self.batch_qa(res, actions, index_name, embed_type, embed_model):
                        logger.debug(f"{total} ready to es_qa")
                        time.sleep(1) # doubao 每分钟请求次数限制导致
                        helpers.bulk(self.es, actions)
                        actions = []
                        res = {}
            if res:
                for q in res.keys():
                    q_vector = get_doubao_embedding(q)

                    if embed_type == "bge_code":
                        q_bge_code_vector = get_bge_code_embedding(q, embed_model)
                        # answer_bge_code_vector = get_bge_code_embedding(qa_data[q], embed_model)

                    data_map = {
                    "kw_question": q,
                    "question": q, 
                    "answer": qa_data[q],
                    "answer_vector": get_doubao_embedding(qa_data[q]),
                    # "answer_bge_code_vector": answer_bge_code_vector,
                    "q_vector": q_vector,
                    # "q_bge_code_vector": q_bge_code_vector if embed_type == "bge_code" else [0.0] * 1536
                    }
                    action = {
                        "_index": index_name,
                        "_source": data_map
                    }
                    actions.append(action)
                helpers.bulk(self.es, actions)
                logger.info(f"last qa actions{len(actions)}")

        except BulkIndexError as e:
            logger.error(f"BulkIndexError: {e}")
            for error in e.errors:
                pass
                logger.error(f"Failed documents:{error}")
        logger.info(f"success insert num of {len(actions)} qa data to es")
    

    def update_qa(self, qa_data, index_name: str = "alpha_qa", embed_type: str = "doubao"):
        for q, a in qa_data.items():
            # 计算向量
            if embed_type == "doubao":
                q_vector = get_doubao_embedding(q)
            
            # 构造文档体
            doc = {
                "kw_question": q,
                "question": q,
                "answer": a,
                "q_vector": q_vector,
                "answer_vector": get_doubao_embedding(a)
            }
            
            # 1. 尝试根据 question 查找已有文档
            query = {
                "query": {
                    "match": {
                        "kw_question": q 
                    }
                }
            }
            resp = self.es.search(index=index_name, body=query, size=1)
            
            if resp["hits"]["total"]["value"] > 0:
                # 已有文档，取出文档 ID 并更新
                doc_id = resp["hits"]["hits"][0]["_id"]
                try:
                    self.es.update(
                        index=index_name,
                        id=doc_id,
                        body={"doc": doc}
                    )
                    logger.info(f"Updated existing QA, index_name:{index_name}, id={doc_id}, question={q!r}")
                except Exception as e:
                    logger.error(f"Failed to update doc id={doc_id}: {e}")
            else:
                # 不存在，则插入新文档
                try:
                    self.es.index(
                        index=index_name,
                        document=doc
                    )
                    logger.info(f"Inserted new QA,  index_name:{index_name}, question={q!r}")
                except Exception as e:
                    logger.error(f"Failed to insert new doc for question={q!r}: {e}")


    def bulk_insert(self, index_name, data):
        """
        批量插入数据到索引
        :param index_name: str, 索引名称
        :param data: list, 数据列表
        """
        actions = []
        for item in data:
            doc = {
                "_index": index_name,
                "_source": {
                    "sheet_name": item['sheet_name'],
                    "text": item['text'],
                    "embedding": item['embedding']
                }
            }
            actions.append(doc)
        if actions:
            bulk(self.es, actions)
            logger.info(f"批量插入 {len(actions)} 条数据成功")

    @perf_counter_timer
    async def search_qa_by_answer_async(self, index_name, answer: str, top_k: int = 5,
                                        embed_model_type: str = "doubao", embed_model=None):
        from aiohttp import BasicAuth
        auth = BasicAuth(config.ES_USER, config.ES_AUTH)
        if embed_model_type == "doubao":
            answer_vector = get_doubao_embedding(answer)  # 注意：必须是同步函数或改成 async 的

            script_score = {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'answer_vector') + 1.0",
                    "params": {"query_vector": answer_vector}
                }
            }

            body = {
                "size": top_k,
                "query": {
                    "script_score": script_score
                }
            }

        url = f"http://{config.ES_HOST}:{config.ES_PORT}/{index_name}/_search"  # 替换为你的 ES 地址

        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    raise Exception(f"Search failed with status {resp.status}")
                res = await resp.json()
                return res["hits"]["hits"]


    def search_qa_by_answer(self, index_name, answer:str, top_k:int=5, embed_model_type:str="doubao", embed_model=None):
        """"""
        if embed_model_type == "doubao":
            answer_vector = get_doubao_embedding(answer)
            script_score= {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'answer_vector') + 1.0",
                        "params": {
                            "query_vector": answer_vector
                        }
                    }
                }
        body = {
            "size": top_k,
            "query": {
                "script_score": script_score
            }
        }
        res = self.es.search(index=index_name, body=body)
        logger.info(f"search_qa_by_answer cost: {res.get('took')} took")
        return res["hits"]["hits"]

    def search_qa_question(
        self,
        index_name,
        query: str,
        top_k: int = 5,
        embed_model_type: str = "doubao",
        embed_model=None,
        search_type: str = "keyword",
    ):
        """
        检索最相关的 topk 个 question。

        search_type 支持:
        - vector / vectorsearch: 向量相似度检索
        - bm25 / keyword: BM25 / 关键词匹配
        - hybrid: BM25 和向量召回后融合排序
        """
        search_type = (search_type or "keyword").lower()

        def build_keyword_body(size):
            return {
                "size": size,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["kw_question^3", "question^2", "answer"],
                    }
                },
            }

        def get_vector_script_score():
            if embed_model_type == "doubao":
                query_vector = get_doubao_embedding(query)
                return {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'q_vector') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
            elif embed_model_type == "bgem3":
                query_vector = get_bgem3_embedding(embed_model, query)
                return {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'q_bgem3_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            elif embed_model_type == "bge_code":
                query_vector = get_bge_code_embedding(query, embed_model)
                return {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'q_bge_code_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }

            # 使用 BGE zh模型进行向量化
            bgezh_model = None
            query_vector = get_embedding(query, bgezh_model)
            return {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'q_bgezh_vector') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }

        def build_vector_body(size):
            return {
                "size": size,
                "query": {
                    "script_score": get_vector_script_score()
                }
            }

        if search_type in {"bm25", "keyword", "keywords", "match"}:
            res = self.es.search(index=index_name, body=build_keyword_body(top_k))
            result = res["hits"]["hits"]
            logger.info(f"search_qa_question bm25 result: {[r['_source']['question'] for r in result]}")
            logger.info(f"search_qa_question bm25 cost: {res.get('took')} took")
            return result

        if search_type in {"hybrid", "mixed", "mix"}:
            candidate_size = max(top_k * 2, top_k)
            keyword_res = self.es.search(index=index_name, body=build_keyword_body(candidate_size))
            vector_res = self.es.search(index=index_name, body=build_vector_body(candidate_size))
            keyword_hits = keyword_res["hits"]["hits"]
            vector_hits = vector_res["hits"]["hits"]
            keyword_max = max((hit.get("_score") or 0 for hit in keyword_hits), default=0) or 1
            vector_max = max((hit.get("_score") or 0 for hit in vector_hits), default=0) or 1
            merged_hits = {}

            for hit in keyword_hits:
                doc_id = hit.get("_id")
                if not doc_id:
                    continue
                merged_hits[doc_id] = hit
                merged_hits[doc_id]["_score"] = (hit.get("_score") or 0) / keyword_max

            for hit in vector_hits:
                doc_id = hit.get("_id")
                if not doc_id:
                    continue
                vector_score = (hit.get("_score") or 0) / vector_max
                if doc_id in merged_hits:
                    merged_hits[doc_id]["_score"] += vector_score
                else:
                    merged_hits[doc_id] = hit
                    merged_hits[doc_id]["_score"] = vector_score

            logger.info(
                f"search_qa_question hybrid cost: keyword={keyword_res.get('took')} vector={vector_res.get('took')} took"
            )
            return sorted(merged_hits.values(), key=lambda item: item.get("_score") or 0, reverse=True)[:top_k]

        if search_type not in {"vector", "vectorsearch", "vector_search"}:
            raise ValueError(f"Unsupported search_type: {search_type}")

        res = self.es.search(index=index_name, body=build_vector_body(top_k))
        logger.info(f"search_qa_question vector cost: {res.get('took')} took")
        return res["hits"]["hits"]


    def delete_qa(self,qa_data:dict, index_name:str="alpha_qa"):
        """
        删除指定问题的QA
        :param qa_data: {q1:a1,q2:a2}
        :param index_name: 索引名称，默认为 "alpha_qa"
        """
        try:
            for question, answer in qa_data.items():
                query = {
                    "query": {
                        "match": {
                            "kw_question": question
                        }
                    }
                }
                resp = self.es.search(index=index_name, body=query, size=1)
                if not len(resp['hits']['hits']) == 1:
                    logger.info(f"Not found qa: {question}")
                    return
                print(f"resP: {resp['hits']['hits'][0]['_source']['kw_question']}")

                response = self.es.delete_by_query(
                    index=index_name,  # 指定索引名称
                    body={
                        "query": {
                            "match": {
                                "kw_question": question
                            }
                        }
                    }
                )

            logger.info(f"Deleted {response['deleted']} documents.")
        except Exception:
            logger.error(f"delete qa error {traceback.format_exc()}")
        
    
    def update_doctor(self, doctor_data:dict, index_name:str="alpha_doctor", embed_model:str="doubao"):
        """
        @param: doctor_data, {2: {"所在区域":"", ...}}
        @param: index_name, es index
        """
        actions = []
        try:
            for doctor_id, doc in doctor_data.items():
                text_for_embedding = doc["姓名"]+ "擅长：" + doc["擅长"]
                if embed_model == "doubao":
                    # 使用 doubao 模型进行向量化
                    doc["goodvector"] = [float(v) for v in get_doubao_embedding(text_for_embedding)]
                    doc["vector"] = [float(v) for v in get_doubao_embedding(doc["出诊地点"]+text_for_embedding)]
                else:
                    # 使用其他模型进行向量化
                    embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                    doc["goodvector"] = get_embedding(embed_model, text_for_embedding)
                self.es.index(index=index_name, document=doc)
            logger.info(f"doctor data {doctor_data.keys()} update to index {index_name} success")
        except Exception:
            logger.error(f"doctor_data {doctor_data.keys()} update to index {index_name} err for {traceback.format_exc()}")
    

    def delete_doctor(self, doctor_ids:List[str], index_name:str="alpha_doctor"):
        """删除指定id的医生，默认医生id唯一标识医生"""
        for id in doctor_ids:
            try:
                query = {
                    "query": {
                        "match": {
                            "ID": id
                        }
                    }
                }
                resp = self.es.search(index=index_name, body=query, size=1)
                assert len(resp['hits']['hits']) == 1
                print(f"resP: {resp['hits']['hits'][0]['_source']['姓名']}")

                response = self.es.delete_by_query(
                    index=index_name,  # 指定索引名称
                    body={
                        "query": {
                            "match": {
                                "ID": id
                            }
                        }
                    }
                )

                logger.info(f"Deleted {response['deleted']} doctor.")
            except Exception:
                logger.error(f"delete doctor error {traceback.format_exc()}")

    def update_disease(self, index_name:str,data:list):
        """增量更新病种医生: disease: doctor_names
        """
        for doc in data:
            if "primary_disease" in index_name: 
                doc = {"doctors": doc["专家姓名"], "primary_disease": doc["疾病种类"], "primary_disease_vector": get_doubao_embedding(doc["疾病种类"])}
            else:
                doc= {"doctors": doc["专家姓名"], "secondary_disease": doc["疾病种类"], "kw_secondary_disease": doc["疾病种类"]}
            self.es.index(index=index_name, document=doc)

    def delete_disease(self, index_name: str, data: list):
        """删除指定专家和病种组合的病种医生数据"""
        disease_field = "primary_disease" if "primary_disease" in index_name else "secondary_disease"
        for doc in data:
            try:
                doctor_name = doc["专家姓名"]
                disease_name = doc["疾病种类"]
                response = self.es.delete_by_query(
                    index=index_name,
                    body={
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"doctors": doctor_name}},
                                    {"match": {disease_field: disease_name}},
                                ]
                            }
                        }
                    },
                )
                logger.info(
                    f"Deleted {response['deleted']} disease docs from {index_name}, "
                    f"doctor={doctor_name}, disease={disease_name}"
                )
            except Exception:
                logger.error(f"delete disease error {traceback.format_exc()}")
    
    def get_mapping(self):
        self.drug_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "category": {"type": "keyword"},
                    "indications": {"type": "text"},
                    "contraindications": {"type": "text"},
                    "adverse_effects": {"type": "text"},
                    "interactions": {"type": "text"},
                    "cautions": {"type": "text"},
                    "pregnancy_lactation": {"type": "text"},
                    "references": {"type": "keyword"}
                }
            }
        }

        # 疾病库 mapping
        self.disease_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "synonyms": {"type": "keyword"},
                    "etiology": {"type": "text"},
                    "pathogenesis": {"type": "text"},
                    "clinical_manifestations": {"type": "text"},
                    "diagnosis": {"type": "text"},
                    "differential": {"type": "text"},
                    "treatment_wm": {"type": "text"},
                    "treatment_tcm": {"type": "text"},
                    "red_flags": {"type": "text"},
                    "references": {"type": "keyword"}
                }
            }
}

    def create_indexes(self, indexes: List[Dict[str, Dict[str, Any]]], need_del:bool=False):
        for item in indexes:
            for index_name, mapping in item.items():
                if need_del:
                    self.delete_index(index_name)
                if not self.es.indices.exists(index=index_name):
                    self.es.indices.create(index=index_name, body=mapping)
                    logger.info(f"创建 {index_name} 完成")
                else:
                    logger.info(f"{index_name} 已存在")

    def delete_index(self, index_name):
        if not self.es.indices.exists(index=index_name):
            logger.info(f"{index_name} not found")
            return
        self.es.indices.delete(index=index_name)
        logger.info(f"deleted {index_name}")


    def insert_data(self, doc, index_name):
        self.es.index(index=index_name, document=doc)
    
    def search_fulltext(self):
        pass

    def search_vector(self):
        pass

    def hybrid_search(self):
        """默认使用"""
        pass


    # -------- BM25 / 全文检索（multi_match） --------
    def bm25_search(self,index: str, query: str, size=10):
        fields = {
            "drug_index": ["name^3","indications","contraindications","adverse_effects","interactions","cautions","pregnancy_lactation"],
            "disease_index": ["name^3","synonyms^2","clinical_manifestations","etiology","diagnosis","differential","treatment_wm","treatment_tcm","red_flags"]
        }[index]
        body = {
            "size": size,
            "query": {"multi_match": {"query": query, "fields": fields}}
        }
        res = self.es.search(index=index, body=body)
        return [{"_id":h["_id"],"_score":h["_score"],"_source":h["_source"]} for h in res["hits"]["hits"]]

    # -------- 向量检索（script_score + cosineSimilarity） --------
    def vector_search_script(self, index: str, query: str, field="content_vector", size=10, embed_model_type: str="doubao"):
        qvec = self.get_embedding(embed_model_type, query)
        body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        # doc[field] 与 qvec 都已单位化 => cosineSimilarity ∈ [-1,1]，再 +1 让分数为正
                        "source": "cosineSimilarity(params.qvec, doc[params.field]) + 1.0",
                        "params": {"qvec": qvec, "field": field}
                    }
                }
            }
        }
        res = self.es.search(index=index, body=body)
        return [{"_id":h["_id"],"_score":h["_score"],"_source":h["_source"]} for h in res["hits"]["hits"]]

    #TODO: 名字不好
    def vector_search_selector(self, index: str, query: str, size:int=8, embed_type: str="", model=None):
        try:
            if embed_type == EMBED_TYPE.DOUBAO:
                query_vector = get_doubao_embedding(query)
                script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'doubao_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                    }
                )

            elif embed_type == EMBED_TYPE.BGECODE:
                query_vector = get_bge_code_embedding(query, model)
                script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'bgecode_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                    }
                )

            else:
                query_vector = get_bgem3_embedding(model, query)
                script_query = Q(
                    'script_score',
                    query=Q('match_all'),
                    script={
                        'source': "cosineSimilarity(params.query_vector, 'bgem3_vector') + 1.0",
                        'params': {'query_vector': query_vector}
                    }
                )
            body = {
                "size": size,
                "query": script_query
            }
            search = Search(using=self.es, index=index).query(script_query)
            res = search.execute()["hits"]['hits']
            # res = self.es.search(index=index, body=body)
            print(res[0], type(res[0]))
            # logger.info(f"vector seach cost: {res.get('took')} took")
        except Exception as e:
            logger.debug(traceback.format_exc(e))
        return res
        
        

    def hybrid_retrieve_rerank(self, index: str, query: str, top_k=10, first_stage_size=100):
        """
        混合检索 + cross-encoder rerank
        index: "drug_index" / "disease_index"
        query: 查询语句
        top_k: 返回数量
        first_stage_size: 初检候选数量
        """
        # 1️⃣ 初检索
        bm25_results = self.bm25_search(index, query, size=first_stage_size)
        vec_results = self.vector_search_script(index, query, size=first_stage_size)

        # 2️⃣ 合并候选集 (去重)
        candidate_dict = {}
        for hit in bm25_results + vec_results:
            candidate_dict[hit["_id"]] = hit["_source"]

        candidates = list(candidate_dict.values())
        if len(candidates)==0:
            return []

        # 3️⃣ 构造 reranker 输入: [(query, doc_text)]
        doc_texts = []
        for doc in candidates:
            content = ""
            if index=="drug_index":
                content = "；".join(filter(None, [
                    doc.get("name",""), doc.get("indications",""), doc.get("contraindications",""),
                    doc.get("adverse_effects",""), doc.get("interactions",""), doc.get("cautions",""),
                    doc.get("pregnancy_lactation","")
                ]))
            elif index=="disease_index":
                content = "；".join(filter(None, [
                    doc.get("name",""), doc.get("etiology",""), doc.get("clinical_manifestations",""),
                    doc.get("diagnosis",""), doc.get("differential",""), doc.get("treatment_wm",""),
                    doc.get("treatment_tcm",""), doc.get("red_flags","")
                ]))
            doc_texts.append(content)

        rerank_inputs = [(query, d) for d in doc_texts]

        # 4️⃣ rerank 得分
        scores = self.bge_reranker.predict(rerank_inputs)

        # 5️⃣ 按得分排序
        final_hits = [{"_source":c, "_score":float(s)} for c,s in zip(candidates, scores)]
        final_hits.sort(key=lambda x: x["_score"], reverse=True)

        return final_hits[:top_k]

def main() -> None:
    args = parse_args()
    if args.print_mapping:
        print(json.dumps(build_index_body(args.dims, args.analyzer), ensure_ascii=False, indent=2))
        return
    print("This module provides reusable ES RAG helpers. Use --print-mapping to inspect the index body.")


if __name__ == "__main__":
    main()
else:
    class _LazyElasticsearchHandler:
        _instance: ElasticsearchHandler | None = None

        def _get(self) -> ElasticsearchHandler:
            if self._instance is None:
                self._instance = ElasticsearchHandler()
            return self._instance

        def __getattr__(self, name: str) -> Any:
            return getattr(self._get(), name)

    es_handler = _LazyElasticsearchHandler()
