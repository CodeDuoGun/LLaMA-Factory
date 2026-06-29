#!/usr/bin/env python3
"""Search Wu Weiping prescriptions by disease, syndrome, and symptom vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medical.config import config
from medical.rag.es.es_processor import (
    DEFAULT_INDEX_NAME,
    _load_elasticsearch,
    embed_prescription_query,
    retrieve_wuweiping_prescription_by_vector,
)


def make_client() -> Any:
    es_class = _load_elasticsearch()
    host = getattr(config, "ES_HOST", "localhost")
    port = getattr(config, "ES_PORT", "9200")
    user = getattr(config, "ES_USER", "")
    password = getattr(config, "ES_AUTH", "")
    auth = (user, password) if user else None
    return es_class(f"http://{host}:{port}", basic_auth=auth, request_timeout=120)


def hit_to_result(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source", {})
    return {
        "id": hit.get("_id", ""),
        "score": hit.get("_score", 0),
        "病名": source.get("diagnosis_result", ""),
        "证型": source.get("syndrome_result", ""),
        "症状": source.get("clinical_symptoms_text", ""),
        "处方结果": source.get("prescription_text", ""),
        "模板方结果": source.get("template_prescription_text", ""),
        "模板方名称": source.get("template_name", ""),
    }


def search(
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    clinical_symptoms: str,
    top_k: int,
    embedding_model_type: str,
) -> dict[str, Any]:
    client = make_client()
    query_vector = embed_prescription_query(clinical_symptoms, embedding_model_type=embedding_model_type)
    result = retrieve_wuweiping_prescription_by_vector(
        client,
        index_name=index_name,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        query_vector=[float(item) for item in query_vector],
        top_k=top_k,
    )
    return {
        "same_disease_syndrome_count": result["same_disease_syndrome_count"],
        "results": [hit_to_result(hit) for hit in result["case_hits"]],
        "template_candidates": result["template_candidates"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Wu Weiping prescriptions.")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--disease", required=True, help="病名，对应 diagnosis_sickness")
    parser.add_argument("--syndrome", required=True, help="证型名，对应 diagnosis_disease")
    parser.add_argument("--symptoms", required=True, help="患者症状文本")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="doubao", choices=["doubao", "bge", "bge_large_zh", "bge_code", "bgem3"])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = search(
        index_name=args.index,
        diagnosis_result=args.disease,
        syndrome_result=args.syndrome,
        clinical_symptoms=args.symptoms,
        top_k=args.top_k,
        embedding_model_type=args.embedding_model,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
