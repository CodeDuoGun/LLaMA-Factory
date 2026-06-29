from __future__ import annotations

import json
from typing import Any

from medical.rag.es.es_processor import (
    DEFAULT_INDEX_NAME,
    _load_elasticsearch,
    bm25_search,
    aggregate_template_candidates,
    embed_prescription_query,
    evaluate_retrieval,
    hybrid_search,
    retrieve_prescription_by_disease_syndrome_symptoms,
    retrieve_wuweiping_prescription_by_vector,
)

from medical.config import config


METRIC_LABELS = [
    ("template_top1_hit_rate", "模板 Top-1 命中率"),
    ("template_top3_recall", "模板 Top-3 召回率"),
    ("final_drug_jaccard", "最终药物 Jaccard"),
    ("core_drug_recall", "核心药物召回率"),
    ("added_drug_f1", "加药 F1"),
    ("removed_drug_f1", "减药 F1"),
    ("dose_hit_rate", "剂量命中率"),
    ("dose_in_history_range_rate", "剂量落入历史范围比例"),
    ("hallucinated_drug_count", "幻觉药物数量"),
    ("high_risk_rule_accuracy", "高风险规则触发准确率"),
]


def parse_json_input(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc.msg}") from exc


def format_metric_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    return [[label, metrics.get(key, 0)] for key, label in METRIC_LABELS]


def hits_to_rows(hits: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        rows.append(
            [
                rank,
                hit.get("_id", ""),
                round(float(hit.get("rerank_score", hit.get("rrf_score", hit.get("_score", 0))) or 0), 6),
                source.get("diagnosis_result", ""),
                source.get("syndrome_result", ""),
                source.get("clinical_symptoms_text", ""),
                source.get("prescription_text", "") or (source.get("text") or "")[:240],
                source.get("template_prescription_text", "") or source.get("template_name", ""),
            ]
        )
    return rows


def templates_to_rows(templates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for rank, template in enumerate(templates, start=1):
        rows.append(
            [
                rank,
                template.get("template_name", ""),
                template.get("case_count", 0),
                template.get("score", 0),
                "、".join(template.get("drug_names") or []),
            ]
        )
    return rows


def _parse_chunk_types(value: str) -> list[str] | None:
    chunk_types = [item.strip() for item in (value or "").split(",") if item.strip()]
    return chunk_types or None


def _parse_vector(value: str) -> list[float] | None:
    data = parse_json_input(value)
    if not data:
        return None
    if not isinstance(data, list):
        raise ValueError("query_vector 必须是 JSON 数组。")
    return [float(item) for item in data]


def _make_client() -> Any:
    es_class = _load_elasticsearch()
    host = getattr(config, "ES_HOST", "localhost")
    port = getattr(config, "ES_PORT", "9200")
    user = getattr(config, "ES_USER", "")
    password = getattr(config, "ES_AUTH", "")
    auth = (user, password) if user else None
    return es_class(f"http://{host}:{port}", basic_auth=auth, request_timeout=80)


def _embed_symptoms(clinical_symptoms: str) -> list[float]:
    return [float(item) for item in embed_prescription_query(clinical_symptoms)]


def run_recall_test(
    index_name: str,
    diagnosis_result: str,
    syndrome_result: str,
    clinical_symptoms: str,
    top_k: int,
    gold_cases_json: str,
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], dict[str, Any]]:
    client = _make_client()
    top_k = int(top_k or 10)

    def search_one(case: dict[str, Any], size: int) -> list[dict[str, Any]]:
        case_diagnosis = case.get("diagnosis_result") or diagnosis_result or None
        case_syndrome = case.get("syndrome_result") or syndrome_result or None
        case_symptoms = case.get("clinical_symptoms") or case.get("query_text") or clinical_symptoms
        result = retrieve_wuweiping_prescription_by_vector(
            client,
            index_name=index_name,
            diagnosis_result=case_diagnosis,
            syndrome_result=case_syndrome,
            query_vector=_embed_symptoms(case_symptoms),
            top_k=size,
        )
        return result["case_hits"]

    gold_cases = parse_json_input(gold_cases_json)
    if isinstance(gold_cases, list) and gold_cases:
        metrics = evaluate_retrieval(gold_cases, search_one, top_k=top_k)
        first_hits = search_one(gold_cases[0], top_k)
        templates = aggregate_template_candidates(first_hits)
        return (
            format_metric_rows(metrics),
            hits_to_rows(first_hits),
            templates_to_rows(templates),
            {"metrics": metrics, "details": metrics.get("details", []), "template_candidates": templates},
        )

    result = retrieve_wuweiping_prescription_by_vector(
        client,
        index_name=index_name,
        diagnosis_result=diagnosis_result,
        syndrome_result=syndrome_result,
        query_vector=_embed_symptoms(clinical_symptoms),
        top_k=top_k,
    )
    return (
        format_metric_rows({}),
        hits_to_rows(result["case_hits"]),
        templates_to_rows(result["template_candidates"]),
        result,
    )


def create_prescription_rag_tab() -> dict[str, Any]:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is required to create the prescription RAG UI.") from exc

    elem_dict = {}
    with gr.Row():
        index_name = gr.Textbox(value=DEFAULT_INDEX_NAME, label="索引", scale=2)
        top_k = gr.Number(value=5, label="Top K", precision=0, scale=1)
    with gr.Row():
        diagnosis_result = gr.Textbox(label="病名", scale=1)
        syndrome_result = gr.Textbox(label="证型名", scale=1)
    clinical_symptoms = gr.Textbox(label="患者症状", lines=4)
    gold_cases = gr.Textbox(label="评测样本 JSON 数组（可选）", lines=8)
    run_btn = gr.Button(value="检索处方", variant="primary")
    metrics = gr.Dataframe(headers=["指标", "值"], label="召回指标", interactive=False)
    hits = gr.Dataframe(
        headers=["Rank", "ID", "Score", "病名", "证型", "患者症状", "处方结果", "模板方结果"],
        label="Top5 处方召回结果",
        interactive=False,
        wrap=True,
    )
    templates = gr.Dataframe(
        headers=["Rank", "模板方", "命中病历数", "聚合得分", "模板药物"],
        label="最相似模板方",
        interactive=False,
        wrap=True,
    )
    detail = gr.JSON(label="详情")

    inputs = [
        index_name,
        diagnosis_result,
        syndrome_result,
        clinical_symptoms,
        top_k,
        gold_cases,
    ]
    run_btn.click(run_recall_test, inputs=inputs, outputs=[metrics, hits, templates, detail])

    elem_dict.update(
        dict(
            index_name=index_name,
            top_k=top_k,
            diagnosis_result=diagnosis_result,
            syndrome_result=syndrome_result,
            clinical_symptoms=clinical_symptoms,
            gold_cases=gold_cases,
            run_btn=run_btn,
            metrics=metrics,
            hits=hits,
            templates=templates,
            detail=detail,
        )
    )
    return elem_dict
