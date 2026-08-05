# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patient-isolated prescription RAG evaluation, persistence, and background jobs."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from medical.data_utils.prescription_rag_agents import (
    DiseaseDifferentiationResult,
    PrescriptionProposal,
    PrescriptionRAGAgents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "medical/eval_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "medical/analysis/output/prescription_rag_eval"
DEFAULT_INDEX_NAME = "alpha_medical_prescription_rag_v1"
EVAL_FILENAMES = ("prescription_rag_eval.jsonl", "eval_cases.jsonl")


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _drug_name(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("name") or item.get("drug_name") or item.get("show_name") or "").strip()
    if hasattr(item, "name"):
        return str(getattr(item, "name") or "").strip()
    return ""


def _drug_names(items: Iterable[Any] | None) -> set[str]:
    return {name for name in (_drug_name(item) for item in items or []) if name}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _dose_map(items: Iterable[Any] | None) -> dict[str, float]:
    result = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _drug_name(item)
        dose = _as_float(item.get("dose") if item.get("dose") is not None else item.get("drug_weight"))
        if name and dose is not None:
            result[name] = dose
    return result


def _prescription_drugs(value: Any) -> list[dict[str, Any]]:
    """Convert raw medical-record ``ps`` prescriptions into evaluation drugs."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    prescriptions = value if isinstance(value, list) else [value]
    drugs: list[dict[str, Any]] = []
    for prescription in prescriptions:
        if not isinstance(prescription, dict):
            continue
        usage_type = str(prescription.get("usage_type") or "").strip()
        if usage_type and usage_type != "内服":
            continue
        items = prescription.get("prescription_items") or {}
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except json.JSONDecodeError:
                continue
        if not isinstance(items, dict):
            continue
        drug_list = items.get("drugList") or items.get("drug_list") or items.get("drugs") or []
        for item in drug_list:
            if not isinstance(item, dict):
                continue
            name = _drug_name(item)
            if not name:
                continue
            dose = _as_float(
                item.get("dose")
                if item.get("dose") is not None
                else item.get("drug_weight")
            )
            drugs.append(
                {
                    "name": name,
                    "dose": dose,
                    "unit": str(item.get("unit") or item.get("unit_name") or "g").strip() or "g",
                }
            )
    return drugs


def _diagnosis_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _normalize_eval_case(case: dict[str, Any], line_number: int) -> dict[str, Any]:
    """Accept both dedicated evaluation rows and normalized full medical records."""
    normalized = dict(case)
    query_id = case.get("query_id") or case.get("order_sn") or case.get("record_id") or case.get("id")
    patient_id = case.get("patient_id") or case.get("user_info_id")
    if not _nonempty(query_id) or not _nonempty(patient_id):
        raise ValueError(
            f"评测数据第 {line_number} 行必须包含 patient_id，且至少包含 "
            "query_id、order_sn、record_id、id 中的一个"
        )
    normalized["query_id"] = str(query_id)
    normalized["patient_id"] = str(patient_id)
    normalized["split"] = str(case.get("split") or "test")
    normalized.setdefault(
        "chief_complaint",
        case.get("ai_patient_appeal") or case.get("doc_ass_stu_appeal") or case.get("patient_appeal") or "",
    )
    normalized.setdefault(
        "present_history",
        case.get("ai_new_medical_history") or case.get("new_medical_history") or "",
    )
    normalized.setdefault("past_history", case.get("ai_old_medical_history") or case.get("old_medical_history") or "")
    normalized.setdefault("tongue_pulse", case.get("admin_face_describe") or "")
    normalized.setdefault("special_population", [item for item in (case.get("special_history"), case.get("birth_detail")) if _nonempty(item)])

    gold = dict(case.get("gold") or {})
    gold.setdefault(
        "western_diagnosis",
        _diagnosis_list(case.get("ai_diagnosis_illness") or case.get("diagnosis_illness")),
    )
    gold.setdefault(
        "tcm_syndrome",
        _diagnosis_list(case.get("ai_diagnosis_disease") or case.get("diagnosis_disease")),
    )
    gold.setdefault(
        "tcm_disease",
        _diagnosis_list(case.get("ai_diagnosis_sickness") or case.get("diagnosis_sickness")),
    )
    if not _nonempty(gold.get("prescription")) and _nonempty(case.get("ps")):
        gold["prescription"] = _prescription_drugs(case.get("ps"))
    normalized["gold"] = gold
    normalized["__source_line"] = line_number
    return normalized


def _visit_sort_key(case: dict[str, Any]) -> tuple[str, int]:
    visit_time = case.get("see_doc_time") or case.get("start_time") or case.get("created_at") or ""
    return str(visit_time), int(case.get("__source_line") or 0)


def _enrich_prior_records(cases: list[dict[str, Any]]) -> None:
    """Build leakage-safe prior visits from earlier rows of the same patient."""
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_patient[str(case["patient_id"])].append(case)
    for patient_cases in by_patient.values():
        patient_cases.sort(key=_visit_sort_key)
        prior_records: list[dict[str, Any]] = []
        previous_prescription: list[dict[str, Any]] = []
        for case in patient_cases:
            if not _nonempty(case.get("prior_records")):
                case["prior_records"] = list(prior_records)
            if not _nonempty(case.get("previous_prescription")) and previous_prescription:
                case["previous_prescription"] = list(previous_prescription)
            gold = case.get("gold") or {}
            current_prescription = gold.get("prescription") or []
            prior_records.append(
                {
                    "query_id": case["query_id"],
                    "visit_time": case.get("see_doc_time") or case.get("start_time") or "",
                    "chief_complaint": case.get("chief_complaint") or "",
                    "present_history": case.get("present_history") or "",
                    "diagnosis": "；".join(
                        str(value)
                        for field in ("western_diagnosis", "tcm_disease", "tcm_syndrome")
                        for value in gold.get(field) or []
                        if value
                    ),
                    "prescription": current_prescription,
                }
            )
            previous_prescription = current_prescription


def _f1(expected: set[str], predicted: set[str]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    tp = len(expected & predicted)
    precision = tp / len(predicted)
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def _relevance(case: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in case.get("similar_cases") or []:
        if isinstance(item, dict):
            case_id = str(item.get("case_id") or item.get("id") or "")
            relevance = _as_float(item.get("relevance"))
            if case_id:
                values[case_id] = relevance if relevance is not None else 1.0
        elif item not in (None, ""):
            values[str(item)] = 1.0
    for case_id in case.get("relevant_case_ids") or []:
        values.setdefault(str(case_id), 1.0)
    return values


def _recall_at(hit_ids: list[str], relevant: dict[str, float], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(hit_ids[:k]) & set(relevant)) / len(relevant)


def _ndcg_at(hit_ids: list[str], relevant: dict[str, float], k: int = 10) -> float | None:
    if not relevant:
        return None
    dcg = sum((2 ** relevant.get(case_id, 0.0) - 1) / math.log2(rank + 1) for rank, case_id in enumerate(hit_ids[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _gold_prescription(case: dict[str, Any]) -> list[dict[str, Any]]:
    gold = case.get("gold") or {}
    return (
        case.get("actual_prescription")
        or case.get("gold_prescription")
        or gold.get("prescription")
        or gold.get("drugs")
        or []
    )


def _core_drugs(case: dict[str, Any]) -> set[str]:
    gold = case.get("gold") or {}
    return _drug_names(case.get("core_drugs") or gold.get("core_drugs") or [])


def _previous_prescription(case: dict[str, Any]) -> list[dict[str, Any]]:
    return case.get("previous_prescription") or (case.get("gold") or {}).get("previous_prescription") or []


def _gold_changes(case: dict[str, Any]) -> tuple[set[str], set[str]]:
    gold = case.get("gold") or {}
    added = _drug_names(case.get("added_drugs") or gold.get("added_drugs") or [])
    removed = _drug_names(case.get("removed_drugs") or gold.get("removed_drugs") or [])
    if not added and not removed and _previous_prescription(case):
        previous = _drug_names(_previous_prescription(case))
        current = _drug_names(_gold_prescription(case))
        return current - previous, previous - current
    return added, removed


def _forbidden_drugs(case: dict[str, Any]) -> set[str]:
    safety = case.get("safety_rules") or {}
    return _drug_names(case.get("forbidden_drugs") or safety.get("forbidden_drugs") or [])


def _forbidden_pairs(case: dict[str, Any]) -> list[set[str]]:
    safety = case.get("safety_rules") or {}
    pairs = case.get("forbidden_pairs") or safety.get("forbidden_pairs") or []
    return [_drug_names(pair) for pair in pairs if len(_drug_names(pair)) >= 2]


def _doctor_final_prescription(case: dict[str, Any]) -> list[dict[str, Any]]:
    review = case.get("doctor_review") or {}
    return case.get("doctor_final_prescription") or review.get("final_prescription") or _gold_prescription(case)


def evaluate_case(
    case: dict[str, Any],
    proposal: PrescriptionProposal,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate all requested metrics for one case without using model-generated explanations as labels."""
    hit_ids = [str(hit.get("case_id") or hit.get("_id") or "") for hit in hits]
    relevance = _relevance(case)
    expected = _drug_names(_gold_prescription(case))
    predicted = _drug_names(proposal.drugs) if not proposal.refused else set()
    tp = len(expected & predicted)
    precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
    prescription_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    core = _core_drugs(case)
    expected_added, expected_removed = _gold_changes(case)
    gold_section = case.get("gold") or {}
    adjustment_evaluable = bool(
        _previous_prescription(case)
        or "added_drugs" in case
        or "removed_drugs" in case
        or "added_drugs" in gold_section
        or "removed_drugs" in gold_section
    )
    previous = _drug_names(_previous_prescription(case))
    predicted_added = _drug_names(proposal.added_drugs) or (predicted - previous if previous else set())
    predicted_removed = _drug_names(proposal.removed_drugs) or (previous - predicted if previous else set())

    predicted_doses = _dose_map(item.model_dump() for item in proposal.drugs)
    dose_ranges = case.get("dose_ranges") or (case.get("gold") or {}).get("dose_ranges") or {}
    dose_results = []
    for name, bounds in dose_ranges.items():
        if not isinstance(bounds, dict):
            continue
        low = _as_float(bounds.get("min"))
        high = _as_float(bounds.get("max"))
        dose = predicted_doses.get(str(name))
        if low is not None and high is not None:
            dose_results.append(1.0 if dose is not None and low <= dose <= high else 0.0)

    forbidden = _forbidden_drugs(case)
    forbidden_pairs = _forbidden_pairs(case)
    violations = sorted(predicted & forbidden)
    pair_violations = [sorted(pair) for pair in forbidden_pairs if pair <= predicted]
    contraindication_evaluable = bool(forbidden or forbidden_pairs)
    retrieved_ids = set(hit_ids)
    unsupported = []
    for drug in proposal.drugs:
        evidence = {str(item) for item in drug.evidence_case_ids}
        if not evidence or not (evidence & retrieved_ids):
            unsupported.append(drug.name)

    doctor_final = _doctor_final_prescription(case)
    doctor_drug_modifications = None
    doctor_dose_modifications = None
    if doctor_final:
        doctor_names = _drug_names(doctor_final)
        doctor_drug_modifications = len(predicted ^ doctor_names)
        doctor_doses = _dose_map(doctor_final)
        doctor_dose_modifications = sum(
            1
            for name in predicted & doctor_names
            if name in predicted_doses and name in doctor_doses and not math.isclose(predicted_doses[name], doctor_doses[name])
        )

    return {
        "recall_at_5": _recall_at(hit_ids, relevance, 5),
        "recall_at_10": _recall_at(hit_ids, relevance, 10),
        "ndcg_at_10": _ndcg_at(hit_ids, relevance, 10),
        "prescription_precision": round(precision, 4),
        "prescription_recall": round(recall, 4),
        "prescription_f1": round(prescription_f1, 4),
        "core_drug_hit_rate": round(len(core & predicted) / len(core), 4) if core else None,
        "added_drug_f1": round(_f1(expected_added, predicted_added), 4) if adjustment_evaluable else None,
        "removed_drug_f1": round(_f1(expected_removed, predicted_removed), 4) if adjustment_evaluable else None,
        "adjustment_f1": (
            round((_f1(expected_added, predicted_added) + _f1(expected_removed, predicted_removed)) / 2, 4)
            if adjustment_evaluable
            else None
        ),
        "dose_range_hit_rate": _mean(dose_results),
        "contraindication_evaluable": contraindication_evaluable,
        "contraindication_violation": bool(violations or pair_violations) if contraindication_evaluable else None,
        "contraindication_violations": violations,
        "forbidden_pair_violations": pair_violations,
        "unsupported_drug_count": len(set(unsupported)),
        "unsupported_drugs": sorted(set(unsupported)),
        "refused": proposal.refused,
        "doctor_drug_modifications": doctor_drug_modifications,
        "doctor_dose_modifications": doctor_dose_modifications,
        "expected_drug_count": len(expected),
        "predicted_drug_count": len(predicted),
        "true_positive_drug_count": tp,
    }


def aggregate_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval metrics by macro average and prescription metrics by micro counts."""
    completed = [item for item in details if not item.get("error")]
    expected_count = sum(item["metrics"]["expected_drug_count"] for item in completed)
    predicted_count = sum(item["metrics"]["predicted_drug_count"] for item in completed)
    tp = sum(item["metrics"]["true_positive_drug_count"] for item in completed)
    precision = tp / predicted_count if predicted_count else (1.0 if not expected_count else 0.0)
    recall = tp / expected_count if expected_count else (1.0 if not predicted_count else 0.0)
    prescription_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metric = lambda name: _mean(item["metrics"].get(name) for item in completed)
    contraindication_cases = [
        item for item in completed if item["metrics"].get("contraindication_evaluable")
    ]
    doctor_modification_cases = [
        item for item in completed if item["metrics"].get("doctor_drug_modifications") is not None
    ]
    return {
        "case_count": len(details),
        "completed_case_count": len(completed),
        "failed_case_count": len(details) - len(completed),
        "similar_case_recall_at_5": metric("recall_at_5"),
        "similar_case_recall_at_10": metric("recall_at_10"),
        "similar_case_ndcg_at_10": metric("ndcg_at_10"),
        "prescription_precision": round(precision, 4),
        "prescription_recall": round(recall, 4),
        "prescription_f1": round(prescription_f1, 4),
        "core_drug_hit_rate": metric("core_drug_hit_rate"),
        "added_drug_f1": metric("added_drug_f1"),
        "removed_drug_f1": metric("removed_drug_f1"),
        "adjustment_f1": metric("adjustment_f1"),
        "dose_range_hit_rate": metric("dose_range_hit_rate"),
        "contraindication_violation_rate": _mean(
            1.0 if item["metrics"]["contraindication_violation"] else 0.0 for item in contraindication_cases
        ),
        "contraindication_evaluable_case_count": len(contraindication_cases),
        "unsupported_drug_count": sum(item["metrics"]["unsupported_drug_count"] for item in completed),
        "unsupported_drugs_per_case": _mean(item["metrics"]["unsupported_drug_count"] for item in completed),
        "system_refusal_rate": _mean(1.0 if item["metrics"]["refused"] else 0.0 for item in completed),
        "doctor_modification_case_count": len(doctor_modification_cases),
        "doctor_drug_modifications": (
            sum(item["metrics"]["doctor_drug_modifications"] or 0 for item in doctor_modification_cases)
            if doctor_modification_cases
            else None
        ),
        "doctor_drug_modifications_per_case": metric("doctor_drug_modifications"),
        "doctor_dose_modifications": (
            sum(item["metrics"]["doctor_dose_modifications"] or 0 for item in doctor_modification_cases)
            if doctor_modification_cases
            else None
        ),
        "doctor_dose_modifications_per_case": metric("doctor_dose_modifications"),
    }


class EvalDatasetRepository:
    """Discover one patient-isolated JSONL evaluation dataset per doctor."""

    def __init__(self, root: Path | str = DEFAULT_EVAL_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()

    def discover(self, doctor_id: str, doctor_name: str = "") -> Path | None:
        candidates = []
        if doctor_name:
            candidates.append(self.root / f"doctor_{doctor_id}_{doctor_name}")
        if self.root.is_dir():
            candidates.extend(sorted(self.root.glob(f"doctor_{doctor_id}_*")))
        seen = set()
        for directory in candidates:
            if directory in seen or not directory.is_dir():
                continue
            seen.add(directory)
            for filename in EVAL_FILENAMES:
                path = directory / filename
                if path.is_file():
                    return path
        return None

    @staticmethod
    def load(path: Path) -> list[dict[str, Any]]:
        cases = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"评测数据第 {line_number} 行不是合法JSON: {error}") from error
                if not isinstance(case, dict):
                    raise ValueError(f"评测数据第 {line_number} 行必须是JSON对象")
                cases.append(_normalize_eval_case(case, line_number))
        _enrich_prior_records(cases)
        for case in cases:
            case.pop("__source_line", None)
        return cases

    @staticmethod
    def validate_patient_splits(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        patient_splits: dict[str, set[str]] = defaultdict(set)
        for case in cases:
            patient_splits[str(case["patient_id"])].add(str(case.get("split") or "test"))
        return [
            {"patient_id": patient_id, "splits": sorted(splits)}
            for patient_id, splits in patient_splits.items()
            if len(splits) > 1
        ]

    def info(self, doctor_id: str, doctor_name: str = "") -> dict[str, Any]:
        path = self.discover(doctor_id, doctor_name)
        if path is None:
            return {
                "available": False,
                "expected_path": str(
                    self.root / f"doctor_{doctor_id}_{doctor_name or '<医生姓名>'}" / EVAL_FILENAMES[0]
                ),
                "case_count": 0,
                "patient_count": 0,
                "split_leaks": [],
            }
        cases = self.load(path)
        return {
            "available": True,
            "path": str(path),
            "case_count": len(cases),
            "patient_count": len({str(case["patient_id"]) for case in cases}),
            "split_leaks": self.validate_patient_splits(cases),
            "split_counts": dict(
                sorted(
                    {
                        split: sum(1 for case in cases if str(case.get("split") or "test") == split)
                        for split in {str(case.get("split") or "test") for case in cases}
                    }.items()
                )
            ),
        }


class ElasticsearchCaseRetriever:
    """Use the existing hybrid ES retrieval and remove the current patient's own cases."""

    def __init__(self) -> None:
        from medical.config import config
        from medical.rag.es.es_processor import _load_elasticsearch

        es_class = _load_elasticsearch()
        host = getattr(config, "ES_HOST", "localhost")
        port = getattr(config, "ES_PORT", "9200")
        user = getattr(config, "ES_USER", "")
        password = getattr(config, "ES_AUTH", "")
        auth = (user, password) if user else None
        self.client = es_class(f"http://{host}:{port}", basic_auth=auth, request_timeout=80)

    @staticmethod
    def _query_text(result: DiseaseDifferentiationResult) -> str:
        sections = [
            ("西医诊断", result.western_diagnoses),
            ("中医疾病", result.tcm_diseases),
            ("中医证候", result.tcm_syndromes),
            ("关键症状", result.key_symptoms),
            ("病因", result.etiologies),
            ("病机", result.pathogenesis),
            ("病位", result.disease_locations),
            ("治则", result.treatment_principles),
            ("治法", result.treatment_methods),
        ]
        return "\n".join(f"{label}：{'、'.join(values)}" for label, values in sections if values)

    @staticmethod
    def _sanitize_hit(hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source") or {}
        metadata = source.get("metadata") or {}
        drugs = source.get("drugs") or metadata.get("drugs") or []
        return {
            "case_id": str(hit.get("_id") or source.get("case_id") or source.get("chunk_id") or ""),
            "score": round(float(hit.get("rerank_score", hit.get("rrf_score", hit.get("_score", 0))) or 0), 6),
            "western_diagnosis": source.get("western_diagnosis") or source.get("diagnosis_result") or "",
            "tcm_disease": source.get("tcm_disease") or "",
            "tcm_syndrome": source.get("tcm_syndrome") or source.get("syndrome_result") or "",
            "clinical_summary": source.get("clinical_symptoms_text") or source.get("clinical_text") or "",
            "treatment_principles": source.get("treatment_principle") or [],
            "prescription_text": source.get("prescription_text") or "",
            "drugs": [
                {
                    "name": _drug_name(item),
                    "dose": item.get("dose") if isinstance(item, dict) else None,
                    "unit": item.get("unit") if isinstance(item, dict) else "",
                }
                for item in drugs
                if _drug_name(item)
            ],
            "followup_outcome": source.get("followup_outcome") or "",
            "quality_score": source.get("quality_score"),
        }

    def retrieve(
        self,
        index_name: str,
        result: DiseaseDifferentiationResult,
        patient_id: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        from medical.rag.es.es_processor import embed_prescription_query, hybrid_search

        query_text = self._query_text(result)
        if not query_text:
            return []
        query_vector = [float(value) for value in embed_prescription_query(query_text)]
        diagnosis = (result.western_diagnoses or result.tcm_diseases or [None])[0]
        syndrome = (result.tcm_syndromes or [None])[0]
        search_size = max(50, top_k * 5)
        search_levels = [(diagnosis, syndrome), (diagnosis, None), (None, syndrome), (None, None)]
        filtered = []
        seen_hit_ids = set()
        for level_diagnosis, level_syndrome in search_levels:
            level_hits = hybrid_search(
                self.client,
                index_name,
                query_text,
                query_vector,
                diagnosis_result=level_diagnosis,
                syndrome_result=level_syndrome,
                chunk_types=["case_prescription"],
                top_k=search_size,
            )
            for hit in level_hits:
                hit_id = str(hit.get("_id") or "")
                metadata = (hit.get("_source") or {}).get("metadata") or {}
                if not hit_id or hit_id in seen_hit_ids or str(metadata.get("patient_id") or "") == str(patient_id):
                    continue
                seen_hit_ids.add(hit_id)
                filtered.append(self._sanitize_hit(hit))
                if len(filtered) >= top_k:
                    break
            if len(filtered) >= top_k:
                break
        return filtered


class PrescriptionRAGEvaluator:
    """Sequentially run differentiation, retrieval, prescription and requested metrics."""

    def __init__(
        self,
        agents: PrescriptionRAGAgents | None = None,
        retriever: ElasticsearchCaseRetriever | None = None,
    ) -> None:
        self.agents = agents or PrescriptionRAGAgents()
        self.retriever = retriever or ElasticsearchCaseRetriever()

    def run(
        self,
        cases: list[dict[str, Any]],
        index_name: str,
        *,
        top_k: int = 10,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        details = []
        for index, case in enumerate(cases, start=1):
            query_id = str(case.get("query_id"))
            try:
                differentiation = self.agents.differentiate(case)
                hits = self.retriever.retrieve(index_name, differentiation, case.get("patient_id"), top_k=top_k)
                proposal = self.agents.prescribe(case, differentiation, hits)
                metrics = evaluate_case(case, proposal, hits)
                details.append(
                    {
                        "query_id": query_id,
                        "differentiation": differentiation.model_dump(),
                        "retrieved_cases": hits,
                        "proposal": proposal.model_dump(),
                        "metrics": metrics,
                    }
                )
            except Exception as error:
                details.append({"query_id": query_id, "error": f"{type(error).__name__}: {error}"})
            if progress:
                progress(index, len(cases))
        return {"metrics": aggregate_metrics(details), "details": details}


class PrescriptionRAGEvaluationJobs:
    """Single-worker background evaluation queue used by the FastAPI dashboard."""

    def __init__(
        self,
        dataset_repository: EvalDatasetRepository | None = None,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        evaluator_factory: Callable[[], PrescriptionRAGEvaluator] = PrescriptionRAGEvaluator,
    ) -> None:
        self.datasets = dataset_repository or EvalDatasetRepository(os.getenv("MEDICAL_RAG_EVAL_ROOT", DEFAULT_EVAL_ROOT))
        self.output_root = Path(output_root)
        self.evaluator_factory = evaluator_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prescription-rag-eval")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def start(
        self,
        doctor_id: str,
        doctor_name: str,
        *,
        index_name: str = DEFAULT_INDEX_NAME,
        limit: int = 0,
        top_k: int = 10,
    ) -> dict[str, Any]:
        info = self.datasets.info(doctor_id, doctor_name)
        if not info["available"]:
            raise FileNotFoundError(info["expected_path"])
        if info["split_leaks"]:
            raise ValueError("同一患者出现在多个split中，请先修复患者级数据泄漏")
        path = Path(info["path"])
        cases = self.datasets.load(path)
        cases = [case for case in cases if str(case.get("split") or "test") == "test"]
        if limit > 0:
            cases = cases[:limit]
        if not cases:
            raise ValueError("评测数据中没有 split=test 的样本")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "doctor_id": doctor_id,
            "doctor_name": doctor_name,
            "index_name": index_name,
            "case_count": len(cases),
            "completed": 0,
            "progress": 0.0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job_id, cases, index_name, top_k)
        return dict(job)

    def _run(self, job_id: str, cases: list[dict[str, Any]], index_name: str, top_k: int) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")

        def update(completed: int, total: int) -> None:
            with self.lock:
                self.jobs[job_id]["completed"] = completed
                self.jobs[job_id]["progress"] = round(completed / total, 4)

        try:
            result = self.evaluator_factory().run(cases, index_name, top_k=top_k, progress=update)
            output_dir = self.output_root / f"doctor_{self.jobs[job_id]['doctor_id']}_{self.jobs[job_id]['doctor_name']}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"evaluation_{job_id}.json"
            payload = {
                "job": {key: value for key, value in self.jobs[job_id].items() if key != "result"},
                **result,
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with self.lock:
                self.jobs[job_id].update(
                    {
                        "status": "completed",
                        "completed": len(cases),
                        "progress": 1.0,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "output_path": str(output_path),
                        "result": result,
                    }
                )
        except Exception as error:
            with self.lock:
                self.jobs[job_id].update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self.jobs[job_id], ensure_ascii=False))
