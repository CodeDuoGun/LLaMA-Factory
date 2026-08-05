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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical.analysis.prescription_rag_eval import (
    EvalDatasetRepository,
    aggregate_metrics,
    evaluate_case,
)
from medical.data_utils.prescription_rag_agents import (
    PrescriptionDrug,
    PrescriptionProposal,
    PrescriptionRAGAgents,
    PrescriptionRAGAgentSettings,
)


def _case() -> dict:
    return {
        "query_id": "q1",
        "patient_id": "patient-1",
        "chief_complaint": "胃脘胀满",
        "present_history": "纳差乏力",
        "gold": {
            "prescription": [
                {"name": "白术", "dose": 10},
                {"name": "茯苓", "dose": 20},
                {"name": "陈皮", "dose": 6},
            ],
            "core_drugs": ["白术", "茯苓"],
        },
        "previous_prescription": [{"name": "白术", "dose": 10}, {"name": "甘草", "dose": 6}],
        "similar_cases": [{"case_id": "c1", "relevance": 3}, {"case_id": "c2", "relevance": 2}],
        "dose_ranges": {"白术": {"min": 9, "max": 11}, "茯苓": {"min": 19, "max": 21}},
        "safety_rules": {"forbidden_drugs": ["附子"], "forbidden_pairs": [["甘草", "甘遂"]]},
        "doctor_review": {
            "final_prescription": [
                {"name": "白术", "dose": 10},
                {"name": "茯苓", "dose": 20},
                {"name": "陈皮", "dose": 6},
            ]
        },
    }


def test_evaluate_case_covers_retrieval_prescription_safety_and_edits() -> None:
    proposal = PrescriptionProposal(
        drugs=[
            PrescriptionDrug(name="白术", dose=10, evidence_case_ids=["c1"]),
            PrescriptionDrug(name="茯苓", dose=18, evidence_case_ids=["c1"]),
            PrescriptionDrug(name="附子", dose=3, evidence_case_ids=["not-retrieved"]),
        ],
        added_drugs=["茯苓", "附子"],
        removed_drugs=["甘草"],
    )
    metrics = evaluate_case(_case(), proposal, [{"case_id": "c1"}, {"case_id": "x"}, {"case_id": "c2"}])

    assert metrics["recall_at_5"] == 1
    assert metrics["ndcg_at_10"] == pytest.approx(0.9558, abs=0.0001)
    assert metrics["prescription_precision"] == pytest.approx(2 / 3, abs=0.0001)
    assert metrics["prescription_recall"] == pytest.approx(2 / 3, abs=0.0001)
    assert metrics["core_drug_hit_rate"] == 1
    assert metrics["dose_range_hit_rate"] == 0.5
    assert metrics["contraindication_violation"] is True
    assert metrics["unsupported_drugs"] == ["附子"]
    assert metrics["doctor_drug_modifications"] == 2
    assert metrics["doctor_dose_modifications"] == 1


def test_aggregate_metrics_counts_refusal_and_failed_cases() -> None:
    successful = evaluate_case(_case(), PrescriptionProposal(refused=True, refusal_reason="证据不足"), [])
    result = aggregate_metrics(
        [
            {"query_id": "q1", "metrics": successful},
            {"query_id": "q2", "error": "model unavailable"},
        ]
    )

    assert result["case_count"] == 2
    assert result["completed_case_count"] == 1
    assert result["failed_case_count"] == 1
    assert result["system_refusal_rate"] == 1
    assert result["prescription_recall"] == 0


def test_dataset_repository_discovers_file_and_rejects_cross_split_patient(tmp_path: Path) -> None:
    path = tmp_path / "doctor_43_朱子奇" / "prescription_rag_eval.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"query_id": "q1", "patient_id": "p1", "split": "train"},
        {"query_id": "q2", "patient_id": "p1", "split": "test"},
        {"query_id": "q3", "patient_id": "p2", "split": "test"},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    info = EvalDatasetRepository(tmp_path).info("43", "朱子奇")

    assert info["available"] is True
    assert info["case_count"] == 3
    assert info["patient_count"] == 2
    assert info["split_leaks"] == [{"patient_id": "p1", "splits": ["test", "train"]}]


def test_dataset_repository_accepts_full_medical_records_and_builds_prior_visits(tmp_path: Path) -> None:
    path = tmp_path / "doctor_43_朱子奇" / "prescription_rag_eval.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "id": 101,
            "order_sn": "visit-1",
            "patient_id": "patient-1",
            "see_doc_time": "2026-01-01 09:00:00",
            "doc_ass_stu_appeal": "胃胀3天",
            "new_medical_history": "胃胀，纳差。",
            "diagnosis_illness": "慢性胃炎",
            "diagnosis_disease": "脾胃虚弱证",
            "diagnosis_sickness": "胃痞病",
            "ps": [
                {
                    "usage_type": "内服",
                    "prescription_items": {
                        "drugList": [{"show_name": "白术", "drug_weight": "10", "unit_name": "g"}]
                    },
                },
                {
                    "usage_type": "外用",
                    "prescription_items": {
                        "drugList": [{"show_name": "苦参", "drug_weight": "20", "unit_name": "g"}]
                    },
                },
            ],
        },
        {
            "id": 102,
            "order_sn": "visit-2",
            "patient_id": "patient-1",
            "see_doc_time": "2026-01-10 09:00:00",
            "patient_appeal": "胃胀反复10天",
            "new_medical_history": "胃胀较前减轻。",
            "ps": [
                {
                    "usage_type": "内服",
                    "prescription_items": {
                        "drugList": [{"show_name": "茯苓", "drug_weight": 15, "unit_name": "克"}]
                    },
                }
            ],
        },
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    cases = EvalDatasetRepository(tmp_path).load(path)

    assert cases[0]["query_id"] == "visit-1"
    assert cases[0]["chief_complaint"] == "胃胀3天"
    assert cases[0]["gold"]["western_diagnosis"] == ["慢性胃炎"]
    assert cases[0]["gold"]["prescription"] == [{"name": "白术", "dose": 10.0, "unit": "g"}]
    assert cases[1]["prior_records"][0]["query_id"] == "visit-1"
    assert cases[1]["previous_prescription"] == [{"name": "白术", "dose": 10.0, "unit": "g"}]


class _FakeResponse:
    def __init__(self, content: dict) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": json.dumps(self.content, ensure_ascii=False)}}]}


def test_agents_do_not_send_gold_diagnosis_or_prescription_to_differentiation() -> None:
    requests = []

    def request_post(*args, **kwargs):
        requests.append(kwargs["json"])
        return _FakeResponse(
            {
                "western_diagnoses": [],
                "tcm_diseases": ["胃痞病"],
                "tcm_syndromes": ["脾胃虚弱证"],
                "etiologies": [],
                "pathogenesis": ["脾气亏虚"],
                "disease_locations": ["脾", "胃"],
                "disease_stage": "",
                "disease_course": "",
                "treatment_principles": ["健脾和胃"],
                "treatment_methods": ["益气健脾"],
                "key_symptoms": ["胃脘胀满", "纳差"],
                "evidence_limits": [],
            }
        )

    agents = PrescriptionRAGAgents(
        PrescriptionRAGAgentSettings(api_key="key", base_url="https://example.test/v1", model="model"),
        request_post=request_post,
    )
    case = _case()
    case["gold"]["western_diagnosis"] = ["绝不能泄漏的诊断"]
    case["gold"]["prescription"].append({"name": "绝不能泄漏的药"})

    result = agents.differentiate(case)
    sent = requests[0]["messages"][1]["content"]

    assert result.tcm_diseases == ["胃痞病"]
    assert "绝不能泄漏的诊断" not in sent
    assert "绝不能泄漏的药" not in sent


def test_frontend_contains_prescription_rag_evaluation_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "static/index.html").read_text(encoding="utf-8")
    javascript = (root / "static/app.js").read_text(encoding="utf-8")

    assert 'data-view="prescription-rag-eval"' in html
    assert 'id="prescription-rag-eval-form"' in html
    assert "/api/prescription-rag-eval/run" in javascript
    assert "similar_case_recall_at_10" in javascript
