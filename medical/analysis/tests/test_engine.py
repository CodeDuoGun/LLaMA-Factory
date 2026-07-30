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

# ruff: noqa: E402, I001

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from medical.analysis.engine import (  # noqa: E402
    MedicalAnalysis,
    canonical_disease,
    classify_outcome,
    extract_symptoms,
    normalize_diagnosis_value,
    normalize_sickness_value,
    normalize_syndrome_value,
)


def _drug(drug_id: int, name: str, dose: int) -> dict:
    return {"drug_id": drug_id, "drug_name": name, "drug_weight": dose, "unit_name": "g"}


def _record(visit_id: int, date: str, first: str, drugs: list[dict], history: str) -> dict:
    return {
        "id": visit_id,
        "patient_id": 10086,
        "start_time": date,
        "is_first": first,
        "diagnosis_illness": "胰腺癌肝转移",
        "new_medical_history": history,
        "patient_appeal": "胰腺癌",
        "ps": [
            {
                "usage_type": "内服",
                "drug_process_name": "饮片",
                "use_limit": "14_1_2",
                "prescription_items": {"drugList": drugs},
            }
        ],
    }


def test_normalization_and_negation() -> None:
    assert canonical_disease("胰腺癌术后，肝转移") == "胰腺癌"
    assert extract_symptoms("乏力，纳差，无腹痛，大便偏干") == ["乏力", "纳差", "便秘"]
    assert classify_outcome("药后乏力较前减轻，整体感觉良好") == "改善"


def test_diagnosis_value_normalizes_separator_order_and_duplicates() -> None:
    assert normalize_diagnosis_value("湿毒蕴肤症") == "湿毒蕴肤症"
    assert normalize_diagnosis_value("血热瘀滞症") == "血热瘀滞症"
    assert normalize_diagnosis_value("脾肺气虚证，痰瘀互结证") == "痰瘀互结证、脾肺气虚证"
    assert normalize_diagnosis_value("痰瘀互结证/脾肺气虚证；脾肺气虚证") == "痰瘀互结证、脾肺气虚证"
    assert normalize_diagnosis_value("甲状腺癌、瘿瘤") == normalize_diagnosis_value("甲状腺癌、瘿瘤病")
    assert normalize_diagnosis_value("咳嗽病") == "咳嗽"
    assert normalize_diagnosis_value("肺结节") == normalize_diagnosis_value("肺结节病") == "肺结节"
    assert {
        normalize_diagnosis_value(value)
        for value in ("间质肺纤维化", "肺间质纤维化", "肺间质性纤维化", ".双肺间质性纤维化")
    } == {"肺间质纤维化"}
    assert normalize_diagnosis_value("双肺间质性改变伴纤维") == "双肺间质性改变伴纤维化"
    assert normalize_diagnosis_value("双肺间质性改变伴纤维化") == "双肺间质性改变伴纤维化"
    assert normalize_diagnosis_value("慢性菱缩性胃炎") == "慢性萎缩性胃炎"
    assert normalize_diagnosis_value("支气管扩张、肺结节") == normalize_diagnosis_value("支气管扩张、肺结节病")
    assert normalize_diagnosis_value("{'code': '', 'name': '', 'type': '2'}") == ""
    assert normalize_diagnosis_value({"code": "", "name": "", "type": "2"}) == ""
    assert normalize_diagnosis_value('["肺结节"]') == ""
    assert normalize_diagnosis_value("痰瘀阻肺证肝气郁结证") == "痰瘀阻肺证、肝气郁结证"
    assert normalize_diagnosis_value("痰瘀阻肺 肝气郁结") == "痰瘀阻肺证、肝气郁结证"
    assert normalize_diagnosis_value("痰瘀阻肺+肝气郁结") == "痰瘀阻肺证、肝气郁结证"
    assert normalize_diagnosis_value("肝气不舒痰瘀阻肺") == "痰瘀阻肺证、肝气不舒证"
    assert normalize_diagnosis_value("痰瘀阻肺证肝脾不和证") == "痰瘀阻肺证、肝脾不和证"
    assert normalize_diagnosis_value("痰瘀阻肺证\n肝脾不和证") == "痰瘀阻肺证、肝脾不和证"
    assert normalize_diagnosis_value("？,耳鸣病") == "耳鸣病"
    assert normalize_diagnosis_value("。,耳鸣病,?,.") == "耳鸣病"
    assert normalize_diagnosis_value("慢性浅表性胃炎伴肠化、诊断") == "慢性浅表性胃炎伴肠化"
    assert normalize_diagnosis_value("?") == ""
    assert normalize_diagnosis_value("？、。、?、.、--") == ""
    assert normalize_diagnosis_value("l、123") == ""
    assert normalize_diagnosis_value("hsaxg") == ""
    assert normalize_diagnosis_value("肝脾血瘀证,l") == "l、肝脾血瘀证"
    assert normalize_diagnosis_value("？2型糖尿病？,HPV感染") == "2型糖尿病、HPV感染"
    assert normalize_diagnosis_value("多囊卵巢综合征[Stein-Leventhal综合征]") == "多囊卵巢综合征"


def test_syndrome_value_normalizes_suffix_order_and_concatenation() -> None:
    assert normalize_syndrome_value("肝气不舒") == "肝气不舒证"
    assert normalize_syndrome_value("肝气不舒证") == "肝气不舒证"
    assert normalize_syndrome_value("湿毒蕴肤症") == "湿毒蕴肤证"
    assert normalize_syndrome_value("血热瘀滞症") == "血热瘀滞证"
    assert normalize_syndrome_value("脾肺气虚，痰瘀互结证") == "痰瘀互结证、脾肺气虚证"
    assert normalize_syndrome_value("痰瘀互结/脾肺气虚") == "痰瘀互结证、脾肺气虚证"
    assert normalize_syndrome_value("脾肺气虚证痰瘀互结证") == "痰瘀互结证、脾肺气虚证"
    assert normalize_syndrome_value("痰瘀互结证、脾肺气虚、痰瘀互结") == "痰瘀互结证、脾肺气虚证"
    assert normalize_syndrome_value("未明确") == "未明确"
    assert normalize_syndrome_value("{'code': '', 'name': '', 'type': '2'}") == ""
    assert normalize_syndrome_value({"code": "", "name": "", "type": "2"}) == ""
    assert normalize_syndrome_value('["肝气不舒"]') == ""
    assert normalize_syndrome_value("肝脾血瘀证,l") == "l、肝脾血瘀证"
    assert normalize_syndrome_value("肝脾血瘀,l") == "l、肝脾血瘀证"
    assert normalize_syndrome_value("l") == ""
    assert normalize_syndrome_value("hsaxg") == ""
    assert normalize_syndrome_value("?") == ""
    assert normalize_syndrome_value("待填写证") == ""
    assert normalize_syndrome_value("待填写") == ""
    assert normalize_syndrome_value("待填写证、肝气不舒") == "肝气不舒证"


def test_sickness_value_uses_tcm_disease_aliases_only() -> None:
    assert normalize_sickness_value("咳嗽病") == "咳嗽"
    assert normalize_sickness_value("肺结节病") == "肺结节病"
    assert normalize_sickness_value("咳嗽病、胃痞病、咳嗽病") == "咳嗽、胃痞病"


def test_longitudinal_difference_uses_patient_id() -> None:
    records = [
        _record(1, "2026-01-01 09:00:00", "初诊", [_drug(1, "黄芪", 20), _drug(2, "白术", 10)], "乏力、纳差"),
        _record(
            2,
            "2026-01-20 09:00:00",
            "复诊",
            [_drug(1, "黄芪", 30), _drug(3, "火麻仁", 15)],
            "药后乏力减轻，大便偏干",
        ),
    ]
    analysis = MedicalAnalysis(records)
    pair = analysis.revisit_pairs[0]

    assert pair["added"] == ["火麻仁"]
    assert pair["removed"] == ["白术"]
    assert pair["dose_changes"][0]["direction"] == "增加"
    assert pair["outcome"] == "改善"
    assert pair["patient_id"] == "10086"
    timeline = analysis.patient_timeline(pair["patient_id"])
    assert timeline["patient_id"] == "10086"
    assert timeline["visit_count"] == 2
    assert timeline["timeline"][0]["date"] == "2026-01-01"
    assert timeline["timeline"][0]["new_medical_history"] == "乏力、纳差"
    assert {(drug["drug_name"], drug["dose"], drug["unit"]) for drug in timeline["timeline"][0]["drugs"]} == {
        ("黄芪", 20.0, "g"),
        ("白术", 10.0, "g"),
    }
    assert timeline["timeline"][1]["date"] == "2026-01-20"
    assert timeline["timeline"][1]["new_medical_history"] == "药后乏力减轻，大便偏干"


def test_patient_timeline_returns_all_prescriptions_with_usage_type() -> None:
    record = _record(1, "2026-01-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
    record["ps"].extend(
        [
            {
                "usage_type": "外用",
                "drug_process_name": "散剂",
                "prescription_items": {"drugList": [_drug(2, "炉甘石", 30)]},
            },
            {
                "usage_type": "代茶饮",
                "drug_process_name": "饮片",
                "prescription_items": {"drugList": [_drug(3, "菊花", 6)]},
            },
        ]
    )

    visit = MedicalAnalysis([record]).patient_timeline("10086")["timeline"][0]

    assert [prescription["usage_type"] for prescription in visit["prescriptions"]] == ["内服", "外用", "代茶饮"]
    assert [prescription["drugs"][0]["drug_name"] for prescription in visit["prescriptions"]] == [
        "黄芪",
        "炉甘石",
        "菊花",
    ]
    assert visit["drugs"][0]["drug_name"] == "黄芪"


def test_patients_are_grouped_and_filtered_by_visit_type() -> None:
    initial_only = _record(1, "2026-01-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
    revisit_only = _record(2, "2026-01-02 09:00:00", "复诊", [_drug(1, "黄芪", 20)], "乏力减轻")
    both_initial = _record(3, "2026-01-03 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "咳嗽")
    both_revisit = _record(4, "2026-01-04 09:00:00", "复诊", [_drug(1, "黄芪", 20)], "咳嗽减轻")
    initial_only["patient_id"] = 1
    revisit_only["patient_id"] = 2
    both_initial["patient_id"] = both_revisit["patient_id"] = 3
    analysis = MedicalAnalysis([initial_only, revisit_only, both_initial, both_revisit])

    all_patients = analysis.patients()["items"]
    initial_patients = analysis.patients(visit_type="初诊")["items"]
    revisit_patients = analysis.patients(visit_type="复诊")["items"]

    assert {item["patient_id"] for item in all_patients} == {"1", "2", "3"}
    assert {item["patient_id"] for item in initial_patients} == {"1", "2"}
    assert {item["patient_id"] for item in revisit_patients} == {"3"}
    patient_three = next(item for item in all_patients if item["patient_id"] == "3")
    assert patient_three["patient_type"] == "复诊"
    assert patient_three["visit_count"] == 2


def test_symptom_drug_association_support() -> None:
    records = []
    for patient_id in range(4):
        first = _record(
            patient_id * 2 + 1, f"2026-01-0{patient_id + 1} 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力"
        )
        second = _record(
            patient_id * 2 + 2,
            f"2026-02-0{patient_id + 1} 09:00:00",
            "复诊",
            [_drug(1, "黄芪", 20), _drug(3, "火麻仁", 15)],
            "大便偏干",
        )
        first["patient_id"] = second["patient_id"] = patient_id
        records.extend((first, second))

    rows = MedicalAnalysis(records).symptom_drug_associations(symptom="便秘", minimum_support=3)["items"]
    assert rows[0]["drug_name"] == "火麻仁"
    assert rows[0]["support"] == 4
    assert rows[0]["patient_count"] == 4


def test_diagnosis_illness_is_not_merged() -> None:
    lung_cancer = _record(101, "2026-03-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
    adenocarcinoma = _record(102, "2026-03-02 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
    lung_cancer["patient_id"] = 1
    adenocarcinoma["patient_id"] = 2
    lung_cancer["diagnosis_illness"] = "肺癌"
    adenocarcinoma["diagnosis_illness"] = "肺腺癌"

    analysis = MedicalAnalysis([lung_cancer, adenocarcinoma])

    assert {item["disease"] for item in analysis.diseases()["items"]} == {"肺癌", "肺腺癌"}
    assert analysis.disease_detail("肺癌")["visit_count"] == 1
    assert analysis.disease_detail("肺腺癌")["visit_count"] == 1


def test_overview_contains_three_independent_diagnosis_distributions() -> None:
    first = _record(111, "2026-03-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
    second = _record(112, "2026-03-02 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "纳差")
    first["patient_id"] = 1
    second["patient_id"] = 2
    first.update(
        {
            "diagnosis_illness": "胃癌，慢性胃炎",
            "diagnosis_sickness": "胃积病；胃痞病",
            "diagnosis_disease": "脾胃虚弱证/气血两虚证",
        }
    )
    second.update(
        {
            "diagnosis_illness": "慢性胃炎、胃癌",
            "diagnosis_sickness": "胃痞病、胃积病",
            "diagnosis_disease": "气血两虚证，脾胃虚弱证",
        }
    )

    distributions = MedicalAnalysis([first, second]).overview()["diagnosis_distributions"]

    assert distributions["diagnosis_illness"][0] == {"value": "慢性胃炎、胃癌", "visit_count": 2, "patient_count": 2}
    assert distributions["diagnosis_sickness"][0] == {"value": "胃痞病、胃积病", "visit_count": 2, "patient_count": 2}
    assert distributions["diagnosis_disease"][0] == {
        "value": "气血两虚证、脾胃虚弱证",
        "visit_count": 2,
        "patient_count": 2,
    }


def test_overview_returns_complete_diagnosis_distribution() -> None:
    records = []
    for index in range(12):
        record = _record(300 + index, "2026-03-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力")
        record["patient_id"] = index
        record["diagnosis_illness"] = f"西医诊断{index}"
        records.append(record)

    rows = MedicalAnalysis(records).overview()["diagnosis_distributions"]["diagnosis_illness"]

    assert len(rows) == 12


def test_disease_case_contains_required_clinical_fields() -> None:
    record = _record(201, "2026-04-01 09:00:00", "初诊", [_drug(1, "黄芪", 20)], "乏力，纳差")
    record.update(
        {
            "patient_sex": "女",
            "patient_age": "56",
            "allergic_history": "青霉素",
            "family_history": "无",
            "personal_history": "无吸烟史",
            "old_medical_history": "高血压",
        }
    )
    case = MedicalAnalysis([record]).visit_case(201)

    assert case["diagnosis_illness"] == "胰腺癌肝转移"
    assert case["sex"] == "女"
    assert case["age"] == "56"
    assert case["new_medical_history"] == "乏力，纳差"
    assert case["allergic_history"] == "青霉素"
    assert case["family_history"] == "无"
    assert case["personal_history"] == "无吸烟史"
    assert case["old_medical_history"] == "高血压"
