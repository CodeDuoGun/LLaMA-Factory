# Copyright 2026 the LlamaFactory team.
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

import json
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from medical.analysis.base_formula import (  # noqa: E402
    WeightedTransaction,
    build_base_formula_catalog,
    discover_base_formulas,
    discover_base_formulas_by_dimensions,
    formula_dimension_metadata,
    formula_strata,
    multidimensional_formula_strata,
    weighted_fp_growth,
)
from medical.analysis.engine import MedicalAnalysis  # noqa: E402
from medical.analysis.app import app, base_formula_query  # noqa: E402


def _record(
    visit_id: int,
    patient_id: int,
    drugs: list[str],
    syndrome: str = "肝胃不和证",
    tcm_disease: str = "胃痞病",
    visit_type: str = "初诊",
) -> dict:
    return {
        "id": visit_id,
        "patient_id": patient_id,
        "patient_name": f"患者{patient_id}",
        "patient_mobile": f"1380000{patient_id:04d}",
        "patient_idcard": f"11010119900101{patient_id:04d}",
        "start_time": "2026-01-01 09:00:00",
        "is_first": visit_type,
        "diagnosis_illness": "慢性萎缩性胃炎",
        "diagnosis_sickness": tcm_disease,
        "diagnosis_disease": syndrome,
        "ps": [
            {
                "usage_type": "内服",
                "drug_process_name": "饮片",
                "prescription_items": {
                    "drugList": [
                        {
                            "drug_id": index + 1,
                            "drug_name": drug,
                            "drug_weight": 10 + index,
                            "unit_name": "g",
                        }
                        for index, drug in enumerate(drugs)
                    ]
                },
            }
        ],
    }


def test_weighted_fp_growth_finds_weighted_itemsets() -> None:
    transactions = [
        WeightedTransaction(frozenset(("白术", "茯苓", "甘草")), 0.5),
        WeightedTransaction(frozenset(("白术", "茯苓")), 0.5),
        WeightedTransaction(frozenset(("白术", "甘草")), 1.0),
    ]

    patterns = weighted_fp_growth(transactions, minimum_weight=1.0)

    assert patterns[frozenset(("白术",))] == 2.0
    assert patterns[frozenset(("白术", "茯苓"))] == 1.0
    assert frozenset(("茯苓", "甘草")) not in patterns


def test_patient_weighting_prevents_frequent_revisitor_domination() -> None:
    records = []
    for visit_id in range(1, 11):
        records.append(_record(visit_id, 1, ["稀有药", "白术"]))
    records.extend((_record(20, 2, ["白术", "茯苓"]), _record(30, 3, ["白术", "茯苓"])))

    report = discover_base_formulas(
        MedicalAnalysis(records),
        "慢性萎缩性胃炎",
        minimum_support=0.5,
        component_count=2,
        bootstrap_rounds=20,
    )
    supports = {row["drug"]: row["support"] for row in report["frequent_drugs"]}

    assert supports["白术"] == 1.0
    assert supports["茯苓"] == 0.6667
    assert "稀有药" not in supports
    assert report["scope"]["visit_count"] == 12
    assert report["scope"]["patient_count"] == 3


def test_report_is_filtered_and_contains_no_direct_identifiers() -> None:
    records = [
        _record(1, 1, ["白术", "茯苓", "甘草"]),
        _record(2, 2, ["白术", "茯苓", "陈皮"]),
        _record(3, 3, ["白术", "茯苓", "厚朴"], syndrome="脾胃虚弱证"),
    ]
    report = discover_base_formulas(
        MedicalAnalysis(records),
        "慢性萎缩性胃炎",
        syndrome="肝胃不和证",
        minimum_support=0.5,
        component_count=2,
        bootstrap_rounds=20,
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["scope"]["visit_count"] == 2
    assert report["representative_base_formula"]["drugs"] == ["白术", "茯苓"]
    assert report["representative_base_formula"]["support"] == 1.0
    assert report["mining_summary"]["frequent_itemset_count"] > 0
    assert "Fewer than 30 patients" in " ".join(report["warnings"])
    assert report["latent_base_formulas"]["components"]
    assert "患者1" not in serialized
    assert "1380000" not in serialized
    assert "110101" not in serialized


def test_strata_use_structured_syndrome_only() -> None:
    structured = _record(1, 1, ["白术", "茯苓"])
    unspecified = _record(2, 2, ["白术", "陈皮"], syndrome="")
    unspecified["diagnosis_sickness"] = "胃痞病"
    analysis = MedicalAnalysis([structured, unspecified])

    default_items = formula_strata(analysis, minimum_patients=1)["items"]
    all_items = formula_strata(analysis, minimum_patients=1, include_unspecified_syndrome=True)["items"]

    assert [(item["disease"], item["syndrome"]) for item in default_items] == [("慢性萎缩性胃炎", "肝胃不和证")]
    assert {item["syndrome"] for item in all_items} == {"肝胃不和证", "未明确"}
    assert "胃痞病" not in {item["syndrome"] for item in all_items}


def test_catalog_mines_eligible_groups_and_marks_small_groups() -> None:
    records = [
        _record(1, 1, ["白术", "茯苓", "甘草"]),
        _record(2, 2, ["白术", "茯苓", "陈皮"]),
        _record(3, 3, ["白术", "厚朴"], syndrome="脾胃虚弱证"),
    ]

    catalog = build_base_formula_catalog(
        MedicalAnalysis(records),
        minimum_patients=2,
        minimum_support=0.5,
        maximum_pattern_length=4,
        component_count=2,
        bootstrap_rounds=20,
    )
    by_syndrome = {item["syndrome"]: item for item in catalog["items"]}

    assert catalog["summary"] == {"stratum_count": 2, "eligible_count": 1, "insufficient_sample_count": 1}
    assert by_syndrome["肝胃不和证"]["status"] == "eligible"
    assert by_syndrome["肝胃不和证"]["base_formula"]["representative_base_formula"]["drugs"] == ["白术", "茯苓"]
    assert by_syndrome["脾胃虚弱证"]["status"] == "insufficient_sample"
    assert by_syndrome["脾胃虚弱证"]["base_formula"] is None
    assert "patient_id" not in json.dumps(catalog, ensure_ascii=False)


def test_multidimensional_strata_and_discovery_use_exact_values() -> None:
    records = [
        _record(1, 1, ["白术", "茯苓", "甘草"]),
        _record(2, 2, ["白术", "茯苓", "陈皮"]),
        _record(3, 3, ["白术", "厚朴"], visit_type="复诊"),
    ]
    service = MedicalAnalysis(records)
    dimensions = ["diagnosis_sickness", "diagnosis_disease", "is_first"]
    strata = multidimensional_formula_strata(service, dimensions, minimum_patients=2)

    assert strata["stratum_count"] == 2
    assert strata["items"][0]["dimension_values"] == {
        "diagnosis_sickness": "胃痞病",
        "diagnosis_disease": "肝胃不和证",
        "is_first": "初诊",
    }
    assert strata["items"][0]["status"] == "eligible"
    report = discover_base_formulas_by_dimensions(
        service,
        strata["items"][0]["dimension_values"],
        minimum_support=0.5,
        maximum_pattern_length=4,
        component_count=2,
        bootstrap_rounds=20,
    )
    assert report["scope"]["dimensions"] == dimensions
    assert report["representative_base_formula"]["drugs"] == ["白术", "茯苓"]


def test_dimension_metadata_marks_unavailable_dimensions_disabled() -> None:
    metadata = formula_dimension_metadata(MedicalAnalysis([_record(1, 1, ["白术", "茯苓"])]))
    by_name = {item["name"]: item for item in metadata["dimensions"]}

    assert {name for name, item in by_name.items() if item["supported"]} == {
        "diagnosis_illness",
        "diagnosis_sickness",
        "diagnosis_disease",
        "is_first",
    }
    assert by_name["disease_course"]["supported"] is False
    assert by_name["etiology_pathogenesis"]["reason"] == "暂无结构化数据支持"


def test_base_formula_query_api_returns_ready_and_insufficient_results() -> None:
    records = [
        _record(1, 1, ["白术", "茯苓", "甘草"]),
        _record(2, 2, ["白术", "茯苓", "陈皮"]),
        _record(3, 3, ["白术", "厚朴"], syndrome="脾胃虚弱证"),
    ]
    service = MedicalAnalysis(records)

    class Catalog:
        @staticmethod
        def get(doctor=None):
            return service

    state = SimpleNamespace(catalog=Catalog(), base_formula_cache={})
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    common = {
        "request": request,
        "minimum_patients": 2,
        "minimum_support": 0.5,
        "maximum_pattern_length": 4,
        "pattern_limit": 10,
        "component_count": 2,
        "bootstrap_rounds": 20,
        "visit_type": None,
        "process_type": None,
        "doctor": None,
    }

    ready = base_formula_query(disease="慢性萎缩性胃炎", syndrome="肝胃不和证", **common)
    insufficient = base_formula_query(disease="慢性萎缩性胃炎", syndrome="脾胃虚弱证", **common)

    assert ready["status"] == "ready"
    assert ready["base_formula"]["representative_base_formula"]["drugs"] == ["白术", "茯苓"]
    assert insufficient["status"] == "insufficient_sample"
    assert insufficient["base_formula"] is None
    assert len(state.base_formula_cache) == 1
    assert {route.path for route in app.routes} >= {
        "/api/base-formulas",
        "/api/base-formulas/strata",
        "/api/base-formulas/dimensions",
        "/api/base-formulas/multidimensional/strata",
        "/api/base-formulas/multidimensional",
    }
