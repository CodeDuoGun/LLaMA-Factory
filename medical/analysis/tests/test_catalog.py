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

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from medical.analysis.catalog import DoctorCatalog


def _record(record_id: int, doctor_id: int, doctor_name: str) -> dict:
    return {
        "id": record_id,
        "patient_id": record_id,
        "doctor_id": doctor_id,
        "doctor_name": doctor_name,
        "start_time": "2026-01-01 09:00:00",
        "is_first": "初诊",
        "diagnosis_illness": "肺癌",
        "new_medical_history": "乏力",
        "ps": [],
    }


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def test_catalog_discovers_doctors_and_merges_supplements(tmp_path: Path) -> None:
    _write(
        tmp_path / "online_alpha" / "甲医生_AI医生分身混合问诊数据_1_20260101000000.json", [_record(1, 1, "甲医生")]
    )
    _write(
        tmp_path / "online_alpha" / "后续补充文件.json",
        [_record(1, 1, "甲医生"), _record(2, 1, "甲医生")],
    )
    _write(tmp_path / "online_beta" / "乙医生_AI医生分身混合问诊数据_2_20260101000000.json", [_record(3, 2, "乙医生")])
    wuweiping = _record(4, 0, "")
    wuweiping["doctor_id"] = None
    _write(tmp_path / "zongyuan_wuweiping" / "wuweiping_record_20260525.json", [wuweiping])

    catalog = DoctorCatalog.discover(tmp_path, default_doctor="alpha")
    doctors = catalog.list_doctors()
    alpha = catalog.get("alpha")
    beta = catalog.get("beta")

    assert doctors["default_doctor"] == "alpha"
    assert {item["key"] for item in doctors["items"]} == {"alpha", "beta", "wuweiping"}
    assert next(item["doctor_name"] for item in doctors["items"] if item["key"] == "alpha") == "甲医生"
    assert next(item["doctor_name"] for item in doctors["items"] if item["key"] == "wuweiping") == "吴卫平"
    assert alpha.raw_record_count == 2
    assert beta.raw_record_count == 1
    assert catalog.sources["alpha"].duplicate_records == 1
    assert next(iter(alpha.patient_visits)) != next(iter(beta.patient_visits))
