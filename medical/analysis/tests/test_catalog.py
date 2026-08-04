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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


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


def test_catalog_discovers_flat_online_doctor_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "甲医生_AI医生分身混合问诊数据_1_20260101000000.json",
        [_record(1, 1, "甲医生")],
    )
    _write(
        tmp_path / "乙医生_AI医生分身混合问诊数据_2_20260101000000.json",
        [_record(2, 2, "乙医生")],
    )
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "._甲医生.json").write_bytes(b"AppleDouble")

    catalog = DoctorCatalog.discover(tmp_path, default_doctor="2")
    doctors = catalog.list_doctors()

    assert doctors["default_doctor"] == "2"
    assert {item["key"] for item in doctors["items"]} == {"1", "2"}
    assert {item["doctor_name"] for item in doctors["items"]} == {"甲医生", "乙医生"}
    assert catalog.get("1").raw_record_count == 1
    assert catalog.get("2").raw_record_count == 1


def test_catalog_discovers_only_normalized_processed_doctor_jsonl(tmp_path: Path) -> None:
    doctor_dir = tmp_path / "doctor_43_朱子奇"
    _write(tmp_path / "filter.json", [_record(99, 99, "非医生辅助数据")])
    _write_jsonl(doctor_dir / "medical_records_ai.jsonl", [_record(1, 43, "朱子奇")])
    normalized_record = _record(2, 43, "朱子奇")
    normalized_record["ai_diagnosis_illness"] = "胃癌"
    _write_jsonl(doctor_dir / "medical_records_ai_normalized.jsonl", [normalized_record])

    catalog = DoctorCatalog.discover(tmp_path, default_doctor="43")
    doctors = catalog.list_doctors()

    assert doctors["default_doctor"] == "43"
    assert doctors["items"] == [
        {
            "key": "43",
            "doctor_name": "朱子奇",
            "doctor_id": "43",
            "file_count": 1,
            "record_count": None,
            "loaded": False,
            "duplicate_records": 0,
        }
    ]
    assert catalog.sources["43"].files == [doctor_dir / "medical_records_ai_normalized.jsonl"]
    assert catalog.get("43").raw_record_count == 1
