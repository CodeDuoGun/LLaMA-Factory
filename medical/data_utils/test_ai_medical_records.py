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

import os


os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from medical.data_utils.ai_medical_records import AIMedicalRecordRepository, record_to_row  # noqa: E402
from medical.data_utils.db_manager import DBManager  # noqa: E402


def _record() -> dict:
    return {
        "id": 13031,
        "order_sn": "A00013035",
        "patient_id": 9724,
        "patient_appeal": "胃痛",
        "ps": [{"name": "处方一"}],
        "ai_new_medical_history": "胃脘隐痛",
        "ai_inspection_report_img": {"reports": []},
        "ai_complaint_history_consistent": True,
        "ai_processing_complete": True,
        "ai_processing_version": "v1",
    }


def test_record_to_row_excludes_ps_and_sets_operator() -> None:
    row = record_to_row(_record())

    assert row["medical_record_id"] == 13031
    assert row["patient_id"] == 9724
    assert row["order_sn"] == "A00013035"
    assert "ps" not in row["record_data"]
    assert row["operator"] == "AI"
    assert row["record_data"]["patient_appeal"] == "胃痛"
    assert "ai_new_medical_history" not in row["record_data"]
    assert row["ai_new_medical_history"] == "胃脘隐痛"


def test_repository_imports_record_without_ps_and_skips_existing_order_sn() -> None:
    manager = DBManager("sqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()

    duplicate_record = _record() | {"id": 13032, "patient_appeal": "重复咨询单"}
    first_count = repository.batch_insert_missing([_record(), duplicate_record])
    second_count = repository.batch_insert_missing([_record()])
    row = repository.get_by_medical_record_id(13031)
    inquiry_row = repository.get_by_inquiry_id("A00013035")

    assert first_count == 1
    assert second_count == 0
    assert row["order_sn"] == "A00013035"
    assert row["record_data"]["patient_appeal"] == "胃痛"
    assert row["operator"] == "AI"
    assert "ai_new_medical_history" not in row["record_data"]
    assert row["ai_new_medical_history"] == "胃脘隐痛"
    assert "ps" not in row["record_data"]
    assert inquiry_row["medical_record_id"] == 13031


def test_repository_updates_field_and_marks_operator_as_human() -> None:
    manager = DBManager("sqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()
    repository.batch_insert_missing([_record()])

    updated_admin_field = repository.update_field_by_order_sn("A00013035", "patient_appeal", "人工修改后的主诉")
    updated_ai_field = repository.update_field_by_order_sn(
        "A00013035",
        "ai_new_medical_history",
        "人工修改后的现病史",
    )
    row = repository.get_by_order_sn("A00013035")

    assert updated_admin_field is True
    assert updated_ai_field is True
    assert row["record_data"]["patient_appeal"] == "人工修改后的主诉"
    assert row["ai_new_medical_history"] == "人工修改后的现病史"
    assert row["operator"] == "HUMAN"


def test_repository_clear_removes_all_rows() -> None:
    manager = DBManager("sqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()
    repository.batch_insert_missing([_record()])

    deleted_count = repository.clear()

    assert deleted_count == 1
    assert repository.get_by_order_sn("A00013035") is None


def test_import_script_defaults_to_doctor_records() -> None:
    from medical.data_utils.import_ai_medical_records import DEFAULT_INPUT, build_parser

    parser = build_parser()
    import_args = parser.parse_args(["import"])
    clear_args = parser.parse_args(["clear", "--yes"])
    record_args = parser.parse_args(["--table-name", "records", "record", "13031"])

    assert DEFAULT_INPUT.as_posix().endswith("medical/processed_data/doctor_43_朱子奇/medical_records_ai.jsonl")
    assert import_args.input == DEFAULT_INPUT
    assert import_args.table_name == "ai_cleaned_medical_records"
    assert clear_args.command == "clear"
    assert clear_args.yes is True
    assert record_args.command == "record"
    assert record_args.medical_record_id == 13031
    assert record_args.table_name == "records"
