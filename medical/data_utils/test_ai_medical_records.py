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

from sqlalchemy import Column, Integer, MetaData, Table


os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from medical.data_utils.ai_medical_records import (  # noqa: E402
    DEFAULT_TABLE_NAME,
    AIMedicalRecordRepository,
    add_column_with_type_default,
    default_value_for_column_type,
    drop_column,
    record_to_row,
)
from medical.data_utils.db_manager import DBManager  # noqa: E402
from medical.data_utils.import_ai_medical_records import DEFAULT_INPUT, build_parser  # noqa: E402


def _record(
    record_id: int = 13031,
    order_sn: str = "A00013035",
    doctor_id: int = 43,
    appeal: str = "胃痛",
    version: str = "v1",
) -> dict:
    return {
        "id": record_id,
        "order_sn": order_sn,
        "doctor_id": doctor_id,
        "patient_id": 9724,
        "patient_appeal": appeal,
        "ps": [{"name": "处方一"}],
        "ai_new_medical_history": "胃脘隐痛",
        "ai_inspection_report_img": {"reports": []},
        "ai_complaint_history_consistent": True,
        "ai_processing_complete": True,
        "ai_processing_version": version,
        "ai_treatment_principle": "清热利湿",
    }


def test_record_to_row_maps_flat_fields_and_treatment_principle() -> None:
    row = record_to_row(_record())

    assert row["source_record_id"] == 13031
    assert row["patient_id"] == 9724
    assert row["order_sn"] == "A00013035"
    assert row["ps"] == [{"name": "处方一"}]
    assert row["patient_appeal"] == "胃痛"
    assert row["ai_new_medical_history"] == "胃脘隐痛"
    assert row["treatment_principle"] == "清热利湿"
    assert row["operator"] == "AI"


def test_add_column_uses_type_default_for_existing_rows() -> None:
    manager = DBManager("sqlite+pysqlite:///:memory:")
    table = Table("records", MetaData(), Column("id", Integer, primary_key=True))
    manager.create_table(table)
    manager.insert("records", {"id": 1})

    assert default_value_for_column_type("string") == ""
    assert add_column_with_type_default(manager, "records", "treatment_principle", "string") is True
    assert add_column_with_type_default(manager, "records", "treatment_principle", "string") is False
    assert manager.select_one("records", filters={"id": 1})["treatment_principle"] == ""

    assert drop_column(manager, "records", "treatment_principle") is True
    assert drop_column(manager, "records", "treatment_principle") is False


def test_repository_upserts_by_order_sn_and_deletes_doctor_rows() -> None:
    manager = DBManager("sqlite+pysqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()

    assert repository.batch_upsert_by_order_sn(
        [_record(), _record(13032, "A00013036", 52, "头痛")]
    ) == 2
    assert repository.batch_upsert_by_order_sn([_record(13033, appeal="胃痛加重", version="v2")]) == 1

    rows = manager.select(repository.table.name, filters={"order_sn": "A00013035"})
    assert len(rows) == 1
    assert rows[0]["source_record_id"] == 13033
    assert rows[0]["patient_appeal"] == "胃痛加重"
    assert rows[0]["ai_processing_version"] == "v2"

    assert repository.delete_by_doctor_id(43) == 1
    assert repository.get_by_order_sn("A00013035") is None
    assert repository.get_by_order_sn("A00013036") is not None


def test_repository_does_not_update_human_labeled_record() -> None:
    manager = DBManager("sqlite+pysqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()
    manager.insert(
        repository.table.name,
        {
            "order_sn": "A00013035",
            "ai_processing_version": "v1",
            "patient_appeal": "人工标注的主诉",
            "treatment_principle": "",
            "status": "labeled",
            "operator": "HUMAN",
        },
    )

    assert repository.batch_upsert_by_order_sn([_record(appeal="AI 更新的主诉")]) == 0
    row = repository.get_by_order_sn("A00013035")

    assert row["patient_appeal"] == "人工标注的主诉"
    assert row["status"] == "labeled"
    assert row["operator"] == "HUMAN"


def test_repository_updates_field_and_marks_operator_as_human() -> None:
    manager = DBManager("sqlite+pysqlite:///:memory:")
    repository = AIMedicalRecordRepository(manager)
    repository.create_table()
    repository.batch_upsert_by_order_sn([_record()])

    assert repository.update_field_by_order_sn("A00013035", "patient_appeal", "人工修改后的主诉") is True
    row = repository.get_by_order_sn("A00013035")

    assert row["patient_appeal"] == "人工修改后的主诉"
    assert row["operator"] == "HUMAN"


def test_import_script_commands() -> None:
    parser = build_parser()
    import_args = parser.parse_args(["import"])
    add_column_args = parser.parse_args(["add-column", "treatment_principle", "--type", "string"])
    delete_args = parser.parse_args(["delete-doctor", "1314", "--yes"])

    assert DEFAULT_INPUT.as_posix().endswith("medical/processed_data/doctor_43_朱子奇/medical_records_ai.jsonl")
    assert import_args.input == DEFAULT_INPUT
    assert import_args.table_name == DEFAULT_TABLE_NAME
    assert add_column_args.column_name == "treatment_principle"
    assert add_column_args.column_type == "string"
    assert delete_args.doctor_id == 1314
    assert delete_args.yes is True
