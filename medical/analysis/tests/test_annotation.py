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
from pathlib import Path

import pytest
from sqlalchemy import JSON, BigInteger, Column, DateTime, MetaData, String, Table, Text


os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from medical.analysis.annotation import (
    ANNOTATION_FIELD_DEFINITIONS,
    ANNOTATION_FIELDS,
    DEFAULT_ANNOTATION_TABLE,
    EXCLUDED_ANNOTATION_FIELDS,
    MedicalAnnotationRepository,
    MedicalAnnotationService,
)
from medical.data_utils.ai_medical_records import AI_FIELDS
from medical.data_utils.db_manager import DBManager


def record(record_id: int, patient_id: int, order_sn: str, doctor_id: int = 1314) -> dict:
    return {
        "id": record_id + 100,
        "source_record_id": record_id,
        "patient_id": patient_id,
        "order_sn": order_sn,
        "doctor_id": doctor_id,
        "ai_processing_version": "v1",
        "operator": "AI",
        "see_doc_time": f"2026-01-{record_id:02d}",
        "patient_appeal": "原始主诉",
        "doc_ass_stu_appeal": "医生原始主诉",
        "new_medical_history": "原始现病史",
        "ai_patient_appeal": "AI 主诉",
        "ai_new_medical_history": "AI 现病史",
        "ai_key_symptoms": ["乏力"],
    }


def build_annotation_table() -> Table:
    metadata = MetaData()
    originals = {item[2] for item in ANNOTATION_FIELD_DEFINITIONS if item[2]}
    json_fields = {item[0] for item in ANNOTATION_FIELD_DEFINITIONS if item[3] == "json"}
    return Table(
        DEFAULT_ANNOTATION_TABLE,
        metadata,
        Column("id", BigInteger, primary_key=True),
        Column("source_record_id", BigInteger),
        Column("order_sn", String(50), nullable=False),
        Column("ai_processing_version", String(64), nullable=False),
        Column("patient_id", BigInteger),
        Column("doctor_id", BigInteger),
        Column("operator", String(100)),
        Column("status", String(16)),
        Column("label_person", String(100)),
        Column("see_doc_time", String(19)),
        Column("created_at", DateTime),
        Column("is_first", String(10)),
        Column("patient_sex", String(10)),
        Column("patient_age", String(50)),
        Column("admin_report_img", JSON),
        Column("admin_face_img", JSON),
        Column("doc_ass_stu_appeal", Text),
        *(Column(name, JSON if name in {"inspection_report_img", "tongue_face_img"} else Text) for name in originals),
        *(Column(name, JSON if name in json_fields else Text) for name in ANNOTATION_FIELDS),
    )


def service_with_records(*records: dict) -> tuple[MedicalAnnotationService, MedicalAnnotationRepository]:
    manager = DBManager("sqlite:///:memory:")
    table = build_annotation_table()
    manager.create_table(table)
    manager.insert_many(DEFAULT_ANNOTATION_TABLE, records)
    repository = MedicalAnnotationRepository(manager)
    return MedicalAnnotationService(repository), repository


def test_annotation_fields_are_all_ai_fields_except_explicit_exclusions() -> None:
    assert set(ANNOTATION_FIELDS) == set(AI_FIELDS) - EXCLUDED_ANNOTATION_FIELDS
    assert len(ANNOTATION_FIELDS) == 21


def test_patient_list_is_grouped_and_scoped_to_doctor() -> None:
    service, repository = service_with_records(
        record(1, 10, "A1"),
        record(2, 10, "A2"),
        record(3, 20, "B1", doctor_id=43),
    )
    repository.update_ai_fields_by_id(101, {"ai_patient_appeal": "人工主诉"}, allowed_fields=ANNOTATION_FIELDS)

    result = service.patients("1314")

    assert result["items"] == [
        {"patient_id": "10", "record_count": 2, "human_record_count": 1},
    ]


def test_patient_list_supports_progressive_pagination() -> None:
    service, _ = service_with_records(
        record(1, 10, "A1"),
        record(2, 20, "A2"),
        record(3, 30, "A3"),
    )

    first_page = service.patients("1314", limit=2)
    second_page = service.patients("1314", limit=2, offset=2)

    assert [item["patient_id"] for item in first_page["items"]] == ["10", "20"]
    assert first_page["total"] == 3
    assert first_page["has_more"] is True
    assert [item["patient_id"] for item in second_page["items"]] == ["30"]
    assert second_page["has_more"] is False


def test_save_updates_only_ai_columns_and_marks_labeled_by_human() -> None:
    service, repository = service_with_records(record(1, 10, "A1"))

    saved = service.save(
        101,
        {"ai_patient_appeal": "人工审核主诉", "ai_key_symptoms": ["乏力", "纳差"]},
        "1314",
    )
    database_row = repository.get_by_id(101)

    assert saved["operator"] == "HUMAN"
    assert saved["status"] == "labeled"
    assert database_row["operator"] == "HUMAN"
    assert database_row["status"] == "labeled"
    assert database_row["ai_patient_appeal"] == "人工审核主诉"
    assert database_row["ai_key_symptoms"] == ["乏力", "纳差"]
    assert database_row["patient_appeal"] == "原始主诉"
    assert database_row["doc_ass_stu_appeal"] == "医生原始主诉"
    assert database_row["new_medical_history"] == "原始现病史"


def test_patient_record_formats_ai_image_results_for_display() -> None:
    image_record = record(1, 10, "A1")
    image_record.update(
        {
            "tongue_face_img": [{"img": "https://example.com/tongue.jpg", "time": "2026-01-01"}],
            "inspection_report_img": [{"img": "https://example.com/report.jpg"}],
            "ai_tongue_face_img": {
                "tongue": {"legal": "是", "side": "正面", "tongue_body": {"color": "淡红"}},
            },
            "ai_inspection_report_img": {
                "reports": [
                    {
                        "是否有效检查报告": True,
                        "图片类别": "检验报告",
                        "report_name": "血常规",
                        "解析结果": "白细胞升高",
                    }
                ]
            },
        }
    )
    service, _ = service_with_records(image_record)

    result = service.patient_records("1314", 10)["records"][0]

    assert result["original"]["tongue_face_img"][0]["img"].endswith("tongue.jpg")
    assert result["original"]["doc_ass_stu_appeal"] == "医生原始主诉"
    assert "舌面方向：正面" in result["ai_display"]["ai_tongue_face_img"]["tongue"]
    assert "报告名称：血常规" in result["ai_display"]["ai_inspection_report_img"]
    assert "报告异常指标：无异常指标" in result["ai_display"]["ai_inspection_report_img"]
    assert "解析结果" not in result["ai_display"]["ai_inspection_report_img"]


def test_image_annotation_fields_follow_current_history() -> None:
    field_names = [item[0] for item in ANNOTATION_FIELD_DEFINITIONS]

    assert field_names[1:4] == [
        "ai_new_medical_history",
        "ai_inspection_report_img",
        "ai_tongue_face_img",
    ]


def test_original_images_use_first_non_empty_source_field() -> None:
    image_record = record(1, 10, "A1")
    image_record.update(
        {
            "inspection_report_img": [],
            "admin_report_img": [{"img": "https://example.com/admin-report.jpg"}],
            "tongue_face_img": [{"img": "https://example.com/patient-face.jpg"}],
            "admin_face_img": [{"img": "https://example.com/admin-face.jpg"}],
        }
    )
    service, _ = service_with_records(image_record)

    result = service.patient_records("1314", 10)["records"][0]

    assert result["original"]["inspection_report_img"][0]["img"].endswith("admin-report.jpg")
    assert result["original_sources"]["ai_inspection_report_img"] == "admin_report_img"
    assert result["original"]["tongue_face_img"][0]["img"].endswith("patient-face.jpg")
    assert result["original_sources"]["ai_tongue_face_img"] == "tongue_face_img"


def test_save_by_primary_key_does_not_update_another_ai_version() -> None:
    first_version = record(1, 10, "A1")
    second_version = record(2, 10, "A1")
    second_version["ai_processing_version"] = "v2"
    service, repository = service_with_records(first_version, second_version)

    service.save(101, {"ai_patient_appeal": "只审核 v1"}, "1314")

    assert repository.get_by_id(101)["operator"] == "HUMAN"
    assert repository.get_by_id(101)["ai_patient_appeal"] == "只审核 v1"
    assert repository.get_by_id(102)["operator"] == "AI"
    assert repository.get_by_id(102)["ai_patient_appeal"] == "AI 主诉"


def test_save_rejects_excluded_and_original_fields() -> None:
    service, _ = service_with_records(record(1, 10, "A1"))

    with pytest.raises(ValueError, match="不支持修改 AI 字段"):
        service.save(101, {"ai_processing_complete": False}, "1314")
    with pytest.raises(ValueError, match="不支持修改 AI 字段"):
        service.save(101, {"patient_appeal": "不得修改"}, "1314")


def test_annotation_frontend_has_patient_workflow_and_save_endpoint() -> None:
    module_dir = Path(__file__).resolve().parents[1]
    html = (module_dir / "static" / "index.html").read_text(encoding="utf-8")
    script = (module_dir / "static" / "app.js").read_text(encoding="utf-8")

    assert 'data-view="annotation"' in html
    assert 'id="annotation-patient-list"' in html
    assert 'id="annotation-records"' in html
    assert 'id="annotation-load-more"' in html
    assert 'id="annotation-image-dialog"' in html
    assert 'id="annotation-image-previous"' in html
    assert 'id="annotation-image-next"' in html
    assert "原始字段仅供核对" in html
    assert 'api("/api/annotations/save"' in script
    assert 'return `<details class="annotation-record panel"' in script
    assert "annotationRows" in script
    assert "AI 未生成内容，可在此补充…" in script
    assert "annotationOriginalContent" in script
    assert "record.original_sources" in script
    assert "annotationAiPreview" in script
    assert "annotation-json-value-editor" in script
    assert "syncAnnotationJsonPreviewValue" in script
    assert "annotation-key-grid" in script
    assert "data-annotation-json-path" in script
    assert "openAnnotationImagePreview" in script
    assert "stepAnnotationImagePreview" in script
    assert "doc_ass_stu_appeal" in script
    assert "record_id: record.annotation_record_id" in script
    assert "saveAnnotationRecord" in script
