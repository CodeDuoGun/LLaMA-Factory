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

"""病历 AI 字段人工审核的字段定义和数据库查询服务."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from sqlalchemy import Table, text

from medical.data_utils.medical_record_formatters import (
    format_inspection_result_text,
    format_tongue_face_result_descriptions,
)


if TYPE_CHECKING:
    from medical.data_utils.db_manager import DBManager


DEFAULT_ANNOTATION_TABLE = "mlops_annotation_medical_record"


EXCLUDED_ANNOTATION_FIELDS = {
    "ai_complaint_history_issues",
    "ai_complaint_history_consistent",
    "ai_processing_complete",
    "ai_processing_version",
    "ai_record_identity",
}

ANNOTATION_FIELD_DEFINITIONS = (
    ("ai_patient_appeal", "主诉", "patient_appeal", "text"),
    ("ai_new_medical_history", "现病史", "new_medical_history", "long_text"),
    ("ai_inspection_report_img", "检查报告解析", "inspection_report_img", "json"),
    ("ai_tongue_face_img", "舌面图解析", "tongue_face_img", "json"),
    ("ai_old_medical_history", "既往史", "old_medical_history", "long_text"),
    ("ai_allergic_history", "过敏史", "allergic_history", "long_text"),
    ("ai_personal_history", "个人史", "personal_history", "long_text"),
    ("ai_special_history", "特殊史", "special_history", "long_text"),
    ("ai_family_history", "家族史", "family_history", "long_text"),
    ("ai_diagnosis_illness", "西医诊断", "diagnosis_illness", "text"),
    ("ai_diagnosis_illness_reason", "西医诊断理由", None, "long_text"),
    ("ai_diagnosis_sickness", "中医诊断", "diagnosis_sickness", "text"),
    ("ai_diagnosis_sickness_reason", "中医诊断理由", None, "long_text"),
    ("ai_diagnosis_disease", "中医证候", "diagnosis_disease", "text"),
    ("ai_diagnosis_disease_reason", "中医证候理由", None, "long_text"),
    ("ai_disease_stage", "疾病阶段", None, "text"),
    ("ai_disease_course", "疾病病程", None, "long_text"),
    ("ai_etiology", "病因", None, "json"),
    ("ai_pathogenesis", "病机", None, "json"),
    ("ai_disease_location", "病位", None, "json"),
    ("ai_key_symptoms", "关键症状", None, "json"),
)
ANNOTATION_FIELDS = tuple(item[0] for item in ANNOTATION_FIELD_DEFINITIONS)

ORIGINAL_IMAGE_FIELD_PRIORITIES = {
    "ai_inspection_report_img": ("inspection_report_img", "admin_report_img"),
    "ai_tongue_face_img": ("tongue_face_img", "admin_face_img"),
}

ANNOTATION_RECORD_COLUMNS = (
    "id",
    "source_record_id",
    "order_sn",
    "ai_processing_version",
    "patient_id",
    "doctor_id",
    "operator",
    "status",
    "label_person",
    "see_doc_time",
    "created_at",
    "is_first",
    "patient_sex",
    "patient_age",
    "admin_report_img",
    "admin_face_img",
    "doc_ass_stu_appeal",
    *(item[2] for item in ANNOTATION_FIELD_DEFINITIONS if item[2]),
    *ANNOTATION_FIELDS,
)


class MedicalAnnotationRepository:
    """真实标注表的只读检索与 AI 字段安全更新封装."""

    def __init__(self, manager: DBManager, *, table_name: str = DEFAULT_ANNOTATION_TABLE) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
            raise ValueError(f"非法标注表名: {table_name}")
        self.manager = manager
        self.table_name = table_name
        self._table: Table | None = None

    @property
    def table(self) -> Table:
        if self._table is None:
            self._table = self.manager.refresh_table(self.table_name)
        return self._table

    def get_by_patient_id(self, doctor_id: int, patient_id: int, *, limit: int = 500) -> list[dict[str, Any]]:
        return self.manager.select(
            self.table_name,
            filters={"doctor_id": doctor_id, "patient_id": patient_id},
            columns=ANNOTATION_RECORD_COLUMNS,
            limit=limit,
            order_by=["created_at", "id"],
        )

    def get_by_id(self, record_id: int) -> dict[str, Any] | None:
        return self.manager.select_one(
            self.table_name,
            filters={"id": record_id},
            columns=ANNOTATION_RECORD_COLUMNS,
        )

    def list_patients(
        self,
        doctor_id: str,
        *,
        query: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """按患者聚合读取标注进度，避免使用热重载期间可能残留的反射元数据."""
        query_filter = " AND CAST(patient_id AS CHAR) LIKE :query" if query else ""
        parameters: dict[str, Any] = {
            "doctor_id": int(doctor_id),
            "limit": limit,
            "offset": offset,
        }
        if query:
            parameters["query"] = f"%{query}%"
        grouped_sql = f"""
            FROM `{self.table_name}`
            WHERE doctor_id = :doctor_id{query_filter}
            GROUP BY patient_id
        """
        statement = text(
            f"""
            SELECT patient_id,
                   COUNT(*) AS record_count,
                   SUM(CASE WHEN operator = 'HUMAN' THEN 1 ELSE 0 END) AS human_record_count
            {grouped_sql}
            ORDER BY patient_id
            LIMIT :limit OFFSET :offset
            """
        )
        count_statement = text(f"SELECT COUNT(*) FROM (SELECT patient_id {grouped_sql}) AS grouped_patients")
        with self.manager.connection() as connection:
            items = [dict(row) for row in connection.execute(statement, parameters).mappings().all()]
            total = int(connection.execute(count_statement, parameters).scalar_one())
        return items, total

    def update_ai_fields_by_id(
        self,
        record_id: int,
        fields: Mapping[str, Any],
        *,
        allowed_fields: Iterable[str],
    ) -> bool:
        editable_fields = set(allowed_fields)
        invalid_fields = sorted(set(fields) - editable_fields)
        if invalid_fields:
            raise ValueError(f"不支持修改 AI 字段: {', '.join(invalid_fields)}")
        if not fields:
            raise ValueError("至少需要提交一个 AI 字段")
        data = dict(fields)
        data["status"] = "labeled"
        data["operator"] = "HUMAN"
        return self.manager.update(self.table_name, data, filters={"id": record_id}) > 0


class MedicalAnnotationService:
    """按医生、患者组织 AI 清洗病历，供人工逐字段审核."""

    def __init__(self, repository: MedicalAnnotationRepository) -> None:
        self.repository = repository

    def field_metadata(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "label": label, "original_field": original_field, "kind": kind}
            for name, label, original_field, kind in ANNOTATION_FIELD_DEFINITIONS
        ]

    def patients(self, doctor_id: str, *, query: str = "", limit: int = 200, offset: int = 0) -> dict[str, Any]:
        items, total = self.repository.list_patients(
            doctor_id,
            query=query,
            limit=limit,
            offset=offset,
        )
        for item in items:
            item["patient_id"] = str(item["patient_id"])
            item["record_count"] = int(item["record_count"] or 0)
            item["human_record_count"] = int(item["human_record_count"] or 0)
        return {
            "items": items,
            "count": len(items),
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

    def patient_records(self, doctor_id: str, patient_id: int) -> dict[str, Any]:
        rows = self.repository.get_by_patient_id(int(doctor_id), patient_id, limit=500)
        records = [self._serialize_record(row) for row in rows]
        if not records:
            raise KeyError(patient_id)
        records.sort(key=lambda row: (row["visit_date"], row["medical_record_id"]))
        return {"patient_id": str(patient_id), "record_count": len(records), "records": records}

    def save(self, record_id: int, fields: dict[str, Any], doctor_id: str) -> dict[str, Any]:
        row = self.repository.get_by_id(record_id)
        if row is None or not self._belongs_to_doctor(row, doctor_id):
            raise KeyError(record_id)
        updated = self.repository.update_ai_fields_by_id(
            record_id,
            fields,
            allowed_fields=ANNOTATION_FIELDS,
        )
        if not updated:
            raise KeyError(record_id)
        saved = self.repository.get_by_id(record_id)
        return self._serialize_record(saved)

    @staticmethod
    def _belongs_to_doctor(row: dict[str, Any], doctor_id: str) -> bool:
        return str(row.get("doctor_id", "")) == str(doctor_id)

    @staticmethod
    def _has_original_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"", "null", "[]", "{}"}
        if isinstance(value, (list, dict, tuple, set)):
            return bool(value)
        return True

    def _serialize_record(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            tongue_face_descriptions = format_tongue_face_result_descriptions(row.get("ai_tongue_face_img"))
        except (TypeError, ValueError):
            tongue_face_descriptions = {}
        try:
            inspection_description = format_inspection_result_text(row.get("ai_inspection_report_img"))
        except (TypeError, ValueError):
            inspection_description = ""
        original = {field[2]: row.get(field[2]) for field in ANNOTATION_FIELD_DEFINITIONS if field[2]}
        original["doc_ass_stu_appeal"] = row.get("doc_ass_stu_appeal")
        original_sources = {}
        for ai_field, source_fields in ORIGINAL_IMAGE_FIELD_PRIORITIES.items():
            selected_source = next(
                (source_field for source_field in source_fields if self._has_original_value(row.get(source_field))),
                source_fields[0],
            )
            original[source_fields[0]] = row.get(selected_source)
            original_sources[ai_field] = selected_source
        return {
            "annotation_record_id": row["id"],
            "medical_record_id": row.get("source_record_id"),
            "patient_id": str(row["patient_id"]),
            "order_sn": row["order_sn"],
            "ai_processing_version": row.get("ai_processing_version"),
            "operator": row["operator"],
            "status": row.get("status"),
            "label_person": row.get("label_person"),
            "visit_date": str(row.get("see_doc_time") or row.get("created_at") or ""),
            "is_first": row.get("is_first"),
            "patient_sex": row.get("patient_sex"),
            "patient_age": row.get("patient_age"),
            "original": original,
            "original_sources": original_sources,
            "ai": {field_name: row.get(field_name) for field_name in ANNOTATION_FIELDS},
            "ai_display": {
                "ai_tongue_face_img": tongue_face_descriptions,
                "ai_inspection_report_img": inspection_description,
            },
        }
