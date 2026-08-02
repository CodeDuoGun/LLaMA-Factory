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

"""AI 清洗病历的数据表、批量写入和检索服务."""

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime, Index, MetaData, String, Table, Text, UniqueConstraint, func, insert


if TYPE_CHECKING:
    from medical.data_utils.db_manager import DBManager


DEFAULT_TABLE_NAME = "ai_cleaned_medical_records"
DEFAULT_BATCH_SIZE = 100
DEFAULT_AI_OPERATOR = "AI"
DEFAULT_HUMAN_OPERATOR = "HUMAN"

AI_TEXT_FIELDS = (
    "ai_new_medical_history",
    "ai_patient_appeal",
    "ai_old_medical_history",
    "ai_allergic_history",
    "ai_personal_history",
    "ai_special_history",
    "ai_family_history",
    "ai_diagnosis_illness",
    "ai_diagnosis_illness_reason",
    "ai_diagnosis_disease",
    "ai_diagnosis_disease_reason",
    "ai_diagnosis_sickness",
    "ai_diagnosis_sickness_reason",
    "ai_disease_stage",
    "ai_disease_course",
    "ai_record_identity",
)
AI_JSON_FIELDS = (
    "ai_inspection_report_img",
    "ai_tongue_face_img",
    "ai_complaint_history_issues",
    "ai_etiology",
    "ai_pathogenesis",
    "ai_disease_location",
    "ai_key_symptoms",
)
AI_BOOLEAN_FIELDS = (
    "ai_complaint_history_consistent",
    "ai_processing_complete",
)
AI_STRING_FIELDS = ("ai_processing_version",)
AI_FIELDS = AI_TEXT_FIELDS + AI_JSON_FIELDS + AI_BOOLEAN_FIELDS + AI_STRING_FIELDS


def build_ai_medical_records_table(metadata: MetaData | None = None, *, table_name: str = DEFAULT_TABLE_NAME) -> Table:
    """构建 AI 清洗病历表定义，一条 JSONL 记录对应一行数据."""
    metadata = metadata or MetaData()
    table = Table(
        table_name,
        metadata,
        Column("medical_record_id", BigInteger, primary_key=True, comment="病历 ID，对应源数据 id"),
        Column("patient_id", BigInteger, nullable=False, comment="患者 ID，对应源数据 patient_id"),
        Column("order_sn", String(64), nullable=False, comment="咨询单 ID，对应源数据 order_sn"),
        *(Column(name, Text) for name in AI_TEXT_FIELDS),
        *(Column(name, JSON) for name in AI_JSON_FIELDS),
        *(Column(name, Boolean) for name in AI_BOOLEAN_FIELDS),
        *(Column(name, String(32)) for name in AI_STRING_FIELDS),
        Column("record_data", JSON, nullable=False, comment="除 ps 和 ai_* 字段外的原始病历记录"),
        Column("operator", String(16), nullable=False, comment="数据操作来源：AI 或 HUMAN"),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
        UniqueConstraint("order_sn", name=f"uq_{table_name}_order_sn"),
        comment="AI 清洗后的病历记录，不包含处方 ps",
    )
    Index(f"ix_{table_name}_patient_id", table.c.patient_id)
    Index(f"ix_{table_name}_order_sn", table.c.order_sn)
    return table


def _extract_record_ids(record: Mapping[str, Any]) -> tuple[int, int, str]:
    medical_record_id = record.get("medical_record_id") or record.get("record_id") or record.get("id")
    patient_id = record.get("patient_id")
    order_sn = record.get("order_sn") or record.get("inquiry_id")
    missing = [
        name
        for name, value in (
            ("medical_record_id/id", medical_record_id),
            ("patient_id", patient_id),
            ("order_sn/inquiry_id", order_sn),
        )
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"病历缺少必填字段: {', '.join(missing)}")
    return int(medical_record_id), int(patient_id), str(order_sn)


def record_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """把一条 JSONL 病历转换为一条数据库行，排除 ps 字段."""
    medical_record_id, patient_id, order_sn = _extract_record_ids(record)
    return {
        "medical_record_id": medical_record_id,
        "patient_id": patient_id,
        "order_sn": order_sn,
        "record_data": {
            field_name: field_value
            for field_name, field_value in record.items()
            if field_name != "ps" and field_name not in AI_FIELDS
        },
        "operator": DEFAULT_AI_OPERATOR,
        **{field_name: record.get(field_name) for field_name in AI_FIELDS},
    }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL，并在错误中标记行号."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            yield value


def _batched(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class AIMedicalRecordRepository:
    """AI 清洗病历的批量写入及索引检索服务."""

    def __init__(self, manager: "DBManager", *, table_name: str = DEFAULT_TABLE_NAME) -> None:
        self.manager = manager
        self.table = build_ai_medical_records_table(table_name=table_name)

    def create_table(self) -> None:
        """不存在时创建数据表."""
        self.manager.create_table(self.table, checkfirst=True)

    def batch_insert_missing(self, records: Iterable[Mapping[str, Any]], *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """分批新增病历；order_sn 已存在的数据会跳过，返回新增病历数."""
        total = 0
        for batch in _batched((record_to_row(record) for record in records), batch_size):
            rows = self._exclude_existing_rows(batch)
            if not rows:
                continue
            with self.manager.connection(transactional=True) as connection:
                connection.execute(insert(self.table), rows)
            total += len(rows)
        return total

    def import_jsonl(self, path: str | Path, *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """创建表并批量导入 JSONL，已存在 order_sn 的记录会跳过."""
        self.create_table()
        return self.batch_insert_missing(iter_jsonl(path), batch_size=batch_size)

    def clear(self) -> int:
        """清空数据表，返回删除行数."""
        self.create_table()
        return self.manager.delete(self.table.name, filters=None, allow_all=True)

    def get_by_patient_id(self, patient_id: int, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """按患者 ID 查询病历列表."""
        return self.manager.select(
            self.table.name,
            filters={"patient_id": patient_id},
            limit=limit,
            offset=offset,
            order_by=["medical_record_id"],
        )

    def get_by_order_sn(self, order_sn: str) -> dict[str, Any] | None:
        """按咨询单 ID 查询单条病历."""
        return self.manager.select_one(self.table.name, filters={"order_sn": order_sn})

    def get_by_inquiry_id(self, inquiry_id: str) -> dict[str, Any] | None:
        """按咨询单 ID 查询单条病历，兼容旧方法名."""
        return self.get_by_order_sn(inquiry_id)

    def get_by_medical_record_id(self, medical_record_id: int) -> dict[str, Any] | None:
        """按病历 ID 查询单条病历."""
        return self.manager.select_one(self.table.name, filters={"medical_record_id": medical_record_id})

    def update_field_by_order_sn(self, order_sn: str, field_name: str, field_value: Any) -> bool:
        """更新单个字段，并把整条记录来源标记为 HUMAN.

        AI_FIELDS 中的字段更新独立列；其他业务字段更新 record_data。
        """
        row = self.get_by_order_sn(order_sn)
        if row is None:
            return False
        if field_name in {"medical_record_id", "order_sn", "record_data", "operator", "created_at", "updated_at"}:
            raise ValueError(f"不支持通过 update_field_by_order_sn 修改字段: {field_name}")

        data: dict[str, Any] = {"operator": DEFAULT_HUMAN_OPERATOR}
        if field_name in AI_FIELDS:
            data[field_name] = field_value
        elif field_name == "patient_id":
            data["patient_id"] = int(field_value)
            record_data = dict(row.get("record_data") or {})
            record_data[field_name] = field_value
            data["record_data"] = record_data
        else:
            record_data = dict(row.get("record_data") or {})
            record_data[field_name] = field_value
            data["record_data"] = record_data

        return self.manager.update(self.table.name, data, filters={"order_sn": order_sn}) > 0

    def _exclude_existing_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_rows: list[dict[str, Any]] = []
        seen_order_sn: set[str] = set()
        for row in rows:
            order_sn = row["order_sn"]
            if order_sn in seen_order_sn:
                continue
            seen_order_sn.add(order_sn)
            unique_rows.append(row)

        existing_rows = self.manager.select(
            self.table.name,
            filters={"order_sn": list(seen_order_sn)},
            columns=["order_sn"],
        )
        existing_order_sn = {row["order_sn"] for row in existing_rows}
        return [row for row in unique_rows if row["order_sn"] not in existing_order_sn]
