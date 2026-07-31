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
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime, Index, MetaData, String, Table, Text, func
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from medical.data_utils.db_manager import DBManager


DEFAULT_TABLE_NAME = "ai_medical_records"
DEFAULT_BATCH_SIZE = 100

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
    """构建 AI 清洗病历表定义."""
    metadata = metadata or MetaData()
    table = Table(
        table_name,
        metadata,
        Column("medical_record_id", BigInteger, primary_key=True, comment="病历 ID，对应源数据 id"),
        Column("patient_id", BigInteger, nullable=False, comment="患者 ID，对应源数据 patient_id"),
        Column("inquiry_id", String(64), nullable=False, comment="问诊单 ID，对应源数据 order_sn"),
        *(Column(name, Text) for name in AI_TEXT_FIELDS),
        *(Column(name, JSON) for name in AI_JSON_FIELDS),
        *(Column(name, Boolean) for name in AI_BOOLEAN_FIELDS),
        *(Column(name, String(32)) for name in AI_STRING_FIELDS),
        Column("ai_data", JSON, nullable=False, comment="完整 ai_* 字段，兼容后续新增字段"),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
        comment="AI 清洗后的病历内容",
    )
    Index(f"ix_{table_name}_patient_id", table.c.patient_id)
    Index(f"ix_{table_name}_inquiry_id", table.c.inquiry_id)
    Index(f"ix_{table_name}_ai_record_identity", table.c.ai_record_identity)
    return table


def record_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """把一条 JSONL 病历转换为数据库行."""
    medical_record_id = record.get("medical_record_id") or record.get("record_id") or record.get("id")
    patient_id = record.get("patient_id")
    inquiry_id = record.get("inquiry_id") or record.get("order_sn")
    missing = [
        name
        for name, value in (
            ("medical_record_id/id", medical_record_id),
            ("patient_id", patient_id),
            ("inquiry_id/order_sn", inquiry_id),
        )
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"病历缺少必填字段: {', '.join(missing)}")

    ai_data = {key: value for key, value in record.items() if key.startswith("ai_")}
    row = {
        "medical_record_id": int(medical_record_id),
        "patient_id": int(patient_id),
        "inquiry_id": str(inquiry_id),
        "ai_data": ai_data,
    }
    row.update({name: record.get(name) for name in AI_FIELDS})
    return row


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

    def __init__(self, manager: DBManager, *, table_name: str = DEFAULT_TABLE_NAME) -> None:
        self.manager = manager
        self.table = build_ai_medical_records_table(table_name=table_name)

    def create_table(self) -> None:
        """不存在时创建数据表."""
        self.manager.create_table(self.table, checkfirst=True)

    def batch_upsert(self, records: Iterable[Mapping[str, Any]], *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """分批新增或更新病历，返回处理条数."""
        total = 0
        rows = (record_to_row(record) for record in records)
        for batch in _batched(rows, batch_size):
            self._upsert_batch(batch)
            total += len(batch)
        return total

    def import_jsonl(self, path: str | Path, *, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """创建表并批量导入 JSONL."""
        self.create_table()
        return self.batch_upsert(iter_jsonl(path), batch_size=batch_size)

    def get_by_patient_id(self, patient_id: int, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """按患者 ID 查询病历列表."""
        return self.manager.select(
            self.table.name,
            filters={"patient_id": patient_id},
            limit=limit,
            offset=offset,
            order_by=["medical_record_id"],
        )

    def get_by_inquiry_id(self, inquiry_id: str) -> list[dict[str, Any]]:
        """按问诊单 ID 查询病历列表."""
        return self.manager.select(
            self.table.name,
            filters={"inquiry_id": inquiry_id},
            order_by=["medical_record_id"],
        )

    def get_by_medical_record_id(self, medical_record_id: int) -> dict[str, Any] | None:
        """按病历 ID 查询单条病历."""
        return self.manager.select_one(self.table.name, filters={"medical_record_id": medical_record_id})

    def _upsert_batch(self, rows: list[dict[str, Any]]) -> None:
        dialect = self.manager.engine.dialect.name
        if dialect == "mysql":
            statement = mysql_insert(self.table).values(rows)
            update_values = {
                column.name: getattr(statement.inserted, column.name)
                for column in self.table.columns
                if column.name not in {"medical_record_id", "created_at"}
            }
            update_values["updated_at"] = func.now()
            statement = statement.on_duplicate_key_update(**update_values)
        elif dialect == "sqlite":
            statement = sqlite_insert(self.table).values(rows)
            update_values = {
                column.name: getattr(statement.excluded, column.name)
                for column in self.table.columns
                if column.name not in {"medical_record_id", "created_at"}
            }
            update_values["updated_at"] = datetime.now()
            statement = statement.on_conflict_do_update(
                index_elements=[self.table.c.medical_record_id],
                set_=update_values,
            )
        else:
            raise NotImplementedError(f"暂不支持 {dialect} 数据库的批量 upsert")

        with self.manager.connection(transactional=True) as connection:
            connection.execute(statement)
