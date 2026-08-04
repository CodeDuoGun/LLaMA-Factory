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

"""AI 病例标注记录的数据表、批量写入和检索服务."""

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    insert,
)


if TYPE_CHECKING:
    from medical.data_utils.db_manager import DBManager


DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"
DEFAULT_BATCH_SIZE = 100
DEFAULT_AI_OPERATOR = "AI"
DEFAULT_HUMAN_OPERATOR = "HUMAN"
DEFAULT_AI_PROCESSING_VERSION = "2026-07-28"
DEFAULT_STATUS = "unprocessed"

ORIGINAL_TEXT_FIELDS = (
    "patient_appeal",
    "doc_ass_stu_appeal",
    "new_medical_history",
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "birth_detail",
    "family_history",
    "admin_face_describe",
    "diagnosis_illness",
    "diagnosis_disease",
    "diagnosis_sickness",
    "disposition",
)
ORIGINAL_STRING_FIELDS = (
    "order_sn",
    "inquiry_method",
    "is_first",
    "patient_name",
    "patient_sex",
    "patient_age",
    "patient_idcard",
    "patient_mobile",
    "is_old_medical_history",
    "is_allergic_history",
    "is_personal_history",
    "is_special",
    "is_child_history",
    "is_marriage_history",
    "is_family_history",
    "see_doc_time",
    "doctor_name",
    "sampling_type",
)
ORIGINAL_BIGINT_FIELDS = (
    "source_record_id",
    "user_info_id",
    "patient_id",
    "doctor_id",
)
ORIGINAL_NUMERIC_FIELDS = (
    "patient_height",
    "patient_weight",
)
ORIGINAL_DATETIME_FIELDS = (
    "created_at",
    "start_time",
    "stop_time",
)
ORIGINAL_JSON_FIELDS = (
    "inspection_report_img",
    "tongue_face_img",
    "admin_report_img",
    "admin_face_img",
    "ps",
    "im_msg",
)
ORIGINAL_FIELDS = (
    ORIGINAL_STRING_FIELDS
    + ORIGINAL_BIGINT_FIELDS
    + ORIGINAL_NUMERIC_FIELDS
    + ORIGINAL_DATETIME_FIELDS
    + ORIGINAL_TEXT_FIELDS
    + ORIGINAL_JSON_FIELDS
)

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


def _bigint_type() -> BigInteger:
    return BigInteger().with_variant(Integer, "sqlite")


def build_ai_medical_records_table(metadata: MetaData | None = None, *, table_name: str = DEFAULT_TABLE_NAME) -> Table:
    """构建 AI 病例标注表定义，一条 JSONL 记录对应一行数据."""
    metadata = metadata or MetaData()
    table = Table(
        table_name,
        metadata,
        Column("id", _bigint_type(), primary_key=True, autoincrement=True, comment="主键 ID"),
        Column("source_record_id", _bigint_type(), comment="来源电子病历记录 ID"),
        Column("order_sn", String(50), nullable=False, comment="订单编号；与 AI 处理版本组成唯一标识"),
        Column("inquiry_method", String(20), comment="问诊方式（文件中为中文名称）"),
        Column("is_first", String(10), comment="初复诊类型（文件中为初诊或复诊）"),
        Column("created_at", DateTime, comment="挂号订单创建时间"),
        Column("start_time", DateTime, comment="问诊开始时间"),
        Column("stop_time", DateTime, comment="问诊结束时间"),
        Column("user_info_id", _bigint_type(), comment="就诊用户信息 ID【xlys_passport_user_info.id】"),
        Column("patient_id", _bigint_type(), comment="就诊人 ID【xlys_passport_patient.id】"),
        Column("patient_name", String(255), comment="就诊人姓名"),
        Column("patient_sex", String(10), comment="就诊人性别（文件中为中文名称）"),
        Column("patient_age", String(50), comment="就诊人年龄"),
        Column("patient_idcard", String(50), comment="就诊人身份证号"),
        Column("patient_mobile", String(50), comment="就诊人手机号"),
        Column("patient_height", Numeric(10, 2), comment="就诊人身高"),
        Column("patient_weight", Numeric(10, 2), comment="就诊人体重"),
        Column("patient_appeal", Text, comment="就诊人主诉"),
        Column("doc_ass_stu_appeal", Text, comment="医生、医助或学生撰写的主诉"),
        Column("new_medical_history", Text, comment="现病史"),
        Column("is_old_medical_history", String(10), comment="是否存在既往病史"),
        Column("old_medical_history", Text, comment="既往史"),
        Column("is_allergic_history", String(10), comment="是否存在过敏史"),
        Column("allergic_history", Text, comment="过敏史"),
        Column("is_personal_history", String(10), comment="是否存在个人史"),
        Column("personal_history", Text, comment="个人史"),
        Column("is_special", String(10), comment="是否处于特殊时期"),
        Column("special_history", Text, comment="特殊时期内容"),
        Column("is_child_history", String(10), comment="生育史（已育或未育）"),
        Column("birth_detail", Text, comment="生育详情（子女人数和状况）"),
        Column("is_marriage_history", String(10), comment="婚姻史（已婚或未婚）"),
        Column("is_family_history", String(10), comment="是否存在家族史"),
        Column("family_history", Text, comment="家族史"),
        Column("inspection_report_img", JSON, comment="患者上传检查资料 JSON（不可修改）"),
        Column("tongue_face_img", JSON, comment="患者上传舌照或面照 JSON（不可修改）"),
        Column("admin_report_img", JSON, comment="大后台或医助上传检查资料 JSON（可修改）"),
        Column("admin_face_img", JSON, comment="大后台或医助上传舌照或面照 JSON（可修改）"),
        Column("admin_face_describe", Text, comment="舌象面部表现"),
        Column("diagnosis_illness", Text, comment="中医辨病或西医诊断"),
        Column("diagnosis_disease", Text, comment="中医辨证"),
        Column("diagnosis_sickness", Text, comment="中医病名"),
        Column("disposition", Text, comment="处置或处理意见"),
        Column("see_doc_time", String(19), comment="就诊时间原始文本（保留 MySQL 零日期）"),
        Column("doctor_id", _bigint_type(), comment="医生 ID【xlys_doctor.id】"),
        Column("doctor_name", String(255), comment="医生姓名"),
        Column("ps", JSON, comment="有效处方列表 JSON"),
        Column("im_msg", JSON, comment="问诊消息列表 JSON"),
        Column("sampling_type", String(16), comment="数据抽样类型"),
        *(Column(name, Text) for name in AI_TEXT_FIELDS),
        *(Column(name, JSON) for name in AI_JSON_FIELDS),
        *(Column(name, Boolean) for name in AI_BOOLEAN_FIELDS),
        Column(
            "ai_processing_version",
            String(64),
            nullable=False,
            comment="AI处理版本；与订单编号组成唯一标识",
        ),
        Column(
            "status",
            String(16),
            default=DEFAULT_STATUS,
            comment="数据状态：unprocessed=未处理，valid=有效，difficult=疑难",
        ),
        Column("operator", String(100), comment="操作人"),
        Column("label_person", String(100), comment="标注员"),
        UniqueConstraint("order_sn", "ai_processing_version", name="anno_med_order_version_uk"),
        comment="AI病例标注记录",
    )
    Index("anno_med_doctor_patient_idx", table.c.doctor_id, table.c.patient_id)
    Index("anno_med_doctor_time_idx", table.c.doctor_id, table.c.created_at)
    Index("anno_med_status_idx", table.c.status)
    return table


def _extract_record_key(record: Mapping[str, Any]) -> tuple[int | None, str, str]:
    source_record_id = (
        record.get("source_record_id")
        or record.get("medical_record_id")
        or record.get("record_id")
        or record.get("id")
    )
    order_sn = record.get("order_sn") or record.get("inquiry_id")
    ai_processing_version = record.get("ai_processing_version") or DEFAULT_AI_PROCESSING_VERSION
    missing = [
        name
        for name, value in (
            ("ai_processing_version", ai_processing_version),
            ("order_sn/inquiry_id", order_sn),
        )
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"病历缺少必填字段: {', '.join(missing)}")
    return (
        int(source_record_id) if source_record_id not in (None, "") else None,
        str(order_sn),
        str(ai_processing_version),
    )


def record_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """把一条 JSONL 病历转换为最新标注表行，字段直接落列."""
    source_record_id, order_sn, ai_processing_version = _extract_record_key(record)
    row = {
        field_name: record.get(field_name)
        for field_name in ORIGINAL_FIELDS + AI_FIELDS
        if field_name not in {"source_record_id", "order_sn", "ai_processing_version"}
    }
    row.update(
        {
            "source_record_id": source_record_id,
            "order_sn": order_sn,
            "ai_processing_version": ai_processing_version,
            "status": record.get("status") or DEFAULT_STATUS,
            "operator": record.get("operator") or DEFAULT_AI_OPERATOR,
            "label_person": record.get("label_person"),
        }
    )
    return {
        key: value
        for key, value in row.items()
        if key not in {"id", "medical_record_id", "record_id", "inquiry_id"}
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
    """AI 病例标注记录的批量写入及索引检索服务."""

    def __init__(self, manager: "DBManager", *, table_name: str = DEFAULT_TABLE_NAME) -> None:
        self.manager = manager
        self.table = build_ai_medical_records_table(table_name=table_name)

    def create_table(self) -> None:
        """不存在时创建数据表."""
        self.manager.create_table(self.table, checkfirst=True)

    def batch_insert_missing(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """分批新增病历；order_sn + ai_processing_version 已存在的数据会跳过."""
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
        """创建表并批量导入 JSONL，已存在 order_sn + ai_processing_version 的记录会跳过."""
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
            order_by=["source_record_id"],
        )

    def get_by_order_sn(self, order_sn: str, *, ai_processing_version: str | None = None) -> dict[str, Any] | None:
        """按咨询单 ID 查询单条病历."""
        filters = {"order_sn": order_sn}
        if ai_processing_version:
            filters["ai_processing_version"] = ai_processing_version
        return self.manager.select_one(self.table.name, filters=filters)

    def get_by_inquiry_id(self, inquiry_id: str) -> dict[str, Any] | None:
        """按咨询单 ID 查询单条病历，兼容旧方法名."""
        return self.get_by_order_sn(inquiry_id)

    def get_by_medical_record_id(self, medical_record_id: int) -> dict[str, Any] | None:
        """按来源病历 ID 查询单条病历，兼容旧方法名."""
        return self.manager.select_one(self.table.name, filters={"source_record_id": medical_record_id})

    def update_field_by_order_sn(self, order_sn: str, field_name: str, field_value: Any) -> bool:
        """更新单个字段，并把整条记录来源标记为 HUMAN.

        AI_FIELDS 和最新标注表中的原始字段都直接更新独立列。
        """
        row = self.get_by_order_sn(order_sn)
        if row is None:
            return False
        if field_name in {"id", "source_record_id", "order_sn", "operator"}:
            raise ValueError(f"不支持通过 update_field_by_order_sn 修改字段: {field_name}")

        data: dict[str, Any] = {"operator": DEFAULT_HUMAN_OPERATOR}
        if field_name not in self.table.c:
            raise ValueError(f"表 {self.table.name} 不存在字段: {field_name}")
        data[field_name] = field_value

        return self.manager.update(
            self.table.name,
            data,
            filters={"order_sn": order_sn, "ai_processing_version": row["ai_processing_version"]},
        ) > 0

    def _exclude_existing_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["order_sn"], row["ai_processing_version"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_rows.append(row)

        existing_rows = self.manager.select(
            self.table.name,
            filters={
                "order_sn": [key[0] for key in seen_keys],
                "ai_processing_version": [key[1] for key in seen_keys],
            },
            columns=["order_sn", "ai_processing_version"],
        )
        existing_keys = {(row["order_sn"], row["ai_processing_version"]) for row in existing_rows}
        return [row for row in unique_rows if (row["order_sn"], row["ai_processing_version"]) not in existing_keys]
