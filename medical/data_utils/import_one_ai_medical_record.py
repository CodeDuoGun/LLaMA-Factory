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

"""导入或预览单条 AI 清洗病历."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from medical.data_utils.ai_medical_records import (
    AI_FIELDS,
    DEFAULT_TABLE_NAME,
    AIMedicalRecordRepository,
    iter_jsonl,
    record_to_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "medical/processed_data/doctor_43_朱子奇/medical_records_ai.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数."""
    parser = argparse.ArgumentParser(description="导入或预览单条 AI 清洗病历")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="AI 清洗后病历表名")
    parser.add_argument("--database-url", help="测试数据库 URL；不设置时读取 config.yaml 或环境变量 DATABASE_URL")
    parser.add_argument("--line-number", type=int, default=1, help="读取第几条非空 JSONL 记录，默认第 1 条")
    parser.add_argument("--order-sn", help="按咨询单 ID 选择记录；设置后优先于 --line-number")
    parser.add_argument("--dry-run", action="store_true", help="只打印拆分结果，不写入数据库")
    parser.add_argument("--preview-limit", type=int, default=20, help="预览 record_data 字段数量")
    return parser


def load_one_record(path: Path, *, line_number: int, order_sn: str | None) -> dict[str, Any]:
    """从 JSONL 中读取一条记录."""
    if line_number <= 0:
        raise ValueError("--line-number 必须大于 0")

    for index, record in enumerate(iter_jsonl(path), start=1):
        if order_sn:
            if record.get("order_sn") == order_sn:
                return record
        elif index == line_number:
            return record

    if order_sn:
        raise ValueError(f"{path} 中未找到 order_sn={order_sn} 的记录")
    raise ValueError(f"{path} 中未找到第 {line_number} 条记录")


def print_preview(record: dict[str, Any], *, preview_limit: int) -> None:
    """打印单条记录拆分后的摘要."""
    row = record_to_row(record)
    preview_keys = list(row["record_data"])[:preview_limit]
    preview_ai_fields = [field_name for field_name in AI_FIELDS if field_name in row][:preview_limit]
    print(
        json.dumps(
            {
                "medical_record_id": row["medical_record_id"],
                "order_sn": row["order_sn"],
                "patient_id": row["patient_id"],
                "has_ps_in_source": "ps" in record,
                "has_ps_in_record_data": "ps" in row["record_data"],
                "record_data_fields": len(row["record_data"]),
                "operator": row["operator"],
                "preview_record_data": {key: row["record_data"][key] for key in preview_keys},
                "preview_ai_columns": {key: row[key] for key in preview_ai_fields},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def print_write_result(
    *,
    inserted_count: int,
    table_name: str,
    order_sn: str,
    row: dict[str, Any] | None,
    preview_limit: int,
) -> None:
    """打印实际写入后从数据库查回的结果."""
    preview_keys = list(row["record_data"])[:preview_limit] if row else []
    preview_ai_fields = [field_name for field_name in AI_FIELDS if row and field_name in row][:preview_limit]
    print(
        json.dumps(
            {
                "inserted_records": inserted_count,
                "table_name": table_name,
                "order_sn": order_sn,
                "stored": row is not None,
                "stored_record_data_fields": len(row["record_data"]) if row else 0,
                "stored_operator": row["operator"] if row else None,
                "stored_preview_record_data": {key: row["record_data"][key] for key in preview_keys} if row else {},
                "stored_preview_ai_columns": {key: row[key] for key in preview_ai_fields} if row else {},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main() -> None:
    """运行单条导入或预览."""
    args = build_parser().parse_args()
    record = load_one_record(args.input, line_number=args.line_number, order_sn=args.order_sn)
    order_sn = record.get("order_sn") or record.get("inquiry_id")
    print_preview(record, preview_limit=args.preview_limit)

    if args.dry_run:
        print("dry-run: 未写入数据库")
        return

    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url

    from medical.data_utils.db_manager import DBManager

    manager = DBManager(args.database_url)
    repository = AIMedicalRecordRepository(manager, table_name=args.table_name)
    repository.create_table()
    count = repository.batch_insert_missing([record], batch_size=1)
    row = repository.get_by_order_sn(str(order_sn))
    print_write_result(
        inserted_count=count,
        table_name=repository.table.name,
        order_sn=str(order_sn),
        row=row,
        preview_limit=args.preview_limit,
    )


if __name__ == "__main__":
    main()
