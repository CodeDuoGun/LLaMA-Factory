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

"""用朱子奇标注 JSONL 中的治疗原则覆盖标注病历表.

标注文件与数据库记录通过 ``doctor_id + order_sn`` 对齐，只更新数据库的
``ai_treatment_principle`` 字段。
脚本默认只处理朱子奇（doctor_id=43）的标注文件，写入前必须显式传入
``--yes``；使用 ``--dry-run`` 可只做文件和数据库预检。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from sqlalchemy import Table

    from medical.data_utils.db_manager import DBManager


DEFAULT_LABELED_FILE = Path(
    "/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records/"
    "doctor_43_朱子奇/medical_records_labeled.jsonl"
)
DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"
SOURCE_FIELD = "ai_treatment_principle"
TARGET_FIELD = "ai_treatment_principle"
RecordKey = tuple[str, str]


@dataclass(frozen=True)
class TreatmentPrincipleUpdate:
    """一条待写入的治疗原则及其来源位置."""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    value: str | None
    path: Path
    line_number: int
    source_field: str


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _record_key(record: Mapping[str, Any], *, path: Path, line_number: int) -> RecordKey:
    doctor_id = _text(record.get("doctor_id"))
    order_sn = _text(record.get("order_sn"))
    if not doctor_id or not order_sn:
        raise ValueError(f"{path} 第 {line_number} 行缺少 doctor_id/order_sn，无法与标注表记录匹配")
    return doctor_id, order_sn


def load_updates(path: Path) -> dict[RecordKey, TreatmentPrincipleUpdate]:
    """读取 JSONL 中的治疗原则，按 ``doctor_id + order_sn`` 建立更新清单."""
    if not path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {path}")

    updates: dict[RecordKey, TreatmentPrincipleUpdate] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")

            key = _record_key(record, path=path, line_number=line_number)
            if SOURCE_FIELD not in record:
                raise ValueError(f"{path} 第 {line_number} 行缺少 {SOURCE_FIELD} 字段")
            value = record[SOURCE_FIELD]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{path} 第 {line_number} 行的 {SOURCE_FIELD} 必须是字符串或 null")
            if key in updates:
                previous = updates[key]
                raise ValueError(
                    f"标注数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}: "
                    f"{previous.path}:{previous.line_number} 和 {path}:{line_number}"
                )
            updates[key] = TreatmentPrincipleUpdate(
                key=key,
                doctor_id=record["doctor_id"],
                order_sn=key[1],
                value=value,
                path=path,
                line_number=line_number,
                source_field=SOURCE_FIELD,
            )

    if not updates:
        raise ValueError(f"标注文件中没有可更新记录: {path}")
    return updates


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _preflight_database_rows(
    manager: DBManager,
    table_name: str,
    updates: list[TreatmentPrincipleUpdate],
    batch_size: int,
) -> Table:
    """校验字段、重复行及全部目标键，成功后才允许写入."""
    table = manager.refresh_table(table_name)
    required_columns = {TARGET_FIELD, "doctor_id", "order_sn"}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    expected = {item.key for item in updates}
    found: set[RecordKey] = set()
    for batch in _chunks(updates, batch_size):
        rows = manager.select(
            table_name,
            filters={"order_sn": [item.order_sn for item in batch]},
            columns=["doctor_id", "order_sn"],
        )
        for row in rows:
            key = (_text(row.get("doctor_id")), _text(row.get("order_sn")))
            if key not in expected:
                continue
            if key in found:
                raise ValueError(f"数据表中存在重复 doctor_id/order_sn={key[0]}/{key[1]}")
            found.add(key)

    missing = sorted(expected - found)
    if missing:
        preview = ", ".join(f"{doctor_id}/{order_sn}" for doctor_id, order_sn in missing[:20])
        suffix = "" if len(missing) <= 20 else f" 等 {len(missing)} 条"
        raise ValueError(f"数据表中找不到对应记录: {preview}{suffix}")
    return table


def update_database(
    manager: DBManager,
    table_name: str,
    updates: list[TreatmentPrincipleUpdate],
    *,
    batch_size: int,
) -> tuple[int, int]:
    """只覆盖 ``ai_treatment_principle``，返回（尝试数，实际变更数）."""
    from sqlalchemy import update

    table = _preflight_database_rows(manager, table_name, updates, batch_size)
    attempted = 0
    changed = 0
    for batch in _chunks(updates, batch_size):
        with manager.connection(transactional=True) as connection:
            for item in batch:
                statement = (
                    update(table)
                    .where(table.c.doctor_id == item.doctor_id)
                    .where(table.c.order_sn == item.order_sn)
                    .values(**{TARGET_FIELD: item.value})
                )
                result = connection.execute(statement)
                attempted += 1
                changed += max(result.rowcount or 0, 0)
    return attempted, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用朱子奇标注数据覆盖标注表的 ai_treatment_principle 字段")
    parser.add_argument("--labeled-file", type=Path, default=DEFAULT_LABELED_FILE, help="待读取的标注 JSONL 文件")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="只校验文件和数据库匹配关系，不更新数据库")
    parser.add_argument("--yes", action="store_true", help="确认覆盖数据库字段")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    """
    PYTHONPATH=. python medical/data_utils/update_labeled_treatment_principle.py \
  --labeled-file "/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records/doctor_43_朱子奇/medical_records_labeled.jsonl" \
  --yes \
  --batch-size 200
    """
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")

    updates = list(load_updates(args.labeled_file).values())
    source_fields = sorted({item.source_field for item in updates})
    print(f"文件校验完成：待覆盖 {TARGET_FIELD} {len(updates)} 条；来源字段：{', '.join(source_fields)}")

    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        if args.dry_run:
            table = _preflight_database_rows(manager, args.table_name, updates, args.batch_size)
            print(f"数据库预检通过：{table.name}；未写入数据")
            return
        attempted, changed = update_database(manager, args.table_name, updates, batch_size=args.batch_size)
    finally:
        manager.close()

    print(f"写入完成：尝试 {attempted} 条，数据库实际变更 {changed} 条；仅覆盖 {TARGET_FIELD} 字段")


if __name__ == "__main__":
    main()
