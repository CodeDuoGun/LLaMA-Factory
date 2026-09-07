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

"""用分析引擎归一化标注数据中的三个 AI 诊断字段并更新 MySQL.

脚本读取 ``labeled_records`` 下的所有医生标注 JSONL，使用
``doctor_id + order_sn`` 匹配 ``mlops_annotation_medical_record``，然后只更新：

* ``ai_diagnosis_illness`` → ``normalize_diagnosis_value``
* ``ai_diagnosis_disease`` → ``normalize_syndrome_value``
* ``ai_diagnosis_sickness`` → ``normalize_sickness_value``

本脚本只使用 ``medical.analysis.engine`` 中的规则，不加载 Excel 标准词表。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Table, update

from medical.analysis.engine import (
    normalize_diagnosis_value,
    normalize_sickness_value,
    normalize_syndrome_value,
)
from medical.data_utils.ai_medical_records import DEFAULT_TABLE_NAME
from medical.data_utils.db_manager import DBManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELED_DIR = Path("/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records")

DIAGNOSIS_FIELDS: tuple[str, ...] = (
    "ai_diagnosis_illness",
    "ai_diagnosis_disease",
    "ai_diagnosis_sickness",
)
NORMALIZERS: dict[str, Callable[[Any], str]] = {
    "ai_diagnosis_illness": normalize_diagnosis_value,
    "ai_diagnosis_disease": normalize_syndrome_value,
    "ai_diagnosis_sickness": normalize_sickness_value,
}
RecordKey = tuple[str, str]


@dataclass(frozen=True)
class DiagnosisUpdate:
    """One normalized diagnosis row derived from a labeled record."""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    values: dict[str, str]
    path: Path
    line_number: int


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _record_key(record: Mapping[str, Any], *, path: Path, position: int) -> RecordKey:
    doctor_id = _text(record.get("doctor_id"))
    order_sn = _text(record.get("order_sn"))
    if not doctor_id or not order_sn:
        raise ValueError(f"{path} 第 {position} 条记录缺少 doctor_id/order_sn")
    return doctor_id, order_sn


def _iter_json_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
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
                yield line_number, record
        return

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是合法 JSON: {exc}") from exc
    records = value if isinstance(value, list) else [value] if isinstance(value, dict) else None
    if records is None:
        raise ValueError(f"{path} 顶层必须是 JSON 数组或对象")
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {position} 条记录必须是 JSON 对象")
        yield position, record


def _data_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def load_updates(root: Path, doctor_ids: set[str] | None = None) -> list[DiagnosisUpdate]:
    """Read all labeled records and normalize the three requested fields."""
    updates: list[DiagnosisUpdate] = []
    seen: set[RecordKey] = set()
    files = _data_files(root)
    if not files:
        raise FileNotFoundError(f"标注目录中没有 JSON/JSONL 文件: {root}")

    for path in files:
        for line_number, record in _iter_json_records(path):
            key = _record_key(record, path=path, position=line_number)
            if doctor_ids is not None and key[0] not in doctor_ids:
                continue
            if key in seen:
                raise ValueError(f"标注数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}")
            missing_fields = [field for field in DIAGNOSIS_FIELDS if field not in record]
            if missing_fields:
                raise ValueError(f"{path} 第 {line_number} 条记录缺少字段: {', '.join(missing_fields)}")

            values = {field: NORMALIZERS[field](record[field]) for field in DIAGNOSIS_FIELDS}
            updates.append(
                DiagnosisUpdate(
                    key=key,
                    doctor_id=record["doctor_id"],
                    order_sn=key[1],
                    values=values,
                    path=path,
                    line_number=line_number,
                )
            )
            seen.add(key)

    if not updates:
        raise ValueError(f"标注目录中没有符合 doctor_id 筛选条件的记录: {root}")
    return updates


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _preflight_database_rows(
    manager: DBManager,
    table_name: str,
    updates: Iterable[DiagnosisUpdate],
    batch_size: int,
) -> Table:
    """Validate the target table and every ``doctor_id + order_sn`` pair."""
    table = manager.refresh_table(table_name)
    required_columns = {"doctor_id", "order_sn", *DIAGNOSIS_FIELDS}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    update_list = list(updates)
    expected = {item.key for item in update_list}
    found: set[RecordKey] = set()
    for batch in _chunks(update_list, batch_size):
        rows = manager.select(
            table_name,
            filters={"order_sn": [item.order_sn for item in batch]},
            columns=["doctor_id", "order_sn"],
        )
        for row in rows:
            key = (_text(row.get("doctor_id")), _text(row.get("order_sn")))
            if key in expected:
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
    updates: list[DiagnosisUpdate],
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Update only the three diagnosis columns and return (attempted, changed)."""
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
                    .values(**item.values)
                )
                result = connection.execute(statement)
                attempted += 1
                changed += max(result.rowcount or 0, 0)
    return attempted, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="归一化标注表中的三个 AI 诊断字段")
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_LABELED_DIR)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--doctor-id",
        action="append",
        dest="doctor_ids",
        help="只处理指定医生；可重复传入。默认处理标注目录中的所有医生",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="只归一化和预检，不更新数据库")
    parser.add_argument("--yes", action="store_true", help="确认将三个归一化字段写入数据库")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")

    doctor_ids = {_text(value) for value in (args.doctor_ids or []) if _text(value)} or None
    updates = load_updates(args.labeled_dir, doctor_ids)
    nonempty_by_field = Counter()
    for item in updates:
        for field in DIAGNOSIS_FIELDS:
            # This is only a report; the actual update always writes all three fields.
            if item.values[field] != "":
                nonempty_by_field[field] += 1
    print(f"读取并归一化完成：{len(updates)} 条；非空结果：{dict(nonempty_by_field)}")

    manager = DBManager()
    try:
        if args.dry_run:
            table = _preflight_database_rows(manager, args.table_name, updates, args.batch_size)
            print(f"数据库预检通过：{table.name}；未执行写入")
            return
        attempted, changed = update_database(
            manager,
            args.table_name,
            updates,
            batch_size=args.batch_size,
        )
    finally:
        manager.close()

    print(f"写入完成：尝试 {attempted} 条，数据库实际变更 {changed} 条；仅更新三个 AI 诊断字段")


if __name__ == "__main__":
    main()
