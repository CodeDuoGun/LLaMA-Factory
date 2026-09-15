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

"""用标注 JSONL 覆盖标注病历表的 ``ai_new_medical_history`` 字段.

每个医生目录中的 ``medical_records_labeled.jsonl`` 与数据库记录通过
``doctor_id + order_sn`` 对齐。脚本默认处理 ``data/labeled_records`` 下的
所有医生，也可以用 ``--doctor-id`` 只处理指定医生。
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


DEFAULT_LABELED_DIR = Path("/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records")
LABELED_FILE_NAME = "medical_records_labeled.jsonl"
# Keep this utility lightweight: importing ``ai_medical_records`` only to read
# this constant would also import optional LangChain dependencies.
DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"

RecordKey = tuple[str, str]


@dataclass(frozen=True)
class MedicalHistoryUpdate:
    """One labeled medical-history value and its source location."""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    ai_new_medical_history: str | None
    path: Path
    line_number: int


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _record_key(record: Mapping[str, Any], *, path: Path, line_number: int) -> RecordKey:
    doctor_id = _text(record.get("doctor_id"))
    order_sn = _text(record.get("order_sn"))
    if not doctor_id or not order_sn:
        raise ValueError(f"{path} 第 {line_number} 行缺少 doctor_id/order_sn，无法与标注表记录匹配")
    return doctor_id, order_sn


def _labeled_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {root}")
    return sorted(path for path in root.glob(f"doctor_*/{LABELED_FILE_NAME}") if path.is_file())


def load_updates(root: Path, doctor_ids: set[str] | None = None) -> dict[RecordKey, MedicalHistoryUpdate]:
    """Load labeled values keyed by ``doctor_id + order_sn``."""
    files = _labeled_files(root)
    if not files:
        raise FileNotFoundError(f"标注目录中没有 doctor_*/{LABELED_FILE_NAME}: {root}")

    updates: dict[RecordKey, MedicalHistoryUpdate] = {}
    for path in files:
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
                if doctor_ids is not None and key[0] not in doctor_ids:
                    continue
                if "ai_new_medical_history" not in record:
                    raise ValueError(f"{path} 第 {line_number} 行缺少 ai_new_medical_history 字段")
                value = record["ai_new_medical_history"]
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{path} 第 {line_number} 行的 ai_new_medical_history 必须是字符串或 null")
                if key in updates:
                    previous = updates[key]
                    raise ValueError(
                        f"标注数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}: "
                        f"{previous.path}:{previous.line_number} 和 {path}:{line_number}"
                    )
                updates[key] = MedicalHistoryUpdate(
                    key=key,
                    doctor_id=record["doctor_id"],
                    order_sn=key[1],
                    ai_new_medical_history=value,
                    path=path,
                    line_number=line_number,
                )

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
    updates: list[MedicalHistoryUpdate],
    batch_size: int,
) -> Table:
    """Check the table and every target row before performing any update."""
    table = manager.refresh_table(table_name)
    required_columns = {"ai_new_medical_history", "doctor_id", "order_sn"}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    expected = {item.key for item in updates}
    found: set[RecordKey] = set()
    for batch in _chunks(updates, batch_size):
        rows = manager.select(
            table_name,
            filters={"order_sn": [item.order_sn for item in batch]},
            columns=["order_sn", "doctor_id"],
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
    updates: list[MedicalHistoryUpdate],
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Overwrite only ``ai_new_medical_history`` and return (attempted, changed)."""
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
                    .values(ai_new_medical_history=item.ai_new_medical_history)
                )
                result = connection.execute(statement)
                attempted += 1
                changed += max(result.rowcount or 0, 0)
    return attempted, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用标注 JSONL 覆盖标注病历表的 ai_new_medical_history 字段")
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_LABELED_DIR)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--doctor-id",
        action="append",
        dest="doctor_ids",
        help="只处理指定医生；可重复传入。默认处理标注目录中的所有医生",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="只校验文件和数据库匹配关系，不更新数据库")
    parser.add_argument("--yes", action="store_true", help="确认覆盖数据库字段")
    return parser


def main() -> None:
    """Validate labeled records and optionally update the annotation table."""
    args = build_parser().parse_args()
    """
    python medical/data_utils/update_labeled_ai_medical_history.py --yes --batch-size 200
    """
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")

    doctor_ids = {_text(value) for value in (args.doctor_ids or []) if _text(value)} or None
    updates = list(load_updates(args.labeled_dir, doctor_ids).values())
    print(f"文件校验完成：待覆盖 ai_new_medical_history {len(updates)} 条")

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

    print(f"写入完成：尝试 {attempted} 条，数据库实际变更 {changed} 条；仅覆盖 ai_new_medical_history 字段")


if __name__ == "__main__":
    main()
