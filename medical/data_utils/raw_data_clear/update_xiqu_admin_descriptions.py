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

"""只将西区处理后病历中的两个检查描述字段更新到标注病历表。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "medical/processed_data"
DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"
RECORD_FILE_NAME = "medical_records_ai.jsonl"
DESCRIPTION_FIELDS = ("admin_face_describe", "admin_report_describe")

RecordKey = tuple[str, str]


@dataclass(frozen=True)
class DescriptionUpdate:
    """两项检查描述及其对应的病历主键。"""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    admin_face_describe: str | None
    admin_report_describe: str | None
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


def _input_files(processed_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        if not input_file.is_file():
            raise FileNotFoundError(f"病历文件不存在: {input_file}")
        return [input_file]

    if not processed_dir.is_dir():
        raise FileNotFoundError(f"处理后数据目录不存在: {processed_dir}")
    files = sorted(processed_dir.glob(f"doctor_二期西区_*/{RECORD_FILE_NAME}"))
    if not files:
        raise FileNotFoundError(f"目录中没有西区医生的 doctor_二期西区_*/{RECORD_FILE_NAME}: {processed_dir}")
    return files


def load_updates(
    processed_dir: Path,
    *,
    input_file: Path | None = None,
    doctor_ids: set[str] | None = None,
) -> list[DescriptionUpdate]:
    """Load西区 JSONL descriptions keyed by ``doctor_id + order_sn``."""
    updates: dict[RecordKey, DescriptionUpdate] = {}
    for path in _input_files(processed_dir, input_file):
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

                missing_fields = [field for field in DESCRIPTION_FIELDS if field not in record]
                if missing_fields:
                    raise ValueError(f"{path} 第 {line_number} 行缺少字段: {', '.join(missing_fields)}")
                values = {field: record[field] for field in DESCRIPTION_FIELDS}
                for field, value in values.items():
                    if value is not None and not isinstance(value, str):
                        raise ValueError(f"{path} 第 {line_number} 行的 {field} 必须是字符串或 null")

                if key in updates:
                    previous = updates[key]
                    raise ValueError(
                        f"源数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}: "
                        f"{previous.path}:{previous.line_number} 和 {path}:{line_number}"
                    )
                updates[key] = DescriptionUpdate(
                    key=key,
                    doctor_id=record["doctor_id"],
                    order_sn=key[1],
                    admin_face_describe=values["admin_face_describe"],
                    admin_report_describe=values["admin_report_describe"],
                    path=path,
                    line_number=line_number,
                )

    if not updates:
        raise ValueError("没有符合筛选条件的西区病历记录")
    return list(updates.values())


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def preflight_database_rows(
    manager: DBManager,
    table_name: str,
    updates: list[DescriptionUpdate],
    batch_size: int,
) -> Table:
    """校验字段存在，且每条源记录只匹配一条数据库记录。"""
    table = manager.refresh_table(table_name)
    required_columns = {"doctor_id", "order_sn", *DESCRIPTION_FIELDS}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    expected = {item.key for item in updates}
    found: set[RecordKey] = set()
    for batch in _chunks(updates, batch_size):
        expected_batch = {item.key for item in batch}
        rows = manager.select(
            table_name,
            filters={"order_sn": [item.order_sn for item in batch]},
            columns=["doctor_id", "order_sn"],
        )
        for row in rows:
            key = (_text(row.get("doctor_id")), _text(row.get("order_sn")))
            if key not in expected_batch:
                continue
            if key in found:
                raise ValueError(f"标注病历表中存在重复 doctor_id/order_sn={key[0]}/{key[1]}")
            found.add(key)

    missing = sorted(expected - found)
    if missing:
        preview = ", ".join(f"{doctor_id}/{order_sn}" for doctor_id, order_sn in missing[:20])
        suffix = "" if len(missing) <= 20 else f" 等 {len(missing)} 条"
        raise ValueError(f"标注病历表中找不到对应记录: {preview}{suffix}")
    return table


def update_database(
    manager: DBManager,
    table_name: str,
    updates: list[DescriptionUpdate],
    *,
    batch_size: int,
) -> tuple[int, int]:
    """仅更新两个检查描述字段，返回 (尝试数, 影响行数)。"""
    from sqlalchemy import update

    table = preflight_database_rows(manager, table_name, updates, batch_size)
    attempted = 0
    changed = 0
    for batch in _chunks(updates, batch_size):
        with manager.connection(transactional=True) as connection:
            for item in batch:
                statement = (
                    update(table)
                    .where(table.c.doctor_id == item.doctor_id)
                    .where(table.c.order_sn == item.order_sn)
                    .values(
                        admin_face_describe=item.admin_face_describe,
                        admin_report_describe=item.admin_report_describe,
                    )
                )
                result = connection.execute(statement)
                attempted += 1
                changed += max(result.rowcount or 0, 0)
    return attempted, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只更新标注病历表中的体格检查和辅助检查描述字段")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--input-file", type=Path, help="只读取指定 JSONL；默认读取 processed_data 下全部西区医生文件")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--doctor-id",
        action="append",
        dest="doctor_ids",
        help="只处理指定医生 ID；可重复传入",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="只校验源数据与数据库匹配关系，不写入数据库")
    parser.add_argument("--yes", action="store_true", help="确认将两个检查描述字段写入标注病历表")
    return parser


def main() -> None:
    """校验西区处理后数据，并按需更新标注病历表的两个指定字段。"""
    args = build_parser().parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")

    doctor_ids = {_text(value) for value in (args.doctor_ids or []) if _text(value)} or None
    updates = load_updates(args.processed_dir, input_file=args.input_file, doctor_ids=doctor_ids)
    print(f"源数据校验完成：待更新标注病历 {len(updates)} 条；字段仅为 {', '.join(DESCRIPTION_FIELDS)}")

    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        if args.dry_run:
            table = preflight_database_rows(manager, args.table_name, updates, args.batch_size)
            print(f"数据库预检通过：{table.name}；未写入数据")
            return
        attempted, changed = update_database(
            manager,
            args.table_name,
            updates,
            batch_size=args.batch_size,
        )
    finally:
        manager.close()

    print(
        f"写入完成：尝试 {attempted} 条，数据库实际变更 {changed} 条；"
        f"仅更新 {', '.join(DESCRIPTION_FIELDS)}"
    )


if __name__ == "__main__":
    """
    PYTHONPATH=. python medical/data_utils/raw_data_clear/update_xiqu_admin_descriptions.py --yes
    """
    main()
