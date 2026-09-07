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

"""用原始病历处方重新格式化并更新 MySQL 标注表的 ``ps`` 字段.

标注记录和原始记录通过 ``doctor_id + order_sn`` 对齐。标注文件只作为更新
白名单，处方内容始终取自原始病历的 ``ps``，并复用
``medical.data_utils.ai_medical_records.format_ps`` 进行格式化。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Table, update

from medical.data_utils.ai_medical_records import DEFAULT_TABLE_NAME, format_ps
from medical.data_utils.db_manager import DBManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELED_DIR = Path("/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records")
DEFAULT_RAW_DIR = PROJECT_ROOT / "medical/data/202301_online"


RecordKey = tuple[str, str]


@dataclass(frozen=True)
class LabeledRecord:
    """A labeled record and its source location."""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    path: Path
    line_number: int


@dataclass(frozen=True)
class RawPrescription:
    """The raw prescription and its source location."""

    ps: list[dict[str, Any]]
    path: Path
    record_number: int


@dataclass(frozen=True)
class PrescriptionUpdate:
    """One database ``ps`` update derived from a raw record."""

    key: RecordKey
    doctor_id: Any
    order_sn: str
    ps: list[dict[str, Any]]
    labeled_path: Path
    raw_path: Path


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _record_key(record: Mapping[str, Any], *, path: Path, position: int) -> RecordKey:
    doctor_id = _text(record.get("doctor_id"))
    order_sn = _text(record.get("order_sn"))
    if not doctor_id or not order_sn:
        raise ValueError(f"{path} 第 {position} 条记录缺少 doctor_id/order_sn，无法与标注表记录匹配")
    return doctor_id, order_sn


def _iter_json_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(position, record)`` pairs from JSONL or JSON files."""
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
                yield line_number, value
        return

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是合法 JSON: {exc}") from exc

    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = [value]
    else:
        raise ValueError(f"{path} 顶层必须是 JSON 数组或对象")

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {position} 条记录必须是 JSON 对象")
        yield position, record


def _data_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl"})


def load_labeled_records(root: Path, doctor_ids: set[str] | None = None) -> dict[RecordKey, LabeledRecord]:
    """Load the labeled-record whitelist keyed by ``doctor_id + order_sn``."""
    records: dict[RecordKey, LabeledRecord] = {}
    files = _data_files(root)
    if not files:
        raise FileNotFoundError(f"标注目录中没有 JSON/JSONL 文件: {root}")

    for path in files:
        for line_number, record in _iter_json_records(path):
            key = _record_key(record, path=path, position=line_number)
            if doctor_ids is not None and key[0] not in doctor_ids:
                continue
            if key in records:
                previous = records[key]
                raise ValueError(
                    f"标注数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}: "
                    f"{previous.path}:{previous.line_number} 和 {path}:{line_number}"
                )
            records[key] = LabeledRecord(
                key=key,
                doctor_id=record["doctor_id"],
                order_sn=key[1],
                path=path,
                line_number=line_number,
            )

    if not records:
        raise ValueError(f"标注目录中没有符合 doctor_id 筛选条件的记录: {root}")
    return records


def load_raw_prescriptions(root: Path, target_keys: set[RecordKey]) -> dict[RecordKey, RawPrescription]:
    """Find raw prescriptions for the requested labeled-record keys."""
    prescriptions: dict[RecordKey, RawPrescription] = {}
    for path in _data_files(root):
        for record_number, record in _iter_json_records(path):
            try:
                key = _record_key(record, path=path, position=record_number)
            except ValueError:
                # Ignore unrelated malformed rows; only requested keys need to be matched.
                continue
            if key not in target_keys:
                continue
            if key in prescriptions:
                previous = prescriptions[key]
                raise ValueError(
                    f"原始数据存在重复 doctor_id/order_sn={key[0]}/{key[1]}: "
                    f"{previous.path}:{previous.record_number} 和 {path}:{record_number}"
                )
            raw_ps = record.get("ps")
            if not isinstance(raw_ps, list):
                raise ValueError(
                    f"原始数据 {path} 第 {record_number} 条记录的 ps 不是数组: doctor_id={key[0]}, order_sn={key[1]}"
                )
            prescriptions[key] = RawPrescription(
                ps=raw_ps,
                path=path,
                record_number=record_number,
            )

    missing = sorted(target_keys - prescriptions.keys())
    if missing:
        preview = ", ".join(f"{doctor_id}/{order_sn}" for doctor_id, order_sn in missing[:20])
        suffix = "" if len(missing) <= 20 else f" 等 {len(missing)} 条"
        raise ValueError(f"原始数据中找不到对应记录: {preview}{suffix}")
    return prescriptions


def build_update_plan(
    labeled_records: Mapping[RecordKey, LabeledRecord],
    raw_prescriptions: Mapping[RecordKey, RawPrescription],
) -> list[PrescriptionUpdate]:
    """Format raw ``ps`` values and build the database update plan."""
    updates: list[PrescriptionUpdate] = []
    for key, labeled in labeled_records.items():
        raw = raw_prescriptions[key]
        try:
            formatted_ps = format_ps(raw.ps)
        except Exception as exc:
            raise ValueError(
                f"格式化处方失败: doctor_id={key[0]}, order_sn={key[1]}, 文件={raw.path} 第 {raw.record_number} 条"
            ) from exc
        updates.append(
            PrescriptionUpdate(
                key=key,
                doctor_id=labeled.doctor_id,
                order_sn=labeled.order_sn,
                ps=formatted_ps,
                labeled_path=labeled.path,
                raw_path=raw.path,
            )
        )
    return updates


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _preflight_database_rows(
    manager: DBManager,
    table_name: str,
    updates: Iterable[PrescriptionUpdate],
    batch_size: int,
) -> Table:
    """Validate that every planned key exists before any write is attempted."""
    table = manager.refresh_table(table_name)
    required_columns = {"ps", "doctor_id", "order_sn"}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    update_list = list(updates)
    expected = {item.key for item in update_list}
    found: dict[RecordKey, dict[str, Any]] = {}
    for batch in _chunks(update_list, batch_size):
        order_sns = [item.order_sn for item in batch]
        rows = manager.select(
            table_name,
            filters={"order_sn": order_sns},
            columns=["order_sn", "doctor_id"],
        )
        for row in rows:
            key = (_text(row.get("doctor_id")), _text(row.get("order_sn")))
            if key in expected:
                if key in found:
                    raise ValueError(f"数据表中存在重复 doctor_id/order_sn={key[0]}/{key[1]}")
                found[key] = row

    missing = sorted(expected - found.keys())
    if missing:
        preview = ", ".join(f"{doctor_id}/{order_sn}" for doctor_id, order_sn in missing[:20])
        suffix = "" if len(missing) <= 20 else f" 等 {len(missing)} 条"
        raise ValueError(f"数据表中找不到对应记录: {preview}{suffix}")
    return table


def update_database(
    manager: DBManager,
    table_name: str,
    updates: list[PrescriptionUpdate],
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Update only ``ps`` in batches and return (attempted, changed)."""
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
                    .values(ps=item.ps)
                )
                result = connection.execute(statement)
                attempted += 1
                changed += max(result.rowcount or 0, 0)
    return attempted, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用原始病历处方格式更新标注表的 ps 字段")
    parser.add_argument("--labeled-dir", type=Path, default=DEFAULT_LABELED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--doctor-id",
        action="append",
        dest="doctor_ids",
        help="只处理指定医生；可重复传入。默认处理标注目录中的所有医生",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="只匹配和格式化，不更新数据库")
    parser.add_argument("--yes", action="store_true", help="确认将格式化后的 ps 写入数据库")
    return parser


def main() -> None:
    """
    PYTHONPATH=. conda run --no-capture-output -n llamafactory \
  python medical/data_utils/update_labeled_prescription_ps.py \
  --yes \
  --batch-size 200

  # 只处理某一个医生
  PYTHONPATH=. conda run --no-capture-output -n llamafactory \
  python medical/data_utils/update_labeled_prescription_ps.py \
  --yes \
  --batch-size 200 \
  --doctor-id 97
    """
    args = build_parser().parse_args()
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")

    doctor_ids = {_text(value) for value in (args.doctor_ids or []) if _text(value)} or None
    labeled_records = load_labeled_records(args.labeled_dir, doctor_ids)
    raw_prescriptions = load_raw_prescriptions(args.raw_dir, set(labeled_records))
    updates = build_update_plan(labeled_records, raw_prescriptions)
    print(
        f"匹配完成：标注记录 {len(labeled_records)} 条，"
        f"原始处方 {len(raw_prescriptions)} 条，待更新 ps {len(updates)} 条"
    )

    manager = DBManager()
    try:
        if args.dry_run:
            table = _preflight_database_rows(manager, args.table_name, updates, args.batch_size)
            print(f"数据库预检通过：{table.name}")
            return
        attempted, changed = update_database(
            manager,
            args.table_name,
            updates,
            batch_size=args.batch_size,
        )
        print(f"数据库预检通过：{args.table_name}")
    finally:
        manager.close()

    print(f"写入完成：尝试 {attempted} 条，数据库实际变更 {changed} 条；仅更新 ps 字段")


if __name__ == "__main__":
    main()
