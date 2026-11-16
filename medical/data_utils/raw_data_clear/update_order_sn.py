# Copyright 2026 The LlamaFactory team.
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

"""Update only ``order_sn`` in the annotation table from processed West District JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from sqlalchemy import Table

    from medical.data_utils.db_manager import DBManager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "medical/processed_data"
DEFAULT_XIQU_SOURCE_DIR = PROJECT_ROOT / "medical/data/二期西区"
DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"
RecordKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class OrderSnUpdate:
    """One final order number and the patient/visit identity of its database row."""

    key: RecordKey
    doctor_id: int
    patient_id: int | None
    patient_idcard: str
    order_sn: str
    visit_date: str | None
    visit_time: datetime | None
    path: Path
    line_number: int


@dataclass(frozen=True)
class PreflightResult:
    """Matched rows, missing input records, and rows needing an order number update."""

    table: Table
    matched_count: int
    unchanged_count: int
    skipped_missing: list[OrderSnUpdate]
    updates: list[tuple[OrderSnUpdate, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只更新病历标注表中的 order_sn 字段。")
    parser.add_argument(
        "--input",
        dest="input_files",
        type=Path,
        action="append",
        help="输入主 medical_records_ai.jsonl；可重复指定。省略时扫描二期西区医生文件",
    )
    parser.add_argument("--doctor-name", "--doctorname", dest="doctor_name", help="只处理指定医生")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--xiqu-source-dir", type=Path, default=DEFAULT_XIQU_SOURCE_DIR)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=200)
    confirmation = parser.add_mutually_exclusive_group(required=True)
    confirmation.add_argument("--dry-run", action="store_true", help="只预检，不写数据库")
    confirmation.add_argument("--yes", action="store_true", help="确认更新数据库 order_sn 字段")
    return parser


def discover_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input_files:
        if args.doctor_name:
            raise ValueError("使用 --input 时不能同时指定 --doctor-name")
        paths = [path.expanduser().resolve() for path in args.input_files]
        if len(paths) != len(set(paths)):
            raise ValueError("输入文件列表中存在重复路径")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("输入 JSONL 不存在: " + ", ".join(map(str, missing)))
        return paths

    from medical.data_utils.raw_data_clear import clear_xiqu_records as xiqu

    if args.doctor_name:
        return [xiqu._output_file_for_doctor(args.processed_dir, args.doctor_name)]

    doctor_names = xiqu.discover_doctors(args.xiqu_source_dir)
    if not doctor_names:
        raise FileNotFoundError(f"在 {args.xiqu_source_dir} 下没有找到二期西区医生源表")
    paths = [xiqu._output_file_for_doctor(args.processed_dir, name) for name in doctor_names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下主 JSONL 不存在: " + ", ".join(map(str, missing)))
    return paths


def _positive_int(value: Any, field: str, path: Path, line_number: int) -> int | None:
    if value is None or not str(value).strip():
        return None
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{path}:{line_number} 的 {field} 不是有效整数")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{path}:{line_number} 的 {field} 不是有效整数") from None
    return result if result > 0 else None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _record_visit_time(record: Mapping[str, Any]) -> datetime | None:
    for field in ("created_at", "start_time", "see_doc_time"):
        visit_time = _parse_datetime(record.get(field))
        if visit_time is not None:
            return visit_time
    return None


def _format_visit_time(item: OrderSnUpdate) -> str:
    if item.visit_time is not None:
        return item.visit_time.strftime("%Y-%m-%d %H:%M:%S")
    return item.visit_date or "未记录"


def _record_key(record: Mapping[str, Any], path: Path, line_number: int) -> tuple[RecordKey, OrderSnUpdate]:
    doctor_id = _positive_int(record.get("doctor_id"), "doctor_id", path, line_number)
    if doctor_id is None:
        raise ValueError(f"{path}:{line_number} 缺少有效 doctor_id")
    patient_id = _positive_int(record.get("patient_id"), "patient_id", path, line_number)
    patient_idcard = "" if record.get("patient_idcard") is None else str(record.get("patient_idcard"))
    normalized_card = re.sub(r"\s+", "", patient_idcard).upper()
    if patient_id is None and not normalized_card:
        raise ValueError(f"{path}:{line_number} 缺少 patient_id 和 patient_idcard")
    order_sn = record.get("order_sn")
    if not isinstance(order_sn, str) or not order_sn.strip():
        raise ValueError(f"{path}:{line_number} 缺少有效 order_sn")

    visit_time = _record_visit_time(record)
    if visit_time is not None:
        visit_time = visit_time.replace(tzinfo=None, microsecond=0)
    record_date = visit_time.strftime("%Y%m%d") if visit_time else None
    patient_key = ("patient_id", str(patient_id)) if patient_id is not None else ("patient_idcard", normalized_card)
    key = (str(doctor_id), patient_key[0], patient_key[1], order_sn)
    item = OrderSnUpdate(
        key=key,
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_idcard=patient_idcard,
        order_sn=order_sn,
        visit_date=record_date,
        visit_time=visit_time,
        path=path,
        line_number=line_number,
    )
    return key, item


def load_updates(paths: list[Path]) -> list[OrderSnUpdate]:
    updates: list[OrderSnUpdate] = []
    seen_keys: dict[RecordKey, OrderSnUpdate] = {}
    target_owners: dict[str, RecordKey] = {}
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} 不是有效 JSON: {error.msg}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} 的 JSON 根节点不是对象")
                key, item = _record_key(record, path, line_number)
                previous = seen_keys.get(key)
                if previous is not None:
                    print(
                        f"[去重] {path}:{line_number} 与 {previous.path}:{previous.line_number} "
                        "是同一医生、患者和目标 order_sn，合并处理"
                    )
                    continue
                target_owner = target_owners.get(item.order_sn)
                if target_owner is not None and target_owner != key:
                    raise ValueError(
                        f"输入数据中目标 order_sn 重复，无法满足数据表唯一约束: {item.order_sn}"
                    )
                seen_keys[key] = item
                target_owners[item.order_sn] = key
                updates.append(item)
    if not updates:
        raise ValueError("输入 JSONL 中没有可更新的病历记录")
    return updates


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _database_patient_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("patient_id")
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _same_patient(item: OrderSnUpdate, row: Mapping[str, Any]) -> bool:
    row_patient_id = _database_patient_id(row)
    if item.patient_id is not None and row_patient_id == item.patient_id:
        return True
    source_card = re.sub(r"\s+", "", item.patient_idcard).upper()
    row_card = "" if row.get("patient_idcard") is None else str(row.get("patient_idcard"))
    return bool(source_card and source_card == re.sub(r"\s+", "", row_card).upper())


def _database_visit_time(row: Mapping[str, Any]) -> datetime | None:
    return _record_visit_time(row)


def _same_visit(item: OrderSnUpdate, row: Mapping[str, Any]) -> bool:
    row_time = _database_visit_time(row)
    if item.visit_time and row_time:
        normalized_row_time = row_time.replace(tzinfo=None, microsecond=0)
        return item.visit_time == normalized_row_time
    if item.visit_date and row_time:
        return item.visit_date == row_time.strftime("%Y%m%d")
    return True


def preflight_database(
    manager: DBManager,
    table_name: str,
    updates: list[OrderSnUpdate],
    batch_size: int,
) -> PreflightResult:
    table = manager.refresh_table(table_name)
    required = {"id", "doctor_id", "patient_id", "patient_idcard", "order_sn"}
    missing_columns = required.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    time_columns = [name for name in ("created_at", "start_time", "see_doc_time") if name in table.c]
    columns = ["id", "doctor_id", "patient_id", "patient_idcard", "order_sn", *time_columns]
    matched: list[tuple[OrderSnUpdate, Mapping[str, Any]]] = []
    skipped: list[OrderSnUpdate] = []
    claimed_rows: dict[Any, OrderSnUpdate] = {}
    for batch in _chunks(updates, batch_size):
        rows_by_doctor: dict[int, dict[Any, Mapping[str, Any]]] = defaultdict(dict)
        items_by_doctor: dict[int, list[OrderSnUpdate]] = defaultdict(list)
        for item in batch:
            items_by_doctor[item.doctor_id].append(item)

        for doctor_id, doctor_items in items_by_doctor.items():
            patient_ids = list(dict.fromkeys(item.patient_id for item in doctor_items if item.patient_id is not None))
            patient_cards = list(
                dict.fromkeys(
                    card
                    for item in doctor_items
                    for card in (
                        item.patient_idcard,
                        re.sub(r"\s+", "", item.patient_idcard).upper(),
                    )
                    if card
                )
            )
            for patient_id_batch in _chunks(patient_ids, batch_size):
                if not patient_id_batch:
                    continue
                rows = manager.select(
                    table_name,
                    filters={"doctor_id": doctor_id, "patient_id": patient_id_batch},
                    columns=columns,
                )
                for row in rows:
                    rows_by_doctor[doctor_id][row.get("id")] = row
            for card_batch in _chunks(patient_cards, batch_size):
                if not card_batch:
                    continue
                rows = manager.select(
                    table_name,
                    filters={"doctor_id": doctor_id, "patient_idcard": card_batch},
                    columns=columns,
                )
                for row in rows:
                    rows_by_doctor[doctor_id][row.get("id")] = row

        for item in batch:
            matches_by_id = {
                row_id: row
                for row_id, row in rows_by_doctor[item.doctor_id].items()
                if _same_patient(item, row) and _same_visit(item, row)
            }

            if not matches_by_id:
                skipped.append(item)
                continue
            if len(matches_by_id) > 1:
                raise ValueError(
                    f"{item.path}:{item.line_number} 的医生、患者和就诊时间对应多个数据库病历，"
                    f"doctor_id={item.doctor_id}, patient_id={item.patient_id}, "
                    f"就诊时间={_format_visit_time(item)}；未更新任何记录"
                )
            row_id, row = next(iter(matches_by_id.items()))
            previous = claimed_rows.get(row_id)
            if previous is not None:
                raise ValueError(
                    f"数据库同一行同时匹配到 {previous.path}:{previous.line_number} "
                    f"和 {item.path}:{item.line_number}；未更新任何记录"
                )
            claimed_rows[row_id] = item
            matched.append((item, row))

    changed = [(item, row) for item, row in matched if str(row.get("order_sn") or "") != item.order_sn]
    target_orders = list(dict.fromkeys(item.order_sn for item, _ in changed))
    target_owners: dict[str, set[Any]] = defaultdict(set)
    for batch in _chunks(target_orders, batch_size):
        rows = manager.select(table_name, filters={"order_sn": batch}, columns=["id", "order_sn"])
        for row in rows:
            target_owners[str(row.get("order_sn") or "")].add(row.get("id"))

    for item, row in changed:
        occupied_by = target_owners.get(item.order_sn, set()) - {row.get("id")}
        if occupied_by:
            raise ValueError(
                f"目标 order_sn={item.order_sn} 已被其他数据库行占用，"
                f"doctor_id={item.doctor_id}；未更新任何记录"
            )

    return PreflightResult(
        table=table,
        matched_count=len(matched),
        unchanged_count=len(matched) - len(changed),
        skipped_missing=skipped,
        updates=[(item, row.get("id")) for item, row in changed],
    )


def update_database(manager: DBManager, preflight: PreflightResult) -> tuple[int, int]:
    from sqlalchemy import update

    attempted = 0
    changed = 0
    with manager.connection(transactional=True) as connection:
        for item, row_id in preflight.updates:
            statement = (
                update(preflight.table)
                .where(preflight.table.c.id == row_id)
                .values(order_sn=item.order_sn)
            )
            result = connection.execute(statement)
            if result.rowcount is not None and result.rowcount > 1:
                raise ValueError(f"更新数据库行 id={row_id} 影响多行，已回滚")
            attempted += 1
            changed += max(result.rowcount or 0, 0)
    return attempted, changed


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")
    input_files = discover_input_files(args)
    updates = load_updates(input_files)
    print(
        f"读取完成：输入文件 {len(input_files)} 个，病历 {len(updates)} 条；"
        "按 doctor_id + 患者 + 就诊时间定位原病历，order_sn 作为原样目标值"
    )

    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        preflight = preflight_database(manager, args.table_name, updates, args.batch_size)
        print(
            f"数据库预检通过：匹配 {preflight.matched_count} 条，"
            f"无匹配跳过 {len(preflight.skipped_missing)} 条，"
            f"order_sn 已一致 {preflight.unchanged_count} 条，"
            f"待更新 {len(preflight.updates)} 条。"
        )
        for item in preflight.skipped_missing:
            print(
                f"[跳过] 数据表中找不到同一病历：doctor_id={item.doctor_id}, "
                f"patient_id={item.patient_id}, 就诊时间={_format_visit_time(item)}, "
                f"目标order_sn={item.order_sn}, 来源={item.path}:{item.line_number}"
            )
        if args.dry_run:
            print("dry-run：未写入数据库，也未修改本地 JSONL 文件。")
            return
        attempted, changed = update_database(manager, preflight)
    finally:
        manager.close()

    print(
        f"写入完成：尝试更新 {attempted} 条，数据库报告变更 {changed} 条；"
        "仅更新 order_sn，本地 JSONL 未修改。"
    )


if __name__ == "__main__":
    main()
