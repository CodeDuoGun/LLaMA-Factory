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

"""Update only the prescription ``ps`` column in the annotation table from processed JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
DEFAULT_ZONGYUAN_SOURCE_DIR = PROJECT_ROOT / "medical/data/二期总院"
DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"
TARGET_FIELD = "ps"
PatientKey = tuple[str, str]
RecordKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class PrescriptionUpdate:
    """One prescription value keyed by doctor, patient, and order number."""

    key: RecordKey
    doctor_id: int
    order_sn: str
    ps: list[Any]
    path: Path
    line_number: int


@dataclass(frozen=True)
class DatabasePreflight:
    """Validated matches, missing keys, and rows whose prescription differs."""

    table: Table
    matched_count: int
    unchanged_count: int
    skipped_missing: list[RecordKey]
    updates: list[tuple[PrescriptionUpdate, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只把主 medical_records_ai.jsonl 中的 ps 字段更新到病历标注表。"
    )
    parser.add_argument(
        "--input",
        dest="input_files",
        type=Path,
        action="append",
        help=(
            "输入 medical_records_ai.jsonl；可重复指定多个文件。"
            "省略时扫描二期西区和二期总院医生文件"
        ),
    )
    parser.add_argument("--doctor-name", "--doctorname", dest="doctor_name", help="只处理指定医生")
    parser.add_argument(
        "--department",
        choices=("xiqu", "zongyuan"),
        help="限制部门；可单独指定，也可与 --doctor-name 配合",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="主 JSONL 所在目录")
    parser.add_argument(
        "--xiqu-source-dir",
        type=Path,
        default=DEFAULT_XIQU_SOURCE_DIR,
        help="二期西区源表目录，用于发现医生",
    )
    parser.add_argument(
        "--zongyuan-source-dir",
        type=Path,
        default=DEFAULT_ZONGYUAN_SOURCE_DIR,
        help="二期总院源表目录，用于发现医生",
    )
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="病历标注表名称")
    parser.add_argument("--batch-size", type=int, default=200, help="数据库预检批次大小")
    confirmation = parser.add_mutually_exclusive_group(required=True)
    confirmation.add_argument(
        "--dry-run", action="store_true", help="只预检匹配关系和差异，不写数据库"
    )
    confirmation.add_argument("--yes", action="store_true", help="确认只更新数据库中的 ps 列")
    return parser


def discover_input_files(args: argparse.Namespace) -> list[Path]:
    """Resolve explicit inputs or discover the Phase 2 doctors' main JSONL files."""
    if args.input_files:
        if args.doctor_name or args.department:
            raise ValueError("使用 --input 时不能同时指定 --doctor-name 或 --department")
        paths = [path.expanduser().resolve() for path in args.input_files]
        if len(paths) != len(set(paths)):
            raise ValueError("输入文件列表中存在重复路径")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("输入 JSONL 不存在: " + ", ".join(map(str, missing)))
        return paths

    from medical.data_utils.raw_data_clear import clear_xiqu_records as xiqu
    from medical.data_utils.raw_data_clear import clear_zongyuan_records as zongyuan

    source_files: list[tuple[str, str, Path, Path]] = []
    if args.doctor_name:
        doctor_name = args.doctor_name
        if args.department in (None, "xiqu") and doctor_name in xiqu.discover_doctors(args.xiqu_source_dir):
            source_files.append(
                (
                    "xiqu",
                    doctor_name,
                    args.xiqu_source_dir / f"{doctor_name} 病历.xlsx",
                    args.xiqu_source_dir / f"{doctor_name}.xlsx",
                )
            )
        if args.department in (None, "zongyuan"):
            workbook_files = [
                path
                for path in args.zongyuan_source_dir.glob("*.xlsx")
                if path.is_file() and not path.name.startswith("~$")
            ]
            record_prefix = f"{doctor_name}-门诊病历-"
            prescription_prefix = f"{doctor_name}-门诊收费明细-"
            records = sorted(path for path in workbook_files if path.name.startswith(record_prefix))
            prescriptions = sorted(path for path in workbook_files if path.name.startswith(prescription_prefix))
            if records or prescriptions:
                if len(records) != 1 or len(prescriptions) != 1:
                    raise ValueError(
                        f"医生“{doctor_name}”的二期总院源表未能一一配对："
                        f"病历表 {len(records)} 个，收费明细表 {len(prescriptions)} 个"
                    )
                source_files.append(("zongyuan", doctor_name, records[0], prescriptions[0]))
    else:
        if args.department in (None, "xiqu"):
            for doctor_name in xiqu.discover_doctors(args.xiqu_source_dir):
                source_files.append(
                    (
                        "xiqu",
                        doctor_name,
                        args.xiqu_source_dir / f"{doctor_name} 病历.xlsx",
                        args.xiqu_source_dir / f"{doctor_name}.xlsx",
                    )
                )
        if args.department in (None, "zongyuan"):
            for doctor_name, record_path, prescription_path in zongyuan.discover_phase2_doctor_files(
                args.zongyuan_source_dir
            ):
                source_files.append(("zongyuan", doctor_name, record_path, prescription_path))

    if args.doctor_name and not source_files:
        raise ValueError(f"未找到医生“{args.doctor_name}”的源表")
    if args.doctor_name and len(source_files) > 1:
        raise ValueError(
            f"医生“{args.doctor_name}”在多个部门出现，请指定 --department xiqu 或 zongyuan"
        )

    paths = []
    for department, doctor_name, _, _ in source_files:
        if department == "xiqu":
            paths.append(xiqu._output_file_for_doctor(args.processed_dir, doctor_name))
        else:
            paths.append(
                args.processed_dir / f"doctor_二期总院_{doctor_name}" / "medical_records_ai.jsonl"
            )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下主 JSONL 不存在: " + ", ".join(map(str, missing)))
    if not paths:
        raise FileNotFoundError("没有找到可更新的医生病历 JSONL")
    return paths


def _doctor_id(value: Any, *, path: Path, line_number: int) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{path}:{line_number} 的 doctor_id 不是有效整数")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{path}:{line_number} 缺少有效的 doctor_id") from None
    if result <= 0:
        raise ValueError(f"{path}:{line_number} 的 doctor_id 必须为正整数")
    return result


def _patient_identity(
    record: Mapping[str, Any], *, path: Path, line_number: int
) -> PatientKey:
    raw_patient_id = record.get("patient_id")
    patient_id = None
    if raw_patient_id is not None and str(raw_patient_id).strip():
        if isinstance(raw_patient_id, bool) or (
            isinstance(raw_patient_id, float) and not raw_patient_id.is_integer()
        ):
            raise ValueError(f"{path}:{line_number} 的 patient_id 不是有效整数")
        try:
            patient_id = int(raw_patient_id)
        except (TypeError, ValueError):
            raise ValueError(f"{path}:{line_number} 的 patient_id 不是有效整数") from None
        if patient_id <= 0:
            patient_id = None

    patient_idcard = "" if record.get("patient_idcard") is None else str(record.get("patient_idcard"))
    normalized_idcard = "".join(patient_idcard.split()).upper()
    if patient_id is not None:
        return "patient_id", str(patient_id)
    if normalized_idcard:
        return "patient_idcard", normalized_idcard
    raise ValueError(f"{path}:{line_number} 缺少 patient_id 和 patient_idcard，无法确认患者身份")


def _record_key(
    record: Mapping[str, Any], *, path: Path, line_number: int
) -> tuple[RecordKey, int, str]:
    doctor_id = _doctor_id(record.get("doctor_id"), path=path, line_number=line_number)
    order_sn = "" if record.get("order_sn") is None else str(record.get("order_sn")).strip()
    if not order_sn:
        raise ValueError(f"{path}:{line_number} 缺少有效的 order_sn")
    patient_key = _patient_identity(record, path=path, line_number=line_number)
    return (str(doctor_id), *patient_key, order_sn), doctor_id, order_sn


def load_updates(paths: list[Path]) -> list[PrescriptionUpdate]:
    """Load only ``doctor_id``, ``order_sn`` and ``ps`` from each input JSONL."""
    updates: list[PrescriptionUpdate] = []
    seen: dict[RecordKey, PrescriptionUpdate] = {}
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

                key, doctor_id, order_sn = _record_key(record, path=path, line_number=line_number)
                if TARGET_FIELD not in record or not isinstance(record[TARGET_FIELD], list):
                    raise ValueError(f"{path}:{line_number} 的 ps 缺失或不是数组")
                if key in seen:
                    previous = seen[key]
                    if previous.ps != record[TARGET_FIELD]:
                        raise ValueError(
                            "同一医生、患者和 order_sn 的输入病历处方不一致："
                            f"{previous.path}:{previous.line_number} 和 {path}:{line_number}"
                        )
                    print(
                        f"[去重] {path}:{line_number} 与 {previous.path}:{previous.line_number} "
                        "属于同一医生、患者和 order_sn，处方相同，合并为一条"
                    )
                    continue
                item = PrescriptionUpdate(
                    key=key,
                    doctor_id=doctor_id,
                    order_sn=order_sn,
                    ps=record[TARGET_FIELD],
                    path=path,
                    line_number=line_number,
                )
                updates.append(item)
                seen[key] = item

    if not updates:
        raise ValueError("输入 JSONL 中没有可更新的病历记录")
    return updates


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _database_key(row: Mapping[str, Any]) -> RecordKey:
    if isinstance(row.get("doctor_id"), bool):
        return "", "", "", ""
    try:
        doctor_id = int(row.get("doctor_id"))
    except (TypeError, ValueError):
        return "", "", "", ""
    raw_patient_id = row.get("patient_id")
    patient_id = None
    if raw_patient_id is not None and not isinstance(raw_patient_id, bool):
        try:
            patient_id = int(raw_patient_id)
        except (TypeError, ValueError):
            patient_id = None
    if patient_id is not None and patient_id > 0:
        patient_key = ("patient_id", str(patient_id))
    else:
        patient_idcard = "" if row.get("patient_idcard") is None else str(row.get("patient_idcard"))
        normalized_idcard = "".join(patient_idcard.split()).upper()
        if not normalized_idcard:
            return "", "", "", ""
        patient_key = ("patient_idcard", normalized_idcard)
    order_sn = "" if row.get("order_sn") is None else str(row.get("order_sn")).strip()
    return str(doctor_id), *patient_key, order_sn


def preflight_database(
    manager: DBManager,
    table_name: str,
    updates: list[PrescriptionUpdate],
    batch_size: int,
) -> DatabasePreflight:
    """Find exact matches and collect missing rows to skip without aborting the batch."""
    table = manager.refresh_table(table_name)
    required_columns = {"id", "doctor_id", "patient_id", "patient_idcard", "order_sn", TARGET_FIELD}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    expected = {item.key for item in updates}
    rows_by_key: dict[RecordKey, Mapping[str, Any]] = {}
    order_numbers = list(dict.fromkeys(item.order_sn for item in updates))
    # 获取数据表中该病历的id
    for batch in _chunks(order_numbers, batch_size):
        rows = manager.select(
            table_name,
            filters={"order_sn": batch},
            columns=["id", "doctor_id", "patient_id", "patient_idcard", "order_sn", TARGET_FIELD],
        )
        for row in rows:
            key = _database_key(row)
            if key not in expected:
                continue
            if key in rows_by_key:
                raise ValueError(
                    f"数据表中 doctor_id={key[0]}, order_sn={key[-1]} 对应多个同患者病历"
                )
            rows_by_key[key] = row

    missing = sorted(expected - rows_by_key.keys())
    rows_to_update = [
        (item, rows_by_key[item.key]["id"])
        for item in updates
        if item.key in rows_by_key and rows_by_key[item.key].get(TARGET_FIELD) != item.ps
    ]
    return DatabasePreflight(
        table=table,
        matched_count=len(rows_by_key),
        unchanged_count=len(rows_by_key) - len(rows_to_update),
        skipped_missing=missing,
        updates=rows_to_update,
    )


def update_database(manager: DBManager, preflight: DatabasePreflight) -> tuple[int, int]:
    """Commit a transaction that changes only ``ps`` for preflighted records."""
    from sqlalchemy import update

    attempted = 0
    changed = 0
    with manager.connection(transactional=True) as connection:
        for item, row_id in preflight.updates:
            statement = (
                update(preflight.table)
                .where(preflight.table.c.id == row_id)
                .values(**{TARGET_FIELD: item.ps})
            )
            result = connection.execute(statement)
            if result.rowcount is not None and result.rowcount > 1:
                raise ValueError(
                    f"更新 doctor_id/order_sn={item.doctor_id}/{item.order_sn} 影响多行，已回滚本次事务"
                )
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
        "更新键为 doctor_id + 患者 + order_sn"
    )

    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        preflight = preflight_database(manager, args.table_name, updates, args.batch_size)
        print(
            f"数据库预检通过：{args.table_name} 匹配 {preflight.matched_count} 条，"
            f"无匹配而跳过 {len(preflight.skipped_missing)} 条，"
            f"ps 已一致 {preflight.unchanged_count} 条，待更新 {len(preflight.updates)} 条。"
        )
        for key in preflight.skipped_missing:
            print(
                f"[跳过] 数据表中无完全匹配病历（医生+患者+order_sn）："
                f"doctor_id={key[0]}, order_sn={key[-1]}"
            )
        if args.dry_run:
            print("dry-run：未写入数据库，也未修改本地 JSONL 文件。")
            return
        attempted, changed = update_database(manager, preflight)
    finally:
        manager.close()

    print(
        f"写入完成：尝试更新 {attempted} 条，数据库报告变更 {changed} 条；"
        f"仅更新 {TARGET_FIELD} 列，本地 JSONL 未修改。"
    )


if __name__ == "__main__":
    main()
