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

"""Rebuild only ``order_sn`` values in processed Phase 2 West District JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "medical/processed_data"
DEFAULT_XIQU_SOURCE_DIR = PROJECT_ROOT / "medical/data/二期西区"


@dataclass(frozen=True)
class PlannedRecord:
    path: Path
    line_number: int
    old_order_sn: str
    base_candidate: str
    new_order_sn: str = ""


@dataclass(frozen=True)
class FilePlan:
    path: Path
    original_bytes: bytes
    new_bytes: bytes
    record_count: int
    changed_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只重新生成二期西区 medical_records_ai.jsonl 中每条病历的 order_sn。"
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--xiqu-source-dir", type=Path, default=DEFAULT_XIQU_SOURCE_DIR)
    parser.add_argument("--doctor-name", "--doctorname", dest="doctor_name", help="只处理指定医生")
    confirmation = parser.add_mutually_exclusive_group(required=True)
    confirmation.add_argument("--dry-run", action="store_true", help="只计算并校验，不写文件")
    confirmation.add_argument("--yes", action="store_true", help="确认写回 order_sn")
    return parser


def discover_input_files(args: argparse.Namespace) -> list[Path]:
    from medical.data_utils.raw_data_clear import clear_xiqu_records as xiqu

    if args.doctor_name:
        record_source = args.xiqu_source_dir / f"{args.doctor_name} 病历.xlsx"
        prescription_source = args.xiqu_source_dir / f"{args.doctor_name}.xlsx"
        if not record_source.is_file() or not prescription_source.is_file():
            raise ValueError(f"医生 {args.doctor_name} 不在二期西区医生列表中")
        return [xiqu._output_file_for_doctor(args.processed_dir, args.doctor_name)]

    doctor_names = xiqu.discover_doctors(args.xiqu_source_dir)
    if not doctor_names:
        raise FileNotFoundError(f"在 {args.xiqu_source_dir} 下没有找到二期西区医生源表")
    paths = [xiqu._output_file_for_doctor(args.processed_dir, name) for name in doctor_names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下主 JSONL 不存在: " + ", ".join(map(str, missing)))
    return paths


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


def _patient_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _source_order_index(path: Path) -> dict[tuple[str, str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"二期西区源病历不存在: {path}")
    result: dict[tuple[str, str], set[str]] = {}
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        sheet.reset_dimensions()
        columns: dict[str, int] = {}
        for row_number, values in enumerate(sheet.iter_rows(values_only=True), start=1):
            if row_number == 4:
                columns = {str(value).strip(): index for index, value in enumerate(values) if value is not None}
                missing = {"时间", "患者姓名", "门诊号"} - columns.keys()
                if missing:
                    raise ValueError(f"{path} 缺少字段: {', '.join(sorted(missing))}")
                continue
            if row_number < 5:
                continue
            visit_time = _parse_datetime(values[columns["时间"]])
            patient_name = _patient_name(values[columns["患者姓名"]])
            outpatient_no = re.sub(
                r"\.0$", "", str(values[columns["门诊号"]] or "").strip()
            ).replace("_", "")
            if visit_time is None or not patient_name or not outpatient_no:
                continue
            key = (visit_time.strftime("%Y%m%d%H%M%S"), patient_name)
            result.setdefault(key, set()).add(outpatient_no)
    finally:
        workbook.close()
    return result


def _base_order_candidate(
    record: dict[str, Any], path: Path, line_number: int, source_orders: dict[tuple[str, str], set[str]]
) -> str:
    if record.get("order_sn") is None or not str(record["order_sn"]).strip():
        raise ValueError(f"{path}:{line_number} 缺少有效 order_sn")
    visit_time = _parse_datetime(record.get("created_at") or record.get("start_time"))
    if visit_time is None:
        raise ValueError(f"{path}:{line_number} 缺少有效就诊时间，无法匹配源病历")
    key = (visit_time.strftime("%Y%m%d%H%M%S"), _patient_name(record.get("patient_name")))
    outpatient_numbers = source_orders.get(key, set())
    if len(outpatient_numbers) != 1:
        raise ValueError(f"{path}:{line_number} 源病历匹配到 {len(outpatient_numbers)} 个门诊号，未写入任何文件")
    return f"{visit_time:%Y%m%d}{next(iter(outpatient_numbers))}"


def replace_top_level_field(raw_json: str, field_name: str, value: Any) -> str:
    """Replace one top-level JSON value without reserializing neighboring fields."""
    decoder = json.JSONDecoder()
    opening = raw_json.find("{")
    if opening < 0 or not raw_json.lstrip().startswith("{"):
        raise ValueError("JSONL 记录根节点不是对象")

    index = opening + 1
    field_span: tuple[int, int] | None = None
    has_fields = False
    while True:
        while index < len(raw_json) and raw_json[index].isspace():
            index += 1
        if index >= len(raw_json):
            raise ValueError("JSON 对象未正确结束")
        if raw_json[index] == "}":
            if field_span is not None:
                break
            insert_at = index
            while insert_at > 0 and raw_json[insert_at - 1].isspace():
                insert_at -= 1
            separator = "," if has_fields else ""
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return (
                raw_json[:insert_at]
                + separator
                + json.dumps(field_name, ensure_ascii=False)
                + ":"
                + encoded
                + raw_json[insert_at:]
            )
        if raw_json[index] == ",":
            index += 1
            continue

        key, key_end = decoder.raw_decode(raw_json, index)
        if not isinstance(key, str):
            raise ValueError("JSON 对象字段名不是字符串")
        index = key_end
        while index < len(raw_json) and raw_json[index].isspace():
            index += 1
        if index >= len(raw_json) or raw_json[index] != ":":
            raise ValueError("JSON 对象字段缺少冒号")
        index += 1
        while index < len(raw_json) and raw_json[index].isspace():
            index += 1
        value_start = index
        _, value_end = decoder.raw_decode(raw_json, value_start)
        if key == field_name:
            if field_span is not None:
                raise ValueError(f"JSON 对象含有重复的 {field_name} 字段")
            field_span = (value_start, value_end)
        has_fields = True
        index = value_end

    if field_span is None:
        raise AssertionError(f"找不到 JSON 字段 {field_name}")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return raw_json[: field_span[0]] + encoded + raw_json[field_span[1] :]


def prepare_file(path: Path, source_dir: Path) -> tuple[FilePlan, list[PlannedRecord]]:
    doctor_name = path.parent.name.rsplit("_", 1)[-1]
    source_orders = _source_order_index(source_dir / f"{doctor_name} 病历.xlsx")
    original_bytes = path.read_bytes()
    original_lines = original_bytes.decode("utf-8").splitlines(keepends=True)
    planned_records: list[PlannedRecord] = []

    for line_number, raw_line in enumerate(original_lines, start=1):
        if not raw_line.strip():
            continue
        ending = "\r\n" if raw_line.endswith("\r\n") else "\n" if raw_line.endswith("\n") else ""
        content = raw_line[: -len(ending)] if ending else raw_line
        try:
            record = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} 不是有效 JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} 的 JSON 根节点不是对象")
        if "order_sn" not in record:
            raise ValueError(f"{path}:{line_number} 缺少 order_sn 字段")
        candidate = _base_order_candidate(record, path, line_number, source_orders)
        planned_records.append(
            PlannedRecord(
                path=path,
                line_number=line_number,
                old_order_sn=str(record["order_sn"]),
                base_candidate=candidate,
            )
        )

    return FilePlan(
        path=path,
        original_bytes=original_bytes,
        new_bytes=b"",
        record_count=len(planned_records),
        changed_count=0,
    ), planned_records


def assign_unique_order_sns(records: list[PlannedRecord]) -> list[PlannedRecord]:
    candidate_counts = Counter(record.base_candidate for record in records)
    reserved_unique = {
        candidate for candidate, count in candidate_counts.items() if count == 1
    }
    used: set[str] = set()
    result = []
    for record in records:
        candidate = record.base_candidate
        if candidate in used:
            sequence = 2
            base = candidate
            candidate = f"{base}{sequence:02d}"
            while candidate in used or candidate in reserved_unique:
                sequence += 1
                candidate = f"{base}{sequence:02d}"
        used.add(candidate)
        result.append(
            PlannedRecord(
                path=record.path,
                line_number=record.line_number,
                old_order_sn=record.old_order_sn,
                base_candidate=record.base_candidate,
                new_order_sn=candidate,
            )
        )
    return result


def build_file_plans(paths: list[Path], source_dir: Path) -> list[FilePlan]:
    plans_and_records = [prepare_file(path, source_dir) for path in paths]
    all_records = [record for _, records in plans_and_records for record in records]
    assigned = assign_unique_order_sns(all_records)
    assigned_by_path_line = {(record.path, record.line_number): record for record in assigned}

    plans = []
    for file_plan, records in plans_and_records:
        records_by_line = {record.line_number: record for record in records}
        output_lines = []
        changed_count = 0
        for line_number, raw_line in enumerate(
            file_plan.original_bytes.decode("utf-8").splitlines(keepends=True), start=1
        ):
            if not raw_line.strip():
                output_lines.append(raw_line)
                continue
            record = records_by_line[line_number]
            ending = "\r\n" if raw_line.endswith("\r\n") else "\n" if raw_line.endswith("\n") else ""
            content = raw_line[: -len(ending)] if ending else raw_line
            assigned_record = assigned_by_path_line[(record.path, record.line_number)]
            if assigned_record.new_order_sn == record.old_order_sn:
                output_lines.append(raw_line)
                continue
            new_content = replace_top_level_field(content, "order_sn", assigned_record.new_order_sn)
            original_record = json.loads(content)
            updated_record = json.loads(new_content)
            original_other = {key: value for key, value in original_record.items() if key != "order_sn"}
            updated_other = {key: value for key, value in updated_record.items() if key != "order_sn"}
            if original_other != updated_other:
                raise AssertionError(f"{record.path}:{record.line_number} 除 order_sn 外字段发生变化")
            output_lines.append(new_content + ending)
            changed_count += 1
        plans.append(
            FilePlan(
                path=file_plan.path,
                original_bytes=file_plan.original_bytes,
                new_bytes="".join(output_lines).encode("utf-8"),
                record_count=file_plan.record_count,
                changed_count=changed_count,
            )
        )
    return plans


def atomic_replace_many(plans: list[FilePlan]) -> None:
    temporary_paths: dict[Path, Path] = {}
    replaced: list[FilePlan] = []
    modes = {plan.path: stat.S_IMODE(plan.path.stat().st_mode) for plan in plans}
    try:
        for plan in plans:
            with tempfile.NamedTemporaryFile(
                "wb", dir=plan.path.parent, prefix=f".{plan.path.name}.", suffix=".tmp", delete=False
            ) as file:
                temp_path = Path(file.name)
                file.write(plan.new_bytes)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temp_path, modes[plan.path])
            temporary_paths[plan.path] = temp_path

        for plan in plans:
            if plan.path.read_bytes() != plan.original_bytes:
                raise RuntimeError(f"{plan.path} 在写入前发生变化，已取消提交")
        for plan in plans:
            os.replace(temporary_paths.pop(plan.path), plan.path)
            replaced.append(plan)
    except Exception:
        for plan in reversed(replaced):
            with tempfile.NamedTemporaryFile(
                "wb", dir=plan.path.parent, prefix=f".{plan.path.name}.rollback.", suffix=".tmp", delete=False
            ) as file:
                rollback_path = Path(file.name)
                file.write(plan.original_bytes)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(rollback_path, modes[plan.path])
            os.replace(rollback_path, plan.path)
        raise
    finally:
        for temp_path in temporary_paths.values():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    paths = discover_input_files(args)
    plans = build_file_plans(paths, args.xiqu_source_dir)
    total_records = sum(plan.record_count for plan in plans)
    total_changed = sum(plan.changed_count for plan in plans)
    print(
        f"预检完成：医生文件 {len(plans)} 个，病历 {total_records} 条，"
        f"order_sn 待更新 {total_changed} 条。"
    )
    for plan in plans:
        print(f"{plan.path}: 病历 {plan.record_count} 条，order_sn 更新 {plan.changed_count} 条")

    if args.dry_run:
        print("dry-run：未写回文件，其他病历字段均保持原样。")
        return
    changed_plans = [plan for plan in plans if plan.original_bytes != plan.new_bytes]
    if not changed_plans:
        print("所有 order_sn 已符合新格式，无需写入。")
        return
    atomic_replace_many(changed_plans)
    print(f"已写回 {len(changed_plans)} 个文件；仅修改 order_sn 字段。")


if __name__ == "__main__":
    """
    python medical/data_utils/raw_data_clear/reprocess_xiqu_order_sn.py --dry-run
    python medical/data_utils/raw_data_clear/reprocess_xiqu_order_sn.py --doctorname 李佩文 --yes
    """
    main()
