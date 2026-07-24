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

"""Export order numbers of online consultation records with missing diagnoses."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "medical" / "data"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_DIR / "missdata_records.xlsx"
ONLINE_FILE_MARKER = "_AI医生分身混合问诊数据_"


def is_empty(value: Any) -> bool:
    """Return whether a diagnosis field should be considered empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return False


def contains_chinese(value: Any) -> bool:
    """Return whether a value contains at least one Chinese character."""
    if not isinstance(value, str):
        return False
    chinese_ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2FA1F),
    )
    return any(start <= ord(char) <= end for char in value for start, end in chinese_ranges)


def discover_online_files(data_dir: Path) -> list[Path]:
    """Find JSON exports in online_* directories and online export files at the data root."""
    files = []
    for path in data_dir.rglob("*.json"):
        relative_parts = path.relative_to(data_dir).parts
        if any(part.startswith(".") or part == "__MACOSX" for part in relative_parts):
            continue
        in_online_directory = any(part.startswith("online_") for part in relative_parts[:-1])
        if in_online_directory or ONLINE_FILE_MARKER in path.name:
            files.append(path)
    return sorted(set(files))


def doctor_name_from_file(path: Path) -> str:
    """Infer the doctor name when a source record does not contain doctor_name."""
    if ONLINE_FILE_MARKER in path.name:
        return path.name.split(ONLINE_FILE_MARKER, maxsplit=1)[0].strip()
    return path.parent.name.removeprefix("online_").strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Online doctor data must be a JSON list: {path}")
    return [record for record in records if isinstance(record, dict)]


def collect_missing_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Group unique order_sn values by doctor when any diagnosis field is empty."""
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        fallback_doctor = doctor_name_from_file(path)
        for record in load_records(path):
            doctor = str(record.get("doctor_name") or fallback_doctor).strip() or "未知医生"
            grouped.setdefault(doctor, [])
            diagnosis_illness = record.get("diagnosis_illness")
            if not is_empty(diagnosis_illness) and contains_chinese(diagnosis_illness):
                continue

            order_sn = str(record.get("order_sn") or "").strip()
            if not order_sn:
                continue
            if order_sn not in seen[doctor]:
                grouped[doctor].append(order_sn)
                seen[doctor].add(order_sn)

    return [{"doctor": doctor, "miss_data": grouped[doctor]} for doctor in sorted(grouped)]


def collect_missing_history_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Group unique order_sn values by doctor when new_medical_history is empty."""
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        fallback_doctor = doctor_name_from_file(path)
        for record in load_records(path):
            doctor = str(record.get("doctor_name") or fallback_doctor).strip() or "未知医生"
            grouped.setdefault(doctor, [])
            if not is_empty(record.get("new_medical_history")):
                continue

            order_sn = str(record.get("order_sn") or "").strip()
            if not order_sn:
                continue
            if order_sn not in seen[doctor]:
                grouped[doctor].append(order_sn)
                seen[doctor].add(order_sn)

    return [{"doctor": doctor, "miss_data": grouped[doctor]} for doctor in sorted(grouped)]


def _write_sheet(workbook: Workbook, title: str, result: list[dict[str, Any]], table_name: str) -> int:
    sheet = workbook.create_sheet(title)
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A2"
    sheet.append(["序号", "医生", "order_sn"])

    row_number = 0
    for item in result:
        for order_sn in item["miss_data"]:
            row_number += 1
            sheet.append([row_number, item["doctor"], order_sn])

    header_fill = PatternFill("solid", fgColor="1D6B50")
    header_font = Font(color="FFFFFF", bold=True)
    bottom_border = Border(bottom=Side(style="thin", color="D9D4C7"))
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.row_dimensions[1].height = 24

    for row in sheet.iter_rows(min_row=2):
        row[0].alignment = Alignment(horizontal="right")
        row[1].alignment = Alignment(horizontal="left")
        row[2].alignment = Alignment(horizontal="left")
        row[2].number_format = "@"
        for cell in row:
            cell.border = bottom_border

    sheet.column_dimensions["A"].width = 10
    sheet.column_dimensions["B"].width = 16
    sheet.column_dimensions["C"].width = 26
    last_row = max(sheet.max_row, 1)
    table = Table(displayName=table_name, ref=f"A1:C{last_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium4",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    sheet.add_table(table)
    return row_number


def export_missing_workbook(data_dir: Path, output_file: Path) -> dict[str, int]:
    """Export diagnosis and medical-history omissions to two sheets in one workbook."""
    source_files = discover_online_files(data_dir)
    if not source_files:
        raise FileNotFoundError(f"No online doctor JSON files found under: {data_dir}")

    diagnosis_result = collect_missing_records(source_files)
    history_result = collect_missing_history_records(source_files)
    workbook = Workbook()
    workbook.remove(workbook.active)
    diagnosis_count = _write_sheet(workbook, "诊断缺失", diagnosis_result, "MissingDiagnosis")
    history_count = _write_sheet(workbook, "现病史缺失", history_result, "MissingMedicalHistory")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_file)
    return {"diagnosis": diagnosis_count, "new_medical_history": history_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 online 医生数据中诊断字段不完整记录的 order_sn。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"数据目录，默认：{DEFAULT_DATA_DIR}")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE, help=f"输出文件，默认：{DEFAULT_OUTPUT_FILE}"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = discover_online_files(args.data_dir)
    counts = export_missing_workbook(args.data_dir, args.output)
    print(f"Scanned {len(source_files)} online doctor files.")
    print(
        f"Exported {counts['diagnosis']} missing diagnoses and "
        f"{counts['new_medical_history']} missing medical histories to {args.output}"
    )


if __name__ == "__main__":
    main()
