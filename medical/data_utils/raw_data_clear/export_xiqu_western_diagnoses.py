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

"""统计西区医生病历中的去重西医诊断，并导出为 Excel."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "medical/processed_data"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_DIR / "二期西区_西医诊断去重.xlsx"
DIAGNOSIS_NUMBER_PREFIX = re.compile(r"^(?:\d+|[一二三四五六七八九十]+)[.、．]\s*")
DIAGNOSIS_LABEL_PREFIX = re.compile(r"^西医诊断\s*[:：]\s*")


def split_diagnoses(value: Any) -> list[str]:
    """拆分西医诊断列表，同时保留括号内的分隔符."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [part for item in value for part in split_diagnoses(item)]
        return list(dict.fromkeys(parts))
    if not isinstance(value, str):
        value = str(value)

    value = re.sub(r"\s+(?=(?:\d+|[一二三四五六七八九十]+)[.、．])", "\n", value.strip())
    pieces: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    for character in value.strip():
        if character in "（(【[":
            bracket_depth += 1
        elif character in "）)】]":
            bracket_depth = max(0, bracket_depth - 1)

        if bracket_depth == 0 and character in "；;，,、\n\r":
            piece = normalize_diagnosis("".join(current))
            if piece:
                pieces.append(piece)
            current = []
        else:
            current.append(character)

    piece = normalize_diagnosis("".join(current))
    if piece:
        pieces.append(piece)
    return list(dict.fromkeys(pieces))


def normalize_diagnosis(value: str) -> str:
    """清理首尾空白和诊断序号，不做临床语义合并."""
    diagnosis = re.sub(r"\s+", " ", value).strip()
    diagnosis = DIAGNOSIS_LABEL_PREFIX.sub("", diagnosis).strip()
    diagnosis = DIAGNOSIS_NUMBER_PREFIX.sub("", diagnosis).strip()
    return diagnosis


def discover_doctor_files(data_dir: Path) -> list[Path]:
    """查找 processed_data 下目录名包含“西区”的医生 JSONL."""
    return sorted(
        path
        for path in data_dir.rglob("medical_records_ai.jsonl")
        if path.parent.name.startswith("doctor_") and "西区" in path.parent.name
    )


def collect_diagnoses_by_doctor(data_dir: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    """分别收集每位西区医生的去重西医诊断."""
    doctor_files = discover_doctor_files(data_dir)
    if not doctor_files:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到西区医生的 medical_records_ai.jsonl")

    diagnoses_by_doctor: dict[str, set[str]] = defaultdict(set)
    total_records = 0
    records_without_diagnosis = 0

    for path in doctor_files:
        folder_doctor = path.parent.name.removeprefix("doctor_")
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

                total_records += 1
                doctor = str(record.get("doctor_name") or folder_doctor).strip() or "未知医生"
                diagnoses_by_doctor.setdefault(doctor, set[str]())
                diagnoses = split_diagnoses(record.get("diagnosis_illness"))
                if not diagnoses:
                    records_without_diagnosis += 1
                    continue
                diagnoses_by_doctor[doctor].update(diagnoses)

    return dict(diagnoses_by_doctor), {
        "doctor_files": len(doctor_files),
        "total_records": total_records,
        "records_without_diagnosis": records_without_diagnosis,
    }


def _unique_sheet_name(doctor_name: str, used_names: set[str]) -> str:
    """生成不含 Excel 禁止字符且不超过 31 个字符的工作表名."""
    base_name = re.sub(r"[\\/*?:\[\]]", "_", doctor_name).strip("' ") or "未知医生"
    candidate = base_name[:31]
    suffix = 2
    while candidate.casefold() in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{base_name[: 31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def export_workbook(data_dir: Path, output_file: Path) -> dict[str, int]:
    """按医生分别导出西区去重诊断."""
    diagnoses_by_doctor, summary = collect_diagnoses_by_doctor(data_dir)
    workbook = Workbook()
    workbook.remove(workbook.active)

    header_fill = PatternFill("solid", fgColor="1D6B50")
    header_font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
    body_font = Font(name="Arial", size=10, color="222222")
    bottom_border = Border(bottom=Side(style="thin", color="D9D4C7"))
    used_sheet_names: set[str] = set()
    for sheet_index, (doctor_name, diagnoses) in enumerate(sorted(diagnoses_by_doctor.items()), start=1):
        sheet = workbook.create_sheet(_unique_sheet_name(doctor_name, used_sheet_names))
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "A2"
        sheet.append(["西医诊断"])
        for diagnosis in sorted(diagnoses):
            sheet.append([diagnosis])

        header_cell = sheet[1][0]
        header_cell.fill = header_fill
        header_cell.font = header_font
        header_cell.alignment = Alignment(horizontal="center", vertical="center")
        sheet.row_dimensions[1].height = 24
        for row in sheet.iter_rows(min_row=2):
            cell = row[0]
            cell.font = body_font
            cell.alignment = Alignment(horizontal="left", vertical="center")
            cell.border = bottom_border

        sheet.column_dimensions["A"].width = 56
        table = Table(displayName=f"DoctorDiagnoses{sheet_index}", ref=f"A1:A{sheet.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium4",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        sheet.add_table(table)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_file)
    return {
        **summary,
        "doctors": len(diagnoses_by_doctor),
        "unique_diagnoses": sum(len(diagnoses) for diagnoses in diagnoses_by_doctor.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计西区医生病历中去重后的西医诊断并导出 Excel。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"病历目录，默认：{DEFAULT_DATA_DIR}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help=f"输出文件，默认：{DEFAULT_OUTPUT_FILE}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_workbook(args.data_dir, args.output)
    print(
        f"扫描 {summary['doctor_files']} 份医生文件、{summary['total_records']} 条病历，"
        f"生成 {summary['doctors']} 个医生工作表，共 {summary['unique_diagnoses']} 个医生-诊断组合；"
        f"{summary['records_without_diagnosis']} 条病历没有西医诊断。"
    )
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
