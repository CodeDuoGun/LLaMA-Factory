# Copyright 2026 the LlamaFactory team.
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
# ruff: noqa: D415

"""使用原收费明细重建已处理 JSON 中的 internal_prescriptions。

脚本只替换 internal_prescriptions，其他病历字段原样保留。收费明细中仅“退费标识”
为“否”的药品会进入新处方。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from medical.data_utils.raw_data_clear.clear_zongyuan_records import (
    FEE_CATEGORIES,
    ID_CANDIDATES,
    aggregate_prescriptions,
    clean_cell,
    normalize_id,
    read_excel_all_sheets,
)


def read_records(path: Path) -> list[dict[str, Any]]:
    """读取已处理的 JSON 病历列表。"""
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} 必须包含 JSON 对象列表。")
    return records


def resolve_id_column(prescription_df: pd.DataFrame, id_column: str | None = None) -> str:
    """确定收费明细中用于关联 inquiry_id 的字段。"""
    if id_column:
        if id_column not in prescription_df.columns:
            raise ValueError(f"收费明细不存在指定 ID 字段: {id_column}")
        return id_column

    for candidate in ("问诊单ID", *ID_CANDIDATES):
        if candidate in prescription_df.columns:
            return candidate
    raise ValueError(f"收费明细无法找到 ID 字段，候选字段: 问诊单ID、{'、'.join(ID_CANDIDATES)}")


def prepare_prescription_df(
    prescription_df: pd.DataFrame,
    inquiry_ids: set[str],
    id_column: str | None = None,
) -> pd.DataFrame:
    """校验并预处理需要重建的收费明细。"""
    required_columns = {"费用类别", "退费标识"}
    missing_columns = sorted(required_columns - set(prescription_df.columns))
    if missing_columns:
        raise ValueError(f"收费明细缺少必需字段: {'、'.join(missing_columns)}")

    resolved_id_column = resolve_id_column(prescription_df, id_column)
    prepared_df = prescription_df.copy()
    prepared_df["问诊单ID"] = prepared_df[resolved_id_column].map(normalize_id)
    return prepared_df[prepared_df["问诊单ID"].isin(inquiry_ids)].copy()


def count_drugs(prescriptions: list[dict[str, Any]]) -> int:
    """统计处方中的药品数。"""
    return sum(len(prescription.get("drugs") or []) for prescription in prescriptions)


def rebuild_internal_prescriptions(
    records: list[dict[str, Any]],
    prescription_df: pd.DataFrame,
    fee_categories: set[str],
    id_column: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """只重建 internal_prescriptions，不改变病历的其他字段。"""
    rebuilt_records = copy.deepcopy(records)
    inquiry_ids = {normalize_id(record.get("inquiry_id")) for record in records}
    inquiry_ids.discard("")
    prepared_df = prepare_prescription_df(prescription_df, inquiry_ids, id_column)
    prescription_map = aggregate_prescriptions(prepared_df, fee_categories)

    stats = {
        "record_count": len(records),
        "rebuilt_record_count": 0,
        "unmatched_record_count": 0,
        "before_drug_count": 0,
        "after_drug_count": 0,
    }
    for record in rebuilt_records:
        old_prescriptions = record.get("internal_prescriptions") or []
        stats["before_drug_count"] += count_drugs(old_prescriptions)

        inquiry_id = normalize_id(record.get("inquiry_id"))
        prescription_data = prescription_map.get(inquiry_id)
        if prescription_data is None:
            stats["unmatched_record_count"] += 1
            stats["after_drug_count"] += count_drugs(old_prescriptions)
            continue

        new_prescriptions = [
            prescription for prescription in prescription_data["prescriptions"] if prescription.get("drugs")
        ]
        record["internal_prescriptions"] = new_prescriptions
        stats["rebuilt_record_count"] += 1
        stats["after_drug_count"] += count_drugs(new_prescriptions)

    return rebuilt_records, stats


def default_output_path(input_path: Path) -> Path:
    """生成默认输出路径，避免覆盖原文件。"""
    return input_path.with_name(f"{input_path.stem}_refund_filtered.json")


def write_records(records: list[dict[str, Any]], path: Path) -> None:
    """写出重建后的最终 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def process(args: argparse.Namespace) -> Path:
    """运行 internal_prescriptions 重建流程。"""
    output_path = args.output or default_output_path(args.input)
    records = read_records(args.input)
    prescription_df = read_excel_all_sheets(args.prescription_file)
    fee_categories = {clean_cell(category) for category in args.fee_categories if clean_cell(category)}

    rebuilt_records, stats = rebuild_internal_prescriptions(
        records,
        prescription_df,
        fee_categories,
        args.id_column,
    )
    write_records(rebuilt_records, output_path)

    print(f"✅ 已写出 {stats['record_count']} 条病历 -> {output_path}")
    print(
        "   internal_prescriptions 重建: "
        f"{stats['rebuilt_record_count']} 条；未匹配保留原处方: {stats['unmatched_record_count']} 条"
    )
    print(f"   药品数: {stats['before_drug_count']} -> {stats['after_drug_count']}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="仅重建已处理 JSON 中的 internal_prescriptions。")
    parser.add_argument("--input", type=Path, required=True, help="已处理的最终 JSON 文件。")
    parser.add_argument("--prescription-file", type=Path, required=True, help="原门诊收费明细 Excel。")
    parser.add_argument("--output", type=Path, help="输出 JSON；默认在输入名后添加 _refund_filtered。")
    parser.add_argument("--id-column", help="收费明细的关联 ID 字段；默认自动识别。")
    parser.add_argument("--fee-categories", nargs="*", default=list(FEE_CATEGORIES), help="需要重建的费用类别。")
    return parser.parse_args()


if __name__ == "__main__":
    process(parse_args())
