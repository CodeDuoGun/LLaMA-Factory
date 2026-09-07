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

"""按 analysis engine 的规则归一化标注表中的三个 AI 诊断字段.

本脚本只使用 ``medical.analysis.engine`` 中的三个归一化函数，不读取 Excel
词表，也不修改原始诊断字段。默认扫描标注表全部记录，仅更新发生变化的
``ai_diagnosis_illness``、``ai_diagnosis_disease`` 和
``ai_diagnosis_sickness`` 字段。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from sqlalchemy import Table, update

from medical.analysis.engine import normalize_diagnosis_value, normalize_sickness_value, normalize_syndrome_value


if TYPE_CHECKING:
    from medical.data_utils.db_manager import DBManager


DEFAULT_TABLE_NAME = "mlops_annotation_medical_record"


DIAGNOSIS_NORMALIZERS: dict[str, Callable[[Any], str]] = {
    "ai_diagnosis_illness": normalize_diagnosis_value,
    "ai_diagnosis_disease": normalize_syndrome_value,
    "ai_diagnosis_sickness": normalize_sickness_value,
}


def normalize_record(record: Mapping[str, Any]) -> dict[str, str]:
    """Return only the diagnosis fields whose normalized value changed."""
    changes: dict[str, str] = {}
    for field_name, normalizer in DIAGNOSIS_NORMALIZERS.items():
        before = record.get(field_name)
        after = normalizer(before)
        if after != before:
            changes[field_name] = after
    return changes



def _validate_table(manager: DBManager, table_name: str) -> Table:
    table = manager.refresh_table(table_name)
    required_columns = {"id", *DIAGNOSIS_NORMALIZERS}
    missing_columns = required_columns.difference(table.c.keys())
    if missing_columns:
        raise ValueError(f"数据表 {table_name} 缺少字段: {', '.join(sorted(missing_columns))}")
    return table


def normalize_table(
    manager: DBManager,
    table_name: str,
    *,
    batch_size: int,
    limit: int = 0,
    apply: bool = False,
) -> tuple[int, int, Counter[str]]:
    """Scan the table and optionally update changed diagnosis fields.

    Returns ``(scanned_rows, changed_rows, changed_fields)``. Rows are read in
    primary-key order so the script does not need to hold the full table in memory.
    """
    table = _validate_table(manager, table_name)
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if limit < 0:
        raise ValueError("limit 不能小于 0")

    scanned_rows = 0
    changed_rows = 0
    changed_fields: Counter[str] = Counter()
    offset = 0

    while True:
        current_limit = batch_size if limit <= 0 else min(batch_size, limit - scanned_rows)
        if current_limit <= 0:
            break
        rows = manager.select(
            table_name,
            columns=["id", *DIAGNOSIS_NORMALIZERS],
            limit=current_limit,
            offset=offset,
            order_by=["id"],
        )
        if not rows:
            break

        planned_updates: list[dict[str, Any]] = []
        for row in rows:
            changes = normalize_record(row)
            if not changes:
                continue
            planned_updates.append({"id": row["id"], "changes": changes})
            changed_fields.update(changes)

        if apply and planned_updates:
            with manager.connection(transactional=True) as connection:
                for item in planned_updates:
                    result = connection.execute(
                        update(table).where(table.c.id == item["id"]).values(**item["changes"])
                    )
                    if (result.rowcount or 0) > 0:
                        changed_rows += 1
        else:
            changed_rows += len(planned_updates)

        scanned_rows += len(rows)
        offset += len(rows)
        print(
            f"[PROGRESS] scanned={scanned_rows} planned_changed={changed_rows} "
            f"field_changes={sum(changed_fields.values())}"
        )

        if len(rows) < current_limit:
            break

    return scanned_rows, changed_rows, changed_fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="归一化标注表中的三个 AI 诊断字段")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条；0 表示全表")
    parser.add_argument("--dry-run", action="store_true", help="只扫描和统计，不更新数据库")
    parser.add_argument("--yes", action="store_true", help="确认写回数据库")
    return parser


def main() -> None:
    """
    针对所有医生数据的 三个诊断进行正则
    PYTHONPATH=. conda run --no-capture-output -n llamafactory \
  python medical/data_utils/update_labeled_diagnosis_fields.py \
  --yes \
  --batch-size 500
    """
    args = build_parser().parse_args()
    if args.dry_run and args.yes:
        raise SystemExit("--dry-run 与 --yes 不能同时使用")
    if not args.dry_run and not args.yes:
        raise SystemExit("默认不直接写库；预览请加 --dry-run，确认写入请加 --yes")

    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        scanned, changed, field_changes = normalize_table(
            manager,
            args.table_name,
            batch_size=args.batch_size,
            limit=args.limit,
            apply=args.yes,
        )
    finally:
        manager.close()

    mode = "写入" if args.yes else "预览"
    print(f"{mode}完成：扫描 {scanned} 条，变化记录 {changed} 条")
    for field_name in DIAGNOSIS_NORMALIZERS:
        print(f"  {field_name}: {field_changes[field_name]} 个字段值发生变化")


if __name__ == "__main__":
    main()
