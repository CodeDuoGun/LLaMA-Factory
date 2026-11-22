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

"""Update only ps for existing West District records matched by three exact identifiers."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED = PROJECT_ROOT / "medical/processed_data"
DEFAULT_SOURCE = PROJECT_ROOT / "medical/data/二期西区"
TABLE_NAME = "mlops_annotation_medical_record"


def discover_files(processed_dir: Path, source_dir: Path, doctor_name: str | None = None) -> list[Path]:
    names = {path.stem.removesuffix(" 病历") for path in source_dir.glob("* 病历.xlsx")}
    if doctor_name:
        if doctor_name not in names:
            raise ValueError(f"西区源表中未找到医生: {doctor_name}")
        names = {doctor_name}
    paths = sorted(
        path
        for path in processed_dir.glob("doctor_*/medical_records_ai.jsonl")
        if path.parent.name.rsplit("_", 1)[-1] in names
    )
    if not paths:
        raise FileNotFoundError(f"未找到西区医生{doctor_name or ''}的 medical_records_ai.jsonl")
    return paths


def load_updates(paths: list[Path]) -> dict[tuple[int, int, str], list]:
    updates = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                doctor_id = int(record["doctor_id"])
                patient_id = int(record["patient_id"])
                order_sn = str(record["order_sn"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} 缺少有效 doctor_id、patient_id 或 order_sn") from exc
            if doctor_id <= 0 or patient_id <= 0 or not order_sn or not isinstance(record.get("ps"), list):
                raise ValueError(f"{path}:{line_number} 的匹配键或 ps 无效")
            key = (doctor_id, patient_id, order_sn)
            if key in updates and updates[key] != record["ps"]:
                raise ValueError(f"重复键且 ps 不一致: {key}")
            updates[key] = record["ps"]
    return updates


def write_batch(manager: Any, table: Any, batch: list[tuple[tuple[int, int, str], Any]], updates: dict) -> int:
    """Update one batch in its own transaction and connection."""
    from sqlalchemy import and_, update

    count = 0
    with manager.connection(transactional=True) as connection:
        for key, row_id in batch:
            doctor_id, patient_id, order_sn = key
            statement = (
                update(table)
                .where(
                    and_(
                        table.c.id == row_id,
                        table.c.doctor_id == doctor_id,
                        table.c.patient_id == patient_id,
                        table.c.order_sn == order_sn,
                    )
                )
                .values(ps=updates[key])
            )
            result = connection.execute(statement)
            if result.rowcount != 1:
                raise ValueError(f"更新影响行数异常，当前批次事务回滚: {key} rowcount={result.rowcount}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--doctor-name", help="只预检或更新指定西区医生，例如 李佩文")
    parser.add_argument("--table-name", default=TABLE_NAME)
    parser.add_argument(
        "--batchsize",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=200,
        help="预检查询及写入事务每批处理的病历数，默认 200",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="并发写入批次数，默认 4")
    parser.add_argument("--yes", action="store_true", help="执行更新；默认只预检")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batchsize 必须为正数")
    if args.concurrency < 1:
        parser.error("--concurrency 必须为正数")
    updates = load_updates(discover_files(args.processed_dir, args.source_dir, args.doctor_name))
    from medical.data_utils.db_manager import DBManager

    manager = DBManager()
    try:
        table = manager.refresh_table(args.table_name)
        required = {"id", "doctor_id", "patient_id", "order_sn", "ps"}
        if required - set(table.c.keys()):
            raise ValueError(f"数据表缺少字段: {sorted(required - set(table.c.keys()))}")
        by_order = {}
        for key in updates:
            by_order.setdefault(key[2], set()).add(key)
        matched = {}
        order_numbers = list(by_order)
        for start in range(0, len(order_numbers), args.batch_size):
            rows = manager.select(
                args.table_name,
                filters={"order_sn": order_numbers[start : start + args.batch_size]},
                columns=["id", "doctor_id", "patient_id", "order_sn", "ps"],
            )
            for row in rows:
                try:
                    key = (int(row["doctor_id"]), int(row["patient_id"]), str(row["order_sn"]).strip())
                except (TypeError, ValueError):
                    continue
                if key not in updates:
                    continue
                if key in matched:
                    raise ValueError(f"MySQL 中存在重复匹配病历: {key}")
                matched[key] = row
        missing = sorted(set(updates) - set(matched))
        for doctor_id, patient_id, order_sn in missing:
            print(f"[跳过] MySQL 未找到 doctor_id={doctor_id} patient_id={patient_id} order_sn={order_sn}")
        changed = []
        for key, row in matched.items():
            current = row["ps"]
            if isinstance(current, str):
                try:
                    current = json.loads(current)
                except json.JSONDecodeError:
                    pass
            if current != updates[key]:
                changed.append((key, row["id"]))
        print(
            f"预检: 本地 {len(updates)} 条，匹配 {len(matched)} 条，未找到 {len(missing)} 条，待更新 {len(changed)} 条"
        )
        if not args.yes:
            print("只预检，未写入 MySQL；传入 --yes 执行更新")
            return
        batches = [changed[start : start + args.batch_size] for start in range(0, len(changed), args.batch_size)]
        count = 0
        failures = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(write_batch, manager, table, batch, updates): index
                for index, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                batch_number = futures[future]
                try:
                    written = future.result()
                except Exception as exc:
                    failures.append(batch_number)
                    print(f"[失败] batch={batch_number}/{len(batches)}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                count += written
                print(f"[完成] batch={batch_number}/{len(batches)} 更新={written} 条", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} 个批次失败，已完成 {count} 条；失败批次: {sorted(failures)}")
        print(f"写入完成: 仅更新 ps，共 {count} 条；未插入任何病历")
    finally:
        manager.close()


if __name__ == "__main__":
    """
    python medical/data_utils/raw_data_clear/update_xiqu_ps_mysql.py --doctor-name 李方洁 --batchsize 200 --concurrency 4 --yes
    """
    main()
