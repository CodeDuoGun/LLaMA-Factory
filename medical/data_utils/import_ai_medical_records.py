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

"""批量写入及检索 AI 病例标注记录的命令行脚本."""

import argparse
import json
from pathlib import Path
from typing import Any

from medical.data_utils.ai_medical_records import DEFAULT_TABLE_NAME, AIMedicalRecordRepository
from medical.data_utils.db_manager import db_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "medical/processed_data/doctor_43_朱子奇/medical_records_ai.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数."""
    parser = argparse.ArgumentParser(description="AI 病例标注记录批量写入和检索")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="AI 病例标注记录表名")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import",
        help="创建表并批量写入 JSONL，已存在 order_sn + AI 版本会跳过",
    )
    import_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    import_parser.add_argument("--batch-size", type=int, default=100)

    clear_parser = subparsers.add_parser("clear", help="清空数据表")
    clear_parser.add_argument("--yes", action="store_true", help="确认清空整张表")

    patient_parser = subparsers.add_parser("patient", help="按患者 ID 查询")
    patient_parser.add_argument("patient_id", type=int)
    patient_parser.add_argument("--limit", type=int, default=100)
    patient_parser.add_argument("--offset", type=int, default=0)

    inquiry_parser = subparsers.add_parser("inquiry", help="按问诊单 ID 查询")
    inquiry_parser.add_argument("inquiry_id")

    record_parser = subparsers.add_parser("record", help="按来源病历 ID 查询")
    record_parser.add_argument("source_record_id", type=int)
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    """运行导入或查询命令."""
    args = build_parser().parse_args()
    repository = AIMedicalRecordRepository(
        db_manager,
        table_name=args.table_name,
    )

    if args.command == "import":
        count = repository.import_jsonl(args.input, batch_size=args.batch_size)
        print(
            f"已新增 {count} 条 AI 病例标注记录；"
            f"已存在 order_sn + AI 版本的记录已跳过；数据表: {repository.table.name}"
        )
    elif args.command == "clear":
        if not args.yes:
            raise SystemExit("清空整张表需要显式添加 --yes")
        count = repository.clear()
        print(f"已清空数据表 {repository.table.name}，删除 {count} 行")
    elif args.command == "patient":
        _print_json(repository.get_by_patient_id(args.patient_id, limit=args.limit, offset=args.offset))
    elif args.command == "inquiry":
        _print_json(repository.get_by_inquiry_id(args.inquiry_id))
    else:
        _print_json(repository.get_by_medical_record_id(args.source_record_id))


if __name__ == "__main__":
    main()
