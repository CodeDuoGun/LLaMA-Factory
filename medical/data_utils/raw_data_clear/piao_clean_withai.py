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

"""Clean Park's medical records and write both AI stages back to the input JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from medical.data_utils.raw_data_clear.clear_xiqu_withai import (  # noqa: E402
    DEFAULT_BASE_URL,
    _without_image_references,
    apply_clinical_extraction_result,
    apply_result,
    build_clinical_extraction_payload,
    build_output_schema,
    build_previous_visit_contexts,
    build_record_payload,
    replace_jsonl_records,
    write_jsonl_atomic,
)


DEFAULT_INPUT = PROJECT_ROOT / "medical/processed_data/doctor_677_朴炳奎/medical_records_ai.jsonl"
PROMPT_DIR = PROJECT_ROOT / "medical/data_utils/prompts"
EXTRA_CLINICAL_FIELDS = ("admin_report_describe", "admin_face_describe", "disposition", "ps")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依次清洗、抽取朴老病历 JSONL 中的临床字段")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batchsize", "--batch-size", type=positive_int, default=1, help="每批并发病历数")
    parser.add_argument("--limit", type=positive_int, help="筛选后最多处理的病历数")
    parser.add_argument(
        "--reclean", nargs="?", const=True, action="append",
        metavar="ORDER_SN", help="按 order_sn 重跑，可重复指定。省略 ORDER_SN 时重跑所有病历",
    )
    parser.add_argument("--failed-output", type=Path, help="失败记录路径，默认输入文件名加 _failed.jsonl")
    parser.add_argument("--model", default=os.getenv("MEDICAL_AGENT_MODEL", ""))
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def read_records(path: Path) -> tuple[list[str], dict[int, dict[str, Any]]]:
    with path.open(encoding="utf-8", newline="") as source:
        lines = source.readlines()
    records = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {number} 行不是合法 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {number} 行必须是 JSON 对象")
        patient_id = pdf_patient_id(record)
        if patient_id:
            record["patient_id"] = patient_id
        records[number] = record
    return lines, records


def order_sn(record: dict[str, Any]) -> str:
    value = record.get("order_sn")
    return str(value).strip() if value is not None else ""


def pdf_patient_id(record: dict[str, Any]) -> str:
    source_pdf = str(record.get("source_pdf") or "").strip()
    if source_pdf:
        return Path(source_pdf).stem
    match = re.match(r"^(.*)_(?:\d{4}-\d{2}-\d{2}|undated-\d+)$", order_sn(record))
    if match:
        return match.group(1)
    return str(record.get("patient_id") or "").strip()


class CleaningError(Exception):
    """Identifies the failed prompt without including model output or patient data."""


async def clean_record(
    source: dict[str, Any], previous: dict[str, Any] | None, stages: tuple[Any, Any], message_type: Any
) -> dict[str, Any]:
    record = source.copy()
    payload = build_record_payload(record, previous)
    for field in EXTRA_CLINICAL_FIELDS:
        value = _without_image_references(record.get(field))
        if value not in (None, "", [], {}):
            payload[field] = value

    cleaner, extractor = stages
    try:
        result = await cleaner.ainvoke([message_type(content=json.dumps(payload, ensure_ascii=False))])
        apply_result(record, result.output)
    except Exception as exc:
        raise CleaningError(f"record_clear.txt ({type(exc).__name__})") from exc

    try:
        extraction = build_clinical_extraction_payload(record)
        for field, label in (("admin_report_describe", "检查报告描述"), ("admin_face_describe", "面诊描述")):
            if field in payload:
                extraction["病历信息"][label] = payload[field]
        result = await extractor.ainvoke([message_type(content=json.dumps(extraction, ensure_ascii=False))])
        return apply_clinical_extraction_result(record, result.output)
    except Exception as exc:
        raise CleaningError(f"extract_prompt.txt ({type(exc).__name__})") from exc


async def run(args: argparse.Namespace) -> int:
    path = args.input.resolve()
    failed_path = (args.failed_output or path.with_name(f"{path.stem}_failed.jsonl")).resolve()
    if failed_path == path or (failed_path.exists() and path.samefile(failed_path)):
        raise ValueError("失败文件必须与输入文件不同")

    lines, records = read_records(path)
    reclean = args.reclean or []
    requested = {value for value in reclean if isinstance(value, str)}
    rerun_all = True in reclean
    if requested:
        missing = requested - {order_sn(record) for record in records.values()}
        if missing:
            raise ValueError(f"未找到指定的 order_sn（{len(missing)} 个）")
    selected = []
    for number, record in records.items():
        if not rerun_all:
            if requested and order_sn(record) not in requested:
                continue
            if not requested and any(field.startswith("ai_") for field in record):
                continue
        selected.append(number)
        if len(selected) == args.limit:
            break
    print(f"病历共 {len(records)} 条，本次处理 {len(selected)} 条", flush=True)
    if not selected:
        return 0
    if not args.api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或传入 --api-key")
    if not args.model:
        raise ValueError("请设置 MEDICAL_AGENT_MODEL 或传入 --model")

    from langchain_core.messages import HumanMessage

    from medical.data_utils.structured_agent import AgentModelConfig, StructuredAgent, StructuredOutputMode
    from medical.schema.clear_record_basemodel import ClinicalExtractionResult

    config = AgentModelConfig(
        model=args.model, api_key=args.api_key, base_url=args.base_url, temperature=0,
        thinking=False, timeout=args.timeout, max_retries=args.max_retries,
    )
    agents = []
    for filename, schema, name in (
        ("record_clear.txt", build_output_schema(), "piao_record_cleaner"),
        ("extract_prompt.txt", ClinicalExtractionResult, "piao_clinical_extractor"),
    ):
        prompt = (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"提示词文件为空：{filename}")
        agents.append(StructuredAgent.from_config(
            config, schema, system_prompt=prompt, name=name,
            output_mode=StructuredOutputMode.JSON_OBJECT, validation_retries=1,
        ))
    stages = (agents[0], agents[1])
    previous = build_previous_visit_contexts(records)
    success_count = failure_count = 0
    for start in range(0, len(selected), args.batchsize):
        batch = selected[start : start + args.batchsize]
        results = await asyncio.gather(
            *(clean_record(records[number], previous.get(number), stages, HumanMessage) for number in batch),
            return_exceptions=True,
        )
        changes = set()
        failures = []
        for number, result in zip(batch, results, strict=True):
            if isinstance(result, Exception):
                failures.append(records[number])
                reason = str(result) if isinstance(result, CleaningError) else type(result).__name__
                print(f"第 {number} 行处理失败：{reason}", flush=True)
            elif isinstance(result, BaseException):
                raise result
            else:
                records[number] = result
                changes.add(number)

        if failures:
            saved = read_records(failed_path)[1] if failed_path.exists() else {}
            write_jsonl_atomic(failed_path, [*saved.values(), *failures])
            failure_count += len(failures)
        if changes:
            replace_jsonl_records(path, lines, records, changes)
            for number in changes:
                lines[number - 1] = json.dumps(records[number], ensure_ascii=False, separators=(",", ":")) + "\n"
            success_count += len(changes)
        print(f"已处理 {start + len(batch)}/{len(selected)} 条，成功 {success_count}，失败 {failure_count}", flush=True)

    if failure_count:
        print(f"失败记录已追加到 {failed_path}", flush=True)
    return int(failure_count > 0)


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    try:
        status = asyncio.run(run(args))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
