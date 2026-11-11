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

"""Clean West District medical records and save structured AI fields in place."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "medical/processed_data/doctor_二期西区_于振宣/medical_records_ai.jsonl"
)
PROMPT_PATH = PROJECT_ROOT / "medical/prompts/xiqu_cleaner.txt"
EXTRACTION_PROMPT_PATH = PROJECT_ROOT / "medical/data_utils/prompts/extract_prompt.txt"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CLEANER_OUTPUT_FIELDS = (
    "patient_appeal",
    "new_medical_history",
    "old_medical_history",
    "allergic_history",
    "diagnosis_illness",
    "diagnosis_sickness",
    "diagnosis_disease",
    "marriage_history",
    "birth_detail",
    "personal_history",
    "family_history",
    "special_history",
    "diagnosis_illness_reason",
    "diagnosis_disease_reason",
    "diagnosis_sickness_reason",
)
CLINICAL_EXTRACTION_FIELDS = (
    "etiology",
    "pathogenesis",
    "disease_location",
    "disease_stage",
    "disease_course",
    "key_symptoms",
    "treatment_principle",
)


def load_system_prompt() -> str:
    """Load the cleaner prompt from the project-local text file."""
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"清洗提示词文件不存在：{PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"清洗提示词文件为空：{PROMPT_PATH}")
    return prompt


def load_extraction_prompt() -> str:
    """Load the clinical extraction prompt from the project-local text file."""
    if not EXTRACTION_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"临床信息抽取提示词文件不存在：{EXTRACTION_PROMPT_PATH}")
    prompt = EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"临床信息抽取提示词文件为空：{EXTRACTION_PROMPT_PATH}")
    return prompt


def parse_bool(value: str) -> bool:
    """Parse common command-line spellings for a boolean flag."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "是"}:
        return True
    if normalized in {"0", "false", "no", "n", "否"}:
        return False
    raise argparse.ArgumentTypeError("布尔值请使用 true/false")


def build_output_schema() -> type:
    """Build the structured output model after CLI parsing, keeping --help lightweight."""
    from typing import Annotated

    from pydantic import ConfigDict, Field, StringConstraints, create_model

    non_empty_text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    return create_model(
        "XiquRecordCleaningResult",
        __doc__="Structured fields written to the matching ai_* keys in the source record.",
        __config__=ConfigDict(extra="forbid"),
        patient_appeal=(str, Field(..., description="清洗后的主诉；无有效主诉时为空字符串")),
        new_medical_history=(str, Field(..., description="清洗后的本次现病史")),
        old_medical_history=(str, Field(..., description="清洗后的既往史")),
        allergic_history=(str, Field(..., description="清洗后的过敏史；未说明时为空字符串")),
        diagnosis_illness=(non_empty_text, Field(..., description="非空的西医诊断")),
        diagnosis_sickness=(non_empty_text, Field(..., description="非空的中医病名")),
        diagnosis_disease=(non_empty_text, Field(..., description="非空的中医证候")),
        marriage_history=(Literal["已婚", "未婚", ""], Field(..., description="婚姻状态")),
        birth_detail=(str, Field(..., description="明确的孕产或生育事实；无信息时为空字符串")),
        personal_history=(str, Field(..., description="清洗后的个人史")),
        family_history=(str, Field(..., description="清洗后的家族史")),
        special_history=(
            Literal["妊娠状态", "备孕", "哺乳期", "月经期", "无"],
            Field(..., description="特殊时期；无特殊时期时为无"),
        ),
        diagnosis_illness_reason=(non_empty_text, Field(..., description="西医诊断的来源或修正依据")),
        diagnosis_disease_reason=(non_empty_text, Field(..., description="中医证候的来源或修正依据")),
        diagnosis_sickness_reason=(non_empty_text, Field(..., description="中医病名的来源或修正依据")),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="使用结构化清洗 Agent 清洗西区病历 JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"待清洗 JSONL 文件（默认：{DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MEDICAL_AGENT_MODEL", "gpt-4.1-mini"),
        help="OpenAI 兼容接口模型名称",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="接口 API Key；也可通过 DASHSCOPE_API_KEY 环境变量提供",
    )
    parser.add_argument("--timeout", type=float, default=300, help="单次模型请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="底层请求失败时的重试次数")
    parser.add_argument(
        "--batchsize",
        "--batch-size",
        "--batch_size",
        dest="batchsize",
        type=int,
        default=1,
        help="每批并发处理的病历数，默认 1（顺序处理）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理的非空记录数；其余行原样保留",
    )
    parser.add_argument(
        "--skip",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="跳过已完成清洗和临床抽取的记录，可写 --skip 或 --skip true；默认 false",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=None,
        help="失败记录另存路径；默认与输入文件同目录的 <文件名>_failed.jsonl",
    )
    parser.add_argument(
        "--merge-target",
        type=Path,
        default=None,
        help="将成功清洗结果按 order_sn 回写到指定 JSONL 主文件，只更新 ai_* 字段",
    )
    return parser.parse_args()

# TODO 线下数据还用不到图片
def _without_image_references(value: Any) -> Any:
    """Keep readable report metadata while dropping image paths, URLs, and binary data."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(token in normalized_key for token in ("url", "path", "base64")):
                continue
            if normalized_key in {"image", "img", "file", "文件", "图片", "图片地址"}:
                continue
            cleaned = _without_image_references(item)
            if cleaned is not None:
                result[key] = cleaned
        return result
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _without_image_references(item)) is not None]
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized.startswith(("http://", "https://", "file://", "data:image/")):
            return None
        suffix = Path(normalized.split("?", maxsplit=1)[0]).suffix
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".pdf"}:
            return None
    return value


def _text(value: Any) -> str:
    """Normalize scalar input text without exposing sentinel values."""
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"none", "null", "nan", "<na>"} else value


def _visit_time(record: dict[str, Any]) -> str:
    return _text(record.get("see_doc_time") or record.get("start_time") or record.get("created_at"))


def _visit_datetime(record: dict[str, Any]) -> datetime | None:
    value = _visit_time(record).strip()
    if not value:
        return None
    normalized = value.replace("年", "-").replace("月", "-").replace("日", "")
    normalized = normalized.replace("/", "-").replace(".", "-")
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _patient_key(record: dict[str, Any]) -> str:
    """Build an in-memory patient key without passing identifiers to the model."""
    patient_id = _text(record.get("patient_id") or record.get("user_info_id") or record.get("patient_idcard"))
    if not patient_id:
        return ""
    return f"{_text(record.get('doctor_id'))}:{patient_id}"


def build_previous_visit_contexts(records: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Find each row's closest earlier visit for the same patient in this input file."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for line_number, record in records.items():
        key = _patient_key(record)
        if key:
            grouped.setdefault(key, []).append((line_number, record))

    contexts: dict[int, dict[str, Any]] = {}
    for visits in grouped.values():
        for line_number, record in visits:
            current_time = _visit_datetime(record)
            candidates = []
            for previous_line, previous in visits:
                if previous_line == line_number:
                    continue
                previous_time = _visit_datetime(previous)
                if current_time is not None:
                    if previous_time is None or previous_time > current_time:
                        continue
                    if previous_time == current_time and previous_line > line_number:
                        continue
                elif previous_line >= line_number:
                    continue
                candidates.append((previous_time, previous_line, previous))
            if not candidates:
                continue

            previous_time, _, previous = max(
                candidates,
                key=lambda item: (item[0] is not None, item[0] or datetime.min, item[1]),
            )
            diagnoses = {
                field: _text(previous.get(f"ai_{field}") or previous.get(field))
                for field in ("diagnosis_illness", "diagnosis_disease", "diagnosis_sickness")
            }
            contexts[line_number] = {
                "previous_visit_time": _visit_time(previous),
                "previous_visit_appeal": _text(
                    previous.get("ai_patient_appeal")
                    or previous.get("patient_appeal")
                    or previous.get("doc_ass_stu_appeal")
                ),
                "previous_visit_diagnoses": diagnoses,
            }
    return contexts


def build_record_payload(
    record: dict[str, Any], previous_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Select only the clinical text needed by the cleaner, excluding patient identifiers."""
    keys = (
        "see_doc_time",
        "patient_sex",
        "patient_appeal",
        "doc_ass_stu_appeal",
        "new_medical_history",
        "old_medical_history",
        "allergic_history",
        "personal_history",
        "family_history",
        "special_history",
        "birth_detail",
        "diagnosis_illness",
        "diagnosis_sickness",
        "diagnosis_disease",
    )
    payload = {
        key: _without_image_references(record[key])
        for key in keys
        if key in record and record[key] not in (None, "", [])
    }
    visit_type = _text(record.get("is_first"))
    if visit_type:
        payload["visit_type"] = visit_type
    marriage_history = _text(record.get("marriage_history") or record.get("is_marriage_history"))
    if marriage_history:
        payload["marriage_history"] = marriage_history
    if previous_context:
        payload.update(previous_context)
    return payload


def _cleaned_value(record: dict[str, Any], field: str) -> Any:
    """Prefer the cleaner's result when available, otherwise use the source value."""
    ai_field = f"ai_{field}"
    return record[ai_field] if ai_field in record else record.get(field)


def build_clinical_extraction_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build a de-identified clinical payload for ClinicalExtractionResult."""
    history_fields = {
        "patient_appeal": "主诉",
        "new_medical_history": "现病史",
        "old_medical_history": "既往史",
        "allergic_history": "过敏史",
        "personal_history": "个人史",
        "family_history": "家族史",
        "special_history": "特殊时期",
        "marriage_history": "婚姻史",
        "birth_detail": "生育史",
    }
    clinical_record: dict[str, Any] = {
        label: _without_image_references(_cleaned_value(record, field))
        for field, label in history_fields.items()
        if _cleaned_value(record, field) not in (None, "", [])
    }
    clinical_record["就诊时间"] = _visit_time(record)
    clinical_record["就诊类型"] = _text(record.get("is_first"))
    clinical_record["性别"] = _text(record.get("patient_sex"))

    diagnoses = {
        label: _cleaned_value(record, field)
        for field, label in (
            ("diagnosis_illness", "西医诊断"),
            ("diagnosis_sickness", "中医病名"),
            ("diagnosis_disease", "中医证候"),
        )
        if _cleaned_value(record, field) not in (None, "", [])
    }
    if diagnoses:
        clinical_record["诊断"] = diagnoses

    for key, label in (
        ("ai_inspection_report_img", "检查报告分析"),
        ("ai_tongue_face_img", "舌面患处分析"),
    ):
        value = _without_image_references(record.get(key))
        if value not in (None, "", []):
            clinical_record[label] = value

    disposition = _text(record.get("disposition"))
    if disposition:
        clinical_record["治疗计划"] = disposition
    prescriptions = _without_image_references(record.get("ps"))
    if prescriptions not in (None, "", []):
        clinical_record["处方详情"] = prescriptions

    return {"病历信息": clinical_record, "知识库知识": "无"}


def apply_result(record: dict[str, Any], result: Any) -> dict[str, Any]:
    """Map structured output fields to their ai_* counterparts."""
    for field in CLEANER_OUTPUT_FIELDS:
        value = getattr(result, field)
        record[f"ai_{field}"] = value.strip() if isinstance(value, str) else value
    return record


def apply_clinical_extraction_result(record: dict[str, Any], result: Any) -> dict[str, Any]:
    """Write ClinicalExtractionResult fields to the corresponding ai_* keys."""
    values = result.model_dump()
    for field, value in values.items():
        if isinstance(value, str):
            value = value.strip()
        elif isinstance(value, list):
            value = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        record[f"ai_{field}"] = value
    return record


def is_record_cleaned(record: dict[str, Any]) -> bool:
    """Return true only when both cleaner and clinical extraction outputs exist."""
    required_fields = (*CLEANER_OUTPUT_FIELDS, *CLINICAL_EXTRACTION_FIELDS)
    return all(f"ai_{field}" in record for field in required_fields)


def write_jsonl_atomic(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically write records as JSONL, including an empty list when requested."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def replace_jsonl_records(
    output_path: Path,
    original_lines: list[str],
    records: dict[int, dict[str, Any]],
    changed_line_numbers: set[int],
) -> None:
    """Atomically replace selected JSONL rows and preserve untouched lines."""
    original_mode = output_path.stat().st_mode & 0o777
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            for line_number, line in enumerate(original_lines, start=1):
                if line_number in changed_line_numbers:
                    serialized = json.dumps(records[line_number], ensure_ascii=False, separators=(",", ":"))
                    target.write(serialized + "\n")
                else:
                    target.write(line)
            target.flush()
            os.fsync(target.fileno())

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def index_merge_target(
    target_records: dict[int, dict[str, Any]],
    source_records: dict[int, dict[str, Any]],
    selected_line_numbers: list[int],
) -> dict[str, int]:
    """Match selected retry rows to unique target rows by order_sn before model calls."""
    source_orders = [_text(source_records[line].get("order_sn")) for line in selected_line_numbers]
    if any(not order_sn for order_sn in source_orders):
        raise ValueError("重试文件中存在空 order_sn，无法回写主文件")
    if len(source_orders) != len(set(source_orders)):
        raise ValueError("重试文件中存在重复 order_sn，无法唯一回写主文件")

    wanted = set(source_orders)
    target_order_to_line: dict[str, int] = {}
    duplicate_targets: set[str] = set()
    for line_number, record in target_records.items():
        order_sn = _text(record.get("order_sn"))
        if order_sn not in wanted:
            continue
        if order_sn in target_order_to_line:
            duplicate_targets.add(order_sn)
        else:
            target_order_to_line[order_sn] = line_number

    if duplicate_targets:
        raise ValueError(f"主文件中存在重复 order_sn，无法唯一回写：{', '.join(sorted(duplicate_targets))}")
    missing = wanted - target_order_to_line.keys()
    if missing:
        raise ValueError(f"以下 order_sn 未在主文件中找到：{', '.join(sorted(missing))}")
    return target_order_to_line


def merge_cleaned_records(
    target_path: Path,
    target_lines: list[str],
    target_records: dict[int, dict[str, Any]],
    target_order_to_line: dict[str, int],
    cleaned_records: dict[int, dict[str, Any]],
    successful_line_numbers: set[int],
) -> int:
    """Merge cleaner and extraction outputs into matching target rows by order_sn."""
    changed_lines: set[int] = set()
    fields = (*CLEANER_OUTPUT_FIELDS, *CLINICAL_EXTRACTION_FIELDS)
    for source_line in successful_line_numbers:
        cleaned = cleaned_records[source_line]
        target_line = target_order_to_line[_text(cleaned.get("order_sn"))]
        target = target_records[target_line]
        for field in fields:
            ai_field = f"ai_{field}"
            if ai_field in cleaned:
                target[ai_field] = cleaned[ai_field]
        changed_lines.add(target_line)

    if changed_lines:
        replace_jsonl_records(target_path, target_lines, target_records, changed_lines)
    return len(changed_lines)


async def clean_record(
    line_number: int,
    source_record: dict[str, Any],
    previous_context: dict[str, Any] | None,
    agent: Any,
    extraction_agent: Any,
    human_message_type: Any,
) -> dict[str, Any]:
    """Clean one independent record without mutating the shared input data."""
    record = copy.deepcopy(source_record)
    order_sn = _text(record.get("order_sn"))
    print(f"正在处理第 {line_number} 行，order_sn={order_sn}", flush=True)

    payload = build_record_payload(record, previous_context)
    response = await agent.ainvoke([human_message_type(content=json.dumps(payload, ensure_ascii=False))])
    apply_result(record, response.output)

    extraction_payload = build_clinical_extraction_payload(record)
    print(f"正在抽取临床字段，第 {line_number} 行，order_sn={order_sn}", flush=True)
    extraction_response = await extraction_agent.ainvoke(
        [human_message_type(content=json.dumps(extraction_payload, ensure_ascii=False))]
    )
    apply_clinical_extraction_result(record, extraction_response.output)
    return record


async def clean_file(args: argparse.Namespace) -> int:
    """Clean records in bounded batches, retaining and exporting failed source rows."""
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    if args.batchsize < 1:
        raise ValueError("--batchsize 必须是大于 0 的整数")
    failed_output = getattr(args, "failed_output", None)
    if failed_output is None:
        failed_output = input_path.with_name(f"{input_path.stem}_failed.jsonl")
    else:
        failed_output = failed_output.expanduser().resolve()
    if failed_output == input_path:
        raise ValueError("--failed-output 不能与 --input 指向同一个文件")
    merge_target = getattr(args, "merge_target", None)
    if merge_target is not None:
        merge_target = merge_target.expanduser().resolve()
        if not merge_target.is_file():
            raise FileNotFoundError(f"回写主文件不存在：{merge_target}")
        if merge_target in {input_path, failed_output}:
            raise ValueError("--merge-target 不能与 --input 或 --failed-output 指向同一个文件")

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是合法 JSON；原文件未覆盖") from exc
        if not isinstance(record, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象；原文件未覆盖")
        records[line_number] = record
    previous_contexts = build_previous_visit_contexts(records)

    skip_cleaned = bool(getattr(args, "skip", False))
    line_numbers = [
        line_number
        for line_number, record in records.items()
        if not skip_cleaned or not is_record_cleaned(record)
    ]
    skipped_count = len(records) - len(line_numbers)
    if args.limit is not None:
        line_numbers = line_numbers[: args.limit]
    target_lines: list[str] = []
    target_records: dict[int, dict[str, Any]] = {}
    target_order_to_line: dict[str, int] = {}
    if merge_target is not None and line_numbers:
        target_lines = merge_target.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(target_lines, start=1):
            if not line.strip():
                continue
            try:
                target_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"主文件第 {line_number} 行不是合法 JSON；主文件未覆盖") from exc
            if not isinstance(target_record, dict):
                raise ValueError(f"主文件第 {line_number} 行必须是 JSON 对象；主文件未覆盖")
            target_records[line_number] = target_record
        target_order_to_line = index_merge_target(target_records, records, line_numbers)
    print(
        f"记录 {len(records)} 条，跳过已完成清洗 {skipped_count} 条，"
        f"本次计划处理 {len(line_numbers)} 条"
    )
    if not line_numbers:
        print("没有待处理记录")
        return 0
    if not args.api_key:
        raise ValueError("未提供 API Key，请设置 DASHSCOPE_API_KEY 或使用 --api-key")

    from langchain_core.messages import HumanMessage

    from medical.data_utils.structured_agent import AgentModelConfig, StructuredAgent, StructuredOutputMode
    from medical.schema.clear_record_basemodel import ClinicalExtractionResult

    output_schema = build_output_schema()
    config = AgentModelConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0,
        thinking=False,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    system_prompt = load_system_prompt()
    agent = StructuredAgent.from_config(
        config,
        output_schema,
        system_prompt=system_prompt,
        name="xiqu_medical_record_cleaner",
        output_mode=StructuredOutputMode.JSON_OBJECT,
        validation_retries=1,
    )
    extraction_agent = StructuredAgent.from_config(
        config,
        ClinicalExtractionResult,
        system_prompt=load_extraction_prompt(),
        name="xiqu_clinical_extractor",
        output_mode=StructuredOutputMode.JSON_OBJECT,
        validation_retries=1,
    )

    attempted = 0
    successful_line_numbers: set[int] = set()
    failed_records: list[dict[str, Any]] = []
    for start in range(0, len(line_numbers), args.batchsize):
        batch_line_numbers = line_numbers[start : start + args.batchsize]
        results = await asyncio.gather(
            *(
                clean_record(
                    line_number,
                    records[line_number],
                    previous_contexts.get(line_number),
                    agent,
                    extraction_agent,
                    HumanMessage,
                )
                for line_number in batch_line_numbers
            ),
            return_exceptions=True,
        )

        for line_number, result in zip(batch_line_numbers, results, strict=True):
            if isinstance(result, Exception):
                failed_records.append(records[line_number])
                order_sn = _text(records[line_number].get("order_sn"))
                print(
                    f"[失败] 第 {line_number} 行，order_sn={order_sn}，"
                    f"{type(result).__name__}: {result}；原记录保留"
                )
                continue
            if isinstance(result, BaseException):
                raise result
            records[line_number] = result
            successful_line_numbers.add(line_number)

        attempted += len(batch_line_numbers)
        print(
            f"已处理 {attempted}/{len(line_numbers)} 条，成功 {len(successful_line_numbers)} 条，"
            f"失败 {len(failed_records)} 条"
        )

    if failed_records:
        write_jsonl_atomic(failed_output, failed_records)
        print(f"失败记录已另存 {len(failed_records)} 条，可将该文件作为 --input 重试：{failed_output}")

    merged_count = 0
    if merge_target is not None and successful_line_numbers:
        merged_count = merge_cleaned_records(
            merge_target,
            target_lines,
            target_records,
            target_order_to_line,
            records,
            successful_line_numbers,
        )
        print(f"已按 order_sn 将 {merged_count} 条成功结果回写到主文件：{merge_target}")

    if successful_line_numbers:
        replace_jsonl_records(input_path, lines, records, successful_line_numbers)

    print(
        f"完成：成功 {len(successful_line_numbers)} 条，失败 {len(failed_records)} 条，"
        f"已更新 {input_path}"
        + (f"，已回写主文件 {merged_count} 条" if merge_target is not None else "")
    )
    return len(successful_line_numbers)


def main() -> None:
    """Load project environment and run the cleaner."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    """
    # 重新跑失败case
    python medical/data_utils/raw_data_clear/clear_xiqu_withai.py \
  --input medical/processed_data/doctor_二期西区_李方洁/medical_records_ai_failed.jsonl\
  --merge-target medical/processed_data/doctor_二期西区_李方洁/medical_records_ai.jsonl \
  --failed-output medical/processed_data/doctor_二期西区_李方洁/medical_records_ai_failed_remaining.jsonl \
  --skip true \
  --batchsize 1

    # 跑数据doctor A
    python medical/data_utils/raw_data_clear/clear_xiqu_withai.py --batchsize 6 --input medical/processed_data/doctor_二期西区_李方洁/medical_records_ai.jsonl
    
    """
    args = parse_args()
    asyncio.run(clean_file(args))


if __name__ == "__main__":
    main()
