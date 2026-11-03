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
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_system_prompt() -> str:
    """Load the cleaner prompt from the project-local text file."""
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"清洗提示词文件不存在：{PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"清洗提示词文件为空：{PROMPT_PATH}")
    return prompt


def build_output_schema() -> type:
    """Build the structured output model after CLI parsing, keeping --help lightweight."""
    from typing import Annotated

    from pydantic import ConfigDict, Field, StringConstraints, create_model

    non_empty_text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    return create_model(
        "XiquRecordCleaningResult",
        __doc__="Structured fields written to the matching ai_* keys in the source record.",
        __config__=ConfigDict(extra="forbid"),
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
        "--limit",
        type=int,
        default=None,
        help="最多处理的非空记录数；其余行原样保留",
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


def apply_result(record: dict[str, Any], result: Any) -> dict[str, Any]:
    """Map structured output fields to their ai_* counterparts."""
    for field in (
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
    ):
        value = getattr(result, field)
        record[f"ai_{field}"] = value.strip() if isinstance(value, str) else value
    return record


async def clean_file(args: argparse.Namespace) -> int:
    """Clean records sequentially and atomically replace the specified JSONL file."""
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    if not args.api_key:
        raise ValueError("未提供 API Key，请设置 DASHSCOPE_API_KEY 或使用 --api-key")

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

    from langchain_core.messages import HumanMessage

    from medical.data_utils.structured_agent import AgentModelConfig, StructuredAgent, StructuredOutputMode

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

    original_mode = input_path.stat().st_mode & 0o777
    temp_path: Path | None = None
    processed = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=input_path.parent,
            prefix=f".{input_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    target.write(line)
                    continue

                if args.limit is not None and processed >= args.limit:
                    target.write(line)
                    continue

                try:
                    record = records[line_number]
                    payload = build_record_payload(record, previous_contexts.get(line_number))
                    response = await agent.ainvoke(
                        [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
                    )
                    apply_result(record, response.output)
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                except Exception as exc:
                    raise RuntimeError(
                        f"第 {line_number} 行清洗失败（{type(exc).__name__}）；原文件未覆盖"
                    ) from exc

                processed += 1
                if processed % 20 == 0:
                    print(f"已完成 {processed} 条")

            target.flush()
            os.fsync(target.fileno())

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, input_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"完成：处理 {processed} 条，已写回 {input_path}")
    return processed


def main() -> None:
    """Load project environment and run the cleaner."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    args = parse_args()
    asyncio.run(clean_file(args))


if __name__ == "__main__":
    main()
