#!/usr/bin/env python3
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

"""使用 LangChain 对多医生病历进行结构化 AI 补全和清洗.

默认遍历 ``medical/data/202301_online/`` 下所有医生的 JSON 数据，也支持传入单个
JSON/JSONL 文件。
原始字段不会被覆盖，所有模型生成结果均写入 ``ai_`` 前缀字段。
处理结果按医生增量追加到 ``medical/processed_data/``；成功记录再次运行时自动跳过，
失败记录写入独立 JSONL 并在后续运行中重试。

示例:
    python medical/data_utils/medical_record_agents.py \
        --model gpt-4.1-mini \
        --vlm-model gpt-4.1-mini
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from datetime import date
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, TextIO
from urllib.parse import unquote_to_bytes

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from medical.utils.log import logger
from dotenv import load_dotenv
from medical.schema.clear_record_basemodel import (
    AgentReviewResult,
    ClinicalExtractionResult,
    CurrentVisitHistoryResult,
    DiagnosisResult,
    HistoryCleaningResult,
    ImageClassificationResult,
    InspectionResult,
    TONGUE_FACE_RESULT_FIELD_CN_MAPPING,
    TongueFaceResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical.analysis.engine import normalize_diagnosis_value, normalize_sickness_value, normalize_syndrome_value  # noqa: E402

DEFAULT_INPUT = Path("medical/data/202301_online")
DEFAULT_OUTPUT_DIR = Path("medical/processed_data")
DOCTOR_RECORD_FILE_NAME = "medical_records_ai.jsonl"
FAILURE_FILE_NAME = "medical_record_agent_failures.jsonl"
FILTER_FILE_NAME = "filter.json"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DIAGNOSIS_FIELDS = ("diagnosis_illness", "diagnosis_disease", "diagnosis_sickness")
HISTORY_FIELDS = (
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "family_history",
)
IMAGE_FIELDS = ("tongue_face_img", "admin_face_img", "admin_report_img", "inspection_report_img")
IMAGE_CATEGORIES = ("舌", "面", "患处", "检验检查报告类", "其他类")
TONGUE_FACE_IMAGE_CATEGORIES = ("舌", "面", "患处")
MAX_VLM_IMAGES_PER_REQUEST = 8
OCR_BATCH_CONCURRENCY = 3
MAX_CLASSIFICATION_IMAGES_PER_REQUEST = 8
IMAGE_CLASSIFICATION_BATCH_CONCURRENCY = 2
CLASSIFICATION_IMAGE_MAX_EDGE = 512
CLASSIFICATION_MODEL_NAME = "qwen3.7-flash"
CLASSIFICATION_MODEL_MAX_TOKENS = 512
IMAGE_DEDUP_DOWNLOAD_CONCURRENCY = 8
IMAGE_DEDUP_DOWNLOAD_TIMEOUT = 10.0
CURRENT_VISIT_HISTORY_SYSTEM_PROMPT = (
    "你是临床病历现病史整理助手。输入可能包含同一患者按日期累积的多次就诊记录。"
    "必须以本次就诊时间为锚点，仅提取与本次复诊对应的现病史，不得混入既往就诊段落。"
    "若存在括号日期、初诊/复诊序号等标记，选择与本次时间相同或最近且不晚于本次时间的段落。"
    "整理为规范现病史：保留本次症状、变化、持续时间、相关治疗反应、重要阴性表现及"
    "与本次相关的检查；纠正明确错别字，删除乱码、重复符号和无意义文本。"
    "检查报告中的异常方向符号必须转换为中文文本描述，例如 ↑、⬆️ 写作“升高”，↓、⬇️ 写作“下降”。"
    "不得新增原文没有的症状、诊断、时间或检查结果；无法确定本次段落时返回空字符串。"
)
INVALID_HISTORY_TEXT = (
    "",
    "-",
    "--",
    "/",
    "无特殊",
    "不详",
    "未知",
    "未填写",
    "未提供",
    "暂无",
    "否认",
    "正常",
    "同上",
)

UPWARD_ABNORMAL_SYMBOL_PATTERN = re.compile(r"(?:⬆️|⬆|↑|↗|⇧|▲|△)+")
DOWNWARD_ABNORMAL_SYMBOL_PATTERN = re.compile(r"(?:⬇️|⬇|↓|↘|⇩|▼|▽)+")
FILTER_FILE_LOCK = Lock()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _env(name: str, default: str = "") -> str:
    return _text(os.getenv(name)) or default


def _env_any(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _mask_secret(value: str | None) -> str:
    secret = _text(value)
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


def _clean_generated_text(value: Any) -> str:
    """清除模型偶尔返回的 JSON 字符串外层引号."""
    text = _text(value)
    for _ in range(2):
        if len(text) < 2 or not (text.startswith('"') and text.endswith('"')):
            break
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            break
        if not isinstance(decoded, str):
            break
        text = decoded.strip()
    return "" if text.lower() in {"null", "none"} else text


def _clean_generated_value(value: Any) -> Any:
    if isinstance(value, str) or value is None:
        return _clean_generated_text(value)
    if isinstance(value, list):
        cleaned_items = [_clean_generated_value(item) for item in value]
        return list(dict.fromkeys(item for item in cleaned_items if item not in ("", None)))
    if isinstance(value, dict):
        return {key: _clean_generated_value(item) for key, item in value.items()}
    return value


class _MedicalHistoryHTMLParser(HTMLParser):
    """提取前端富文本中的纯文本，并保留块级标签的段落边界."""

    BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def _append_break(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self._append_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _normalize_medical_history(value: Any) -> str:
    """删除现病史中的前端 HTML 标签，解码实体并规整段落空白."""
    text = _text(value)
    if not text:
        return ""
    parser = _MedicalHistoryHTMLParser()
    parser.feed(text)
    parser.close()
    lines = []
    for line in "".join(parser.parts).replace("\xa0", " ").splitlines():
        normalized = re.sub(r"[\t\f\v ]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def _replace_inspection_report_symbols(value: Any) -> Any:
    """将检查报告常见异常符号转换为中文文本，便于后续病史拼接."""
    if isinstance(value, str):
        text = UPWARD_ABNORMAL_SYMBOL_PATTERN.sub("升高", value)
        return DOWNWARD_ABNORMAL_SYMBOL_PATTERN.sub("下降", text)
    if isinstance(value, list):
        return [_replace_inspection_report_symbols(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_inspection_report_symbols(item) for key, item in value.items()}
    return value


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _chief_complaint(record: dict[str, Any]) -> str:
    return _text(record.get("doc_ass_stu_appeal"))

def _chief_patient_complaint(record: dict[str, Any]) -> str:
    return _text(record.get("doc_ass_stu_appeal"))



def _present_history(record: dict[str, Any]) -> str:
    return _text(record.get("new_medical_history") or record.get("present_history"))


def _latest_chief_complaint(record: dict[str, Any]) -> str:
    return _text(record.get("ai_patient_appeal")) or _chief_complaint(record)


def _latest_present_history(record: dict[str, Any]) -> str:
    return _text(record.get("ai_new_medical_history")) or _present_history(record)


def _diagnosis_present_history(record: dict[str, Any]) -> str:
    """排除病史中由检查报告补入、不可用于诊断生成的内容."""
    history = _latest_present_history(record)
    for marker in ("本次检查资料：", "本次检查资料:"):
        if marker in history:
            history = history.split(marker, 1)[0].rstrip("；;，,。 ")
    return history


def _visit_time(record: dict[str, Any]) -> str:
    return _text(record.get("see_doc_time") or record.get("start_time") or record.get("created_at"))


REPORT_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{4})\s*(?:年|[-/.])"
    r"\s*(?P<month>\d{1,2})(?:\s*(?:月|[-/.])\s*(?P<day>\d{1,2}))?"
)


def _parse_report_date(value: Any) -> tuple[date, str] | None:
    match = REPORT_DATE_PATTERN.search(_text(value))
    if not match:
        return None
    precision = "day" if match.group("day") else "month"
    try:
        parsed = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day") or 1),
        )
    except ValueError:
        return None
    return parsed, precision


def _relative_report_time(report_date: Any, visit_time: Any) -> str:
    """Describe a report date relative to see_doc_time using the largest useful unit."""
    parsed_report = _parse_report_date(report_date)
    parsed_visit = _parse_report_date(visit_time)
    if parsed_report is None or parsed_visit is None:
        return ""
    report_day, report_precision = parsed_report
    visit_day, _ = parsed_visit
    delta_days = (visit_day - report_day).days
    if delta_days == 0:
        return "当天"
    suffix = "前" if delta_days > 0 else "后"
    absolute_days = abs(delta_days)
    if report_precision == "month":
        months = abs((visit_day.year - report_day.year) * 12 + visit_day.month - report_day.month)
        if months >= 12:
            return f"{months // 12}年{suffix}"
        if months:
            return f"{months}个月{suffix}"
    if absolute_days >= 365:
        return f"{absolute_days // 365}年{suffix}"
    if absolute_days >= 30:
        return f"{absolute_days // 30}个月{suffix}"
    return f"{absolute_days}天{suffix}"


def _patient_id(record: dict[str, Any]) -> str:
    patient_id = _text(record.get("patient_id"))
    if not patient_id:
        patient_id = _text(record.get("user_info_id"))
    return patient_id


def _patient_key(record: dict[str, Any]) -> str:
    patient_id = _patient_id(record)
    if not patient_id:
        return ""
    return f"{_text(record.get('doctor_id'))}:{patient_id}"


def _history_payload(record: dict[str, Any], ai: bool = False) -> dict[str, str]:
    prefix = "ai_" if ai else ""
    return {field: _text(record.get(f"{prefix}{field}")) for field in HISTORY_FIELDS}


def _diagnosis_payload(record: dict[str, Any], *, prefer_ai: bool = False) -> dict[str, str]:
    diagnoses = {}
    for field in DIAGNOSIS_FIELDS:
        ai_value = _text(record.get(f"ai_{field}"))
        raw_value = _text(record.get(field))
        diagnoses[field] = (ai_value or raw_value) if prefer_ai else raw_value
    return diagnoses


def _diagnosis_reason_fallback(
    field: str,
    diagnosis: str,
    source_diagnoses: dict[str, str],
    previous_diagnoses: dict[str, str],
    chief_complaint: str,
    present_history: str,
) -> str:
    """在模型遗漏 reason 时生成基于输入上下文的可核验原因."""
    source_labels = {
        "diagnosis_illness": "西医诊断",
        "diagnosis_disease": "中医证候",
        "diagnosis_sickness": "中医病名",
    }
    for source_field, source_value in source_diagnoses.items():
        if source_field != field and diagnosis in source_value:
            return (
                f"原字段为空，从原始{source_labels[source_field]}字段“{source_value}”中"
                f"提取并补全为“{diagnosis}”。"
            )

    def evidence_excerpt(value: str, limit: int = 80) -> str:
        text = re.sub(r"\s+", " ", value).strip()
        return text if len(text) <= limit else f"{text[:limit]}……"

    evidence = []
    if chief_complaint:
        evidence.append(f"本次主诉“{evidence_excerpt(chief_complaint)}”")
    if present_history:
        evidence.append(f"本次现病史“{evidence_excerpt(present_history)}”")
    previous_value = previous_diagnoses.get(field, "")
    if previous_value and diagnosis in previous_value:
        evidence.append(f"上一次同患者诊断“{previous_value}”")
    if evidence:
        return f"原字段为空，依据{'、'.join(evidence)}补全为“{diagnosis}”。"
    return f"原字段为空，依据已有诊断上下文补全为“{diagnosis}”；当前病历未提供更具体的补全依据。"


def _has_any_diagnosis(diagnoses: dict[str, str]) -> bool:
    return any(_text(value) for value in diagnoses.values())


def _should_filter_record(record: Mapping[str, Any]) -> bool:
    """Filter records with insufficient complaint/history/diagnosis context."""
    patient_appeal = _text(record.get("patient_appeal"))
    present_history = _text(record.get("new_medical_history"))
    diagnosis_illness = _text(record.get("diagnosis_illness"))
    if not any((patient_appeal, present_history, diagnosis_illness)):
        return True
    return bool(patient_appeal and len(patient_appeal) <= 20 and not present_history and not diagnosis_illness)


def _filtered_record_identity(record: Mapping[str, Any]) -> str:
    order_sn = _text(record.get("order_sn"))
    if order_sn:
        return f"order_sn:{order_sn}"
    return _record_identity(dict(record))


def write2filter_json(record: Mapping[str, Any], path: Path | None = None) -> bool:
    """Append a filtered record to a JSON array, deduplicated by order_sn or record identity."""
    target = path or (DEFAULT_OUTPUT_DIR / FILTER_FILE_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    filtered_record = dict(record)
    filtered_record.pop("__original_diagnoses", None)
    filtered_record.pop("__previous_diagnoses", None)
    identity = _filtered_record_identity(filtered_record)

    with FILTER_FILE_LOCK:
        records: list[dict[str, Any]] = []
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"过滤文件不是合法 JSON: {target}") from exc
            if not isinstance(existing, list) or not all(isinstance(item, dict) for item in existing):
                raise ValueError(f"过滤文件必须是 JSON 对象数组: {target}")
            records = existing
        if any(_filtered_record_identity(item) == identity for item in records):
            return False
        records.append(filtered_record)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    return True


def _model_dump(model: BaseModel, *, by_alias: bool = False) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 的序列化接口."""
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


def _format_tongue_face_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        items = [_format_tongue_face_value(item) for item in value]
        return "、".join(dict.fromkeys(item for item in items if item))
    return str(value).strip()


def _normalize_tongue_face_display_value(path: str, value: str) -> str:
    if "、" in value:
        items = [_normalize_tongue_face_display_value(path, item) for item in value.split("、")]
        return "、".join(item for item in items if item)

    replacements = {
        "tongue.tongue_coat.root": {
            "有根苔": "有根",
            "无根苔": "无根",
        },
        "tongue.tongue_coat.distribution": {
            "整体": "全舌",
        },
    }
    if value in replacements.get(path, {}):
        return replacements[path][value]
    if path in {
        "tongue.tongue_coat.color",
        "tongue.tongue_coat.thickness",
        "tongue.tongue_coat.moisture",
        "tongue.tongue_coat.texture",
    } and value.endswith("苔"):
        return value[:-1]
    return value


def _iter_tongue_face_leaf_values(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, BaseModel):
        value = _model_dump(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if isinstance(item, BaseModel) or isinstance(item, Mapping):
                yield from _iter_tongue_face_leaf_values(item, path)
            else:
                yield path, item
        return
    yield prefix, value


def format_tongue_face_result_descriptions(result: TongueFaceResult | Mapping[str, Any] | None) -> dict[str, str]:
    """将舌面患处结构化结果拼接为舌象、面象、患处三段文本."""
    if result is None:
        data: dict[str, Any] = {}
    elif isinstance(result, BaseModel):
        data = _model_dump(result)
    else:
        data = dict(result)

    descriptions = {}
    for section in ("tongue", "face", "lesions"):
        parts = []
        for path, value in _iter_tongue_face_leaf_values(data.get(section, {}), section):
            label = TONGUE_FACE_RESULT_FIELD_CN_MAPPING.get(path)
            text = _format_tongue_face_value(value)
            if path.endswith(".legal") and text == "是":
                continue
            if label and text:
                text = _normalize_tongue_face_display_value(path, text)
                parts.append(f"{label}：{text}")
        descriptions[section] = "；".join(parts)
    return descriptions


def format_tongue_face_result_text(result: TongueFaceResult | Mapping[str, Any] | None) -> str:
    """将舌面患处结构化结果转换为适合模型上下文的自然语言."""
    descriptions = format_tongue_face_result_descriptions(result)
    labels = {"tongue": "舌象", "face": "面象", "lesions": "患处"}
    sections = [
        f"{labels[section]}：{description}。"
        for section, description in descriptions.items()
        if description
    ]
    return "\n".join(sections) or "未见有效舌面患处信息。"


def _inspection_field(report: Mapping[str, Any], field: str, alias: str) -> Any:
    return report.get(field) if field in report else report.get(alias)


def format_inspection_result_text(result: InspectionResult | Mapping[str, Any] | None) -> str:
    """将有效检查报告转换为适合模型上下文的自然语言."""
    if result is None:
        data: dict[str, Any] = {}
    elif isinstance(result, BaseModel):
        data = _model_dump(result)
    else:
        data = dict(result)

    descriptions = []
    valid_index = 0
    for report in data.get("reports") or []:
        if isinstance(report, BaseModel):
            report = _model_dump(report)
        if not isinstance(report, Mapping):
            continue
        is_valid = _inspection_field(report, "is_valid_report", "是否有效检查报告")
        if is_valid is not True:
            continue
        valid_index += 1
        fields = (
            ("图片类别", _inspection_field(report, "image_type", "图片类别")),
            ("报告名称", report.get("report_name")),
            ("报告日期", _inspection_field(report, "report_date", "报告日期")),
            ("相对就诊时间", _inspection_field(report, "relative_to_visit_time", "相对就诊时间")),
            ("解析结果", _inspection_field(report, "content", "解析结果")),
            ("异常指标", _inspection_field(report, "abnormal_indicators", "异常指标")),
            ("报告结论", _inspection_field(report, "conclusion", "报告结论")),
        )
        details = [
            f"{label}：{text}"
            for label, value in fields
            if (text := _format_tongue_face_value(value))
        ]
        if details:
            descriptions.append(f"有效检查报告{valid_index}：" + "；".join(details) + "。")

    summary = _format_tongue_face_value(
        data.get("history_evidence_summary")
        if "history_evidence_summary" in data
        else data.get("现病史检查证据摘要")
    )
    if summary:
        descriptions.append(f"现病史检查证据摘要：{summary}。")
    return "\n".join(descriptions) or "未见有效检查报告信息。"


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def _tool_response_format(schema: type[BaseModel]) -> ToolStrategy[Any]:
    """统一通过工具调用生成并校验结构化响应."""
    return ToolStrategy(schema=schema, handle_errors=True)


def _model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema.schema()


async def _invoke_structured_candidate(
    agent: Any,
    messages: list[HumanMessage],
    schema: type[BaseModel],
) -> tuple[BaseModel, str]:
    last_error: StructuredOutputError | None = None
    for attempt in range(2):
        attempt_messages = list(messages)
        if attempt:
            attempt_messages.append(
                HumanMessage(
                    content=(
                        f"上一次 {schema.__name__} 结构化参数校验失败。请重新生成一次，"
                        "严格遵守字段类型；只调用一次结构化输出工具，不得重复调用。"
                    )
                )
            )
        try:
            response = await agent.ainvoke({"messages": attempt_messages})
        except StructuredOutputError as exc:
            last_error = exc
            continue
        structured_response = response.get("structured_response")
        if isinstance(structured_response, schema):
            return structured_response, _agent_response_trace(response)
        if isinstance(structured_response, dict):
            return schema.model_validate(structured_response), _agent_response_trace(response)
        raise RuntimeError(f"Agent 未返回 {schema.__name__} structured_response")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Agent 未返回 {schema.__name__} structured_response")


def _trace_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _model_dump(value)
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _agent_response_trace(response: Any) -> str:
    """Extract observable model messages and tool calls without requesting hidden reasoning."""
    if not isinstance(response, Mapping):
        return _text(response)
    trace = []
    for message in response.get("messages") or []:
        message_type = type(message).__name__
        if message_type in {"HumanMessage", "SystemMessage"}:
            continue
        entry = {
            "message_type": message_type,
            "content": _trace_value(getattr(message, "content", "")),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = _trace_value(tool_calls)
        invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
        if invalid_tool_calls:
            entry["invalid_tool_calls"] = _trace_value(invalid_tool_calls)
        trace.append(entry)
    return _json_for_prompt(trace) if trace else ""


def _review_message(
    *,
    agent_name: str,
    system_prompt: str,
    original_message: HumanMessage,
    schema: type[BaseModel],
    candidate: BaseModel,
    agent_trace: str,
) -> HumanMessage:
    original_content = original_message.content
    original_texts = []
    image_blocks = []
    if isinstance(original_content, str):
        original_texts.append(original_content)
    elif isinstance(original_content, list):
        for block in original_content:
            if isinstance(block, str):
                original_texts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "input_text"}:
                original_texts.append(_text(block.get("text")))
            elif isinstance(block, dict) and block.get("type") in {"image_url", "input_image"}:
                image_blocks.append(block)
    review_text = (
        f"【被评审 Agent】\n{agent_name}\n\n"
        f"【系统提示词】\n{system_prompt}\n\n"
        f"【原始用户输入】\n{chr(10).join(text for text in original_texts if text)}\n\n"
        f"【目标响应模型 JSON Schema】\n{_json_for_prompt(_model_json_schema(schema))}\n\n"
        f"【可观察执行轨迹】\n{agent_trace or '未提供额外执行轨迹，仅审核最终响应。'}\n\n"
        f"【当前结构化结果】\n{_json_for_prompt(_model_dump(candidate))}"
    )
    if not image_blocks:
        return HumanMessage(content=review_text)
    return HumanMessage(content=[{"type": "text", "text": review_text}, *image_blocks])


async def _review_candidate(
    reviewer_agent: Any,
    *,
    agent_name: str,
    system_prompt: str,
    original_message: HumanMessage,
    schema: type[BaseModel],
    candidate: BaseModel,
    agent_trace: str,
) -> AgentReviewResult:
    review, _ = await _invoke_structured_candidate(
        reviewer_agent,
        [
            _review_message(
                agent_name=agent_name,
                system_prompt=system_prompt,
                original_message=original_message,
                schema=schema,
                candidate=candidate,
                agent_trace=agent_trace,
            )
        ],
        AgentReviewResult,
    )
    if not isinstance(review, AgentReviewResult):
        return AgentReviewResult.model_validate(_model_dump(review))
    return review


def _revision_message(schema: type[BaseModel], candidate: BaseModel, review: AgentReviewResult) -> HumanMessage:
    instructions = review.revision_instructions.strip() or "；".join(review.issues)
    if not instructions:
        raise ValueError("评审未通过，但未提供任何具体问题或修改要求")
    return HumanMessage(
        content=(
            f"评审 Agent 判定上一次 {schema.__name__} 结果未通过。\n"
            f"具体问题：{_json_for_prompt(review.issues)}\n"
            f"修改要求：{instructions}\n"
            f"上一次结构化结果：{_json_for_prompt(_model_dump(candidate))}\n"
            "请依据原始输入和上述评审意见修订，返回一份完整的新结果。"
            "不得只返回修改字段，不得添加原始输入没有的信息。"
        )
    )


async def _invoke_structured_agent(
    agent: Any,
    message: HumanMessage,
    schema: type[BaseModel],
    *,
    reviewer_agent: Any | None = None,
    agent_name: str = "",
    system_prompt: str = "",
) -> BaseModel:
    messages = [message]
    for review_round in range(2):
        candidate, agent_trace = await _invoke_structured_candidate(agent, messages, schema)
        if reviewer_agent is None:
            return candidate
        review = await _review_candidate(
            reviewer_agent,
            agent_name=agent_name or schema.__name__,
            system_prompt=system_prompt,
            original_message=message,
            schema=schema,
            candidate=candidate,
            agent_trace=agent_trace,
        )
        if review.passed and review.followed_prompt and review.response_meets_requirements:
            return candidate
        logger.warning(
            f"[REVIEW] agent={agent_name or schema.__name__} round={review_round + 1} "
            f"issues={_json_for_prompt(review.issues)}"
        )
        if review_round == 1:
            raise ValueError(
                f"{agent_name or schema.__name__} 连续两轮未通过评审: {_json_for_prompt(review.issues)}"
            )
        messages = [message, _revision_message(schema, candidate, review)]
    raise RuntimeError(f"{agent_name or schema.__name__} 评审流程异常结束")


def _response_text(response: Any) -> str:
    """兼容 OpenAI 风格响应的字符串或内容块格式."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                texts.append(_text(block.get("text")))
        return "\n".join(text for text in texts if text).strip()
    return _text(content)


async def _invoke_json_object_model(
    model: Any,
    system_prompt: str,
    message: HumanMessage,
    schema: type[BaseModel],
    *,
    reviewer_agent: Any | None = None,
    agent_name: str = "",
) -> BaseModel:
    """调用仅支持 json_object 的 OpenAI 兼容模型，并在客户端完成模型校验."""
    human_messages = [message]
    for review_round in range(2):
        last_error: json.JSONDecodeError | ValidationError | None = None
        candidate = None
        for attempt in range(2):
            attempt_messages: list[Any] = [SystemMessage(content=system_prompt), *human_messages]
            if attempt:
                attempt_messages.append(
                    HumanMessage(
                        content=(
                            f"上一次返回内容无法通过 {schema.__name__} 校验。"
                            "请重新返回一个完整、可解析的 json 对象，所有要求字段均不得省略，"
                            "不要输出 Markdown、注释或额外说明。"
                        )
                    )
                )
            response = await model.ainvoke(attempt_messages)
            raw_text = _response_text(response)
            if raw_text.startswith("```") and raw_text.endswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
            try:
                candidate = schema.model_validate(json.loads(raw_text))
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        if candidate is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"模型未返回可解析的 {schema.__name__} json 对象")
        if reviewer_agent is None:
            return candidate
        review = await _review_candidate(
            reviewer_agent,
            agent_name=agent_name or schema.__name__,
            system_prompt=system_prompt,
            original_message=message,
            schema=schema,
            candidate=candidate,
            agent_trace=raw_text,
        )
        if review.passed and review.followed_prompt and review.response_meets_requirements:
            return candidate
        logger.warning(
            f"[REVIEW] agent={agent_name or schema.__name__} round={review_round + 1} "
            f"issues={_json_for_prompt(review.issues)}"
        )
        if review_round == 1:
            raise ValueError(
                f"{agent_name or schema.__name__} 连续两轮未通过评审: {_json_for_prompt(review.issues)}"
            )
        human_messages = [message, _revision_message(schema, candidate, review)]
    raise RuntimeError(f"{agent_name or schema.__name__} 评审流程异常结束")


def _image_value(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for key in ("img", "url", "image_url", "path", "file_path", "src", "origin_url"):
        value = item.get(key)
        if isinstance(value, dict):
            value = value.get("url")
        if value:
            return _text(value)
    return ""


def _as_images(value: Any) -> list[str]:
    items = value if isinstance(value, list) else ([value] if value else [])
    return [image for image in (_image_value(item) for item in items) if image]


def _record_images(record: dict[str, Any]) -> list[str]:
    """汇总四个图片字段并按首次出现顺序去重."""
    images = []
    seen = set()
    for field in IMAGE_FIELDS:
        for image in _as_images(record.get(field)):
            if image not in seen:
                seen.add(image)
                images.append(image)
    return images


def _to_image_url(value: str, input_dir: Path) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = input_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _multimodal_message(prompt: str, images: list[str], input_dir: Path) -> HumanMessage:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": _to_image_url(image, input_dir)}})
    return HumanMessage(content=content)


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def _read_image_content(
    image: str,
    input_dir: Path,
    client: httpx.AsyncClient | None,
    semaphore: asyncio.Semaphore,
) -> bytes | None:
    """Read image bytes for exact-content deduplication and classification resizing."""
    try:
        if image.startswith(("http://", "https://")):
            if client is None:
                return None
            async with semaphore, client.stream("GET", image) as response:
                response.raise_for_status()
                chunks = []
                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunks.append(chunk)
                return b"".join(chunks) or None

        if image.startswith("data:"):
            header, separator, payload = image.partition(",")
            if not separator:
                return None
            return base64.b64decode(payload) if ";base64" in header.lower() else unquote_to_bytes(payload)
        else:
            path = Path(image).expanduser()
            if not path.is_absolute():
                path = input_dir / path
            if not path.is_file():
                return None
            return await asyncio.to_thread(path.read_bytes)
    except Exception as exc:
        logger.warning(f"图片内容去重读取失败，保留原图片: {type(exc).__name__}")
        return None


def _classification_image_data_url(content: bytes) -> str | None:
    """Resize a classification-only copy while preserving the original downstream image."""
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail(
                (CLASSIFICATION_IMAGE_MAX_EDGE, CLASSIFICATION_IMAGE_MAX_EDGE),
                Image.Resampling.LANCZOS,
            )
            if image.mode != "RGB":
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except (OSError, ValueError):
        return None


async def _image_classification_message(
    images: list[str],
    input_dir: Path,
) -> list[tuple[HumanMessage, list[str]]]:
    """Build at-most-eight-image messages after URL and exact-content deduplication."""
    url_unique_images = list(dict.fromkeys(image for image in (_text(image) for image in images) if image))
    remote_images = any(image.startswith(("http://", "https://")) for image in url_unique_images)
    client = (
        httpx.AsyncClient(
            follow_redirects=True,
            timeout=IMAGE_DEDUP_DOWNLOAD_TIMEOUT,
            limits=httpx.Limits(max_connections=IMAGE_DEDUP_DOWNLOAD_CONCURRENCY),
        )
        if remote_images
        else None
    )
    try:
        semaphore = asyncio.Semaphore(IMAGE_DEDUP_DOWNLOAD_CONCURRENCY)
        image_contents = await asyncio.gather(
            *(_read_image_content(image, input_dir, client, semaphore) for image in url_unique_images)
        )
    finally:
        if client is not None:
            await client.aclose()

    unique_images: list[str] = []
    classification_urls: list[str] = []
    seen_digests = set()
    for image, image_content in zip(url_unique_images, image_contents, strict=True):
        if image_content:
            digest = hashlib.sha256(image_content).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
        unique_images.append(image)
        resized_url = (
            await asyncio.to_thread(_classification_image_data_url, image_content)
            if image_content
            else None
        )
        classification_urls.append(resized_url or _to_image_url(image, input_dir))

    batches = []
    for start in range(0, len(unique_images), MAX_CLASSIFICATION_IMAGES_PER_REQUEST):
        batch_images = unique_images[start : start + MAX_CLASSIFICATION_IMAGES_PER_REQUEST]
        batch_urls = classification_urls[start : start + MAX_CLASSIFICATION_IMAGES_PER_REQUEST]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"下面共有 {len(batch_images)} 张图片。"
                    "请严格按图片序号逐张分类，每张图片返回且仅返回一个分类结果。"
                ),
            }
        ]
        for index, image_url in enumerate(batch_urls, 1):
            content.extend(
                [
                    {"type": "text", "text": f"图片 {index}："},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        batches.append((HumanMessage(content=content), batch_images))
    return batches


def _group_classified_images(
    result: ImageClassificationResult,
    images: list[str],
) -> dict[str, list[str]]:
    """校验分类结果与输入一一对应，并按类别分组."""
    grouped = {category: [] for category in IMAGE_CATEGORIES}
    classified_indexes = set()
    for item in result.images:
        index = item.image_index
        if index > len(images):
            raise ValueError(f"图片分类序号越界: {index}")
        if index in classified_indexes:
            raise ValueError(f"图片分类序号重复: {index}")
        classified_indexes.add(index)
        grouped[item.image_type].append(images[index - 1])

    expected_indexes = set(range(1, len(images) + 1))
    missing_indexes = sorted(expected_indexes - classified_indexes)
    if missing_indexes:
        raise ValueError(f"图片分类结果缺少序号: {missing_indexes}")
    return grouped


class MedicalRecordAgents:
    """病历诊断、五史清洗、视觉分析和病机抽取 Agent 集合."""

    def __init__(
        self,
        text_model: ChatOpenAI,
        vlm_model: ChatOpenAI,
        input_dir: Path,
        diagnosis_model: ChatOpenAI | None = None,
        image_classification_model: ChatOpenAI | None = None,
        reviewer_model: ChatOpenAI | None = None,
        enable_review: bool = False,
        filter_path: Path | None = None,
        stage_timeout: float = 0.0,
    ) -> None:
        self.input_dir = input_dir
        self.filter_path = filter_path or (DEFAULT_OUTPUT_DIR / FILTER_FILE_NAME)
        self.stage_timeout = stage_timeout
        self.vlm_model = vlm_model
        self.ocr_batch_semaphore = asyncio.Semaphore(OCR_BATCH_CONCURRENCY)
        self.diagnosis_agent = self._build_diagnosis_agent(diagnosis_model or text_model)
        self.history_agent = self._build_history_agent(text_model)
        self.extract_agent = self._build_extract_agent(text_model)
        self.reviewer_agent = None
        if enable_review:
            self.reviewer_agent = create_agent(
                model=reviewer_model or vlm_model,
                tools=[],
                system_prompt=_load_prompt("agent_review_prompt.txt"),
                response_format=_tool_response_format(AgentReviewResult),
                name="medical_agent_reviewer",
            )
        self.current_visit_history_agent = create_agent(
            model=text_model,
            tools=[],
            system_prompt=CURRENT_VISIT_HISTORY_SYSTEM_PROMPT,
            response_format=_tool_response_format(CurrentVisitHistoryResult),
            name="current_visit_history_agent",
        )
        self.image_classifier_agent = create_agent(
            model=image_classification_model or vlm_model,
            tools=[],
            system_prompt=_load_prompt("image_classification_prompt.txt"),
            response_format=_tool_response_format(ImageClassificationResult),
            name="medical_image_classifier_agent",
        )
        self.inspection_agent = create_agent(
            model=vlm_model,
            tools=[],
            system_prompt=_load_prompt("inspection_report_analysis_prompt.txt"),
            response_format=_tool_response_format(InspectionResult),
            name="inspection_report_agent",
        )
        self.tongue_face_agent = create_agent(
            model=vlm_model,
            tools=[],
            system_prompt=_load_prompt("tongue_face_analysis_prompt.txt"),
            response_format=_tool_response_format(TongueFaceResult),
            name="tongue_face_agent",
        )

    @staticmethod
    def _build_diagnosis_agent(model: ChatOpenAI) -> Any:
        # Baichuan accepts json_object, but rejects ToolStrategy's tool_choice and
        # ProviderStrategy's JSON Schema request body. Validate the JSON locally.
        return model.bind(response_format={"type": "json_object"})

    @staticmethod
    def _build_history_agent(model: ChatOpenAI) -> Any:
        return create_agent(
            model=model,
            tools=[],
            system_prompt=_load_prompt("clinical_record_cleaning_prompt.txt"),
            response_format=_tool_response_format(HistoryCleaningResult),
            name="history_cleaning_agent",
        )

    @staticmethod
    def _build_extract_agent(model: ChatOpenAI) -> Any:
        return create_agent(
            model=model,
            tools=[],
            system_prompt=_load_prompt("extract_prompt.txt"),
            response_format=_tool_response_format(ClinicalExtractionResult),
            name="clinical_extraction_agent",
        )

    @staticmethod
    def normalize_medical_history(record: dict[str, Any]) -> None:
        record["new_medical_history"] = _normalize_medical_history(record.get("new_medical_history"))

    @staticmethod
    def normalize_diagnoses(record: dict[str, Any]) -> None:
        record["diagnosis_illness"] = normalize_diagnosis_value(record.get("diagnosis_illness"))
        record["diagnosis_sickness"] = normalize_sickness_value(record.get("diagnosis_sickness"))
        # 此时尚未完成诊断类型纠错，不能按字段位置直接追加“证”，否则会把错填在
        # diagnosis_disease 中的西医诊断（如“高血压”）误改为“高血压证”。
        record["diagnosis_disease"] = normalize_diagnosis_value(record.get("diagnosis_disease"))

    async def complete_diagnoses(self, record: dict[str, Any]) -> None:
        for field in DIAGNOSIS_FIELDS:
            record.setdefault(f"ai_{field}", "")
            record.setdefault(f"ai_{field}_reason", "")
        protected_diagnoses = record.get("__original_diagnoses")
        if not isinstance(protected_diagnoses, dict):
            protected_diagnoses = _diagnosis_payload(record)
        source_diagnoses = {field: _text(protected_diagnoses.get(field)) for field in DIAGNOSIS_FIELDS}
        previous_diagnoses = record.get("__previous_diagnoses") or {}
        if not isinstance(previous_diagnoses, dict):
            previous_diagnoses = {}
        previous_diagnoses = {field: _text(previous_diagnoses.get(field)) for field in DIAGNOSIS_FIELDS}
        chief_complaint = _latest_chief_complaint(record)
        present_history = _diagnosis_present_history(record)

        diagnosis_system_prompt = _load_prompt("fill_diagnosis_prompt.txt")
        result = await _invoke_json_object_model(
            self.diagnosis_agent,
            diagnosis_system_prompt,
            HumanMessage(
                content=(
                    f"本次就诊患者主诉：{chief_complaint}\n"
                    f"本次就诊现病史：{present_history}\n"
                    f"三个原始诊断结果：{_json_for_prompt(source_diagnoses)}\n"
                    f"上一次同患者诊断结果（如为空表示无上一诊断可参考）：{_json_for_prompt(previous_diagnoses)}\n"
                    "要求：diagnosis_illness 即使非空也必须清理、拆分、重分类，保留所有有效西医诊断；"
                    "diagnosis_disease 和 diagnosis_sickness 非空时原则上保留，空字段优先从 diagnosis_illness 或其他原始字段中提取对应类别结果。"
                    "无效诊断文本必须丢弃并结合本次主诉、本次现病史和上一次诊断结果补全。最终所有字段值均不能为空。"

                )
            ),
            DiagnosisResult,
            reviewer_agent=self.reviewer_agent,
            agent_name="diagnosis_completion_agent",
        )
        values = _model_dump(result)
        normalizers = {
            "diagnosis_illness": normalize_diagnosis_value,
            "diagnosis_sickness": normalize_sickness_value,
            "diagnosis_disease": normalize_syndrome_value,
        }
        for field in DIAGNOSIS_FIELDS:
            original_value = _clean_generated_text(source_diagnoses.get(field))
            if field != "diagnosis_illness" and original_value:
                record[f"ai_{field}"] = original_value
                record[f"ai_{field}_reason"] = "原字段非空，按要求保留原诊断结果。"
                continue

            record[f"ai_{field}"] = normalizers[field](_clean_generated_text(values.get(field)))
            if not record[f"ai_{field}"]:
                raise ValueError(f"DiagnosisResult.{field} 不能为空")
            reason = _clean_generated_text(values.get(f"{field}_reason"))
            if not reason:
                reason = _diagnosis_reason_fallback(
                    field,
                    record[f"ai_{field}"],
                    source_diagnoses,
                    previous_diagnoses,
                    chief_complaint,
                    present_history,
                )
            record[f"ai_{field}_reason"] = reason

    async def extract_current_visit_history(self, record: dict[str, Any], enabled: bool) -> None:
        record["ai_new_medical_history"] = ""
        history = _present_history(record)
        if not enabled or not history:
            return
        result = await _invoke_structured_agent(
            self.current_visit_history_agent,
            HumanMessage(
                content=(
                    f"本次就诊时间：{_visit_time(record) or '未记录'}\n"
                    f"就诊类型：{_text(record.get('is_first')) or '未记录'}\n"
                    f"患者主诉：{_chief_patient_complaint(record) or '未记录'}\n"
                    f"医生撰写主诉：{_chief_complaint(record) or '未记录'}\n"
                    f"累积现病史原文：\n{history}"
                )
            ),
            CurrentVisitHistoryResult,
            reviewer_agent=self.reviewer_agent,
            agent_name="current_visit_history_agent",
            system_prompt=CURRENT_VISIT_HISTORY_SYSTEM_PROMPT,
        )
        record["ai_new_medical_history"] = _replace_inspection_report_symbols(
            _clean_generated_text(result.new_medical_history)
        )

    async def clean_histories(self, record: dict[str, Any]) -> None:
        histories = _history_payload(record)
        inspection_description = format_inspection_result_text(record.get("ai_inspection_report_img"))
        logger.info(f"检查报告分析结果： {inspection_description}")
        tongue_face_description = format_tongue_face_result_text(record.get("ai_tongue_face_img"))
        logger.info(f"舌面分析结果： {tongue_face_description}")
        system_prompt = _load_prompt("clinical_record_cleaning_prompt.txt")
        result = await _invoke_structured_agent(
            self.history_agent,
            HumanMessage(
                content=(
                    f"患者主诉（doc_ass_stu_appeal）：{_chief_complaint(record)}\n"
                    f"本次现病史：{_latest_present_history(record)}\n"
                    f"原始五史：{_json_for_prompt(histories)}\n"
                    f"检查报告图片分析：\n{inspection_description}\n"
                    f"舌面患处图片分析：\n{tongue_face_description}"
                )
            ),
            HistoryCleaningResult,
            reviewer_agent=self.reviewer_agent,
            agent_name="history_cleaning_agent",
            system_prompt=system_prompt,
        )
        values = _model_dump(result)
        record["ai_patient_appeal"] = _clean_generated_text(values.get("patient_appeal"))
        record["ai_new_medical_history"] = _replace_inspection_report_symbols(
            _clean_generated_text(values.get("new_medical_history"))
        )
        record["ai_complaint_history_consistent"] = values.get("complaint_history_consistent")
        record["ai_complaint_history_issues"] = values.get("consistency_issues") or []
        for field in HISTORY_FIELDS:
            original = histories[field]
            if field == "allergic_history" and original in {"无", "无特殊", "否认", "否认过敏"}:
                record[f"ai_{field}"] = "无药物及食物过敏史"
            else:
                record[f"ai_{field}"] = (
                    "" if original in INVALID_HISTORY_TEXT else _clean_generated_text(values.get(field))
                )

    def build_illness_knowledge(self, diagnosis_res:dict):
        """根据不同医生--不同疾病--不同病期---不同治则治法—不同配伍用药知识库召回疾病知识."""
        return ''

    async def _ocr_image_batches(self, images: list[str], prompt: str, *, agent_name: str) -> str:
        """Run OCR batches concurrently and return their text in original image order."""
        batches = list(_chunked(images, MAX_VLM_IMAGES_PER_REQUEST))
        semaphore = getattr(self, "ocr_batch_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(OCR_BATCH_CONCURRENCY)
            self.ocr_batch_semaphore = semaphore

        async def run_batch(batch_index: int, batch_images: list[str]) -> str:
            start_index = (batch_index - 1) * MAX_VLM_IMAGES_PER_REQUEST + 1
            numbered_prompt = (
                f"{prompt}\n"
                f"本批共 {len(batch_images)} 张图片，对应原始图片序号 {start_index} 到 "
                f"{start_index + len(batch_images) - 1}。"
                "请逐张输出可识别文字或可见客观事实；无法识别的图片也要保留图片序号并说明无法识别。"
            )
            started = time.perf_counter()
            logger.info(
                f"[OCR BATCH START] agent={agent_name} batch={batch_index}/{len(batches)} "
                f"images={len(batch_images)}"
            )
            try:
                async with semaphore:
                    response = await self.vlm_model.ainvoke(
                        [_multimodal_message(numbered_prompt, batch_images, self.input_dir)]
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.error(
                    f"[OCR BATCH ERROR] agent={agent_name} batch={batch_index}/{len(batches)} "
                    f"elapsed={elapsed:.1f}s error={type(exc).__name__}: {exc}"
                )
                raise RuntimeError(
                    f"{agent_name} OCR 批次 {batch_index}/{len(batches)} 失败: {type(exc).__name__}: {exc}"
                ) from exc
            elapsed = time.perf_counter() - started
            logger.info(
                f"[OCR BATCH END] agent={agent_name} batch={batch_index}/{len(batches)} "
                f"images={len(batch_images)} elapsed={elapsed:.1f}s"
            )
            text = _response_text(response)
            return f"【{agent_name} OCR 批次 {batch_index}】\n{text}"

        results = await asyncio.gather(
            *(run_batch(batch_index, batch_images) for batch_index, batch_images in enumerate(batches, 1)),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(
                f"{agent_name} OCR 共 {len(errors)}/{len(batches)} 个批次失败: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]
        return "\n\n".join(result for result in results if isinstance(result, str))

    async def _analyze_inspection_batches(self, images: list[str], visit_time: str) -> InspectionResult:
        """Extract one structured result per image batch and merge without a second model call."""
        batches = list(_chunked(images, MAX_VLM_IMAGES_PER_REQUEST))
        semaphore = getattr(self, "ocr_batch_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(OCR_BATCH_CONCURRENCY)
            self.ocr_batch_semaphore = semaphore
        system_prompt = _load_prompt("inspection_report_analysis_prompt.txt")

        async def run_batch(batch_index: int, batch_images: list[str]) -> InspectionResult:
            start_index = (batch_index - 1) * MAX_VLM_IMAGES_PER_REQUEST + 1
            prompt = (
                f"本次就诊时间（see_doc_time）：{visit_time or '未记录'}\n"
                f"本批共 {len(batch_images)} 张图片，对应原始图片序号 {start_index} 到 "
                f"{start_index + len(batch_images) - 1}。请逐张分类并返回结构化结果。\n"
                "为控制耗时和上下文，每张有效报告只保留：报告名称、报告日期、指标名称、"
                "指标结果、单位、异常标记和报告原有结论。content 仅用简短的“指标：结果 单位”"
                "格式记录，不得全文转写、不得重复参考范围、患者信息、医院信息和无关页眉页脚。"
                "abnormal_indicators 只保留报告明确标记的异常指标；conclusion 只保留报告原有结论。"
                "非检查报告或无法识别时只返回图片类别和 is_valid_report=false，其余字段留空。"
            )
            started = time.perf_counter()
            logger.info(
                f"[INSPECTION BATCH START] batch={batch_index}/{len(batches)} images={len(batch_images)}"
            )
            try:
                async with semaphore:
                    result = await _invoke_structured_agent(
                        self.inspection_agent,
                        _multimodal_message(prompt, batch_images, self.input_dir),
                        InspectionResult,
                        reviewer_agent=self.reviewer_agent,
                        agent_name="inspection_report_agent",
                        system_prompt=system_prompt,
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.error(
                    f"[INSPECTION BATCH ERROR] batch={batch_index}/{len(batches)} "
                    f"elapsed={elapsed:.1f}s error={type(exc).__name__}: {exc}"
                )
                raise RuntimeError(
                    f"inspection_report 批次 {batch_index}/{len(batches)} 失败: {type(exc).__name__}: {exc}"
                ) from exc
            elapsed = time.perf_counter() - started
            logger.info(
                f"[INSPECTION BATCH END] batch={batch_index}/{len(batches)} "
                f"images={len(batch_images)} elapsed={elapsed:.1f}s"
            )
            return result

        results = await asyncio.gather(
            *(run_batch(batch_index, batch_images) for batch_index, batch_images in enumerate(batches, 1)),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(
                f"inspection_report 共 {len(errors)}/{len(batches)} 个批次失败: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

        reports = []
        summaries = []
        seen_summaries = set()
        for result in results:
            if not isinstance(result, InspectionResult):
                continue
            reports.extend(result.reports)
            summary = _clean_generated_text(result.history_evidence_summary)
            if summary and summary not in seen_summaries:
                seen_summaries.add(summary)
                summaries.append(summary)
        return InspectionResult(reports=reports, history_evidence_summary="\n".join(summaries))

    async def _extract_tongue_face_from_ocr(self, ocr_text: str) -> TongueFaceResult:
        prompt = (
            "以下内容是多张舌象、面象和患处图片按批次转写后的 OCR/可见事实文本。"
            "请只依据这些文本，按结构化模型提取可见舌象、面象和患处内容；"
            "不得补充文本中没有的疾病、证候、病因或病机推断。\n"
            f"{ocr_text}"
        )
        return await _invoke_structured_agent(
            self.tongue_face_agent,
            HumanMessage(content=prompt),
            TongueFaceResult,
            reviewer_agent=self.reviewer_agent,
            agent_name="tongue_face_agent",
            system_prompt=_load_prompt("tongue_face_analysis_prompt.txt"),
        )

    async def classify_images(self, record: dict[str, Any]) -> dict[str, list[str]]:
        """不依赖字段名，对四个图片字段中的图片逐张分类."""
        images = _record_images(record)
        if not images:
            return {category: [] for category in IMAGE_CATEGORIES}
        batches = await _image_classification_message(images, self.input_dir)
        if not batches:
            return {category: [] for category in IMAGE_CATEGORIES}

        async def classify_batch(
            message: HumanMessage,
            batch_images: list[str],
        ) -> dict[str, list[str]]:
            async with classification_semaphore:
                result = await _invoke_structured_agent(
                    self.image_classifier_agent,
                    message,
                    ImageClassificationResult,
                    reviewer_agent=self.reviewer_agent,
                    agent_name="medical_image_classifier_agent",
                    system_prompt=_load_prompt("image_classification_prompt.txt"),
                )
            return _group_classified_images(result, batch_images)

        classification_semaphore = asyncio.Semaphore(IMAGE_CLASSIFICATION_BATCH_CONCURRENCY)
        batch_results = await asyncio.gather(
            *(classify_batch(message, batch_images) for message, batch_images in batches)
        )
        grouped = {category: [] for category in IMAGE_CATEGORIES}
        for batch_result in batch_results:
            for category in IMAGE_CATEGORIES:
                grouped[category].extend(batch_result[category])
        return grouped

    async def analyze_inspection_images(self, record: dict[str, Any], images: list[str] | None = None) -> None:
        if images is None:
            images = (await self.classify_images(record))["检验检查报告类"]
        if not images:
            record["ai_inspection_report_img"] = _replace_inspection_report_symbols(
                _model_dump(InspectionResult(), by_alias=True)
            )
            return
        visit_time = _visit_time(record)
        logger.info(
            f"[INSPECTION] images={len(images)} batch_size={MAX_VLM_IMAGES_PER_REQUEST} "
            f"concurrency={OCR_BATCH_CONCURRENCY}"
        )
        result = await self._analyze_inspection_batches(images, visit_time)
        for report in result.reports:
            relative_time = _relative_report_time(report.report_date, visit_time)
            if relative_time:
                report.relative_to_visit_time = relative_time
        record["ai_inspection_report_img"] = _replace_inspection_report_symbols(
            _model_dump(result, by_alias=True)
        )

    async def analyze_tongue_face_images(self, record: dict[str, Any], images: list[str] | None = None) -> None:
        record_id = (
            _text(record.get("order_sn"))
            or _text(record.get("id"))
            or _text(record.get("record_id"))
            or "<unknown>"
        )
        if images is None:
            classified_images = await self.classify_images(record)
            images = [image for category in TONGUE_FACE_IMAGE_CATEGORIES for image in classified_images[category]]
        logger.info(f"[TONGUE_FACE] record={record_id} images={len(images)}")
        if not images:
            record["ai_tongue_face_img"] = _model_dump(TongueFaceResult())
            return
        prompt = (
            "按结构化模型分析图片中的可见舌象、面象和患处。分别记录舌体、舌苔、舌下络脉、"
            "面神、面色、光泽、面部形态、五官局部和皮损。只描述图像可见事实，不作疾病、证候、"
            "病因或病机推断；无法辨认的字符串字段返回空字符串，多选字段返回空数组。"
        )
        if len(images) > MAX_VLM_IMAGES_PER_REQUEST:
            logger.info(
                f"[TONGUE_FACE] record={record_id} images={len(images)} exceeds "
                f"{MAX_VLM_IMAGES_PER_REQUEST}, using OCR batches"
            )
            ocr_text = await self._ocr_image_batches(
                images,
                (
                    "你是舌象、面象和患处图片转写助手。请逐张记录图片中清晰可见的客观事实："
                    "舌体、舌苔、舌下络脉、面部神色、五官局部、皮肤或患处形态颜色分布。"
                ),
                agent_name="tongue_face",
            )
            result = await self._extract_tongue_face_from_ocr(ocr_text)
        else:
            logger.info(f"[TONGUE_FACE] record={record_id} building multimodal message")
            message = _multimodal_message(prompt, images, self.input_dir)
            logger.info(f"[TONGUE_FACE] record={record_id} invoking model")
            result = await _invoke_structured_agent(
                self.tongue_face_agent,
                message,
                TongueFaceResult,
                reviewer_agent=self.reviewer_agent,
                agent_name="tongue_face_agent",
                system_prompt=_load_prompt("tongue_face_analysis_prompt.txt"),
            )
        logger.info(f"[TONGUE_FACE] record={record_id} model returned")
        record["ai_tongue_face_img"] = _model_dump(result)

    async def extract_clinical_fields(self, record: dict[str, Any], use_rag:bool=False) -> None:
        diagnoses = {
            field: _text(record.get(f"ai_{field}")) or _text(record.get(field))
            for field in DIAGNOSIS_FIELDS
        }
        histories = {
            field: _text(record.get(f"ai_{field}")) if f"ai_{field}" in record else _text(record.get(field))
            for field in HISTORY_FIELDS
        }
        illness_knowledge = self.build_illness_knowledge(diagnoses) if use_rag else "无"
        inspection_description = format_inspection_result_text(record.get("ai_inspection_report_img"))
        tongue_face_description = format_tongue_face_result_text(record.get("ai_tongue_face_img"))
        system_prompt = _load_prompt("extract_prompt.txt")
        result = await _invoke_structured_agent(
            self.extract_agent,
            HumanMessage(
                content=(
                    f"最新主诉：{_latest_chief_complaint(record)}\n"
                    f"本次现病史：{_latest_present_history(record)}\n"
                    f"清洗后五史：{_json_for_prompt(histories)}\n"
                    f"诊断上下文：{_json_for_prompt(diagnoses)}\n"
                    f"检查报告图片分析：\n{inspection_description}\n"
                    f"舌面患处图片分析：\n{tongue_face_description}\n"
                    f"疾病的知识库信息 {illness_knowledge}\n"
                    f"严格根据知识库知识，提取患者在中医层面的病因、病机、病期、病位、病程、临床症状。若知识库无知识，根据患者病史信息完成上述内容提取。必须返回合法的 JSON 对象，不要输出 Markdown，不要输出额外解释"
                )
            ),
            ClinicalExtractionResult,
            reviewer_agent=self.reviewer_agent,
            agent_name="clinical_extraction_agent",
            system_prompt=system_prompt,
        )
        values = _model_dump(result)
        for field, value in values.items():
            record[f"ai_{field}"] = _clean_generated_value(value)

    async def _run_stage(
        self,
        stage_name: str,
        record_id: str,
        doctor_name: str,
        stage: Callable[[], Any],
    ) -> None:
        started_at = time.monotonic()
        logger.info(f"[STAGE START] record={record_id} doctor_name={doctor_name} stage={stage_name}")
        try:
            result = stage()
            if inspect.isawaitable(result):
                if self.stage_timeout > 0:
                    await asyncio.wait_for(result, timeout=self.stage_timeout)
                else:
                    await result
        except Exception:
            logger.info(
                f"[STAGE FAILED] record={record_id} doctor_name={doctor_name} "
                f"stage={stage_name} elapsed={time.monotonic() - started_at:.1f}s"
            )
            raise
        logger.info(
            f"[STAGE END] record={record_id} doctor_name={doctor_name} "
            f"stage={stage_name} elapsed={time.monotonic() - started_at:.1f}s"
        )

    async def process(
        self,
        record: dict[str, Any],
        *,
        extract_current_history: bool = False,
        use_rag: bool = False,
    ) -> dict[str, Any]:
        output = dict(record)
        # 前端保存的是富文本 HTML。先转成纯文本再参与空值过滤和后续 Agent 调用，
        # 避免标签污染提示词，也避免只有 <p><br></p> 的空病史绕过过滤。
        self.normalize_medical_history(output)
        record_id = (
            _text(record.get("order_sn"))
            or _text(record.get("id"))
            or _text(record.get("record_id"))
            or "<unknown>"
        )
        doctor_name = _text(record.get("doctor_name")) or "<unknown>"
        output["__original_diagnoses"] = _diagnosis_payload(output)
        errors = []

        # 主诉、现病史和西医诊断全部缺失时，不调用任何 Agent
        if _should_filter_record(output):
            write2filter_json(output, self.filter_path)
            output["ai_processing_filtered"] = True
            output["ai_processing_complete"] = False
            output["ai_filter_reason"] = (
                "patient_appeal、new_medical_history、diagnosis_illness 均为空"
            )
            logger.info(f"[STAGE FILTER] record={record_id} doctor_name={doctor_name}")
            return output

        # step1：先做不依赖诊断类型的基础归一化，不额外存储 ai_normalized_*。
        try:
            await self._run_stage(
                "diagnosis_normalization",
                record_id,
                doctor_name,
                lambda: self.normalize_diagnoses(output),
            )
        except Exception as exc:
            errors.append({"stage": "diagnosis_normalization", "error": f"{type(exc).__name__}: {exc}"})
        # step2：图片分类与视觉解析管线。
        async def run_image_pipeline() -> list[dict[str, str]]:
            pipeline_errors = []
            try:
                classified_images: dict[str, list[str]] = {}

                async def classify_image_stage() -> None:
                    nonlocal classified_images
                    classified_images = await self.classify_images(output)

                await self._run_stage("image_classification", record_id, doctor_name, classify_image_stage)
            except Exception as exc:
                pipeline_errors.append(
                    {"stage": "image_classification", "error": f"{type(exc).__name__}: {exc}"}
                )
                classified_images = {category: [] for category in IMAGE_CATEGORIES}

            # 报告类与舌/面/患处类图片互不依赖，并行解析。
            tongue_face_images = [
                image for category in TONGUE_FACE_IMAGE_CATEGORIES for image in classified_images[category]
            ]
            image_stages = {
                "inspection_vlm": (
                    self.analyze_inspection_images,
                    classified_images["检验检查报告类"],
                ),
                "tongue_face_vlm": (self.analyze_tongue_face_images, tongue_face_images),
            }
            tasks = {
                name: asyncio.create_task(
                    self._run_stage(
                        name,
                        record_id,
                        doctor_name,
                        lambda stage=stage, images=images: stage(output, images),
                    )
                )
                for name, (stage, images) in image_stages.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for stage_name, result in zip(tasks, results, strict=True):
                if isinstance(result, Exception):
                    pipeline_errors.append(
                        {"stage": stage_name, "error": f"{type(result).__name__}: {result}"}
                    )
            return pipeline_errors

        # step3：本次复诊病史只依赖文本字段，可与整条视觉管线并发。
        async def run_current_visit_history() -> list[dict[str, str]]:
            logger.debug("【本次复诊现病史内容抽取】】")
            try:
                await self._run_stage(
                    "current_visit_history",
                    record_id,
                    doctor_name,
                    lambda: self.extract_current_visit_history(output, extract_current_history),
                )
            except Exception as exc:
                return [{"stage": "current_visit_history", "error": f"{type(exc).__name__}: {exc}"}]
            return []

        pipeline_results = await asyncio.gather(
            run_image_pipeline(),
            run_current_visit_history(),
        )
        for pipeline_errors in pipeline_results:
            errors.extend(pipeline_errors)

        # step4：等待图片和本次病史均完成后，校验主诉/现病史一致性并清洗五史。
        try:
            await self._run_stage("clinical_cleaning", record_id, doctor_name, lambda: self.clean_histories(output))
        except Exception as exc:
            errors.append({"stage": "clinical_cleaning", "error": f"{type(exc).__name__}: {exc}"})
        # step5：用最新病历补全诊断，再提取知识库字段。
        try:
            await self._run_stage(
                "diagnosis_completion",
                record_id,
                doctor_name,
                lambda: self.complete_diagnoses(output),
            )
        except Exception as exc:
            errors.append({"stage": "diagnosis_completion", "error": f"{type(exc).__name__}: {exc}"})
        try:
            await self._run_stage(
                "clinical_extraction",
                record_id,
                doctor_name,
                lambda: self.extract_clinical_fields(output, use_rag),
            )
        except Exception as exc:
            errors.append({"stage": "clinical_extraction", "error": f"{type(exc).__name__}: {exc}"})
        if errors:
            output["ai_processing_errors"] = errors
        else:
            output.pop("ai_processing_errors", None)
        output.pop("__original_diagnoses", None)
        output.pop("__previous_diagnoses", None)

        return output


def _iter_record_file(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
                yield record
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = next((data[key] for key in ("data", "records", "items") if isinstance(data.get(key), list)), None)
        if records is None:
            raise ValueError(f"{path} 中未找到 data/records/items 病历数组")
    else:
        raise ValueError(f"{path} 必须是 JSON 数组、JSONL 或包含病历数组的 JSON 对象")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {index + 1} 条病历不是 JSON 对象")
        yield record


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_file():
        yield from _iter_record_file(path)
        return
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {path}")

    files = sorted(
        file
        for file in path.glob("*.json")
        if file.is_file() and not file.name.startswith(("._", "."))
    )
    if not files:
        raise ValueError(f"{path} 下没有可读取的 JSON 病历文件")
    for file in files:
        yield from _iter_record_file(file)


def _record_key(record: dict[str, Any], index: int) -> str:
    return _text(record.get("id") or record.get("inquiry_id") or record.get("order_sn")) or str(index)


def _record_identity(record: dict[str, Any], index: int = 0) -> str:
    return f"{_text(record.get('doctor_id'))}:{_record_key(record, index)}"


def _doctor_matches(record: dict[str, Any], doctor_ids: set[str]) -> bool:
    if not doctor_ids:
        return True
    return _text(record.get("doctor_id")) in doctor_ids


def _selected_doctor_quota_reached(
    doctor_id: str, doctor_ids: set[str], attempted_by_doctor: Counter[str], limit: int
) -> bool:
    return bool(doctor_ids and doctor_id and limit > 0 and attempted_by_doctor[doctor_id] >= limit)


def _all_selected_doctors_reached(doctor_ids: set[str], attempted_by_doctor: Counter[str], limit: int) -> bool:
    return bool(doctor_ids and limit > 0 and all(attempted_by_doctor[doctor_id] >= limit for doctor_id in doctor_ids))


def count_patient_visits(path: Path, doctor_ids: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in iter_records(path):
        if not _doctor_matches(record, doctor_ids):
            continue
        patient_key = _patient_key(record)
        if patient_key:
            counts[patient_key] += 1
    return counts


def load_processed_record_keys(output_dir: Path) -> tuple[set[str], set[str]]:
    """Load identities and order numbers already written to result or filter files."""
    processed_ids: set[str] = set()
    processed_order_sns: set[str] = set()
    if not output_dir.is_dir():
        return processed_ids, processed_order_sns
    # 递归读取新的医生子目录，同时兼容旧版直接存储在 output_dir 根目录的 *_ai.jsonl。
    for path in output_dir.rglob("*_ai.jsonl"):
        if path.name == FAILURE_FILE_NAME:
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_result_record = isinstance(record, dict) and (
                    path.name == DOCTOR_RECORD_FILE_NAME or record.get("ai_processing_complete") is True
                )
                if is_result_record:
                    processed_ids.add(_text(record.get("ai_record_identity")) or _record_identity(record))
                    order_sn = _text(record.get("order_sn"))
                    if order_sn:
                        processed_order_sns.add(order_sn)
    for path in output_dir.rglob(FILTER_FILE_NAME):
        try:
            filtered_records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"忽略无法解析的过滤文件: {path}")
            continue
        if not isinstance(filtered_records, list):
            logger.warning(f"忽略非 JSON 数组过滤文件: {path}")
            continue
        for record in filtered_records:
            if not isinstance(record, dict):
                continue
            processed_ids.add(_record_identity(record))
            order_sn = _text(record.get("order_sn"))
            if order_sn:
                processed_order_sns.add(order_sn)
    return processed_ids, processed_order_sns


def load_processed_record_ids(output_dir: Path) -> set[str]:
    """Load processed identities; retained for callers using the original helper."""
    return load_processed_record_keys(output_dir)[0]


def _safe_path_part(value: Any, fallback: str = "unknown") -> str:
    safe_value = re.sub(r'[<>:"/\\|?*\s]+', "_", _text(value)).strip("._")
    return safe_value or fallback


def _doctor_output_dir(output_dir: Path, record: dict[str, Any]) -> Path:
    doctor_id = _text(record.get("doctor_id")) or "unknown"
    doctor_name = _safe_path_part(record.get("doctor_name"))
    return output_dir / f"doctor_{_safe_path_part(doctor_id)}_{doctor_name}"


def _doctor_output_path(output_dir: Path, record: dict[str, Any]) -> Path:
    return _doctor_output_dir(output_dir, record) / DOCTOR_RECORD_FILE_NAME


def _doctor_failure_path(output_dir: Path, record: dict[str, Any]) -> Path:
    return _doctor_output_dir(output_dir, record) / FAILURE_FILE_NAME


def _write_jsonl(file: TextIO, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def build_model(
    model_name: str,
    args: argparse.Namespace,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    resolved_base_url = args.base_url if base_url is None else base_url
    resolved_api_key = args.api_key if api_key is None else api_key
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
    }
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    logger.info(
        f"model_name {model_name}, base_url {resolved_base_url or '<default>'}, "
        f"apikey {_mask_secret(resolved_api_key)}"
    )
    normalized_base_url = _text(resolved_base_url).lower()
    model_id = model_name.lower()
    # 所有 Agent 都用于结构化抽取，统一关闭 thinking mode，避免响应中混入推理内容。
    if "deepseek" in model_id or "deepseek" in normalized_base_url:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        kwargs["extra_body"] = {"enable_thinking": False}
    return ChatOpenAI(**kwargs)


async def _process_record_batch(
    agents: MedicalRecordAgents,
    batch: list[dict[str, Any]],
    previous_diagnoses_by_patient: dict[str, dict[str, str]],
    record_timeout: float,
) -> list[tuple[dict[str, Any], dict[str, Any], Exception | None]]:
    """Process different patients concurrently while preserving each patient's visit order."""
    grouped_items: dict[str, list[dict[str, Any]]] = {}
    for item in batch:
        patient_key = item["patient_key"]
        group_key = patient_key or f"__record__:{item['identity']}"
        grouped_items.setdefault(group_key, []).append(item)

    async def process_patient_visits(
        items: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any], Exception | None]]:
        outcomes = []
        for item in items:
            working_record = dict(item["record"])
            patient_key = item["patient_key"]
            if patient_key and patient_key in previous_diagnoses_by_patient:
                working_record["__previous_diagnoses"] = previous_diagnoses_by_patient[patient_key]

            try:
                process_call = agents.process(
                    working_record,
                    extract_current_history=item["extract_current_history"],
                    use_rag=item["use_rag"],
                )
                if record_timeout > 0:
                    enriched = await asyncio.wait_for(process_call, timeout=record_timeout)
                else:
                    enriched = await process_call
                error = None
            except TimeoutError:
                error = TimeoutError(f"病历处理超过 {record_timeout:g} 秒")
                enriched = dict(working_record)
            except Exception as exc:  # 单条病历失败不能取消同批其他病历
                error = exc
                enriched = dict(working_record)

            if error is not None:
                enriched.pop("__previous_diagnoses", None)
                enriched.pop("__original_diagnoses", None)
                enriched["ai_processing_errors"] = [{"stage": "record", "error": f"{type(error).__name__}: {error}"}]
                enriched["ai_processing_complete"] = False
                enriched["ai_record_identity"] = item["identity"]

            outcomes.append((item, enriched, error))
            if patient_key and enriched.get("ai_processing_filtered") is not True:
                enriched_diagnoses = _diagnosis_payload(enriched, prefer_ai=True)
                if _has_any_diagnosis(enriched_diagnoses):
                    previous_diagnoses_by_patient[patient_key] = enriched_diagnoses
        return outcomes

    grouped_outcomes = await asyncio.gather(
        *(process_patient_visits(items) for items in grouped_items.values()),
        return_exceptions=True,
    )
    outcomes_by_item: dict[int, tuple[dict[str, Any], dict[str, Any], Exception | None]] = {}
    for items, group_outcome in zip(grouped_items.values(), grouped_outcomes, strict=True):
        if isinstance(group_outcome, BaseException):
            for item in items:
                error = RuntimeError(f"批处理任务异常: {type(group_outcome).__name__}: {group_outcome}")
                enriched = dict(item["record"])
                enriched["ai_processing_errors"] = [{"stage": "record", "error": f"{type(error).__name__}: {error}"}]
                enriched["ai_processing_complete"] = False
                enriched["ai_record_identity"] = item["identity"]
                outcomes_by_item[id(item)] = (item, enriched, error)
            continue
        for outcome in group_outcome:
            outcomes_by_item[id(outcome[0])] = outcome
    return [outcomes_by_item[id(item)] for item in batch]


async def process_records(args: argparse.Namespace) -> tuple[int, int, int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doctor_ids = set(args.doctor_id or [])

    text_model = build_model(args.model, args, base_url=args.base_url, api_key=args.api_key)
    diagnosis_model = build_model(
        args.diagnosis_model,
        args,
        base_url=args.diagnosis_base_url,
        api_key=args.diagnosis_api_key,
    )
    vlm_model = build_model(
        args.vlm_model or args.model,
        args,
        base_url=args.vlm_base_url or args.base_url,
        api_key=args.vlm_api_key or args.api_key,
    )
    image_classification_model = build_model(
        getattr(args, "image_classification_model", CLASSIFICATION_MODEL_NAME),
        args,
        base_url=getattr(args, "image_classification_base_url", "") or args.base_url,
        api_key=getattr(args, "image_classification_api_key", "") or args.api_key,
        max_tokens=CLASSIFICATION_MODEL_MAX_TOKENS,
    )
    enable_review = bool(getattr(args, "enable_review", False))
    reviewer_model = None
    if enable_review:
        reviewer_model = build_model(
            getattr(args, "reviewer_model", "") or os.getenv("KIMI_K3", "kimi/kimi-k3"),
            args,
            base_url=getattr(args, "reviewer_base_url", ""),
            api_key=getattr(args, "reviewer_api_key", ""),
        )
    input_dir = args.input if args.input.is_dir() else args.input.parent
    agents = MedicalRecordAgents(
        text_model,
        vlm_model,
        input_dir,
        diagnosis_model=diagnosis_model,
        image_classification_model=image_classification_model,
        reviewer_model=reviewer_model,
        enable_review=enable_review,
        filter_path=args.output_dir / FILTER_FILE_NAME,
        stage_timeout=getattr(args, "timeout", 120.0),
    )
    patient_visit_counts = count_patient_visits(args.input, doctor_ids)
    if args.reprocess:
        processed_ids, processed_order_sns = set(), set()
    else:
        processed_ids, processed_order_sns = load_processed_record_keys(args.output_dir)
    succeeded = 0
    failed = 0
    skipped = 0
    attempted = 0
    attempted_by_doctor: Counter[str] = Counter()
    output_files: dict[Path, TextIO] = {}
    failure_files: dict[Path, TextIO] = {}
    previous_diagnoses_by_patient: dict[str, dict[str, str]] = {}
    batch_size = max(1, int(getattr(args, "batch_size", 8)))
    record_timeout = max(0.0, float(getattr(args, "record_timeout", 0.0)))
    pending_batch: list[dict[str, Any]] = []

    with ExitStack() as stack:

        async def flush_batch() -> None:
            nonlocal failed, skipped, succeeded
            if not pending_batch:
                return

            batch = list(pending_batch)
            pending_batch.clear()
            logger.info(f"[BATCH START] records={len(batch)} concurrency={len(batch)}")
            batch_started_at = time.monotonic()
            outcomes = await _process_record_batch(
                agents,
                batch,
                previous_diagnoses_by_patient,
                record_timeout,
            )
            first_failure: RuntimeError | None = None
            for item, enriched, process_error in outcomes:
                identity = item["identity"]
                order_sn = item["order_sn"]
                record_id = item["record_id"]
                enriched["ai_record_identity"] = identity
                enriched.pop("__previous_diagnoses", None)
                enriched.pop("__original_diagnoses", None)

                if process_error is not None:
                    failure_path = _doctor_failure_path(args.output_dir, enriched)
                    if failure_path not in failure_files:
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_files[failure_path] = stack.enter_context(failure_path.open("a", encoding="utf-8"))
                    _write_jsonl(failure_files[failure_path], enriched)
                    failed += 1
                    logger.error(f"[ERROR] record={record_id}: {type(process_error).__name__}: {process_error}")
                    first_failure = first_failure or RuntimeError(f"病历 {record_id} 处理失败")
                    continue

                if enriched.get("ai_processing_filtered") is True:
                    skipped += 1
                    processed_ids.add(identity)
                    if order_sn:
                        processed_order_sns.add(order_sn)
                    logger.info(f"数据关键字段均为空，【FILTER】 {order_sn or record_id}")
                    continue

                if enriched.get("ai_processing_errors"):
                    failed += 1
                    enriched["ai_processing_complete"] = False
                    failure_path = _doctor_failure_path(args.output_dir, enriched)
                    if failure_path not in failure_files:
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_files[failure_path] = stack.enter_context(failure_path.open("a", encoding="utf-8"))
                    _write_jsonl(failure_files[failure_path], enriched)
                    logger.error(f"[ERROR] record={record_id}: {_json_for_prompt(enriched['ai_processing_errors'])}")
                    first_failure = first_failure or RuntimeError(f"病历 {record_id} 存在处理失败阶段")
                    continue

                succeeded += 1
                enriched["ai_processing_complete"] = True
                enriched["ai_processing_version"] = "2026-07-28"
                output_path = _doctor_output_path(args.output_dir, enriched)
                if output_path not in output_files:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_files[output_path] = stack.enter_context(output_path.open("a", encoding="utf-8"))
                _write_jsonl(output_files[output_path], enriched)
                processed_ids.add(identity)
                if order_sn:
                    processed_order_sns.add(order_sn)

            logger.info(f"[BATCH END] records={len(batch)} elapsed={time.monotonic() - batch_started_at:.1f}s")
            if getattr(args, "interval", 0) > 0:
                await asyncio.sleep(args.interval)
            if first_failure is not None and args.fail_fast:
                raise first_failure

        for index, record in enumerate(iter_records(args.input), 1):
            if not _doctor_matches(record, doctor_ids):
                continue
            doctor_id = _text(record.get("doctor_id"))
            if _selected_doctor_quota_reached(doctor_id, doctor_ids, attempted_by_doctor, args.limit):
                if _all_selected_doctors_reached(doctor_ids, attempted_by_doctor, args.limit):
                    break
                continue
            identity = _record_identity(record, index)
            order_sn = _text(record.get("order_sn"))
            patient_key = _patient_key(record)
            if identity in processed_ids or (order_sn and order_sn in processed_order_sns):
                # Preserve input ordering for the previous-diagnosis dependency.
                await flush_batch()
                skipped_diagnoses = _diagnosis_payload(record, prefer_ai=True)
                if patient_key and _has_any_diagnosis(skipped_diagnoses):
                    previous_diagnoses_by_patient[patient_key] = skipped_diagnoses
                skipped += 1
                logger.info(f"数据已经处理过，【SKIP】 {order_sn}")
                continue
            if not doctor_ids and args.limit and attempted >= args.limit:
                break
            attempted += 1
            if doctor_id:
                attempted_by_doctor[doctor_id] += 1
            record_id = _record_key(record, index)
            extract_current_history = (
                _text(record.get("is_first")) == "复诊"
                and bool(patient_key)
                and patient_visit_counts[patient_key] > 1
            )
            pending_batch.append(
                {
                    "record": record,
                    "record_id": record_id,
                    "identity": identity,
                    "order_sn": order_sn,
                    "patient_key": patient_key,
                    "extract_current_history": extract_current_history,
                    "use_rag": args.use_rag,
                }
            )
            if len(pending_batch) >= batch_size:
                await flush_batch()
        await flush_batch()
    return succeeded, failed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 LangChain 的多医生病历 AI 清洗与结构化脚本")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入目录、.json 或 .jsonl；默认 {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"分医生 JSONL 输出目录；默认 {DEFAULT_OUTPUT_DIR}",
    )
    default_model = _env("MEDICAL_AGENT_MODEL", "gpt-4.1-mini")
    parser.add_argument("--model", default=default_model, help="文本模型")
    parser.add_argument(
        "--diagnosis-model",
        default=_env_any(
            ("MEDICAL_AGENT_DIAGNOSIS_MODEL", "DEEPSEEK_TEXT_MODEL", "BAICHUAN_TEXT_MODEL"),
            "Baichuan4-Turbo",
        ),
        help="诊断补全模型，默认使用独立诊断模型配置",
    )
    parser.add_argument(
        "--diagnosis-base-url",
        default=_env_any(
            ("MEDICAL_AGENT_DIAGNOSIS_BASE_URL", "DEEPSEEK_BASE_URL", "BAICHUAN_BASE_URL"),
            "https://api.baichuan-ai.com/v1/",
        ),
        help="诊断补全模型的 OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--diagnosis-api-key",
        default=_env_any(("MEDICAL_AGENT_DIAGNOSIS_API_KEY", "DEEPSEEK_API_KEY", "BAICHUAN_API_KEY")),
        help="诊断补全模型 API Key",
    )
    parser.add_argument("--vlm-model", default=_env("MEDICAL_AGENT_VLM_MODEL"), help="视觉模型，默认同文本模型")
    parser.add_argument(
        "--image-classification-model",
        default=_env("MEDICAL_AGENT_IMAGE_CLASSIFICATION_MODEL", CLASSIFICATION_MODEL_NAME),
        help=f"图片分类模型，默认 {CLASSIFICATION_MODEL_NAME}",
    )
    parser.add_argument(
        "--base-url",
        "--text-base-url",
        dest="base_url",
        default=_env("DASHSCOPE_BASE_URL", DASHSCOPE_COMPATIBLE_BASE_URL),
        help="文本模型 OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--api-key",
        "--text-api-key",
        dest="api_key",
        default=_env("DASHSCOPE_API_KEY"),
        help="文本模型 API Key",
    )
    parser.add_argument(
        "--vlm-base-url",
        default=_env("MEDICAL_AGENT_VLM_BASE_URL"),
        help="视觉模型 OpenAI 兼容接口地址；默认使用文本模型接口地址",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=_env("MEDICAL_AGENT_VLM_API_KEY"),
        help="视觉模型 API Key；默认使用文本模型 API Key",
    )
    parser.add_argument(
        "--image-classification-base-url",
        default=_env_any(("MEDICAL_AGENT_IMAGE_CLASSIFICATION_BASE_URL", "DASHSCOPE_BASE_URL"), DASHSCOPE_COMPATIBLE_BASE_URL),
        help="图片分类模型接口地址；默认使用 DashScope OpenAI 兼容接口",
    )
    parser.add_argument(
        "--image-classification-api-key",
        default=_env_any(("MEDICAL_AGENT_IMAGE_CLASSIFICATION_API_KEY", "DASHSCOPE_API_KEY")),
        help="图片分类模型 API Key；默认使用 DASHSCOPE_API_KEY",
    )
    parser.add_argument(
        "--reviewer-model",
        default=_env_any(("MEDICAL_AGENT_REVIEWER_MODEL", "KIMI_K3"), "kimi/kimi-k3"),
        help="评审 Agent 模型；默认使用 DashScope 的 KIMI_K3",
    )
    parser.add_argument(
        "--reviewer-base-url",
        default=_env_any(
            ("MEDICAL_AGENT_REVIEWER_BASE_URL", "KIMI_BASE_URL", "DASHSCOPE_BASE_URL"),
            DASHSCOPE_COMPATIBLE_BASE_URL,
        ),
        help="评审 Agent 的 DashScope OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--reviewer-api-key",
        default=_env_any(("MEDICAL_AGENT_REVIEWER_API_KEY", "KIMI_API_KEY", "DASHSCOPE_API_KEY")),
        help="评审 Agent 的 DashScope API Key",
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="启用评审 Agent；默认关闭",
    )
    parser.add_argument("--doctor-id", action="append", help="仅处理指定 doctor_id；可重复传入，默认所有医生")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="处理条数限制；指定 doctor_id 时表示每个医生最多处理多少条，默认所有医生时表示总条数",
    )
    parser.add_argument("--interval", type=float, default=0.0, help="每条病历后的等待秒数")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每批最多并发处理的病历数；同一患者在批内仍按就诊顺序串行，默认 8",
    )
    parser.add_argument(
        "--record-timeout",
        type=float,
        default=0.0,
        help="单条病历全部阶段的总超时秒数，0 表示不额外限制；单阶段仍受 --timeout 限制",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单次模型请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="模型请求重试次数")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="兼容旧参数；所有 Agent 默认关闭 thinking mode",
    )
    parser.add_argument("--reprocess", action="store_true", help="忽略已有成功记录并重新处理")
    parser.add_argument("--fail-fast", action="store_true", help="任一病历失败时立即退出")
    parser.add_argument("--use-rag", type=bool, default=False, help="是否使用病因提取中医知识库")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    succeeded, failed, skipped = asyncio.run(process_records(args))
    logger.info(
        f"处理完成: 成功 {succeeded} 条，失败 {failed} 条，跳过已处理或过滤 {skipped} 条，输出目录 {args.output_dir}"
    )


if __name__ == "__main__":
    main()
