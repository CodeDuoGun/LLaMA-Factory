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
import fcntl
import hashlib
import inspect
import json
import math
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, TextIO
from urllib.parse import unquote_to_bytes
from uuid import UUID

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from medical.data_utils.structured_agent import (
    AgentModelConfig,
    StructuredAgent,
    StructuredOutputMode,
    model_json_schema,
    response_text,
)
from medical.utils.log import logger
from medical.data_utils.medical_record_formatters import (
    format_inspection_result_text,
    format_tongue_face_result_text,
)
from medical.data_utils.ai_medical_records import AI_FIELDS, DEFAULT_AI_OPERATOR, record_to_row
from dotenv import load_dotenv
from medical.schema.clear_record_basemodel import (
    AgentReviewResult,
    ClinicalExtractionResult,
    DiagnosisResult,
    HistoryCleaningResult,
    ImageClassificationResult,
    InspectionResult,
    TongueFaceResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical.analysis.engine import normalize_diagnosis_value, normalize_sickness_value  # noqa: E402
from medical.data_utils.normalize_ai_diagnosis_labels import normalize_ai_diagnosis_to_standard  # noqa: E402

DEFAULT_INPUT = Path("medical/data/202301_online")
DEFAULT_OUTPUT_DIR = Path("medical/processed_data")
MYSQL_RECORD_TABLE = "mlops_annotation_medical_record"
MYSQL_ID_COLUMN = "id"
MYSQL_NON_STAGE_AI_FIELDS = {
    "ai_processing_errors",
    "ai_processing_complete",
    "ai_processing_filtered",
    "ai_filter_reason",
    "ai_record_identity",
    "ai_processing_version",
}
DOCTOR_RECORD_FILE_NAME = "medical_records_ai_clean.jsonl"
FAILURE_FILE_NAME = "medical_record_agent_failures.jsonl"
FILTER_FILE_NAME = "filter.json"
MYSQL_UPDATE_PROGRESS_FILE_NAME = "mysql_stage_update_progress.json"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DIAGNOSIS_FIELDS = ("diagnosis_illness", "diagnosis_disease", "diagnosis_sickness")
STAGE_NAMES = (
    "image_classification",
    "inspection_vlm",
    "tongue_face_vlm",
    "clinical_cleaning",
    "diagnosis_completion",
    "clinical_extraction",
    "diagnosis_normalization",
)
IMAGE_CLEANING_STAGES = (
    "image_classification",
    "inspection_vlm",
    "tongue_face_vlm",
    "clinical_cleaning",
)
HISTORY_FIELDS = (
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "birth_detail",
    "marriage_history",
    "family_history",
)
CLEANED_HISTORY_FIELDS = (
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "family_history",
)
IMAGE_CLEANING_AI_FIELDS = {
    "ai_inspection_report_img",
    "ai_tongue_face_img",
    "ai_patient_appeal",
    "ai_new_medical_history",
    "ai_complaint_history_consistent",
    "ai_complaint_history_issues",
    *(f"ai_{field}" for field in CLEANED_HISTORY_FIELDS),
    *(f"ai_{field}" for field in DIAGNOSIS_FIELDS),
    *(f"ai_{field}_reason" for field in DIAGNOSIS_FIELDS),
}
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
DEFAULT_AGENT_TEMPERATURE = 0.0
DEFAULT_AGENT_MAX_TOKENS = 3600
IMAGE_DEDUP_DOWNLOAD_CONCURRENCY = 6
IMAGE_DEDUP_DOWNLOAD_TIMEOUT = 15.0
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
LAB_REPORT_TYPES = {"检验检测报告", "检验报告"}
EXAMINATION_REPORT_TYPES = {
    "检查报告",
    "CT诊断报告",
    "影像检查报告",
    "病理报告",
    "其他检查报告",
}
DISCHARGE_REPORT_TYPES = {"住院/出院报告"}
OUTPATIENT_RECORD_TYPES = {"门诊病历"}
INVALID_INSPECTION_TYPES = {"非目标医学文档", "非检查报告", "其他/无法识别", "无法识别"}
FILTER_FILE_LOCK = Lock()


def resolve_stage_names(stage_names: Iterable[str] | None) -> tuple[str, ...]:
    """Validate and de-duplicate requested stages while preserving pipeline order."""
    requested = {_text(name) for name in (stage_names or []) if _text(name)}
    if not requested:
        return STAGE_NAMES
    unknown = requested.difference(STAGE_NAMES)
    if unknown:
        raise ValueError(f"未知 stage_name: {', '.join(sorted(unknown))}；可选值: {', '.join(STAGE_NAMES)}")
    if {"inspection_vlm", "tongue_face_vlm"} & requested:
        requested.add("image_classification")
    return tuple(name for name in STAGE_NAMES if name in requested)


def _stage_output_file_name(stage_names: Iterable[str] | None, *, failure: bool = False) -> str:
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d")
    prefix = "medical_record_agent_failures_stage" if failure else "medical_records_ai_stage"
    return f"{prefix}.jsonl"


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


MYSQL_DATABASE_URL = _env("MEDICAL_AGENT_MYSQL_DATABASE_URL") or _env("DATABASE_URL")


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


def _normalize_inspection_finding(report: Any) -> None:
    """Enforce category-specific fields after VLM extraction."""
    image_type = _clean_generated_text(report.image_type)
    report.image_type = image_type
    if image_type in INVALID_INSPECTION_TYPES:
        report.is_valid_report = False
        report.report_name = ""
        report.report_date = ""
        report.relative_to_visit_time = ""
        report.content = ""
        report.abnormal_indicators = []
        report.abnormal_results = []
        report.conclusion = ""
        report.discharge_diagnosis = ""
        report.discharge_condition = ""
        return

    valid_types = LAB_REPORT_TYPES | EXAMINATION_REPORT_TYPES | DISCHARGE_REPORT_TYPES | OUTPATIENT_RECORD_TYPES
    if image_type in valid_types:
        report.is_valid_report = True

    if image_type in LAB_REPORT_TYPES:
        report.abnormal_results = []
        report.discharge_diagnosis = ""
        report.discharge_condition = ""
    elif image_type in EXAMINATION_REPORT_TYPES:
        report.abnormal_indicators = []
        report.discharge_diagnosis = ""
        report.discharge_condition = ""
    elif image_type in DISCHARGE_REPORT_TYPES:
        report.content = ""
        report.abnormal_indicators = []
        report.abnormal_results = []
        report.conclusion = ""
    elif image_type in OUTPATIENT_RECORD_TYPES:
        report.content = ""
        report.abnormal_indicators = []
        report.discharge_diagnosis = ""
        report.discharge_condition = ""

    report.abnormal_indicators = _clean_generated_value(report.abnormal_indicators)
    report.abnormal_results = _clean_generated_value(report.abnormal_results)
    report.discharge_diagnosis = _clean_generated_text(report.discharge_diagnosis)
    report.discharge_condition = _clean_generated_text(report.discharge_condition)


def _json_default(value: Any) -> Any:
    """Serialize common values returned by SQLAlchemy/MySQL without hiding unknown types."""
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (set, frozenset)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default, **kwargs)


def _json_for_prompt(value: Any) -> str:
    return _json_dumps(value, separators=(",", ":"))


def _chief_complaint(record: dict[str, Any]) -> str:
    return _text(record.get("patient_appeal") or record.get("doc_ass_stu_appeal"))

def _present_history(record: dict[str, Any]) -> str:
    return _text(record.get("new_medical_history") or record.get("present_history"))


def _latest_chief_complaint(record: dict[str, Any]) -> str:
    return _text(record.get("ai_patient_appeal")) or _chief_complaint(record)


def _latest_present_history(record: dict[str, Any]) -> str:
    return _text(record.get("ai_new_medical_history")) or _present_history(record)


def _has_effective_history_text(value: Any) -> bool:
    text = _text(value)
    return bool(text and text not in INVALID_HISTORY_TEXT and text not in {"未记录", "不详", "未知"})


def _fallback_history_excerpt(value: Any, *, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", _text(value)).strip("；;，,。 ")
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}……"


def _fallback_patient_appeal(record: dict[str, Any], present_history: str) -> str:
    """模型返回空主诉时，仅使用允许的主诉来源，不从诊断或图片猜测."""
    previous_context = record.get("__previous_record_context") or {}
    if not isinstance(previous_context, Mapping):
        previous_context = {}
    for value in (
        record.get("doc_ass_stu_appeal"),
        record.get("patient_appeal"),
        previous_context.get("chief_complaint"),
    ):
        if _has_effective_history_text(value):
            return _fallback_history_excerpt(value, limit=40)
    return ""


def _record_history_context(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "visit_time": _visit_time(dict(record)),
        "chief_complaint": _text(record.get("ai_patient_appeal")) or _chief_complaint(dict(record)),
        "present_history": _text(record.get("ai_new_medical_history")) or _present_history(dict(record)),
    }


def _fallback_present_history(
    record: dict[str, Any],
    histories: Mapping[str, str],
    inspection_description: str,
    tongue_face_description: str,
) -> str:
    """在模型返回空现病史时，基于输入证据保守补全."""
    original_history = _present_history(record)
    if _has_effective_history_text(original_history):
        return _replace_inspection_report_symbols(_fallback_history_excerpt(original_history, limit=500))

    parts = []
    chief = _text(record.get("doc_ass_stu_appeal")) or _text(record.get("patient_appeal"))
    if chief:
        parts.append(f"本次就诊主要问题为{_fallback_history_excerpt(chief, limit=80)}。")
    for label, field in (
        ("既往史", "old_medical_history"),
        ("过敏史", "allergic_history"),
        ("个人史", "personal_history"),
        ("特殊时期", "special_history"),
        ("家族史", "family_history"),
    ):
        value = histories.get(field, "")
        if _has_effective_history_text(value):
            parts.append(f"{label}：{_fallback_history_excerpt(value, limit=120)}。")
    if inspection_description and inspection_description != "未见有效检查报告信息。":
        parts.append(f"本次检查资料：{_replace_inspection_report_symbols(inspection_description)}")
    if tongue_face_description and tongue_face_description != "未见有效舌面患处信息。":
        parts.append(tongue_face_description)
    if not parts:
        parts.append("现有资料未提供明确现病史细节。")
    return _replace_inspection_report_symbols("".join(parts))


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
    histories = {}
    for field in HISTORY_FIELDS:
        ai_field = f"ai_{field}"
        histories[field] = _text(record.get(ai_field)) if ai and ai_field in record else _text(record.get(field))
    # 原始数据使用 is_marriage_history，统一映射到 HISTORY_FIELDS 中的 marriage_history。
    if not histories["marriage_history"]:
        histories["marriage_history"] = _text(record.get("is_marriage_history"))
    return histories


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
    filtered_record.pop("__previous_record_context", None)
    identity = _filtered_record_identity(filtered_record)

    with FILTER_FILE_LOCK:
        lock_path = target.with_name(f".{target.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
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
                temporary.write_text(_json_dumps(records, indent=2), encoding="utf-8")
                temporary.replace(target)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return True

HAS_DRUGNAME_PROCESS_NAME = ["饮片","颗粒","水丸","蜜丸", "膏方", "粉剂","浓缩丸","水蜜丸","糊丸"]

def _model_dump(model: BaseModel, *, by_alias: bool = False) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 的序列化接口."""
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


def _format_prescription_dose(item: Mapping[str, Any]) -> str:
    """按处方分析口径计算并格式化单味药剂量."""
    try:
        dose = float(item["spec_number"]) * float(item["drug_num"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not math.isfinite(dose):
        return ""
    dose_text = str(int(dose)) if dose.is_integer() else f"{dose:g}"
    return f"{dose_text}{_text(item.get('unit_name')) or 'g'}"


def format_prescription_text(prescriptions: Any) -> str:
    """将 ``ps`` 字段转换为包含处方类型、药物剂量及用法医嘱的详情文本."""
    if not isinstance(prescriptions, list):
        return "未见有效处方信息。"

    descriptions = []
    for prescription in prescriptions:
        if not isinstance(prescription, Mapping):
            continue

        drugs = []
        prescription_items = prescription.get("prescription_items")
        drug_list = prescription_items.get("drugList") if isinstance(prescription_items, Mapping) else []
        for item in drug_list or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("drug_process_name") in HAS_DRUGNAME_PROCESS_NAME:
                name = _text(item.get("drug_name"))
            else:
                name = _text(item.get("drug_name") or item.get("sub_drug_name") or item.get("show_name"))
            drug_id = _text(item.get("drug_id"))
            if not name and not drug_id:
                continue
            dose = _format_prescription_dose(item)
            decoction = _text(item.get("decoction_name"))
            drug_text = name or f"药品ID {drug_id}"
            if dose:
                drug_text += dose
            if decoction:
                drug_text += f"（{decoction}）"
            drugs.append(drug_text)

        details = []
        for label, value in (
            ("用法", prescription.get("usage_type")),
            ("加工类型", prescription.get("drug_process_name")),
        ):
            if text := _text(value):
                details.append(f"{label}：{text}")
        if drugs:
            details.append(f"药物（{len(drugs)}味）：{'、'.join(drugs)}")
        for label, value in (
            ("用法用量", prescription.get("usage_desc")),
            ("医嘱", prescription.get("doctor_advice")),
        ):
            if text := _text(value):
                details.append(f"{label}：{text}")
        if details:
            descriptions.append(f"处方{len(descriptions) + 1}：" + "；".join(details) + "。")

    return "\n".join(descriptions) or "未见有效处方信息。"


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def _model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    return model_json_schema(schema)


async def _invoke_structured_candidate(
    agent: Any,
    messages: list[HumanMessage],
    schema: type[BaseModel],
) -> tuple[BaseModel, str]:
    if isinstance(agent, StructuredAgent):
        result = await agent.ainvoke(messages)
        if not isinstance(result.output, schema):
            return schema.model_validate(_model_dump(result.output)), result.trace
        return result.output, result.trace

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
    return response_text(response)


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
        if isinstance(model, StructuredAgent):
            structured_result = await model.ainvoke(human_messages)
            candidate = structured_result.output
            raw_text = structured_result.trace
        else:
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
    """Build at-most-eight-image messages after URL and exact-content deduplication.

    HTTP URL:
        ├─ 下载成功 → 去重 → 缩放 → Base64 传给分类模型
        └─ 下载失败 → 原始 HTTP URL 直接传给分类模型
    """
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
            self.reviewer_agent = StructuredAgent(
                reviewer_model or vlm_model,
                AgentReviewResult,
                system_prompt=_load_prompt("agent_review_prompt.txt"),
                name="medical_agent_reviewer",
            )
        self.image_classifier_agent = StructuredAgent(
            image_classification_model or vlm_model,
            ImageClassificationResult,
            system_prompt=_load_prompt("image_classification_prompt.txt"),
            name="medical_image_classifier_agent",
        )
        self.inspection_agent = StructuredAgent(
            vlm_model,
            InspectionResult,
            system_prompt=_load_prompt("inspection_report_analysis_prompt.txt"),
            name="inspection_report_agent",
        )
        self.tongue_face_agent = StructuredAgent[TongueFaceResult](
            vlm_model,
            TongueFaceResult,
            system_prompt=_load_prompt("tongue_face_analysis_prompt.txt"),
            name="tongue_face_agent",
        )

    @staticmethod
    def _build_diagnosis_agent(model: ChatOpenAI) -> Any:
        # Baichuan accepts json_object, but rejects ToolStrategy's tool_choice and
        # ProviderStrategy's JSON Schema request body. Validate the JSON locally.
        return StructuredAgent(
            model,
            DiagnosisResult,
            system_prompt=_load_prompt("fill_diagnosis_prompt.txt"),
            name="diagnosis_completion_agent",
            output_mode=StructuredOutputMode.JSON_OBJECT,
        )

    @staticmethod
    def _build_history_agent(model: ChatOpenAI) -> Any:
        return StructuredAgent(
            model,
            HistoryCleaningResult,
            system_prompt=_load_prompt("clinical_record_cleaning_prompt.txt"),
            name="history_cleaning_agent",
        )

    @staticmethod
    def _build_extract_agent(model: ChatOpenAI) -> Any:
        return StructuredAgent(
            model,
            ClinicalExtractionResult,
            system_prompt=_load_prompt("extract_prompt.txt"),
            name="clinical_extraction_agent",
        )

    @staticmethod
    def normalize_medical_history(record: dict[str, Any]) -> None:
        record["new_medical_history"] = _normalize_medical_history(record.get("new_medical_history"))

    @staticmethod
    def normalize_diagnoses(record: dict[str, Any]) -> None:
        record["diagnosis_illness"] = normalize_diagnosis_value(record.get("diagnosis_illness"))
        record["diagnosis_sickness"] = normalize_sickness_value(record.get("diagnosis_sickness"))
        # AI 补全不会回写原始诊断字段，不能按字段位置直接追加“证”，否则会把错填在
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
        for field in DIAGNOSIS_FIELDS:
            original_value = _clean_generated_text(source_diagnoses.get(field))
            if field != "diagnosis_illness" and original_value:
                record[f"ai_{field}"] = normalize_ai_diagnosis_to_standard(f"ai_{field}", original_value)
                record[f"ai_{field}_reason"] = "原字段非空，按要求保留原诊断结果。"
                continue

            record[f"ai_{field}"] = normalize_ai_diagnosis_to_standard(
                f"ai_{field}", _clean_generated_text(values.get(field))
            )
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

    async def clean_histories(self, record: dict[str, Any]) -> None:
        histories = _history_payload(record)
        previous_context = record.get("__previous_record_context") or {}
        if not isinstance(previous_context, Mapping):
            previous_context = {}
        inspection_description = format_inspection_result_text(record.get("ai_inspection_report_img"))
        logger.info(f"检查报告分析结果： {inspection_description}")
        tongue_face_description = format_tongue_face_result_text(record.get("ai_tongue_face_img"))
        logger.info(f"舌面分析结果： {tongue_face_description}")
        system_prompt = _load_prompt("clinical_record_cleaning_prompt.txt")
        result = await _invoke_structured_agent(
            self.history_agent,
            HumanMessage(
                content=(
                    f"医生撰写主诉 doc_ass_stu_appeal：{_text(record.get('doc_ass_stu_appeal')) or '未记录'}\n"
                    f"患者主诉 patient_appeal：{_text(record.get('patient_appeal')) or '未记录'}\n"
                    f"就诊类型 is_first：{_text(record.get('is_first')) or '未记录'}\n"
                    f"原始本次现病史 new_medical_history：{_present_history(record) or '未记录'}\n"
                    "说明：该字段可能包含按日期累积的多次就诊记录，须按本次就诊时间和类型定位。\n"
                    f"上一次就诊时间：{_text(previous_context.get('visit_time')) or '无上一次就诊'}\n"
                    f"上一次看诊主诉：{_text(previous_context.get('chief_complaint')) or '未记录'}\n"
                    f"本次就诊时间：{_visit_time(record) or '未记录'}\n"
                    f"原始病史：{_json_for_prompt(histories)}\n"
                    f"医生书写的舌象及面相 admin_face_describe：{_text(record.get('admin_face_describe')) or '未记录'}\n"
                    f"生育史 birth_detail：{_text(record.get('birth_detail')) or '未记录'}\n"
                    f"婚恋史 is_marriage_history（统一映射为 marriage_history）："
                    f"{histories['marriage_history'] or '未记录'}\n"
                    f"西医诊断 diagnosis_illness：{_text(record.get('diagnosis_illness')) or '未记录'}\n"
                    f"中医证型 diagnosis_disease：{_text(record.get('diagnosis_disease')) or '未记录'}\n"
                    f"中医诊断 diagnosis_sickness：{_text(record.get('diagnosis_sickness')) or '未记录'}\n"
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
        if not _has_effective_history_text(record["ai_patient_appeal"]):
            record["ai_patient_appeal"] = _fallback_patient_appeal(record, record["ai_new_medical_history"])
        if not _has_effective_history_text(record["ai_new_medical_history"]):
            record["ai_new_medical_history"] = _fallback_present_history(
                record,
                histories,
                inspection_description,
                tongue_face_description,
            )
        record["ai_complaint_history_consistent"] = values.get("complaint_history_consistent")
        record["ai_complaint_history_issues"] = values.get("consistency_issues") or []
        for field in CLEANED_HISTORY_FIELDS:
            original = histories[field]
            if field == "allergic_history" and original in {"无", "无特殊", "否认", "否认过敏"}:
                record[f"ai_{field}"] = "无药物及食物过敏史"
            else:
                record[f"ai_{field}"] = (
                    "" if original in INVALID_HISTORY_TEXT else _clean_generated_text(values.get(field))
                )
        for field in DIAGNOSIS_FIELDS:
            cleaned_diagnosis = _clean_generated_text(values.get(field)) or _clean_generated_text(record.get(field))
            record[f"ai_{field}"] = normalize_ai_diagnosis_to_standard(f"ai_{field}", cleaned_diagnosis)
            reason = _clean_generated_text(values.get(f"{field}_reason"))
            if not reason and record[f"ai_{field}"]:
                reason = "模型未提供修正理由，保留并规范化原诊断。"
            record[f"ai_{field}_reason"] = reason

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
                "检验检测报告的 abnormal_indicators 只保留明确异常指标；检查/CT 报告和门诊病历中的"
                "检查项目将明确异常结果写入 abnormal_results；住院/出院报告只提取非空的"
                "discharge_diagnosis 和 discharge_condition。conclusion 只保留原文结论。"
                "非医学文档或无法识别时只返回图片类别和 is_valid_report=false，其余字段留空。"
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
            system_prompt=_load_prompt("tongue_face_analysis_prompt_v2.txt"),
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
            _normalize_inspection_finding(report)
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
        histories = _history_payload(record, ai=True)
        illness_knowledge = self.build_illness_knowledge(diagnoses) if use_rag else "无"
        inspection_description = format_inspection_result_text(record.get("ai_inspection_report_img"))
        tongue_face_description = format_tongue_face_result_text(record.get("ai_tongue_face_img"))
        prescription_description = format_prescription_text(record.get("ps"))
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
                    f"处方详情：\n{prescription_description}\n"
                    f"疾病的知识库信息 {illness_knowledge}\n"
                    "严格根据知识库和本次病历证据，一次性提取患者在中医层面的病因、病机、病期、"
                    "病位、病程、临床症状及治则治法。若知识库无知识，根据患者病史保守完成提取。"
                    "必须返回合法的 JSON 对象，不要输出 Markdown，不要输出额外解释"
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
        use_rag: bool = False,
        stage_names: list[str] | tuple[str, ...] | None = None,
        skip_record_filter: bool = False,
    ) -> dict[str, Any]:
        selected_stages = set(resolve_stage_names(stage_names))
        output = dict(record)
        original_history_present = "new_medical_history" in output
        original_history = output.get("new_medical_history")
        # 前端保存的是富文本 HTML。先转成纯文本再参与空值过滤和后续 Agent 调用，
        # 避免标签污染提示词，也避免只有 <p><br></p> 的空病史绕过过滤。
        # 1. 规则清洗现病史字段内容
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

        # 2. 主诉、现病史和西医诊断全部缺失时，不调用任何 Agent，过滤
        if not skip_record_filter and _should_filter_record(output):
            write2filter_json(output, self.filter_path)
            output["ai_processing_filtered"] = True
            output["ai_processing_complete"] = False
            output["ai_filter_reason"] = (
                "patient_appeal、new_medical_history、diagnosis_illness 均为空"
            )
            logger.info(f"[STAGE FILTER] record={record_id} doctor_name={doctor_name}")
            if original_history_present:
                output["new_medical_history"] = original_history
            else:
                output.pop("new_medical_history", None)
            return output

        # 3：图片分类与视觉解析管线。
        async def run_image_pipeline() -> list[dict[str, str]]:
            pipeline_errors = []
            classified_images: dict[str, list[str]] = {category: [] for category in IMAGE_CATEGORIES}
            if "image_classification" in selected_stages or {
                "inspection_vlm",
                "tongue_face_vlm",
            } & selected_stages:
                try:
                    async def classify_image_stage() -> None:
                        nonlocal classified_images
                        classified_images = await self.classify_images(output)

                    await self._run_stage("image_classification", record_id, doctor_name, classify_image_stage)
                except Exception as exc:
                    pipeline_errors.append(
                        {"stage": "image_classification", "error": f"{type(exc).__name__}: {exc}"}
                    )

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
            image_stages = {name: value for name, value in image_stages.items() if name in selected_stages}
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

        errors.extend(await run_image_pipeline())

        # 4：等待图片完成后，在同一阶段提取本次现病史并完成主诉、现病史和五史清洗。
        if "clinical_cleaning" in selected_stages:
            try:
                await self._run_stage("clinical_cleaning", record_id, doctor_name, lambda: self.clean_histories(output))
            except Exception as exc:
                errors.append({"stage": "clinical_cleaning", "error": f"{type(exc).__name__}: {exc}"})
        # 5：用最新病历补全诊断，再一次性提取知识库字段和治则治法。
        if "diagnosis_completion" in selected_stages:
            try:
                await self._run_stage(
                    "diagnosis_completion",
                    record_id,
                    doctor_name,
                    lambda: self.complete_diagnoses(output),
                )
            except Exception as exc:
                errors.append({"stage": "diagnosis_completion", "error": f"{type(exc).__name__}: {exc}"})
        if "clinical_extraction" in selected_stages:
            try:
                await self._run_stage(
                    "clinical_extraction",
                    record_id,
                    doctor_name,
                    lambda: self.extract_clinical_fields(output, use_rag),
                )
            except Exception as exc:
                errors.append({"stage": "clinical_extraction", "error": f"{type(exc).__name__}: {exc}"})
        # 6：所有 Agent 阶段完成后，最后归一化原始诊断字段。
        if "diagnosis_normalization" in selected_stages:
            try:
                await self._run_stage(
                    "diagnosis_normalization",
                    record_id,
                    doctor_name,
                    lambda: self.normalize_diagnoses(output),
                )
            except Exception as exc:
                errors.append({"stage": "diagnosis_normalization", "error": f"{type(exc).__name__}: {exc}"})

        if errors:
            output["ai_processing_errors"] = errors
        else:
            output.pop("ai_processing_errors", None)
        output.pop("__original_diagnoses", None)
        output.pop("__previous_diagnoses", None)
        output.pop("__previous_record_context", None)
        if original_history_present:
            output["new_medical_history"] = original_history
        else:
            output.pop("new_medical_history", None)

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


def count_planned_records(
    path: Path,
    doctor_ids: set[str],
    limit: int,
    processed_ids: set[str],
    processed_order_sns: set[str],
) -> int:
    """Count records this run will actually submit to an Agent."""
    attempted = 0
    attempted_by_doctor: Counter[str] = Counter()
    for index, record in enumerate(iter_records(path), 1):
        if not _doctor_matches(record, doctor_ids):
            continue
        doctor_id = _text(record.get("doctor_id"))
        if _selected_doctor_quota_reached(doctor_id, doctor_ids, attempted_by_doctor, limit):
            if _all_selected_doctors_reached(doctor_ids, attempted_by_doctor, limit):
                break
            continue
        identity = _record_identity(record, index)
        order_sn = _text(record.get("order_sn"))
        if identity in processed_ids or (order_sn and order_sn in processed_order_sns):
            continue
        if not doctor_ids and limit and attempted >= limit:
            break
        attempted += 1
        if doctor_id:
            attempted_by_doctor[doctor_id] += 1
    return attempted


def _is_missing_history_reprocess_candidate(record: Mapping[str, Any]) -> bool:
    missing_complaint = (
        not _has_effective_history_text(record.get("doc_ass_stu_appeal"))
        and not _has_effective_history_text(record.get("ai_patient_appeal"))
    )
    missing_present_history = (
        not _has_effective_history_text(record.get("new_medical_history"))
        and not _has_effective_history_text(record.get("ai_new_medical_history"))
    )
    return missing_complaint or missing_present_history


def _prepare_reprocess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(record)
    for field in (
        "ai_processing_errors",
        "ai_processing_complete",
        "ai_processing_filtered",
        "ai_filter_reason",
        "ai_record_identity",
        "ai_processing_version",
    ):
        prepared.pop(field, None)
    return prepared


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning(f"忽略无法解析的 JSONL 行: {path}:{line_number}: {error}")
                continue
            if not isinstance(record, dict):
                logger.warning(f"忽略非 JSON 对象行: {path}:{line_number}")
                continue
            yield record


def _candidate_key(record: Mapping[str, Any], fallback: int) -> str:
    return _text(record.get("order_sn")) or _record_identity(dict(record), fallback)


def select_reprocess_records(
    output_dir: Path,
    doctor_ids: set[str],
    *,
    include_failures: bool,
    include_missing_histories: bool,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """选择历史失败记录，以及成功文件中主诉/现病史仍缺失的记录."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempted_by_doctor: Counter[str] = Counter()

    def maybe_add(record: Mapping[str, Any], index: int) -> None:
        record_dict = dict(record)
        if not _doctor_matches(record_dict, doctor_ids):
            return
        doctor_id = _text(record.get("doctor_id"))
        if _selected_doctor_quota_reached(doctor_id, doctor_ids, attempted_by_doctor, limit):
            return
        if not doctor_ids and limit and len(selected) >= limit:
            return
        key = _candidate_key(record, index)
        if key in seen:
            return
        seen.add(key)
        if doctor_id:
            attempted_by_doctor[doctor_id] += 1
        selected.append(_prepare_reprocess_record(record))

    index = 0
    if include_failures:
        for path in sorted(output_dir.rglob(FAILURE_FILE_NAME)):
            for record in _iter_jsonl_records(path):
                index += 1
                maybe_add(record, index)

    if include_missing_histories:
        for path in sorted(output_dir.rglob(DOCTOR_RECORD_FILE_NAME)):
            for record in _iter_jsonl_records(path):
                index += 1
                if _is_missing_history_reprocess_candidate(record):
                    maybe_add(record, index)

    return selected


def count_patient_visits_in_records(records: Iterable[Mapping[str, Any]], doctor_ids: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        record_dict = dict(record)
        if not _doctor_matches(record_dict, doctor_ids):
            continue
        patient_key = _patient_key(record_dict)
        if patient_key:
            counts[patient_key] += 1
    return counts


def count_planned_records_in_records(
    records: Iterable[Mapping[str, Any]],
    doctor_ids: set[str],
    limit: int,
    processed_ids: set[str],
    processed_order_sns: set[str],
) -> int:
    attempted = 0
    attempted_by_doctor: Counter[str] = Counter()
    for index, record in enumerate(records, 1):
        record_dict = dict(record)
        if not _doctor_matches(record_dict, doctor_ids):
            continue
        doctor_id = _text(record_dict.get("doctor_id"))
        if _selected_doctor_quota_reached(doctor_id, doctor_ids, attempted_by_doctor, limit):
            if _all_selected_doctors_reached(doctor_ids, attempted_by_doctor, limit):
                break
            continue
        identity = _record_identity(record_dict, index)
        order_sn = _text(record_dict.get("order_sn"))
        if identity in processed_ids or (order_sn and order_sn in processed_order_sns):
            continue
        if not doctor_ids and limit and attempted >= limit:
            break
        attempted += 1
        if doctor_id:
            attempted_by_doctor[doctor_id] += 1
    return attempted


def _replace_jsonl_records_by_order_sn(path: Path, replacements: list[dict[str, Any]]) -> None:
    if not replacements:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement_by_key = {
        _text(record.get("order_sn")) or _record_identity(record): record
        for record in replacements
    }
    written_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    if path.is_file():
        for index, record in enumerate(_iter_jsonl_records(path), 1):
            key = _text(record.get("order_sn")) or _record_identity(record, index)
            if key in replacement_by_key:
                merged.append(replacement_by_key[key])
                written_keys.add(key)
            else:
                merged.append(record)
    for key, record in replacement_by_key.items():
        if key not in written_keys:
            merged.append(record)

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in merged:
            file.write(_json_dumps(record) + "\n")
    temporary.replace(path)


def _remove_jsonl_records_by_order_sn(path: Path, order_sns: set[str]) -> None:
    if not order_sns or not path.is_file():
        return
    kept = [
        record
        for record in _iter_jsonl_records(path)
        if _text(record.get("order_sn")) not in order_sns
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in kept:
            file.write(_json_dumps(record) + "\n")
    temporary.replace(path)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _progress_log_message(
    *,
    completed: int,
    total: int,
    succeeded: int,
    failed: int,
    filtered: int,
    skipped_existing: int,
    elapsed: float,
    scope: str,
) -> str:
    ratio = min(1.0, completed / total) if total else 1.0
    bar_width = 20
    filled = min(bar_width, round(ratio * bar_width))
    progress_bar = f"{'=' * filled}{'.' * (bar_width - filled)}"
    records_per_minute = completed / elapsed * 60 if completed and elapsed > 0 else 0.0
    eta = (total - completed) / (completed / elapsed) if completed and elapsed > 0 else None
    return (
        f"[PROGRESS] doctor={scope} [{progress_bar}] {ratio * 100:5.1f}% "
        f"completed={completed}/{total} success={succeeded} failed={failed} "
        f"filtered={filtered} skipped_existing={skipped_existing} "
        f"speed={records_per_minute:.2f} records/min elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}"
    )


def load_processed_record_keys(
    output_dir: Path,
    result_file_name: str | None = None,
) -> tuple[set[str], set[str]]:
    """Load identities and order numbers already written to result or filter files."""
    processed_ids: set[str] = set()
    processed_order_sns: set[str] = set()
    if not output_dir.is_dir():
        return processed_ids, processed_order_sns
    # 指定阶段运行时只读取该阶段对应的结果文件，避免其他阶段的结果造成误跳过。
    result_pattern = result_file_name or "*_ai.jsonl"
    for path in output_dir.rglob(result_pattern):
        if path.name == FAILURE_FILE_NAME:
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_result_record = isinstance(record, dict) and (
                    path.name in {DOCTOR_RECORD_FILE_NAME, result_file_name}
                    or record.get("ai_processing_complete") is True
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


def _doctor_output_path(
    output_dir: Path,
    record: dict[str, Any],
    stage_names: Iterable[str] | None = None,
) -> Path:
    return _doctor_output_dir(output_dir, record) / _stage_output_file_name(stage_names)


def _doctor_failure_path(
    output_dir: Path,
    record: dict[str, Any],
    stage_names: Iterable[str] | None = None,
) -> Path:
    return _doctor_output_dir(output_dir, record) / _stage_output_file_name(stage_names, failure=True)


def _write_jsonl(file: TextIO, record: dict[str, Any]) -> None:
    file.write(_json_dumps(record) + "\n")
    file.flush()


def _mysql_row_to_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a database row into the flat record shape consumed by agents."""
    record_data = row.get("record_data")
    record = dict(record_data) if isinstance(record_data, Mapping) else {}
    for key, value in row.items():
        if key != "record_data" and value is not None:
            record[key] = value
    if "id" not in record and row.get("medical_record_id") is not None:
        record["id"] = row["medical_record_id"]
    return record


def _mysql_table_columns(manager: Any, table_name: str) -> set[str]:
    return set(manager.refresh_table(table_name).c.keys())


def _mysql_result_row(record: Mapping[str, Any], columns: set[str], id_column: str) -> dict[str, Any]:
    """Build an output row compatible with either raw tables or ai_cleaned tables."""
    if "record_data" in columns:
        try:
            row = record_to_row(record)
        except ValueError:
            row = {
                "record_data": {key: value for key, value in record.items() if key != "ps" and key not in AI_FIELDS},
                "operator": DEFAULT_AI_OPERATOR,
                **{field_name: record.get(field_name) for field_name in AI_FIELDS},
            }
        row["operator"] = row.get("operator") or DEFAULT_AI_OPERATOR
    else:
        row = dict(record)
    if id_column in columns and id_column not in row and record.get("id") is not None:
        row[id_column] = record.get("id")
    return {key: value for key, value in row.items() if key in columns}


def _mysql_record_key(record: Mapping[str, Any], id_column: str) -> Any:
    return record.get(id_column) or record.get("medical_record_id") or record.get("id")

def _mysql_update_progress_path(args: argparse.Namespace) -> Path:
    """Return the shared local progress JSON used by all MySQL workers."""
    configured = getattr(args, "mysql_progress_file", None)
    if configured:
        return Path(configured)
    return args.output_dir / MYSQL_UPDATE_PROGRESS_FILE_NAME


def _load_mysql_update_progress(path: Path) -> dict[str, set[str]]:
    """Load doctor_id -> processed order_sn set from the shared JSON file.

    A separate lock file is used so four doctor processes can safely read while
    another process is atomically replacing the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            if not path.is_file():
                return {}
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"MySQL 清洗进度文件不是合法 JSON: {path}") from exc
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    if not isinstance(raw, dict):
        raise ValueError(f"MySQL 清洗进度文件必须是 JSON 对象: {path}")

    progress: dict[str, set[str]] = {}
    for doctor_id, order_sns in raw.items():
        if not isinstance(order_sns, list):
            raise ValueError(
                f"MySQL 清洗进度文件中 doctor_id={doctor_id} 的值必须是 order_sn 数组"
            )
        progress[_text(doctor_id)] = {
            _text(order_sn) for order_sn in order_sns if _text(order_sn)
        }
    return progress


def _append_mysql_update_progress(path: Path, doctor_id: Any, order_sn: Any) -> bool:
    """Atomically add one successful doctor_id/order_sn to the shared JSON.

    Returns True when a new order_sn was added, False when it already existed.
    The exclusive file lock makes this safe for the four concurrent workers.
    """
    doctor_key = _text(doctor_id)
    order_key = _text(order_sn)
    if not doctor_key or not order_key:
        raise ValueError("写入 MySQL 清洗进度时 doctor_id/order_sn 不能为空")

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            current: dict[str, list[str]] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"MySQL 清洗进度文件不是合法 JSON: {path}") from exc
                if not isinstance(loaded, dict):
                    raise ValueError(f"MySQL 清洗进度文件必须是 JSON 对象: {path}")
                for key, values in loaded.items():
                    if not isinstance(values, list):
                        raise ValueError(
                            f"MySQL 清洗进度文件中 doctor_id={key} 的值必须是 order_sn 数组"
                        )
                    current[_text(key)] = [
                        _text(value) for value in values if _text(value)
                    ]

            values = current.setdefault(doctor_key, [])
            if order_key in values:
                return False
            values.append(order_key)

            # Keep deterministic output while preserving uniqueness.
            current = {
                key: list(dict.fromkeys(values))
                for key, values in sorted(current.items(), key=lambda item: item[0])
            }
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(_json_dumps(current, indent=2), encoding="utf-8")
            temporary.replace(path)
            return True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _filter_mysql_records_by_progress(
    records: Iterable[Mapping[str, Any]],
    progress: Mapping[str, set[str]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove doctor_id/order_sn pairs that have already completed successfully."""
    pending: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        doctor_id = _text(record.get("doctor_id"))
        order_sn = _text(record.get("order_sn"))
        if doctor_id and order_sn and order_sn in progress.get(doctor_id, set()):
            skipped += 1
            logger.info(f"[MYSQL PROGRESS SKIP] doctor_id={doctor_id} order_sn={order_sn}")
            continue
        pending.append(dict(record))
    return pending, skipped


def _mysql_record_filters(args: argparse.Namespace) -> dict[str, Any]:
    doctor_ids = [
        int(value)
        for value in (getattr(args, "doctor_id", None) or [])
        if _text(value)
    ]

    if not doctor_ids:
        raise ValueError("MySQL 清洗模式必须指定 --doctor-id")

    return {
        "doctor_id": doctor_ids,
        "status": ["unprocessed", "problem"],
    }


def _load_mysql_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the complete fixed MySQL worklist before batch processing."""
    from medical.data_utils.db_manager import DBManager

    manager = DBManager(MYSQL_DATABASE_URL or None)
    try:
        columns = _mysql_table_columns(manager, MYSQL_RECORD_TABLE)
        filters = _mysql_record_filters(args)
        for required_column in filters:
            if required_column not in columns:
                raise ValueError(
                    f"MySQL 表 {MYSQL_RECORD_TABLE} 不存在 {required_column} 字段"
                )

        logger.info(
            f"[MYSQL SELECT] table={MYSQL_RECORD_TABLE} filters={_json_for_prompt(filters)}"
        )
        rows = manager.select(MYSQL_RECORD_TABLE, filters=filters)
        return [_mysql_row_to_record(row) for row in rows]
    finally:
        manager.close()


def _mysql_source_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return _text(record.get("doctor_id")), _text(record.get("order_sn"))


def _load_original_records_for_mysql(
    mysql_records: Iterable[Mapping[str, Any]],
    input_root: Path = DEFAULT_INPUT,
) -> list[dict[str, Any]]:
    """Resolve each selected MySQL row to its raw source record by doctor_id + order_sn.

    Raw/non-AI fields come from ``medical/data/202301_online``. Existing ``ai_*``
    values and the database primary key come from the current MySQL row so a
    selected downstream stage can still consume upstream AI context without
    allowing stale MySQL raw fields to replace the original source data.
    """
    mysql_rows = [dict(record) for record in mysql_records]
    requested: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for position, mysql_record in enumerate(mysql_rows, 1):
        key = _mysql_source_key(mysql_record)
        doctor_id, order_sn = key
        if not doctor_id or not order_sn:
            raise ValueError(
                f"MySQL 查询结果第 {position} 行缺少 doctor_id/order_sn，无法回源原始病历"
            )
        if key in requested:
            raise ValueError(f"MySQL 查询结果存在重复 doctor_id/order_sn: {doctor_id}/{order_sn}")
        requested[key] = mysql_record
        ordered_keys.append(key)

    if not requested:
        return []
    if not input_root.exists():
        raise FileNotFoundError(f"原始病历目录不存在: {input_root}")

    matched_originals: dict[tuple[str, str], dict[str, Any]] = {}
    for original_record in iter_records(input_root):
        key = _mysql_source_key(original_record)
        if key not in requested:
            continue
        if key in matched_originals:
            raise ValueError(f"原始病历存在重复 doctor_id/order_sn: {key[0]}/{key[1]}")
        matched_originals[key] = dict(original_record)
        if len(matched_originals) == len(requested):
            break

    missing = [key for key in ordered_keys if key not in matched_originals]
    if missing:
        preview = ", ".join(f"doctor_id={doctor_id},order_sn={order_sn}" for doctor_id, order_sn in missing[:20])
        raise ValueError(
            f"MySQL 命中 {len(requested)} 条，但在 {input_root} 仅找到 {len(matched_originals)} 条同医生同 order_sn 原始病历；"
            f"缺失 {len(missing)} 条：{preview}"
        )

    resolved: list[dict[str, Any]] = []
    for key in ordered_keys:
        mysql_record = requested[key]
        original_record = matched_originals[key]
        processing_record = dict(original_record)

        # Preserve current upstream AI context from MySQL, but never let MySQL raw
        # clinical fields overwrite the original source record used for cleaning.
        processing_record.update(
            {name: value for name, value in mysql_record.items() if name.startswith("ai_")}
        )
        for name in (MYSQL_ID_COLUMN, "medical_record_id", "id", "status"):
            if name in mysql_record:
                processing_record[name] = mysql_record[name]

        processing_record["__mysql_current_row"] = mysql_record
        resolved.append(processing_record)

    logger.info(
        f"[MYSQL SOURCE MATCH] mysql={len(requested)} original={len(resolved)} root={input_root}"
    )
    return resolved



def export_mysql_records_to_local(args: argparse.Namespace) -> int:
    """导出 MySQL 记录到本地 JSONL 文件，按医生分组."""
    from medical.data_utils.db_manager import DBManager

    output_dir: Path = args.output_dir
    filters = _mysql_record_filters(args)
    export_file_name = _text(getattr(args, "export_file_name", "records.jsonl")) or "records.jsonl"
    if Path(export_file_name).name != export_file_name or export_file_name in {".", ".."}:
        raise ValueError("--export-file-name 只能是文件名，不能包含目录")
    limit = getattr(args, "limit", 0) or 0

    manager = DBManager(MYSQL_DATABASE_URL or None)
    try:
        columns = _mysql_table_columns(manager, MYSQL_RECORD_TABLE)
        id_column = MYSQL_ID_COLUMN if MYSQL_ID_COLUMN in columns else "medical_record_id"

        for required_column in filters:
            if required_column not in columns:
                raise ValueError(
                    f"MySQL 表 {MYSQL_RECORD_TABLE} 不存在 {required_column} 字段，无法按固定条件导出"
                )

        rows = manager.select(
            MYSQL_RECORD_TABLE,
            filters=filters,
            limit=limit if limit > 0 else None,
            order_by=["-id"] if id_column in columns else None,
        )
    finally:
        manager.close()

    doctor_files: dict[str, tuple[TextIO, Path, Path]] = {}
    count = 0

    def get_doctor_file(doctor_id: str, doctor_name: str | None) -> TextIO:
        key = doctor_id
        if key not in doctor_files:
            doctor_dir = output_dir / f"doctor_{_safe_path_part(doctor_id)}_{_safe_path_part(doctor_name)}"
            doctor_dir.mkdir(parents=True, exist_ok=True)
            file_path = doctor_dir / export_file_name
            temporary_path = doctor_dir / f".{file_path.name}.{os.getpid()}.tmp"
            doctor_files[key] = (open(temporary_path, "w", encoding="utf-8"), temporary_path, file_path)
        return doctor_files[key][0]

    export_succeeded = False
    try:
        for row in rows:
            record = _mysql_row_to_record(row)
            doctor_id = _text(record.get("doctor_id")) or "unknown"
            # SQL 已使用统一固定 filter；目录名直接使用记录中的医生姓名。
            doctor_name = record.get("doctor_name")
            file = get_doctor_file(doctor_id, doctor_name)
            _write_jsonl(file, record)
            count += 1
            if count % 1000 == 0:
                logger.info(f"已导出 {count} 条记录...")
        export_succeeded = True
    finally:
        for file, _, _ in doctor_files.values():
            file.close()
        if export_succeeded:
            for _, temporary_path, file_path in doctor_files.values():
                temporary_path.replace(file_path)
        else:
            for _, temporary_path, _ in doctor_files.values():
                temporary_path.unlink(missing_ok=True)

    logger.info(f"导出完成：共 {count} 条记录，分布在 {len(doctor_files)} 个医生目录")
    return count




def _save_mysql_record(args: argparse.Namespace, record: Mapping[str, Any]) -> None:
    from medical.data_utils.db_manager import DBManager

    manager = DBManager(MYSQL_DATABASE_URL or None)
    try:
        columns = _mysql_table_columns(manager, MYSQL_RECORD_TABLE)
        id_column = MYSQL_ID_COLUMN if MYSQL_ID_COLUMN in columns else "medical_record_id"
        row = _mysql_result_row(record, columns, id_column)
        key = _mysql_record_key(record, id_column)
        if key is None:
            raise ValueError(f"无法写入 MySQL：记录缺少主键字段 {id_column}/id/medical_record_id")
        filters = {id_column: key} if id_column in columns else None
        if filters and manager.count(MYSQL_RECORD_TABLE, filters=filters):
            row.pop(id_column, None)
            manager.update(MYSQL_RECORD_TABLE, row, filters=filters)
        else:
            manager.insert(MYSQL_RECORD_TABLE, row)
    finally:
        manager.close()


def _mysql_stage_ai_updates(
    before_record: Mapping[str, Any],
    after_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only stage-produced AI fields whose values changed.

    ``before_record`` is the exact record sent to the selected stages. Because
    MedicalRecordAgents.process only mutates outputs belonging to selected stages,
    this diff naturally limits the UPDATE to those stage outputs. Processing
    metadata is intentionally excluded.
    """
    updates: dict[str, Any] = {}
    names = {
        name
        for name in after_record
        if name.startswith("ai_") and name not in MYSQL_NON_STAGE_AI_FIELDS
    }
    for name in names:
        if after_record.get(name) != before_record.get(name):
            updates[name] = after_record.get(name)
    return updates


def _update_mysql_stage_ai_fields(
    mysql_record: Mapping[str, Any],
    before_record: Mapping[str, Any],
    after_record: Mapping[str, Any],
) -> int:
    """Update only AI fields changed by the selected stages on one existing row."""
    from medical.data_utils.db_manager import DBManager

    manager = DBManager(MYSQL_DATABASE_URL or None)
    try:
        columns = _mysql_table_columns(manager, MYSQL_RECORD_TABLE)
        id_column = MYSQL_ID_COLUMN if MYSQL_ID_COLUMN in columns else "medical_record_id"
        key = _mysql_record_key(mysql_record, id_column)
        if key is None:
            raise ValueError(f"无法更新 MySQL：记录缺少主键字段 {id_column}/id/medical_record_id")
        filters = {id_column: key}
        if not manager.count(MYSQL_RECORD_TABLE, filters=filters):
            raise ValueError(f"无法更新 MySQL：原表中不存在 {id_column}={key} 的记录")

        updates = {
            name: value
            for name, value in _mysql_stage_ai_updates(before_record, after_record).items()
            if name in columns
        }
        if not updates:
            logger.info(f"[MYSQL NOOP] {id_column}={key} 本次 stage 没有产生变化的 ai_ 字段")
            return 0

        manager.update(MYSQL_RECORD_TABLE, updates, filters=filters)
        logger.info(
            f"[MYSQL UPDATE] {id_column}={key} fields={','.join(sorted(updates))}"
        )
        return len(updates)
    finally:
        manager.close()


def build_model(
    model_name: str,
    args: argparse.Namespace,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = DEFAULT_AGENT_TEMPERATURE,
    thinking: bool | None = None,
    max_tokens: int | None = DEFAULT_AGENT_MAX_TOKENS,
) -> ChatOpenAI:
    resolved_base_url = args.base_url if base_url is None else base_url
    resolved_api_key = args.api_key if api_key is None else api_key
    resolved_thinking = getattr(args, "thinking", False) if thinking is None else thinking
    logger.info(
        f"model_name {model_name}, base_url {resolved_base_url or '<default>'}, "
        f"temperature {temperature}, thinking {resolved_thinking}, "
        f"max_tokens {max_tokens or '<provider-default>'}, apikey {_mask_secret(resolved_api_key)}"
    )
    return AgentModelConfig(
        model=model_name,
        base_url=resolved_base_url or "",
        api_key=resolved_api_key or "",
        temperature=temperature,
        thinking=resolved_thinking,
        max_tokens=max_tokens,
        timeout=getattr(args, "timeout", 120.0),
        max_retries=getattr(args, "max_retries", 2),
    ).build()


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
                previous_state = previous_diagnoses_by_patient[patient_key]
                working_record["__previous_diagnoses"] = {
                    field: _text(previous_state.get(field)) for field in DIAGNOSIS_FIELDS
                }
                previous_context = previous_state.get("__record_context")
                if isinstance(previous_context, Mapping):
                    working_record["__previous_record_context"] = dict(previous_context)

            try:
                process_call = agents.process(
                    working_record,
                    use_rag=item["use_rag"],
                    stage_names=item.get("stage_names"),
                    skip_record_filter=bool(item.get("skip_record_filter", False)),
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
                enriched.pop("__previous_record_context", None)
                enriched.pop("__original_diagnoses", None)
                enriched["ai_processing_errors"] = [{"stage": "record", "error": f"{type(error).__name__}: {error}"}]
                enriched["ai_processing_complete"] = False
                enriched["ai_record_identity"] = item["identity"]

            outcomes.append((item, enriched, error))
            if patient_key and enriched.get("ai_processing_filtered") is not True:
                enriched_diagnoses = _diagnosis_payload(enriched, prefer_ai=True)
                previous_diagnoses_by_patient[patient_key] = {
                    **enriched_diagnoses,
                    "__record_context": _record_history_context(enriched),
                }
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


async def process_mysql_records(args: argparse.Namespace) -> tuple[int, int, int]:
    """Process the full MySQL filter result in batches using raw source records.

    Flow:
      1. SELECT every row matching the fixed MySQL worklist filter.
      2. Match each row to medical/data/202301_online by doctor_id + order_sn.
      3. Process matched source records in --batch-size batches.
      4. UPDATE only ai_* fields actually changed by the selected stages.

    No whole-row UPDATE and no ai_processing_complete based skip is performed.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_stage_names = list(getattr(args, "stage_name", None) or [])
    stage_names = resolve_stage_names(requested_stage_names)
    logger.info(f"本次清洗的阶段 {stage_names}")

    if getattr(args, "reprocess_failures", False) or getattr(args, "reprocess_missing_histories", False):
        raise ValueError("mysql 数据源暂不支持 --reprocess-failures 或 --reprocess-missing-histories")
    if getattr(args, "save_to", "mysql") not in ("mysql", "both"):
        raise ValueError("mysql 数据源处理结果必须写回 mysql（--save-to mysql/both）")

    # 1) Read the complete fixed MySQL worklist first.
    #    SELECT * FROM mlops_annotation_medical_record
    #    WHERE doctor_id = 97 AND status IN ('unprocessed', 'problem');
    mysql_records = _load_mysql_records(args)
    logger.info(f"本次从数据库按 filter 条件读取 {len(mysql_records)} 条")
    if not mysql_records:
        return 0, 0, 0

    # Skip records that were successfully handled by an earlier run. This is
    # intentionally done before --limit, so --limit N means N new records.
    progress_path = _mysql_update_progress_path(args)
    completed_progress = _load_mysql_update_progress(progress_path)
    mysql_records, skipped_by_progress = _filter_mysql_records_by_progress(
        mysql_records, completed_progress
    )
    logger.info(
        f"[MYSQL PROGRESS] file={progress_path} skipped={skipped_by_progress} pending={len(mysql_records)}"
    )
    if not mysql_records:
        logger.info("MySQL filter 命中的记录均已存在于本地清洗进度文件，本次无需处理")
        return 0, 0, skipped_by_progress

    # Optional limit is applied after progress skipping.
    if args.limit and args.limit > 0:
        mysql_records = mysql_records[: args.limit]
        logger.info(f"应用 --limit 后待处理 {len(mysql_records)} 条新记录")

    # 2) Raw clinical fields always come from the original dataset.
    source_records = _load_original_records_for_mysql(mysql_records, DEFAULT_INPUT)

    text_model = build_model(args.model, args, base_url=args.base_url, api_key=args.api_key, temperature=0.1)
    diagnosis_model = build_model(
        args.diagnosis_model,
        args,
        base_url=args.diagnosis_base_url,
        api_key=args.diagnosis_api_key,
        max_tokens=1000,
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
        max_tokens=6000,
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

    agents = MedicalRecordAgents(
        text_model,
        vlm_model,
        DEFAULT_INPUT,
        diagnosis_model=diagnosis_model,
        image_classification_model=image_classification_model,
        reviewer_model=reviewer_model,
        enable_review=enable_review,
        filter_path=args.output_dir / FILTER_FILE_NAME,
        stage_timeout=getattr(args, "timeout", 120.0),
    )

    succeeded = 0
    failed = 0
    skipped = skipped_by_progress
    filtered = 0
    completed = 0
    batch_size = max(1, int(getattr(args, "batch_size", 8)))
    record_timeout = max(0.0, float(getattr(args, "record_timeout", 0.0)))
    progress_every = max(1, int(getattr(args, "progress_every", 10)))
    progress_scope = (
        f"doctor_id={','.join(_text(value) for value in (getattr(args, 'doctor_id', None) or [])) or 'unknown'},"
        "status=unprocessed|problem"
    )
    progress_started_at = time.monotonic()
    last_progress_completed = 0
    previous_diagnoses_by_patient: dict[str, dict[str, str]] = {}

    logger.info(
        f"[PROGRESS START] data_source=mysql doctor={progress_scope} target={len(source_records)} "
        f"batch_size={batch_size} progress_every={progress_every}"
    )

    for batch_start in range(0, len(source_records), batch_size):
        records_batch = source_records[batch_start : batch_start + batch_size]
        batch_items: list[dict[str, Any]] = []
        for offset, processing_record in enumerate(records_batch, 1):
            absolute_index = batch_start + offset
            mysql_record = processing_record.get("__mysql_current_row")
            if not isinstance(mysql_record, Mapping):
                raise ValueError("内部错误：处理记录缺少 __mysql_current_row")

            clean_record = dict(processing_record)
            clean_record.pop("__mysql_current_row", None)
            identity = _record_identity(clean_record, absolute_index)
            order_sn = _text(clean_record.get("order_sn"))
            batch_items.append(
                {
                    "record": clean_record,
                    "before_record": dict(clean_record),
                    "mysql_record": dict(mysql_record),
                    "record_id": _record_key(clean_record, absolute_index),
                    "identity": identity,
                    "order_sn": order_sn,
                    "patient_key": _patient_key(clean_record),
                    "use_rag": args.use_rag,
                    "stage_names": stage_names,
                    "skip_record_filter": False,
                }
            )

        logger.info(f"[BATCH START] records={len(batch_items)} concurrency={len(batch_items)}")
        batch_started_at = time.monotonic()
        outcomes = await _process_record_batch(
            agents,
            batch_items,
            previous_diagnoses_by_patient,
            record_timeout,
        )
        first_failure: RuntimeError | None = None

        for item, enriched, process_error in outcomes:
            record_id = item["record_id"]
            order_sn = item["order_sn"]
            enriched.pop("__previous_diagnoses", None)
            enriched.pop("__previous_record_context", None)
            enriched.pop("__original_diagnoses", None)

            if process_error is not None:
                failed += 1
                logger.error(f"[ERROR] record={record_id}: {type(process_error).__name__}: {process_error}")
                first_failure = first_failure or RuntimeError(f"病历 {record_id} 处理失败")
                continue

            if enriched.get("ai_processing_filtered") is True:
                skipped += 1
                filtered += 1
                logger.info(f"数据关键字段均为空，【FILTER】 {order_sn or record_id}")
                continue

            if enriched.get("ai_processing_errors"):
                failed += 1
                logger.error(f"[ERROR] record={record_id}: {_json_for_prompt(enriched['ai_processing_errors'])}")
                first_failure = first_failure or RuntimeError(f"病历 {record_id} 存在处理失败阶段")
                continue

            # 3/4) Never save the whole enriched record. Only changed stage ai_* fields.
            logger.info(f"本次更新病历表内容{enriched.keys()}")
            updated_field_count = _update_mysql_stage_ai_fields(
                item["mysql_record"],
                item["before_record"],
                enriched,
            )
            mysql_record = item["mysql_record"]
            progress_added = _append_mysql_update_progress(
                progress_path,
                mysql_record.get("doctor_id"),
                order_sn,
            )
            logger.info(
                f"[MYSQL PROGRESS SAVE] doctor_id={_text(mysql_record.get('doctor_id'))} "
                f"order_sn={order_sn} updated_fields={updated_field_count} added={progress_added}"
            )
            succeeded += 1

        completed += len(outcomes)
        logger.info(
            f"[BATCH END] records={len(batch_items)} elapsed={time.monotonic() - batch_started_at:.1f}s"
        )
        if (
            completed >= len(source_records)
            or completed - last_progress_completed >= progress_every
        ):
            logger.warning(
                _progress_log_message(
                    completed=completed,
                    total=len(source_records),
                    succeeded=succeeded,
                    failed=failed,
                    filtered=filtered,
                    skipped_existing=0,
                    elapsed=time.monotonic() - progress_started_at,
                    scope=progress_scope,
                )
            )
            last_progress_completed = completed

        if getattr(args, "interval", 0) > 0:
            await asyncio.sleep(args.interval)
        if first_failure is not None and args.fail_fast:
            raise first_failure

    return succeeded, failed, skipped


async def process_records(args: argparse.Namespace) -> tuple[int, int, int]:
    if getattr(args, "data_source", "local") == "mysql":
        return await process_mysql_records(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_stage_names = list(getattr(args, "stage_name", None) or [])
    stage_names = resolve_stage_names(requested_stage_names)
    stage_output_names = stage_names if requested_stage_names else None
    doctor_ids = set(args.doctor_id or [])
    reprocess_selected = bool(
        getattr(args, "reprocess_failures", False) or getattr(args, "reprocess_missing_histories", False)
    )
    source_records = (
        select_reprocess_records(
            args.output_dir,
            doctor_ids,
            include_failures=bool(getattr(args, "reprocess_failures", False)),
            include_missing_histories=bool(getattr(args, "reprocess_missing_histories", False)),
            limit=args.limit,
        )
        if reprocess_selected
        else None
    )


    text_model = build_model(
        args.model,
        args,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    diagnosis_model = build_model(
        args.diagnosis_model,
        args,
        base_url=args.diagnosis_base_url,
        api_key=args.diagnosis_api_key,
        max_tokens=1000,
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
        max_tokens=6000,
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
    if args.reprocess or reprocess_selected:
        processed_ids, processed_order_sns = set(), set()
    else:
        processed_ids, processed_order_sns = load_processed_record_keys(
            args.output_dir,
            _stage_output_file_name(stage_output_names),
        )
    planned_total = (
        len(source_records)
        if source_records is not None
        else count_planned_records(
            args.input,
            doctor_ids,
            args.limit,
            processed_ids,
            processed_order_sns,
        )
    )
    succeeded = 0
    failed = 0
    skipped = 0
    filtered = 0
    skipped_existing = 0
    completed = 0
    attempted = 0
    attempted_by_doctor: Counter[str] = Counter()
    output_files: dict[Path, TextIO] = {}
    failure_files: dict[Path, TextIO] = {}
    replacement_records: dict[Path, list[dict[str, Any]]] = {}
    successful_reprocess_order_sns: dict[Path, set[str]] = {}
    previous_diagnoses_by_patient: dict[str, dict[str, str]] = {}
    batch_size = max(1, int(getattr(args, "batch_size", 8)))
    record_timeout = max(0.0, float(getattr(args, "record_timeout", 0.0)))
    progress_every = max(1, int(getattr(args, "progress_every", 10)))
    progress_scope = ",".join(sorted(doctor_ids)) or "all"
    progress_started_at = time.monotonic()
    last_progress_completed = 0
    pending_batch: list[dict[str, Any]] = []
    if planned_total:
        logger.info(
            f"[PROGRESS START] doctor={progress_scope} target={planned_total} "
            f"batch_size={batch_size} progress_every={progress_every}"
        )

    with ExitStack() as stack:

        async def flush_batch() -> None:
            nonlocal completed, failed, filtered, last_progress_completed, skipped, succeeded
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
                enriched.pop("__previous_record_context", None)
                enriched.pop("__original_diagnoses", None)

                if process_error is not None:
                    failure_path = _doctor_failure_path(args.output_dir, enriched, stage_output_names)
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
                    filtered += 1
                    processed_ids.add(identity)
                    if order_sn:
                        processed_order_sns.add(order_sn)
                    logger.info(f"数据关键字段均为空，【FILTER】 {order_sn or record_id}")
                    continue

                if enriched.get("ai_processing_errors"):
                    failed += 1
                    enriched["ai_processing_complete"] = False
                    failure_path = _doctor_failure_path(args.output_dir, enriched, stage_output_names)
                    if failure_path not in failure_files:
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_files[failure_path] = stack.enter_context(failure_path.open("a", encoding="utf-8"))
                    _write_jsonl(failure_files[failure_path], enriched)
                    logger.error(f"[ERROR] record={record_id}: {_json_for_prompt(enriched['ai_processing_errors'])}")
                    first_failure = first_failure or RuntimeError(f"病历 {record_id} 存在处理失败阶段")
                    continue

                succeeded += 1
                save_to = getattr(args, "save_to", "local")
                enriched["ai_processing_complete"] = True
                enriched["ai_processing_version"] = date.today().isoformat()
                if save_to in ("local", "both") and not reprocess_selected:
                    output_path = _doctor_output_path(args.output_dir, enriched, stage_output_names)
                    if output_path not in output_files:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_files[output_path] = stack.enter_context(output_path.open("a", encoding="utf-8"))
                    _write_jsonl(output_files[output_path], enriched)
                if save_to in ("mysql", "both"):
                    _save_mysql_record(args, enriched)
                if reprocess_selected:
                    output_path = _doctor_output_path(args.output_dir, enriched, stage_output_names)
                    replacement_records.setdefault(output_path, []).append(enriched)
                    if order_sn:
                        failure_path = _doctor_failure_path(args.output_dir, enriched, stage_output_names)
                        successful_reprocess_order_sns.setdefault(failure_path, set()).add(order_sn)
                processed_ids.add(identity)
                if order_sn:
                    processed_order_sns.add(order_sn)

            completed += len(outcomes)
            logger.info(f"[BATCH END] records={len(batch)} elapsed={time.monotonic() - batch_started_at:.1f}s")
            if (
                completed == len(outcomes)
                or completed >= planned_total
                or completed - last_progress_completed >= progress_every
            ):
                logger.warning(
                    _progress_log_message(
                        completed=completed,
                        total=planned_total,
                        succeeded=succeeded,
                        failed=failed,
                        filtered=filtered,
                        skipped_existing=skipped_existing,
                        elapsed=time.monotonic() - progress_started_at,
                        scope=progress_scope,
                    )
                )
                last_progress_completed = completed
            if getattr(args, "interval", 0) > 0:
                await asyncio.sleep(args.interval)
            if first_failure is not None and args.fail_fast:
                raise first_failure

        records_iterable = source_records if source_records is not None else iter_records(args.input)
        for index, record in enumerate(records_iterable, 1):
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
                    previous_diagnoses_by_patient[patient_key] = {
                        **skipped_diagnoses,
                        "__record_context": _record_history_context(record),
                    }
                skipped += 1
                skipped_existing += 1
                logger.info(f"数据已经处理过，【SKIP】 {order_sn}")
                continue
            if not doctor_ids and args.limit and attempted >= args.limit:
                break
            attempted += 1
            if doctor_id:
                attempted_by_doctor[doctor_id] += 1
            record_id = _record_key(record, index)
            pending_batch.append(
                {
                    "record": record,
                    "record_id": record_id,
                    "identity": identity,
                    "order_sn": order_sn,
                    "patient_key": patient_key,
                    "use_rag": args.use_rag,
                    "stage_names": stage_names,
                }
            )
            if len(pending_batch) >= batch_size:
                await flush_batch()
        await flush_batch()
        for path, records in replacement_records.items():
            _replace_jsonl_records_by_order_sn(path, records)
        for path, order_sns in successful_reprocess_order_sns.items():
            _remove_jsonl_records_by_order_sn(path, order_sns)
        if planned_total and completed != last_progress_completed:
            logger.warning(
                _progress_log_message(
                    completed=completed,
                    total=planned_total,
                    succeeded=succeeded,
                    failed=failed,
                    filtered=filtered,
                    skipped_existing=skipped_existing,
                    elapsed=time.monotonic() - progress_started_at,
                    scope=progress_scope,
                )
            )
    return succeeded, failed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 LangChain 的多医生病历 AI 清洗与结构化脚本")
    parser.add_argument(
        "--data-source",
        choices=("local", "mysql"),
        default="local",
        help="数据来源：local 从本地文件读取；mysql 从 MySQL 表读取",
    )
    parser.add_argument(
        "--save-to",
        choices=("local", "mysql", "both"),
        default=None,
        help="处理结果写入位置：local 仅写入本地文件；mysql 仅写入 MySQL 表；both 同时写入，默认跟随 --data-source",
    )
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
    parser.add_argument(
        "--mysql-progress-file",
        type=Path,
        default=None,
        help=(
            "MySQL 清洗成功 order_sn 的共享进度 JSON；"
            f"默认 <output-dir>/{MYSQL_UPDATE_PROGRESS_FILE_NAME}，支持多进程并发安全写入"
        ),
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
    parser.add_argument(
        "--doctor-id",
        action="extend",
        nargs="+",
        default=[],
        help="仅处理指定 doctor_id；MySQL 模式用于构造 doctor_id 过滤条件",
    )
    parser.add_argument(
        "--stage-name",
        action="extend",
        nargs="+",
        choices=STAGE_NAMES,
        default=[],
        help=(
            "仅运行指定清洗阶段，可一次传入一个或多个名称，也可重复使用参数；"
            "不传时运行全部阶段。指定后本地输出文件名会包含 stage 名称"
        ),
    )
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="每完成多少条病历输出一次后台进度日志，默认 10",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单次模型请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="模型请求重试次数")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="开启或关闭 Agent 思考模式，默认关闭；也可使用 --no-thinking",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="thinking",
        action="store_false",
        help="兼容旧参数；等同于 --no-thinking",
    )
    parser.add_argument("--reprocess", action="store_true", help="忽略已有成功记录并重新处理")
    parser.add_argument(
        "--reprocess-failures",
        action="store_true",
        help="重新处理输出目录中的失败 JSONL 记录",
    )
    parser.add_argument(
        "--reprocess-missing-histories",
        action="store_true",
        help=(
            "重新处理成功结果中 doc_ass_stu_appeal/ai_patient_appeal 或 "
            "new_medical_history/ai_new_medical_history 同时为空的记录"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true", help="任一病历失败时立即退出")
    parser.add_argument("--use-rag", type=bool, default=False, help="是否使用病因提取中医知识库")
    parser.add_argument(
        "--export-mode",
        action="store_true",
        help="仅导出模式：将 MySQL 数据导出到本地 JSONL，不执行处理",
    )
    parser.add_argument(
        "--export-file-name",
        default="records.jsonl",
        help="导出的 JSONL 文件名，默认 records.jsonl。仅与 --export-mode 一起使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_source == "local" and not args.input.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    # save_to 默认为与 data_source 一致
    if args.save_to is None:
        args.save_to = args.data_source

    # 导出模式：直接从 MySQL 导出到本地 JSONL
    if getattr(args, "export_mode", False):
        if args.data_source != "mysql":
            raise ValueError("--export-mode 仅在 --data-source mysql 时有效")
        count = export_mysql_records_to_local(args)
        logger.info(f"导出完成，共 {count} 条记录到 {args.output_dir}")
        return

    succeeded, failed, skipped = asyncio.run(process_records(args))
    output_location = (
        str(args.output_dir) if args.save_to in ("local", "both") else f"MySQL 表 {MYSQL_RECORD_TABLE}"
    )
    logger.info(
        f"处理完成: 成功 {succeeded} 条，失败 {failed} 条，跳过已处理或过滤 {skipped} 条，输出位置 {output_location}"
    )


if __name__ == "__main__":
    main()
