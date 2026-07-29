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
import base64
import json
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO
from pydantic import BaseModel, ValidationError

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from medical.utils.log import logger
from dotenv import load_dotenv
from medical.schema.clear_record_basemodel import (
    ClinicalExtractionResult,
    CurrentVisitHistoryResult,
    DiagnosisResult,
    HistoryCleaningResult,
    InspectionResult,
    TONGUE_FACE_RESULT_FIELD_CN_MAPPING,
    TongueFaceResult,
)
load_dotenv()

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical.analysis.engine import normalize_diagnosis_value, normalize_syndrome_value  # noqa: E402

DEFAULT_INPUT = Path("medical/data/202301_online")
DEFAULT_OUTPUT_DIR = Path("medical/processed_data")
DOCTOR_RECORD_FILE_NAME = "medical_records_ai.jsonl"
FAILURE_FILE_NAME = "medical_record_agent_failures.jsonl"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DIAGNOSIS_FIELDS = ("diagnosis_illness", "diagnosis_disease", "diagnosis_sickness")
HISTORY_FIELDS = (
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "family_history",
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


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    return _text(record.get("start_time") or record.get("created_at"))


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


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def _tool_response_format(schema: type[BaseModel]) -> ToolStrategy[Any]:
    """统一通过工具调用生成并校验结构化响应."""
    return ToolStrategy(schema=schema, handle_errors=True)


def _invoke_structured_agent(agent: Any, message: HumanMessage, schema: type[BaseModel]) -> BaseModel:
    last_error: StructuredOutputError | None = None
    logger.debug(f"humanmessage: {message}")
    for attempt in range(2):
        messages = [message]
        if attempt:
            messages.append(
                HumanMessage(
                    content=(
                        f"上一次 {schema.__name__} 结构化参数校验失败。请重新生成一次，"
                        "严格遵守字段类型；只调用一次结构化输出工具，不得重复调用。"
                    )
                )
            )
        try:
            response = agent.invoke({"messages": messages})
        except StructuredOutputError as exc:
            last_error = exc
            continue
        structured_response = response.get("structured_response")
        if isinstance(structured_response, schema):
            return structured_response
        if isinstance(structured_response, dict):
            return schema.model_validate(structured_response)
        raise RuntimeError(f"Agent 未返回 {schema.__name__} structured_response")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Agent 未返回 {schema.__name__} structured_response")


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


def _invoke_json_object_model(
    model: Any,
    system_prompt: str,
    message: HumanMessage,
    schema: type[BaseModel],
) -> BaseModel:
    """调用仅支持 json_object 的 OpenAI 兼容模型，并在客户端完成模型校验."""
    last_error: json.JSONDecodeError | ValidationError | None = None
    for attempt in range(2):
        messages = [SystemMessage(content=system_prompt), message]
        if attempt:
            messages.append(
                HumanMessage(
                    content=(
                        f"上一次返回内容无法通过 {schema.__name__} 校验。"
                        "请重新返回一个完整、可解析的 json 对象，所有要求字段均不得省略，"
                        "不要输出 Markdown、注释或额外说明。"
                    )
                )
            )
        response = model.invoke(messages)
        raw_text = _response_text(response)
        if raw_text.startswith("```") and raw_text.endswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        try:
            return schema.model_validate(json.loads(raw_text))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"模型未返回可解析的 {schema.__name__} json 对象")


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


class MedicalRecordAgents:
    """病历诊断、五史清洗、视觉分析和病机抽取 Agent 集合."""

    def __init__(
        self,
        text_model: ChatOpenAI,
        vlm_model: ChatOpenAI,
        input_dir: Path,
        diagnosis_model: ChatOpenAI | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.diagnosis_agent = self._build_diagnosis_agent(diagnosis_model or text_model)
        self.history_agent = self._build_history_agent(text_model)
        self.extract_agent = self._build_extract_agent(text_model)
        self.current_visit_history_agent = create_agent(
            model=text_model,
            tools=[],
            system_prompt=(
                "你是临床病历现病史整理助手。输入可能包含同一患者按日期累积的多次就诊记录。"
                "必须以本次就诊时间为锚点，仅提取与本次复诊对应的现病史，不得混入既往就诊段落。"
                "若存在括号日期、初诊/复诊序号等标记，选择与本次时间相同或最近且不晚于本次时间的段落。"
                "整理为规范现病史：保留本次症状、变化、持续时间、相关治疗反应、重要阴性表现及"
                "与本次相关的检查；纠正明确错别字，删除乱码、重复符号和无意义文本。"
                "检查报告中的异常方向符号必须转换为中文文本描述，例如 ↑、⬆️ 写作“升高”，↓、⬇️ 写作“下降”。"
                "不得新增原文没有的症状、诊断、时间或检查结果；无法确定本次段落时返回空字符串。"
            ),
            response_format=_tool_response_format(CurrentVisitHistoryResult),
            name="current_visit_history_agent",
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
    def normalize_diagnoses(record: dict[str, Any]) -> None:
        record["diagnosis_illness"] = normalize_diagnosis_value(record.get("diagnosis_illness"))
        record["diagnosis_sickness"] = normalize_diagnosis_value(record.get("diagnosis_sickness"))
        # 此时尚未完成诊断类型纠错，不能按字段位置直接追加“证”，否则会把错填在
        # diagnosis_disease 中的西医诊断（如“高血压”）误改为“高血压证”。
        record["diagnosis_disease"] = normalize_diagnosis_value(record.get("diagnosis_disease"))

    def complete_diagnoses(self, record: dict[str, Any]) -> None:
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

        result = _invoke_json_object_model(
            self.diagnosis_agent,
            _load_prompt("fill_diagnosis_prompt.txt"),
            HumanMessage(
                content=(
                    f"本次就诊患者主诉：{chief_complaint}\n"
                    f"本次就诊现病史：{present_history}\n"
                    f"三个原始诊断结果：{_json_for_prompt(source_diagnoses)}\n"
                    f"原始诊断为空的字段："
                    f"{_json_for_prompt([field for field, value in source_diagnoses.items() if not value])}\n"
                    f"上一次同患者诊断结果（如为空表示无上一诊断可参考）：{_json_for_prompt(previous_diagnoses)}\n"
                    "要求：原始诊断字段非空时必须原样保留，不得改写；"
                    "仅对原始诊断为空的字段补全。补全时先从其他原始诊断字段中提取对应类别结果，"
                    "仍为空再结合本次主诉、本次现病史和上一次诊断结果补全。最终所有字段值均不能为空。"

                )
            ),
            DiagnosisResult,
        )
        values = _model_dump(result)
        normalizers = {
            "diagnosis_illness": normalize_diagnosis_value,
            "diagnosis_sickness": normalize_diagnosis_value,
            "diagnosis_disease": normalize_syndrome_value,
        }
        for field in DIAGNOSIS_FIELDS:
            original_value = _clean_generated_text(source_diagnoses.get(field))
            if original_value:
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

    def extract_current_visit_history(self, record: dict[str, Any], enabled: bool) -> None:
        record["ai_new_medical_history"] = ""
        history = _present_history(record)
        if not enabled or not history:
            return
        result = _invoke_structured_agent(
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
        )
        record["ai_new_medical_history"] = _replace_inspection_report_symbols(
            _clean_generated_text(result.new_medical_history)
        )

    def clean_histories(self, record: dict[str, Any]) -> None:
        histories = _history_payload(record)
        result = _invoke_structured_agent(
            self.history_agent,
            HumanMessage(
                content=(
                    f"患者主诉（doc_ass_stu_appeal）：{_chief_complaint(record)}\n"
                    f"本次现病史：{_latest_present_history(record)}\n"
                    f"原始五史：{_json_for_prompt(histories)}\n"
                    f"检查报告图片分析：{_json_for_prompt(record.get('ai_inspection_report_img', {}))}\n"
                    f"舌面患处图片分析：{_json_for_prompt(record.get('ai_tongue_face_img', {}))}"
                )
            ),
            HistoryCleaningResult,
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

    def analyze_inspection_images(self, record: dict[str, Any]) -> None:
        images = _as_images(record.get("inspection_report_img") or record.get("admin_report_img"))
        if not images:
            record["ai_inspection_report_img"] = _replace_inspection_report_symbols(
                _model_dump(InspectionResult(), by_alias=True)
            )
            return
        prompt = "逐张分类并解析图片，提取有效检查报告中可用于核对或补充本次现病史的客观事实。"
        result = _invoke_structured_agent(
            self.inspection_agent,
            _multimodal_message(prompt, images, self.input_dir),
            InspectionResult,
        )
        record["ai_inspection_report_img"] = _replace_inspection_report_symbols(
            _model_dump(result, by_alias=True)
        )

    def analyze_tongue_face_images(self, record: dict[str, Any]) -> None:
        images = _as_images(record.get("tongue_face_img") or record.get("admin_face_img"))
        if not images:
            record["ai_tongue_face_img"] = _model_dump(TongueFaceResult())
            return
        prompt = (
            "按结构化模型分析图片中的可见舌象、面象和患处。分别记录舌体、舌苔、舌下络脉、"
            "面神、面色、光泽、面部形态、五官局部和皮损。只描述图像可见事实，不作疾病、证候、"
            "病因或病机推断；无法辨认的字符串字段返回空字符串，多选字段返回空数组。"
        )
        result = _invoke_structured_agent(
            self.tongue_face_agent,
            _multimodal_message(prompt, images, self.input_dir),
            TongueFaceResult,
        )
        record["ai_tongue_face_img"] = _model_dump(result)

    def extract_clinical_fields(self, record: dict[str, Any], use_rag:bool=False) -> None:
        diagnoses = {
            field: _text(record.get(f"ai_{field}")) or _text(record.get(field))
            for field in DIAGNOSIS_FIELDS
        }
        histories = {
            field: _text(record.get(f"ai_{field}")) if f"ai_{field}" in record else _text(record.get(field))
            for field in HISTORY_FIELDS
        }
        illness_knowledge = self.build_illness_knowledge(diagnoses) if use_rag else "无"
        result = _invoke_structured_agent(
            self.extract_agent,
            HumanMessage(
                content=(
                    f"最新主诉：{_latest_chief_complaint(record)}\n"
                    f"本次现病史：{_latest_present_history(record)}\n"
                    f"清洗后五史：{_json_for_prompt(histories)}\n"
                    f"诊断上下文：{_json_for_prompt(diagnoses)}\n"
                    f"检查报告图片分析：{_json_for_prompt(record.get('ai_inspection_report_img', {}))}\n"
                    f"舌面患处图片分析：{_json_for_prompt(record.get('ai_tongue_face_img', {}))}\n"
                    f"疾病的知识库信息 {illness_knowledge}\n"
                    f"严格根据知识库知识，提取患者在中医层面的病因、病机、病期、病位、病程、临床症状。若知识库无知识，根据患者病史信息完成上述内容提取。必须返回合法的 JSON 对象，不要输出 Markdown，不要输出额外解释"
                )
            ),
            ClinicalExtractionResult,
        )
        values = _model_dump(result)
        for field, value in values.items():
            record[f"ai_{field}"] = _clean_generated_value(value)

    def process(self, record: dict[str, Any], *, extract_current_history: bool = False, use_rag:bool=False) -> dict[str, Any]:
        output = dict(record)
        output["__original_diagnoses"] = _diagnosis_payload(output)
        errors = []

        # step1：先做不依赖诊断类型的基础归一化，不额外存储 ai_normalized_*。
        try:
            self.normalize_diagnoses(output)
        except Exception as exc:
            errors.append({"stage": "diagnosis_normalization", "error": f"{type(exc).__name__}: {exc}"})

        # step2：两类互不依赖的图片分析并行执行。
        image_stages = {
            "inspection_vlm": self.analyze_inspection_images,
            "tongue_face_vlm": self.analyze_tongue_face_images,
        }
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="medical-vlm") as executor:
            futures = {name: executor.submit(stage, output) for name, stage in image_stages.items()}
            for stage_name, future in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"stage": stage_name, "error": f"{type(exc).__name__}: {exc}"})

        # step3：从累积现病史中抽取本次复诊内容。
        try:
            self.extract_current_visit_history(output, extract_current_history)
        except Exception as exc:
            errors.append({"stage": "current_visit_history", "error": f"{type(exc).__name__}: {exc}"})

        # step4：结合图片结果校验主诉/现病史一致性，并清洗五史。
        try:
            self.clean_histories(output)
        except Exception as exc:
            errors.append({"stage": "clinical_cleaning", "error": f"{type(exc).__name__}: {exc}"})

        # step5：用最新病历补全诊断，再提取知识库字段。
        try:
            self.complete_diagnoses(output)
        except Exception as exc:
            errors.append({"stage": "diagnosis_completion", "error": f"{type(exc).__name__}: {exc}"})

        try:
            self.extract_clinical_fields(output, use_rag)
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


def count_patient_visits(path: Path, doctor_ids: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in iter_records(path):
        if not _doctor_matches(record, doctor_ids):
            continue
        patient_key = _patient_key(record)
        if patient_key:
            counts[patient_key] += 1
    return counts


def load_processed_record_ids(output_dir: Path) -> set[str]:
    processed = set()
    if not output_dir.is_dir():
        return processed
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
                if isinstance(record, dict) and record.get("ai_processing_complete") is True:
                    processed.add(_text(record.get("ai_record_identity")) or _record_identity(record))
    return processed


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
    normalized_base_url = _text(resolved_base_url).lower()
    model_id = model_name.lower()
    is_dashscope_thinking_model = (
        "dashscope" in normalized_base_url
        or "aliyuncs.com" in normalized_base_url
        or model_id.startswith(("qwen", "qwq"))
    )
    if getattr(args, "disable_thinking", False) or is_dashscope_thinking_model:
        # 部分 OpenAI 兼容接口默认开启 thinking mode；结构化输出场景下显式关闭。
        if "deepseek" in model_id:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["extra_body"] = {"enable_thinking": False}
    logger.debug(
        f"model={model_name} base_url={resolved_base_url} extra_body={kwargs.get('extra_body', {})}",
    )
    return ChatOpenAI(**kwargs)


def process_records(args: argparse.Namespace) -> tuple[int, int, int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doctor_ids = set(args.doctor_id or [])

    text_model = build_model(args.model, args)
    diagnosis_model = build_model(
        args.diagnosis_model,
        args,
        base_url=args.diagnosis_base_url,
        api_key=args.diagnosis_api_key,
    )
    vlm_model = build_model(
        args.vlm_model or args.model,
        args,
        base_url=getattr(args, "vlm_base_url", "") or args.base_url,
        api_key=getattr(args, "vlm_api_key", "") or args.api_key,
    )
    input_dir = args.input if args.input.is_dir() else args.input.parent
    agents = MedicalRecordAgents(text_model, vlm_model, input_dir, diagnosis_model=diagnosis_model)
    patient_visit_counts = count_patient_visits(args.input, doctor_ids)
    processed_ids = set() if args.reprocess else load_processed_record_ids(args.output_dir)
    succeeded = 0
    failed = 0
    skipped = 0
    attempted = 0
    output_files: dict[Path, TextIO] = {}
    failure_files: dict[Path, TextIO] = {}
    previous_diagnoses_by_patient: dict[str, dict[str, str]] = {}

    with ExitStack() as stack:
        for index, record in enumerate(iter_records(args.input), 1):
            if not _doctor_matches(record, doctor_ids):
                continue
            identity = _record_identity(record, index)
            patient_key = _patient_key(record)
            if identity in processed_ids:
                skipped_diagnoses = _diagnosis_payload(record, prefer_ai=True)
                if patient_key and _has_any_diagnosis(skipped_diagnoses):
                    previous_diagnoses_by_patient[patient_key] = skipped_diagnoses
                skipped += 1
                continue
            if args.limit and attempted >= args.limit:
                break
            attempted += 1
            record_id = _record_key(record, index)
            extract_current_history = (
                _text(record.get("is_first")) == "复诊"
                and bool(patient_key)
                and patient_visit_counts[patient_key] > 1
            )
            working_record = dict(record)
            if patient_key and patient_key in previous_diagnoses_by_patient:
                working_record["__previous_diagnoses"] = previous_diagnoses_by_patient[patient_key]
            try:
                enriched = agents.process(working_record, extract_current_history=extract_current_history, use_rag=args.use_rag)
            except Exception as exc:  # 单条失败不能中断其他医生/病历
                enriched = dict(working_record)
                enriched.pop("__previous_diagnoses", None)
                enriched.pop("__original_diagnoses", None)
                enriched["ai_processing_errors"] = [
                    {"stage": "record", "error": f"{type(exc).__name__}: {exc}"}
                ]
                enriched["ai_processing_complete"] = False
                enriched["ai_record_identity"] = identity
                failure_path = _doctor_failure_path(args.output_dir, enriched)
                if failure_path not in failure_files:
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_files[failure_path] = stack.enter_context(failure_path.open("a", encoding="utf-8"))
                _write_jsonl(failure_files[failure_path], enriched)
                failed += 1
                logger.error(f"[ERROR] record={record_id}: {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    raise
            else:
                enriched["ai_record_identity"] = identity
                enriched.pop("__previous_diagnoses", None)
                enriched.pop("__original_diagnoses", None)
                if enriched.get("ai_processing_errors"):
                    failed += 1
                    enriched["ai_processing_complete"] = False
                    failure_path = _doctor_failure_path(args.output_dir, enriched)
                    if failure_path not in failure_files:
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_files[failure_path] = stack.enter_context(failure_path.open("a", encoding="utf-8"))
                    _write_jsonl(failure_files[failure_path], enriched)
                    logger.error(f"[ERROR] record={record_id}: {_json_for_prompt(enriched['ai_processing_errors'])}")
                    if args.fail_fast:
                        raise RuntimeError(f"病历 {record_id} 存在处理失败阶段")
                else:
                    succeeded += 1
                    enriched["ai_processing_complete"] = True
                    enriched["ai_processing_version"] = "2026-07-28"
                    output_path = _doctor_output_path(args.output_dir, enriched)
                    if output_path not in output_files:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_files[output_path] = stack.enter_context(output_path.open("a", encoding="utf-8"))
                    _write_jsonl(output_files[output_path], enriched)
                    processed_ids.add(identity)
            enriched_diagnoses = _diagnosis_payload(enriched, prefer_ai=True)
            if patient_key and _has_any_diagnosis(enriched_diagnoses):
                previous_diagnoses_by_patient[patient_key] = enriched_diagnoses
            if args.interval > 0:
                time.sleep(args.interval)
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
    parser.add_argument("--model", default=os.getenv("MEDICAL_AGENT_MODEL", "gpt-4.1-mini"), help="文本模型")
    parser.add_argument(
        "--diagnosis-model",
        default=os.getenv("BAICHUAN_TEXT_MODEL", "Baichuan4-Turbo"),
        help="诊断补全模型，默认使用 Baichuan",
    )
    parser.add_argument(
        "--diagnosis-base-url",
        default=os.getenv("BAICHUAN_BASE_URL", "https://api.baichuan-ai.com/v1/"),
        help="诊断补全模型的 OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--diagnosis-api-key",
        default=os.getenv("BAICHUAN_API_KEY", ""),
        help="诊断补全模型 API Key",
    )
    parser.add_argument("--vlm-model", default=os.getenv("MEDICAL_AGENT_VLM_MODEL", ""), help="视觉模型，默认同文本模型")
    parser.add_argument(
        "--base-url",
        "--text-base-url",
        dest="base_url",
        default=os.getenv("DASHSCOPE_BASE_URL", os.getenv("DASHSCOPE_BASE_URL", "")),
        help="文本模型 OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--api-key",
        "--text-api-key",
        dest="api_key",
        default=os.getenv("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
        help="文本模型 API Key",
    )
    parser.add_argument(
        "--vlm-base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", ""),
        help="视觉模型 OpenAI 兼容接口地址；默认使用文本模型接口地址",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="视觉模型 API Key；默认使用文本模型 API Key",
    )
    parser.add_argument("--doctor-id", action="append", help="仅处理指定 doctor_id；可重复传入，默认所有医生")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条，0 表示不限制")
    parser.add_argument("--interval", type=float, default=0.0, help="每条病历后的等待秒数")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次模型请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="模型请求重试次数")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="关闭模型 thinking mode；DashScope/Qwen 会自动关闭，其他兼容接口可显式启用此参数",
    )
    parser.add_argument("--reprocess", action="store_true", help="忽略已有成功记录并重新处理")
    parser.add_argument("--fail-fast", action="store_true", help="任一病历失败时立即退出")
    parser.add_argument("--use-rag", type=bool, default=False, help="是否使用病因提取中医知识库")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    succeeded, failed, skipped = process_records(args)
    logger.info(
        f"处理完成: 成功 {succeeded} 条，失败 {failed} 条，跳过已处理 {skipped} 条，输出目录 {args.output_dir}"
    )


if __name__ == "__main__":
    main()
