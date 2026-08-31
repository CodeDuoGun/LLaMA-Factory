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

"""病历结构化结果的轻量展示格式化函数."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel

from medical.schema.clear_record_basemodel import (
    TONGUE_FACE_RESULT_FIELD_CN_MAPPING,
    InspectionResult,
    TongueFaceResult,
)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


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
        "tongue.tongue_coat.root": {"有根苔": "有根", "无根苔": "无根"},
        "tongue.tongue_coat.distribution": {"整体": "全舌"},
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
    sections = [f"{labels[section]}：{description}。" for section, description in descriptions.items() if description]
    return "\n".join(sections) or "未见有效舌面患处信息。"


def _inspection_field(report: Mapping[str, Any], field: str, alias: str) -> Any:
    return report.get(field) if field in report else report.get(alias)


def format_inspection_result_text(result: InspectionResult | Mapping[str, Any] | None) -> str:
    """按文档类型保留异常检查事实或非空的出院信息."""
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
        report_time = (
            _inspection_field(report, "relative_to_visit_time", "相对就诊时间")
            or _inspection_field(report, "report_time", "报告时间")
            or _inspection_field(report, "report_date", "报告日期")
        )
        report_name = _format_tongue_face_value(_inspection_field(report, "report_name", "报告名称")) or "未识别"
        image_type = _format_tongue_face_value(_inspection_field(report, "image_type", "图片类别"))
        abnormal_indicators = _format_tongue_face_value(
            _inspection_field(report, "abnormal_indicators", "异常指标")
        )
        abnormal_results = _format_tongue_face_value(
            _inspection_field(report, "abnormal_results", "异常结果")
        )
        # 兼容新增“异常结果”字段前已落库的影像/检查解析结果。
        if not abnormal_results and "abnormal_results" not in report and "异常结果" not in report:
            if image_type in {"检查报告", "CT诊断报告", "影像检查报告", "病理报告", "门诊病历"}:
                abnormal_results = _format_tongue_face_value(
                    _inspection_field(report, "conclusion", "报告结论")
                )
        discharge_diagnosis = _format_tongue_face_value(
            _inspection_field(report, "discharge_diagnosis", "出院诊断")
        )
        discharge_condition = _format_tongue_face_value(
            _inspection_field(report, "discharge_condition", "出院情况")
        )

        details = []
        if abnormal_indicators:
            details.append(f"异常指标：{abnormal_indicators}")
        if abnormal_results:
            details.append(f"异常结果：{abnormal_results}")
        if discharge_diagnosis:
            details.append(f"出院诊断：{discharge_diagnosis}")
        if discharge_condition:
            details.append(f"出院情况：{discharge_condition}")
        if not details:
            continue

        valid_index += 1
        prefix = "有效医学文档" if image_type in {"住院/出院报告", "门诊病历"} else "有效检查报告"
        descriptions.append(
            f"{prefix}{valid_index}："
            f"报告时间：{_format_tongue_face_value(report_time) or '未识别'}；"
            f"报告名称：{report_name}；"
            f"{'；'.join(details)}。"
        )
    return "\n".join(descriptions) or "未见有效检查报告信息。"
