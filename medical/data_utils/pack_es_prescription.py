#!/usr/bin/env python3
"""Pack Wu Weiping prescription records into ES-ready JSONL.

Default behavior does not call an LLM. It keeps an empty
`clinical_symptoms`/`clinical_symptoms_text` field so symptoms can be filled
later, or populated in one pass with `--extract-clinical-symptoms`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from medical.config import config
from medical.utils.log import logger


DEFAULT_INPUT = Path("medical/data/wuweiping_record_20260525.json")
DEFAULT_OUTPUT = Path(f"medical/processed_data/wuweiping_es_prescription_{datetime.now().strftime('%Y%m%d')}.jsonl")
DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")

KEEP_FIELDS = [
    "id",
    "order_sn",
    "is_first",
    "inquiry_method",
    "user_info_id",
    "patient_id",
    "patient_name",
    "patient_sex",
    "patient_age",
    "doc_ass_stu_appeal",
    "new_medical_history",
    "is_old_medical_history",
    "old_medical_history",
    "is_allergic_history",
    "allergic_history",
    "is_personal_history",
    "personal_history",
    "is_special",
    "special_history",
    "is_child_history",
    "birth_detail",
    "is_marriage_history",
    "is_family_history",
    "family_history",
    "inspection_report_img",
    "tongue_face_img",
    "admin_report_img",
    "admin_face_img",
    "admin_face_describe",
    "diagnosis_disease",
    "diagnosis_sickness",
]

CLINICAL_SOURCE_FIELDS = [
    "is_first",
    "patient_sex",
    "patient_age",
    "doc_ass_stu_appeal",
    "new_medical_history",
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "special_history",
    "family_history",
    "admin_face_describe",
]

INSPECTION_IMAGE_FIELDS = ["inspection_report_img", "admin_report_img"]
TONGUE_FACE_IMAGE_FIELDS = ["tongue_face_img", "admin_face_img"]
INTERNAL_USAGE_TYPE = "内服"
EXTERNAL_USAGE_TYPES = {"外用", "外服"}


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported input JSON shape: {path}")


def load_templates(path: Path = DEFAULT_TEMPLATE_FILE) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("templates") or data.get("data") or []
    raise ValueError(f"Unsupported template JSON shape: {path}")


def _drug_name(drug: dict[str, Any]) -> str:
    return str(drug.get("drug_name") or drug.get("show_name") or drug.get("sub_drug_name") or "").strip()


def _drug_text(drug: dict[str, Any]) -> str:
    name = _drug_name(drug)
    if not name:
        return ""
    dose = drug.get("dose") or drug.get("single_dose") or drug.get("quantity") or drug.get("drug_num") or ""
    unit = drug.get("unit_name") or drug.get("unit") or ""
    return f"{name}{dose}{unit}" if dose else name


def format_prescription(prescription: dict[str, Any]) -> str:
    usage_type = str(prescription.get("usage_type") or "").strip()
    prescription_items = prescription.get("prescription_items") or {}
    drug_list = prescription_items.get("drugList") or prescription_items.get("drug_list") or []
    drugs = [_drug_text(drug) for drug in drug_list if isinstance(drug, dict)]
    drugs = [drug for drug in drugs if drug]
    if usage_type and drugs:
        return f"{usage_type}：{'、'.join(drugs)}"
    if usage_type:
        return usage_type
    return "、".join(drugs)


def format_prescriptions(prescriptions: Iterable[dict[str, Any]]) -> list[str]:
    return [text for text in (format_prescription(item) for item in prescriptions or []) if text]


def _prescription_usage_type(prescription: Any) -> str:
    if isinstance(prescription, dict):
        return str(prescription.get("usage_type") or "").strip()
    if isinstance(prescription, str):
        match = re.match(r"^([^:：\s]+)\s*[:：]", prescription.strip())
        if match:
            return match.group(1).strip()
    return ""


def _has_internal_prescription(record: dict[str, Any]) -> bool:
    for prescription in record.get("ps") or []:
        if _prescription_usage_type(prescription) == INTERNAL_USAGE_TYPE:
            return True
    return False


def _has_external_prescription(record: dict[str, Any]) -> bool:
    for prescription in record.get("ps") or []:
        usage_type = _prescription_usage_type(prescription)
        if usage_type in EXTERNAL_USAGE_TYPES or (usage_type and usage_type != INTERNAL_USAGE_TYPE):
            return True
    return False


def _is_external_only_prescription_record(record: dict[str, Any]) -> bool:
    return _has_external_prescription(record) and not _has_internal_prescription(record)


def _parse_record_datetime(value: Any) -> datetime:
    if not value:
        return datetime.min
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for candidate, fmt in (
        (text, "%Y-%m-%d %H:%M:%S"),
        (text[:19], "%Y-%m-%d %H:%M:%S"),
        (text[:19], "%Y-%m-%dT%H:%M:%S"),
        (text[:10], "%Y-%m-%d"),
    ):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return datetime.min


def _record_sort_key(record: dict[str, Any]) -> tuple[datetime, int]:
    time_value = record.get("start_time") or record.get("created_at") or record.get("stop_time")
    try:
        record_id = int(record.get("id") or 0)
    except (TypeError, ValueError):
        record_id = 0
    return (_parse_record_datetime(time_value), record_id)


def _internal_prescription_texts(record: dict[str, Any]) -> list[str]:
    return [
        text
        for prescription in record.get("ps") or []
        if _prescription_usage_type(prescription) == INTERNAL_USAGE_TYPE
        for text in [format_prescription(prescription) if isinstance(prescription, dict) else str(prescription).strip()]
        if text
    ]


def build_linked_internal_prescription(previous_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_inquiry_id": previous_record.get("id"),
        "source_order_sn": previous_record.get("order_sn") or "",
        "source_start_time": previous_record.get("start_time") or "",
        "source_created_at": previous_record.get("created_at") or "",
        "diagnosis_disease": previous_record.get("diagnosis_disease") or "",
        "diagnosis_sickness": previous_record.get("diagnosis_sickness") or "",
        "ps": _internal_prescription_texts(previous_record),
    }


def build_previous_internal_prescription_map(records: Iterable[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    records_by_patient: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        patient_id = record.get("patient_id")
        if patient_id in (None, ""):
            continue
        records_by_patient[patient_id].append(record)

    linked_by_record_id: dict[Any, dict[str, Any]] = {}
    for patient_records in records_by_patient.values():
        previous_internal_record = None
        for record in sorted(patient_records, key=_record_sort_key):
            if _is_external_only_prescription_record(record) and previous_internal_record is not None:
                linked_by_record_id[record.get("id")] = build_linked_internal_prescription(previous_internal_record)
            if _has_internal_prescription(record):
                previous_internal_record = record
    return linked_by_record_id


def _normalize_drug_name(name: Any) -> str:
    value = str(name or "").strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"^\d+[.、]", "", value)
    value = re.sub(r"[\d.]+(?:g|ml|mg|片|粒|袋|支|瓶|盒|丸|克)$", "", value, flags=re.IGNORECASE)
    return value.strip("，,。、；;：: ")


def extract_drug_names_from_prescription(value: Any, internal_only: bool = True) -> set[str]:
    names: set[str] = set()
    if not value:
        return names
    if isinstance(value, dict):
        if "ps" in value:
            names.update(extract_drug_names_from_prescription(value.get("ps"), internal_only=internal_only))
        usage_type = str(value.get("usage_type") or "").strip()
        if internal_only and usage_type and usage_type != "内服":
            return names
        prescription_items = value.get("prescription_items") or {}
        drug_list = prescription_items.get("drugList") or prescription_items.get("drug_list") or []
        for drug in drug_list:
            if isinstance(drug, dict):
                name = _normalize_drug_name(_drug_name(drug))
                if name:
                    names.add(name)
        return names
    if isinstance(value, list):
        for item in value:
            names.update(extract_drug_names_from_prescription(item, internal_only=internal_only))
        return names
    if isinstance(value, str):
        raw_text = value.strip()
        usage_match = re.match(r"^(内服|外用|代茶饮|膏方|丸剂|颗粒剂|处方)\s*[:：]", raw_text)
        if internal_only and usage_match and usage_match.group(1) != "内服":
            return names
        text = re.sub(r"^(内服|外用|代茶饮|膏方|丸剂|颗粒剂|处方)\s*[:：]", "", raw_text)
        for part in re.split(r"[、,，;；\n\r]+", text):
            name = _normalize_drug_name(part)
            if name and name not in {"内服", "外用"}:
                names.add(name)
    return names


def extract_template_drug_names(template: dict[str, Any]) -> set[str]:
    names = set()
    for drug in template.get("drugs") or []:
        if isinstance(drug, dict):
            name = _normalize_drug_name(drug.get("drug_name") or drug.get("name"))
            if name:
                names.add(name)
    return names


def build_template_match_detail(prescription_drugs: set[str], template: dict[str, Any]) -> dict[str, Any]:
    template_drugs = extract_template_drug_names(template)
    overlap = prescription_drugs & template_drugs
    union = prescription_drugs | template_drugs
    score = round(len(overlap) / len(union), 4) if union else 0.0
    return {
        "template_id": template.get("template_id"),
        "template_name": template.get("template_name") or "",
        "score": score,
        "overlap_drugs": sorted(overlap),
        "missing_template_drugs": sorted(template_drugs - prescription_drugs),
        "extra_prescription_drugs": sorted(prescription_drugs - template_drugs),
        "drugs": template.get("drugs") or [],
    }


def find_most_similar_template(prescription: Any, templates: list[dict[str, Any]]) -> dict[str, Any]:
    prescription_drugs = extract_drug_names_from_prescription(prescription)
    if not prescription_drugs or not templates:
        return {}
    candidates = [build_template_match_detail(prescription_drugs, template) for template in templates]
    candidates.sort(key=lambda item: (item["score"], len(item["overlap_drugs"])), reverse=True)
    return candidates[0] if candidates else {}


def _template_match_prescription_source(record: dict[str, Any]) -> Any:
    if extract_drug_names_from_prescription(record.get("ps")):
        return record
    linked_internal_prescription = record.get("linked_internal_prescription") or {}
    if linked_internal_prescription.get("ps"):
        return linked_internal_prescription.get("ps")
    return record


def _template_match_detail_for_record(
    record: dict[str, Any],
    templates: list[dict[str, Any]] | None = None,
    linked_internal_prescription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_match_detail = record.get("matched_template_detail")
    if isinstance(existing_match_detail, dict) and existing_match_detail:
        return existing_match_detail
    if templates is None:
        return {}
    if extract_drug_names_from_prescription(record.get("ps")):
        source = record
    else:
        source = {"ps": linked_internal_prescription.get("ps", []) if linked_internal_prescription else []}
    return find_most_similar_template(source, templates)


def add_most_similar_template_detail(record: dict[str, Any], templates: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(record)
    match_detail = find_most_similar_template(_template_match_prescription_source(record), templates)
    if not match_detail:
        enriched.update(
            {
                "matched_template_id": "",
                "matched_template_name": "",
                "matched_template_score": 0.0,
                "matched_template_detail": {},
            }
        )
        return enriched
    enriched["matched_template_id"] = str(match_detail.get("template_id") or "")
    enriched["matched_template_name"] = match_detail.get("template_name") or ""
    enriched["matched_template_score"] = match_detail.get("score", 0.0)
    enriched["matched_template_detail"] = match_detail
    return enriched


def build_clinical_context(record: dict[str, Any]) -> str:
    parts = []
    for field in CLINICAL_SOURCE_FIELDS:
        value = record.get(field)
        if value:
            parts.append(f"{field}: {value}")
    # inspection_images = _format_image_context(record, INSPECTION_IMAGE_FIELDS)
    # if inspection_images:
    #     parts.append(f"检查报告图片: {inspection_images}")
    # tongue_face_images = _format_image_context(record, TONGUE_FACE_IMAGE_FIELDS)
    # if tongue_face_images:
    #     parts.append(f"舌面图片: {tongue_face_images}")
    return "\n".join(parts)


def _normalized_contains(source_text: str, expected: str) -> bool:
    source = re.sub(r"\s+", "", source_text or "")
    value = re.sub(r"\s+", "", expected or "")
    return bool(value and value in source)


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _clean_association_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_adjustment_associations(items: Any, source_text: str = "") -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    associations = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symptom_change = _clean_association_text(item.get("symptom_change") or item.get("symptom"))
        if source_text and not _normalized_contains(source_text, symptom_change):
            continue
        associations.append(
            {
                "symptom_change": symptom_change,
                "drug_name": _clean_association_text(item.get("drug_name")),
                "adjustment": _clean_association_text(item.get("adjustment")),
                "reason": _clean_association_text(item.get("reason") or item.get("relation")),
                "evidence": _clean_association_text(item.get("evidence")),
                "confidence": _confidence(item.get("confidence")),
            }
        )
    return [item for item in associations if item["symptom_change"] and item["drug_name"]]


def _requires_adjustment_associations(match_detail: dict[str, Any] | None) -> bool:
    if not isinstance(match_detail, dict):
        return False
    return bool(match_detail.get("missing_template_drugs") or match_detail.get("extra_prescription_drugs"))


def _list_text(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return "、".join(_clean_association_text(value) for value in values if _clean_association_text(value))


def _empty_adjustment_association_reason(match_detail: dict[str, Any], source_text: str = "") -> list[dict[str, Any]]:
    missing_drugs = _list_text(match_detail.get("missing_template_drugs"))
    extra_drugs = _list_text(match_detail.get("extra_prescription_drugs"))
    changed_drugs = "、".join(item for item in [missing_drugs, extra_drugs] if item) or "模板差异药物"
    adjustment = "加药/减药" if missing_drugs and extra_drugs else "减药" if missing_drugs else "加药"
    reason_parts = [
        "模型返回为空",
        "matched_template_detail 中存在缺失或新增药物",
        "但模型未能基于原文症状、检查异常、舌面情况与药物药性功效形成明确关联说明",
    ]
    if source_text:
        reason_parts.append("需人工结合原文进一步确认")
    return [
        {
            "symptom_change": "症状与用药调整关联不明确",
            "drug_name": changed_drugs,
            "adjustment": adjustment,
            "reason": "；".join(reason_parts),
            "evidence": source_text[:200],
            "confidence": 0.0,
        }
    ]


def _format_image_context(record: dict[str, Any], fields: list[str]) -> str:
    chunks = []
    for field in fields:
        images = record.get(field) or []
        if not images:
            continue
        urls = []
        for image in images:
            if isinstance(image, dict):
                url = image.get("img") or image.get("url")
            else:
                url = image
            if url:
                urls.append(str(url))
        if urls:
            chunks.append(f"{field}({len(urls)}张): " + "，".join(urls[:3]))
    return "；".join(chunks)


def pack_record(
    record: dict[str, Any],
    clinical_symptoms: str = "",
    clinical_symptoms_text: str = "",
    inspection_abnormalities: str = "",
    tongue_face_findings: str = "",
    symptom_adjustment_associations: list[dict[str, Any]] | None = None,
    templates: list[dict[str, Any]] | None = None,
    linked_internal_prescription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packed = {field: record.get(field, [] if field.endswith("_img") else "") for field in KEEP_FIELDS}
    packed["ps"] = format_prescriptions(record.get("ps") or [])
    packed["linked_internal_prescription"] = linked_internal_prescription or {}
    packed["clinical_symptoms"] = clinical_symptoms
    packed["inspection_abnormalities"] = inspection_abnormalities
    packed["tongue_face_findings"] = tongue_face_findings
    packed["clinical_symptoms_text"] = clinical_symptoms_text or "，".join(
        item for item in [clinical_symptoms, inspection_abnormalities, tongue_face_findings] if item
    )
    packed["symptom_adjustment_associations"] = symptom_adjustment_associations or []
    if templates is not None:
        packed = add_most_similar_template_detail(packed, templates)
    return packed


def pack_records(
    records: Iterable[dict[str, Any]],
    templates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = list(records)
    linked_prescriptions = build_previous_internal_prescription_map(records)
    return [
        pack_record(
            record,
            templates=templates,
            linked_internal_prescription=linked_prescriptions.get(record.get("id")),
        )
        for record in records
    ]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prompt_record_item(
    record: dict[str, Any],
    templates: list[dict[str, Any]] | None = None,
    linked_internal_prescription: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prescriptions = format_prescriptions(record.get("ps") or [])
    item = {
        "id": record.get("id"),
        "text": build_clinical_context(record),
        "diagnosis_disease": record.get("diagnosis_disease", ""),
        "diagnosis_sickness": record.get("diagnosis_sickness", ""),
        "prescriptions": prescriptions,
    }
    if linked_internal_prescription:
        item["linked_internal_prescription"] = linked_internal_prescription
    match_detail = _template_match_detail_for_record(
        record,
        templates=templates,
        linked_internal_prescription=linked_internal_prescription,
    )
    if match_detail:
        item["matched_template_detail"] = match_detail
        item["matched_template"] = {
            "template_id": match_detail.get("template_id") or record.get("matched_template_id"),
            "template_name": match_detail.get("template_name") or record.get("matched_template_name"),
            "overlap_drugs": match_detail.get("overlap_drugs", []),
            "missing_template_drugs": match_detail.get("missing_template_drugs", []),
            "extra_prescription_drugs": match_detail.get("extra_prescription_drugs", []),
            "drugs": match_detail.get("drugs", []),
        }
    return item


def build_symptom_extraction_prompt(
    records: list[dict[str, Any]],
    templates: list[dict[str, Any]] | None = None,
    linked_prescriptions: dict[Any, dict[str, Any]] | None = None,
) -> str:
    linked_prescriptions = linked_prescriptions or {}
    items = [
        _prompt_record_item(
            record,
            templates=templates,
            linked_internal_prescription=linked_prescriptions.get(record.get("id")),
        )
        for record in records
    ]
    # print(f"records {items[0]}")
    prompt = f"""请从以下中医问诊病历中抽取“本次就诊可用于相似病例检索的临床症状”。

抽取规则：
1. 只抽取症状、体征、检查所见异常、舌面情况、诱因和病情变化，去掉诊断名、证候名、处方药名、治疗建议、检查建议。
2. 初诊患者：提取本次主诉、现病史中当前存在的症状和体征。
3. 复诊患者：重点提取本次仍存在、新增、加重、减轻、消失、复发的症状和体征，以及服药后疗效变化。
4. 复诊患者的既往病史、上次症状不能直接当作本次症状，除非文本明确说明“目前仍有、仍存在、尚有、反复、复发、加重、减轻”。
5. 如果文本描述“无明显不适、无新发、症状已消失”，也要如实保留这个当前状态。
6. 检查所见异常必须记录到 inspection_abnormalities；包括化验、影像、皮肤镜、病理、报告描述里的异常结果。正常结果可以不写，除非文本强调“检查未见异常”。
7. 舌面情况必须记录到 tongue_face_findings；包括舌质、舌苔、舌下络脉、面部颜色、皮损外观、照片文字描述等。
8. 如果上下文只有检查报告图片或舌面图片 URL、没有文字描述，不能编造图片内容；可在对应字段写“有检查报告图片，待 OCR/视觉识别”或“有舌面图片，待视觉识别”。
9. clinical_symptoms_text 用于 ES 检索，应合并“当前症状 + 新增症状 + 病情变化 + 疗效反馈 + 检查所见异常 + 舌面情况 + 二便情况 + 饮食情况 + 睡眠情况， 女性患者重点记录月经情况”，不要写成列表，不要包含药名和诊断名。
10. 分析 symptom_adjustment_associations：结合 matched_template_detail 中 missing_template_drugs 和 extra_prescription_drugs，缺失/新增药物，分析“输入文本明确提及的症状变化”和用药加、减、剂量变化之间的可能关联；症状变化必须来自输入文本，每条给出 confidence，范围 0-1。
11. 如果 matched_template_detail.missing_template_drugs 和 extra_prescription_drugs 任意一个不为空，symptom_adjustment_associations 必须为非空数组，至少返回 1 条，不能返回空数组。
12. 生成 symptom_adjustment_associations 时，必须从当前症状、症状变化、检查异常、舌面情况中选择原文出现的短语作为 symptom_change；reason 必须结合该症状/体征和药物的药性、功效或治疗方向，说明为什么可能加用、减去、保留或调整该药。即使无法确定严格因果，也要给出“可能的辨证/治疗方向解释”，不要因为不确定而留空。
13. 如果确实无法从输入文本建立症状/体征与加减药的关联，也不能返回空数组；请返回 1 条原因说明：symptom_change 写“症状与用药调整关联不明确”，drug_name 填 missing_template_drugs 和 extra_prescription_drugs 中涉及的药物，adjustment 写“加药/减药/加药、减药”，reason 写明“字段为空的原因”，例如“原文仅有处方差异，缺少可对应到该药药性、功效或治疗方向的症状变化证据”，confidence 设为 0。
14. drug_name 必须优先覆盖 matched_template_detail.missing_template_drugs 和 extra_prescription_drugs 中的药物；多个药物治疗方向相近时可以合并到一条。
只返回 JSON 数组，不要输出解释。每项格式：
[
  {{
    "id": 原始id,
    "clinical_symptoms": "本次症状摘要",
    "inspection_abnormalities": "检查所见异常",
    "tongue_face_findings": "舌面情况",
    "clinical_symptoms_text": "用于 ES 检索的合并症状文本",
    "symptom_adjustment_associations": [
      {{
        "symptom_change": "症状变化描述",
        "drug_name": "与相似模版方相比，发生加减的药名，如果治疗方向或者药性相同的药物，放在一起，用逗号分隔",
        "adjustment": "加药/减药/剂量增加/剂量减少",
        "reason": "基于当前症状、症状变化、检查异常或舌面情况，并结合药物药性、功效或治疗方向，解释该加减药的可能原因",
        "evidence": "输入文本中的依据短句",
        "confidence": 0.0
      }}
    ]
  }}
]

示例：
输入文本：
复诊。服药后面部红斑较前减轻，丘疹减少，仍有口干，近3日熬夜后鼻翼潮红加重。检查提示白细胞偏高。舌红苔黄腻，面部潮红。

示例输出：
[
  {{
    "id": 0,
    "clinical_symptoms": "面部红斑、丘疹、口干、鼻翼潮红；红斑较前减轻，丘疹减少，鼻翼潮红近3日加重",
    "inspection_abnormalities": "白细胞偏高",
    "tongue_face_findings": "舌红苔黄腻，面部潮红",
    "clinical_symptoms_text": "面部红斑较前减轻，丘疹减少，仍有口干，鼻翼潮红加重，熬夜诱发，白细胞偏高，舌红苔黄腻，面部潮红",
    "symptom_adjustment_associations": [
      {{"symptom_change": "鼻翼潮红近3日加重", "drug_name": "黄芩片", "adjustment": "保留", "reason": "清热解毒以应对潮红加重", "evidence": "鼻翼潮红近3日加重", "confidence": 0.68}}
    ]
  }}
]

待抽取病历：
{json.dumps(items, ensure_ascii=False, indent=2)}
"""
    return prompt


def parse_symptom_response(
    response_text: str,
    source_text_by_id: dict[Any, str] | None = None,
    match_detail_by_id: dict[Any, dict[str, Any]] | None = None,
) -> dict[Any, dict[str, Any]]:
    source_text_by_id = source_text_by_id or {}
    match_detail_by_id = match_detail_by_id or {}
    text = _extract_json_payload(response_text)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    result = {}
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            item_id = item["id"]
            source_text = source_text_by_id.get(item_id, "")
            adjustment_associations = _normalize_adjustment_associations(
                item.get("symptom_adjustment_associations"),
                source_text=source_text,
            )
            match_detail = match_detail_by_id.get(item_id)
            if _requires_adjustment_associations(match_detail) and not adjustment_associations:
                adjustment_associations = _empty_adjustment_association_reason(match_detail or {}, source_text)
            result[item_id] = {
                "clinical_symptoms": str(item.get("clinical_symptoms") or "").strip(),
                "clinical_symptoms_text": str(
                    item.get("clinical_symptoms_text") or item.get("clinical_symptoms") or ""
                ).strip(),
                "inspection_abnormalities": str(item.get("inspection_abnormalities") or "").strip(),
                "tongue_face_findings": str(item.get("tongue_face_findings") or "").strip(),
                "symptom_adjustment_associations": adjustment_associations,
            }
    return result


def _extract_json_payload(response_text: str) -> str:
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    starts = [index for index in (text.find("["), text.find("{")) if index != -1]
    if not starts:
        return text

    decoder = json.JSONDecoder()
    start = min(starts)
    _, end = decoder.raw_decode(text[start:])
    return text[start : start + end]


from openai import OpenAI

def create_llm_call(model: str):

    provider = config.LLM_PROVIDER.lower()
    if provider == "dashscope":
        client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )

    elif provider == "deepseek":
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        model = config.DEEPSEEK_MODEL

    elif provider == "baichuan":
        client = OpenAI(
            api_key=config.BAICHUAN_API_KEY,
            base_url=config.BAICHUAN_BASE_URL,
        )
        model = config.BAICHUAN_TEXT_MODEL

    elif provider == "ark":
        client = OpenAI(
            api_key=config.ARK_API_KEY,
            base_url=config.ARK_BASE_URL,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    def call(prompt: str):
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {
                #     "role": "system",
                #     "content": "你是中医病历结构化抽取助手。"
                # },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )
        logger.debug(f"response: {response.choices[0].message.content[:100]}")
        return response.choices[0].message.content or "[]"
    logger.debug(f"call llm with model {model}")
    return call


def fill_clinical_symptoms(
    records: list[dict[str, Any]],
    llm_call: Callable[[str], str],
    batch_size: int = 10,
    templates: list[dict[str, Any]] | None = None,
    append_output: Path | None = None,
) -> list[dict[str, Any]]:
    packed_records = []
    linked_prescriptions = build_previous_internal_prescription_map(records)
    if append_output is not None:
        append_output.parent.mkdir(parents=True, exist_ok=True)
        append_output.write_text("", encoding="utf-8")

    def _pack_with_default(record: dict[str, Any]) -> dict[str, Any]:
        return pack_record(
            record,
            templates=templates,
            linked_internal_prescription=linked_prescriptions.get(record.get("id")),
        )

    def _extract_batch(batch_records: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        source_text_by_id = {record.get("id"): build_clinical_context(record) for record in batch_records}
        match_detail_by_id = {
            record.get("id"): _template_match_detail_for_record(
                record,
                templates=templates,
                linked_internal_prescription=linked_prescriptions.get(record.get("id")),
            )
            for record in batch_records
        }
        response = llm_call(
            build_symptom_extraction_prompt(
                batch_records,
                templates=templates,
                linked_prescriptions=linked_prescriptions,
            )
        )
        return parse_symptom_response(
            response,
            source_text_by_id=source_text_by_id,
            match_detail_by_id=match_detail_by_id,
        )

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            symptom_map = _extract_batch(batch)
        except Exception as exc:
            logger.warning(
                "failed to extract clinical symptoms for batch ids %s, fallback to single-record mode: %s",
                [record.get("id") for record in batch],
                exc,
            )
            symptom_map = {}
            for record in batch:
                try:
                    symptom_map.update(_extract_batch([record]))
                except Exception as single_exc:
                    logger.warning(
                        "failed to extract clinical symptoms for record %s, use default empty extraction: %s",
                        record.get("id"),
                        single_exc,
                    )
        for record in batch:
            extracted = symptom_map.get(record.get("id"), {})
            if isinstance(extracted, str):
                extracted = {"clinical_symptoms": extracted, "clinical_symptoms_text": extracted}
            if not extracted:
                packed_records.append(_pack_with_default(record))
                continue
            packed_records.append(
                pack_record(
                    record,
                    templates=templates,
                    linked_internal_prescription=linked_prescriptions.get(record.get("id")),
                    **extracted,
                )
            )
        if append_output is not None:
            with append_output.open("a", encoding="utf-8") as f:
                for packed_record in packed_records[-len(batch) :]:
                    f.write(json.dumps(packed_record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return packed_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack prescription records into ES-ready JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N records; 0 means all.")
    parser.add_argument("--extract-clinical-symptoms", action="store_true", help="Use LLM to fill clinical symptoms.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=os.getenv("ARK_MODEL", "qwen3.6-flash"))
    parser.add_argument("--template-file", type=Path, default=DEFAULT_TEMPLATE_FILE)
    parser.add_argument("--no-template-match", action="store_true", help="Skip matched template enrichment.")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Write each processed batch to output immediately; useful for long background runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    templates = None if args.no_template_match else load_templates(args.template_file)
    print(f"templates: {templates[0]}")

    if args.extract_clinical_symptoms:
        packed_records = fill_clinical_symptoms(
            records,
            create_llm_call(args.model),
            batch_size=args.batch_size,
            templates=templates,
            append_output=args.output if args.append_output else None,
        )
    else:
        packed_records = pack_records(records, templates=templates)

    if not (args.extract_clinical_symptoms and args.append_output):
        write_jsonl(packed_records, args.output)
    print(f"wrote {len(packed_records)} records to {args.output}")


if __name__ == "__main__":
    main()
