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
from pathlib import Path
from typing import Any, Callable, Iterable
from medical.config import config
from medical.utils.log import logger


DEFAULT_INPUT = Path("medical/data/wuweiping_record_20260525.json")
DEFAULT_OUTPUT = Path("medical/processed_data/wuweiping_es_prescription.jsonl")
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


def add_most_similar_template_detail(record: dict[str, Any], templates: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(record)
    match_detail = find_most_similar_template(record, templates)
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
    templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    packed = {field: record.get(field, [] if field.endswith("_img") else "") for field in KEEP_FIELDS}
    packed["ps"] = format_prescriptions(record.get("ps") or [])
    packed["clinical_symptoms"] = clinical_symptoms
    packed["inspection_abnormalities"] = inspection_abnormalities
    packed["tongue_face_findings"] = tongue_face_findings
    packed["clinical_symptoms_text"] = clinical_symptoms_text or "，".join(
        item for item in [clinical_symptoms, inspection_abnormalities, tongue_face_findings] if item
    )
    if templates is not None:
        packed = add_most_similar_template_detail(packed, templates)
    return packed


def pack_records(
    records: Iterable[dict[str, Any]],
    templates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [pack_record(record, templates=templates) for record in records]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_symptom_extraction_prompt(records: list[dict[str, Any]]) -> str:
    items = [
        {
            "id": record.get("id"),
            "text": build_clinical_context(record),
            "diagnosis_disease": record.get("diagnosis_disease", ""),
            "diagnosis_sickness": record.get("diagnosis_sickness", ""),
        }
        for record in records
    ]
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
9. clinical_symptoms_text 用于 ES 检索，应合并“当前症状 + 新增症状 + 病情变化 + 疗效反馈 + 检查所见异常 + 舌面情况”，不要写成列表，不要包含药名和诊断名。

只返回 JSON 数组，不要输出解释。每项格式：
[
  {{
    "id": 原始id,
    "clinical_symptoms": "本次症状摘要",
    "inspection_abnormalities": "检查所见异常",
    "tongue_face_findings": "舌面情况",
    "clinical_symptoms_text": "用于 ES 检索的合并症状文本"
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
    "clinical_symptoms_text": "面部红斑较前减轻，丘疹减少，仍有口干，鼻翼潮红加重，熬夜诱发，白细胞偏高，舌红苔黄腻，面部潮红"
  }}
]

待抽取病历：
{json.dumps(items, ensure_ascii=False, indent=2)}
"""
    return prompt


def parse_symptom_response(response_text: str) -> dict[Any, str]:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    result = {}
    for item in data:
        if isinstance(item, dict) and item.get("id") is not None:
            result[item["id"]] = {
                "clinical_symptoms": str(item.get("clinical_symptoms") or "").strip(),
                "clinical_symptoms_text": str(
                    item.get("clinical_symptoms_text") or item.get("clinical_symptoms") or ""
                ).strip(),
                "inspection_abnormalities": str(item.get("inspection_abnormalities") or "").strip(),
                "tongue_face_findings": str(item.get("tongue_face_findings") or "").strip(),
            }
    return result


def create_llm_call(model: str) -> Callable[[str], str]:
    """Create an LLM caller lazily so normal packing has no network dependency."""
    from openai import OpenAI

    api_key = config.DASHSCOPE_API_KEY
    base_url = config.DASHSCOPE_BASE_URL
    if not api_key:
        raise RuntimeError("Set ARK_API_KEY or OPENAI_API_KEY before using --extract-clinical-symptoms.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(prompt: str) -> str:
        logger.debug(f"Calling LLM with model: {model}")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是中医病历结构化抽取助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return completion.choices[0].message.content or "[]"

    return call


def fill_clinical_symptoms(
    records: list[dict[str, Any]],
    llm_call: Callable[[str], str],
    batch_size: int = 10,
    templates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    packed_records = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        response = llm_call(build_symptom_extraction_prompt(batch))
        symptom_map = parse_symptom_response(response)
        for record in batch:
            extracted = symptom_map.get(record.get("id"), {})
            if isinstance(extracted, str):
                extracted = {"clinical_symptoms": extracted, "clinical_symptoms_text": extracted}
            packed_records.append(pack_record(record, templates=templates, **extracted))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    templates = None if args.no_template_match else load_templates(args.template_file)

    if args.extract_clinical_symptoms:
        packed_records = fill_clinical_symptoms(
            records,
            create_llm_call(args.model),
            batch_size=args.batch_size,
            templates=templates,
        )
    else:
        packed_records = pack_records(records, templates=templates)

    write_jsonl(packed_records, args.output)
    print(f"wrote {len(packed_records)} records to {args.output}")


if __name__ == "__main__":
    main()
