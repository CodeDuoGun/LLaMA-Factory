"""
从病历数据medical/data/wuweiping_record_20260525.json中，获取问诊单id、患者性别姓名年龄、主诉、现病史、既往史、过敏史、家族史、婚育史。诊断名称、诊断症候、内服处方详情
"""
import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path("medical/data/wuweiping_record_20260525.json")
DEFAULT_OUTPUT_FILE = Path("medical/data/wuweiping_record_prescription_extract_20260525.json")


def _clean_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _join_history_parts(*parts: Any) -> str:
    cleaned_parts = []
    for part in parts:
        part = _clean_text(part)
        if part in (None, "", "否", "未知"):
            continue
        cleaned_parts.append(str(part))
    return "；".join(cleaned_parts)


def _extract_check_findings(text: Any) -> str:
    text = _clean_text(text)
    if not text:
        return ""

    keywords = (
        "舌",
        "苔",
        "检查",
        "检验",
        "体检",
        "血常规",
        "尿常规",
        "肝肾",
        "肝功能",
        "肾功能",
        "功能",
        "阳性",
        "阴性",
        "正常",
        "↑",
        "↓",
    )
    separators = "。\n；;"
    fragments = [str(text)]
    for separator in separators:
        next_fragments = []
        for fragment in fragments:
            next_fragments.extend(fragment.split(separator))
        fragments = next_fragments

    findings = []
    for fragment in fragments:
        fragment = fragment.strip()
        if fragment and any(keyword in fragment for keyword in keywords):
            findings.append(fragment)
    return "；".join(findings[:8])


def _get_drug_name(drug: dict[str, Any]) -> Any:
    return _clean_text(drug.get("drug_name") or drug.get("show_name") or drug.get("sub_drug_name"))


def extract_internal_prescriptions(record: dict[str, Any]) -> list[dict[str, Any]]:
    internal_prescriptions = []
    for prescription in record.get("ps") or []:
        if prescription.get("usage_type") != "内服":
            continue

        prescription_items = prescription.get("prescription_items") or {}
        drugs = []
        for drug in prescription_items.get("drugList") or []:
            drugs.append(
                {
                    "drug_name": _get_drug_name(drug),
                    "dose": _clean_text(drug.get("drug_num")),
                    "unit": _clean_text(drug.get("unit_name")),
                }
            )

        internal_prescriptions.append(
            {
                "prescription_order_id": prescription.get("prescription_order_id"),
                "drugs": drugs,
            }
        )
    return internal_prescriptions


def extract_record(record: dict[str, Any]) -> dict[str, Any]:
    diagnosis_result = _clean_text(record.get("diagnosis_sickness"))
    present_history = _clean_text(record.get("new_medical_history"))
    exam_findings = _join_history_parts(record.get("admin_face_describe"), _extract_check_findings(present_history))
    return {
        "inquiry_id": record.get("id"),
        "patient": {
            "name": _clean_text(record.get("patient_name")),
            "sex": _clean_text(record.get("patient_sex")),
            "age": _clean_text(record.get("patient_age")),
        },
        "chief_complaint": _clean_text(record.get("patient_appeal")),
        "clinical_info": {
            "patient_symptoms": _clean_text(record.get("patient_appeal")),
            "doctor_symptom_summary": _clean_text(record.get("doc_ass_stu_appeal")),
            "present_history": present_history,
            "exam_findings": exam_findings,
            "report_images": record.get("admin_report_img") or record.get("inspection_report_img") or [],
            "face_tongue_images": record.get("admin_face_img") or record.get("tongue_face_img") or [],
        },
        "present_history": present_history,
        "past_history": _clean_text(record.get("old_medical_history")),
        "allergic_history": _clean_text(record.get("allergic_history")),
        "family_history": _clean_text(record.get("family_history")),
        "marriage_childbearing_history": _join_history_parts(
            record.get("special_history"), record.get("birth_detail")
        ),
        "diagnosis_name": diagnosis_result,
        "diagnosis_result": diagnosis_result,
        "diagnosis_syndrome": _clean_text(record.get("diagnosis_disease")),
        "internal_prescriptions": extract_internal_prescriptions(record),
    }


def read_records(input_file: Path) -> list[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected {input_file} to contain a JSON list.")
    return records


def write_records(records: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract record and internal prescription details.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Input record JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Output extracted JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    extracted_records = [extract_record(record) for record in records]
    write_records(extracted_records, args.output)
    print(f"Extracted {len(extracted_records)} records to {args.output}")

if __name__ == "__main__":
    main()
