#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从吴卫平医生原始问诊数据中抽取可用于处方生成 SFT 的有效病历。

默认输入:
    medical/data/wuweiping_record_20260525.json

默认输出:
    medical/data/sft_generate_prescription/extract_valid_record.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_FILE = Path("medical/data/wuweiping_record_20260525.json")
DEFAULT_OUTPUT_FILE = Path("medical/data/sft_generate_prescription/extract_valid_record.json")
DEFAULT_USAGE_TYPES = ("内服",)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_prescription_items(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_drug(item: dict[str, Any]) -> dict[str, Any] | None:
    drug_name = clean_text(item.get("drug_name") or item.get("show_name") or item.get("sub_drug_name"))
    if not drug_name:
        return None

    dose = clean_text(item.get("drug_num") or item.get("dose") or item.get("quantity") or item.get("drug_weight"))
    unit = clean_text(item.get("unit_name") or item.get("unit"))
    decoction_name = clean_text(item.get("decoction_name"))
    return {
        "drug_name": drug_name,
        "dose": dose,
        "unit": unit,
        "decoction_name": decoction_name 
    }


def extract_prescription(prescription: dict[str, Any]) -> dict[str, Any] | None:
    items = parse_prescription_items(prescription.get("prescription_items"))
    drug_list = items.get("drugList") or items.get("drugs") or []
    if not isinstance(drug_list, list):
        return None

    drugs = []
    for item in drug_list:
        if not isinstance(item, dict):
            continue
        drug = extract_drug(item)
        if drug is not None:
            drugs.append(drug)

    if not drugs:
        return None

    return {
        "prescription_order_id": prescription.get("prescription_order_id") or prescription.get("id"),
        "usage_type": clean_text(prescription.get("usage_type")),
        "drug_process_name": clean_text(prescription.get("drug_process_name")),
        "doctor_advice": clean_text(prescription.get("doctor_advice")),
        "drugs": drugs,
    }


def extract_prescriptions(record: dict[str, Any], usage_types: set[str] | None) -> list[dict[str, Any]]:
    prescriptions = []
    for prescription in record.get("ps") or []:
        if not isinstance(prescription, dict):
            continue
        usage_type = clean_text(prescription.get("usage_type"))
        if usage_types is not None and usage_type not in usage_types:
            continue
        extracted = extract_prescription(prescription)
        if extracted is not None:
            prescriptions.append(extracted)
    return prescriptions


def extract_record(record: dict[str, Any], usage_types: set[str] | None) -> dict[str, Any] | None:
    prescriptions = extract_prescriptions(record, usage_types=usage_types)
    if not prescriptions:
        return None

    return {
        "inquiry_id": record.get("id"),
        "order_sn": clean_text(record.get("order_sn")),
        "inquiry_method": clean_text(record.get("inquiry_method")),
        "is_first": clean_text(record.get("is_first")),
        "patient_name": clean_text(record.get("patient_name")),
        "patient_sex": clean_text(record.get("patient_sex")),
        "patient_age": clean_text(record.get("patient_age")),
        "patient_appeal": clean_text(record.get("patient_appeal")),
        "doc_ass_stu_appeal": clean_text(record.get("doc_ass_stu_appeal")),
        "new_medical_history": clean_text(record.get("new_medical_history")),
        "old_medical_history": clean_text(record.get("old_medical_history")),
        "allergic_history": clean_text(record.get("allergic_history")),
        "family_history": clean_text(record.get("family_history")),
        "personal_history": clean_text(record.get("personal_history")),
        "birth_detail": clean_text(record.get("birth_detail") or record.get("marriage_childbearing_history")),
        "admin_report_describe": clean_text(record.get("admin_report_describe") or record.get("inspection_abnormalities")),
        "admin_face_describe": clean_text(record.get("admin_face_describe")),
        "inspection_report_img": as_list(record.get("inspection_report_img")),
        "tongue_face_img": as_list(record.get("tongue_face_img")),
        "admin_report_img": as_list(record.get("admin_report_img")),
        "admin_face_img": as_list(record.get("admin_face_img")),
        "diagnosis_sickness": clean_text(record.get("diagnosis_sickness")),
        "diagnosis_illness": clean_text(record.get("diagnosis_illness")),
        "diagnosis_disease": clean_text(record.get("diagnosis_disease")),
        "internal_prescriptions": prescriptions,
    }


def extract_valid_records(
    records: list[dict[str, Any]], usage_types: Iterable[str] | None = DEFAULT_USAGE_TYPES
) -> list[dict[str, Any]]:
    usage_type_filter = set(usage_types) if usage_types is not None else None
    valid_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        extracted = extract_record(record, usage_types=usage_type_filter)
        if extracted is not None:
            valid_records.append(extracted)
    return valid_records


def build_stats(source_records: list[dict[str, Any]], valid_records: list[dict[str, Any]]) -> dict[str, int]:
    prescription_count = sum(len(record.get("internal_prescriptions", [])) for record in valid_records)
    drug_count = sum(
        len(prescription.get("drugs", []))
        for record in valid_records
        for prescription in record.get("internal_prescriptions", [])
    )
    return {
        "source_records": len(source_records),
        "valid_records": len(valid_records),
        "internal_prescriptions": prescription_count,
        "drugs": drug_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取可用于处方生成 SFT 的有效病历 JSON。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help=f"输入文件，默认 {DEFAULT_INPUT_FILE}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help=f"输出文件，默认 {DEFAULT_OUTPUT_FILE}")
    parser.add_argument(
        "--usage-type",
        action="append",
        default=["内服"],
        help="要抽取的处方类型，可重复传入；默认只抽取内服以兼容 internal_prescriptions。",
    )
    parser.add_argument(
        "--include-all-usage-types",
        action="store_true",
        help="抽取所有处方类型。开启后忽略 --usage-type。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records = load_json(args.input)
    if not isinstance(source_records, list):
        raise ValueError(f"Input must be a JSON list: {args.input}")

    usage_types = None if args.include_all_usage_types else set(args.usage_type)
    valid_records = extract_valid_records(source_records, usage_types=usage_types)
    save_json(valid_records, args.output)

    stats = build_stats(source_records, valid_records)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
