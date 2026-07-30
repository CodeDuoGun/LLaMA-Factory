#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计吴卫平医生同病种开方规律和模板方加减策略。
"""

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")
DEFAULT_RECORD_FILE = Path("medical/data/wuweiping_record_prescription_extract_20260525.json")
DEFAULT_OUTPUT_FILE = Path("medical/data/wuweiping_prescription_strategy_analysis.json")


DIAGNOSIS_ALIASES = {
    "酒齄鼻": "酒糟鼻",
    "酒渣鼻": "酒糟鼻",
}

SYNDROME_ALIASES = {
    "湿毒蕴肤": "湿毒蕴肤证",
    "湿毒蕴肤症": "湿毒蕴肤证",
    "血热瘀滞症": "血热瘀滞证",
    "脾虚湿蕴证": "脾虚湿蕴肌肤证",
}

SYMPTOM_KEYWORDS = [
    "鼻翼潮红",
    "面部潮红",
    "毛细血管扩张",
    "皮肤屏障受损",
    "脓疱",
    "丘疹",
    "粉刺",
    "红斑",
    "发红",
    "潮红",
    "发烫",
    "灼热",
    "瘙痒",
    "疼痛",
    "刺痛",
    "肿胀",
    "脱屑",
    "干燥",
    "干紧",
    "油腻",
    "口干",
    "口苦",
    "便秘",
    "腹泻",
    "睡眠差",
    "失眠",
    "月经不调",
    "舌红",
    "苔黄",
    "苔白",
    "苔腻",
]


def normalize_name(value: Any, aliases: dict[str, str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return aliases.get(text, text)


def normalize_diagnosis(value: Any) -> str:
    return normalize_name(value, DIAGNOSIS_ALIASES)


def normalize_diagnosis_result(value: Any) -> str:
    if value is None:
        return ""
    parts = [
        normalize_diagnosis(part)
        for part in re.split(r"[,，、;；/]+", str(value))
        if str(part).strip()
    ]
    return ",".join(sorted(set(parts)))


def normalize_syndrome(value: Any) -> str:
    return normalize_name(value, SYNDROME_ALIASES)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_templates(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return data.get("templates", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported template JSON structure: {path}")


def clean_drug_name(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clinical_text(record: dict[str, Any]) -> str:
    clinical_info = record.get("clinical_info") or {}
    parts = [
        clinical_info.get("doctor_symptom_summary"),
        clinical_info.get("patient_symptoms"),
        clinical_info.get("present_history"),
        clinical_info.get("exam_findings"),
        clinical_info.get("inspection_abnormalities"),
        clinical_info.get("tongue_face_findings"),
        record.get("chief_complaint"),
        record.get("present_history"),
    ]
    return "。".join(str(part).strip() for part in parts if str(part or "").strip())


def extract_symptom_tags(text: Any) -> list[str]:
    normalized = re.sub(r"\s+", "", str(text or ""))
    tags = [keyword for keyword in SYMPTOM_KEYWORDS if keyword in normalized]
    return sorted(set(tags), key=tags.index)


def dose_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def drug_set(drugs: list[dict[str, Any]]) -> set[str]:
    return {clean_drug_name(drug.get("drug_name")) for drug in drugs if clean_drug_name(drug.get("drug_name"))}


def drug_dose_map(drugs: list[dict[str, Any]], dose_field: str) -> dict[str, dict[str, Any]]:
    result = {}
    for drug in drugs:
        name = clean_drug_name(drug.get("drug_name"))
        if not name:
            continue
        result[name] = {
            "dose": drug.get(dose_field),
            "unit": drug.get("unit"),
        }
    return result


def match_template(prescription: dict[str, Any], templates: list[dict[str, Any]]) -> dict[str, Any]:
    record_drugs = prescription.get("drugs", [])
    record_names = drug_set(record_drugs)
    best_template = None
    best_ratio = 0.0

    for template in templates:
        template_names = drug_set(template.get("drugs", []))
        if not template_names:
            continue
        union_count = len(record_names | template_names)
        ratio = len(record_names & template_names) / union_count if union_count else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_template = template

    template_drugs = best_template.get("drugs", []) if best_template else []
    template_names = drug_set(template_drugs)
    common_drugs = sorted(record_names & template_names)
    added_drugs = sorted(record_names - template_names)
    removed_drugs = sorted(template_names - record_names)

    record_doses = drug_dose_map(record_drugs, "dose")
    template_doses = drug_dose_map(template_drugs, "quantity")
    dose_changes = []
    for name in common_drugs:
        record_dose = record_doses.get(name, {}).get("dose")
        template_dose = template_doses.get(name, {}).get("dose")
        if dose_key(record_dose) and dose_key(template_dose) and dose_key(record_dose) != dose_key(template_dose):
            dose_changes.append(
                {
                    "drug_name": name,
                    "template_dose": template_dose,
                    "record_dose": record_dose,
                    "unit": record_doses.get(name, {}).get("unit") or template_doses.get(name, {}).get("unit"),
                }
            )

    return {
        "matched_template_id": best_template.get("template_id") if best_template else None,
        "matched_template_name": best_template.get("template_name") if best_template else "",
        "match_ratio": round(best_ratio, 3),
        "common_drugs": common_drugs,
        "added_drugs": added_drugs,
        "removed_drugs": removed_drugs,
        "dose_changes": dose_changes,
    }


def flatten_prescriptions(records: list[dict[str, Any]], templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        diagnosis_name = normalize_diagnosis(record.get("diagnosis_name"))
        diagnosis_result = normalize_diagnosis_result(record.get("diagnosis_result", record.get("diagnosis_name")))
        diagnosis_syndrome = normalize_syndrome(record.get("diagnosis_syndrome"))
        clinical_info = record.get("clinical_info") or {}
        report_images = clinical_info.get("report_images") or []
        face_tongue_images = clinical_info.get("face_tongue_images") or []
        normalized_clinical_info = {
            "patient_symptoms": clinical_info.get("patient_symptoms") or record.get("chief_complaint") or "",
            "doctor_symptom_summary": clinical_info.get("doctor_symptom_summary") or "",
            "present_history": clinical_info.get("present_history") or record.get("present_history") or "",
            "exam_findings": clinical_info.get("exam_findings") or "",
            "report_image_count": len(report_images),
            "face_tongue_image_count": len(face_tongue_images),
            "report_images": report_images,
            "face_tongue_images": face_tongue_images,
        }
        symptom_tags = extract_symptom_tags(clinical_text(record))
        for prescription in record.get("internal_prescriptions", []):
            drugs = prescription.get("drugs", [])
            match = match_template(prescription, templates)
            rows.append(
                {
                    "inquiry_id": record.get("inquiry_id"),
                    "diagnosis_name": diagnosis_name,
                    "diagnosis_result": diagnosis_result,
                    "diagnosis_syndrome": diagnosis_syndrome,
                    "clinical_info": normalized_clinical_info,
                    "symptom_tags": symptom_tags,
                    "prescription_order_id": prescription.get("prescription_order_id"),
                    "drug_count": len(drug_set(drugs)),
                    "drugs": drugs,
                    **match,
                }
            )
    return rows


def top_counter(counter: Counter, total: int, name_key: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for name, count in counter.most_common(limit):
        rows.append({name_key: name, "count": count, "rate": round(count / total, 4) if total else 0})
    return rows


def top_templates(
    counter: Counter,
    total: int,
    template_catalog: dict[str, dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for template_name, count in counter.most_common(limit):
        template = template_catalog.get(template_name, {})
        rows.append(
            {
                "template_id": template.get("template_id"),
                "template_name": template_name,
                "count": count,
                "rate": round(count / total, 4) if total else 0,
                "drugs": template.get("drugs", []),
            }
        )
    return rows


def summarize_match_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [row.get("match_ratio", 0) for row in rows]
    total = len(ratios)
    bucket_defs = [
        ("90%-100%", lambda value: value >= 0.9),
        ("80%-89%", lambda value: 0.8 <= value < 0.9),
        ("70%-79%", lambda value: 0.7 <= value < 0.8),
        ("60%-69%", lambda value: 0.6 <= value < 0.7),
        ("<60%", lambda value: value < 0.6),
    ]
    return {
        "avg": round(sum(ratios) / total, 3) if total else 0,
        "min": round(min(ratios), 3) if total else 0,
        "max": round(max(ratios), 3) if total else 0,
        "exact_count": sum(value == 1 for value in ratios),
        "exact_rate": round(sum(value == 1 for value in ratios) / total, 4) if total else 0,
        "buckets": [
            {
                "range": label,
                "count": sum(matcher(value) for value in ratios),
                "rate": round(sum(matcher(value) for value in ratios) / total, 4) if total else 0,
            }
            for label, matcher in bucket_defs
        ],
    }


def summarize_inquiry_template_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for row in sorted(rows, key=lambda item: (str(item.get("inquiry_id") or ""), str(item.get("prescription_order_id") or ""))):
        matches.append(
            {
                "inquiry_id": row.get("inquiry_id"),
                "prescription_order_id": row.get("prescription_order_id"),
                "diagnosis_result": row.get("diagnosis_result"),
                "diagnosis_syndrome": row.get("diagnosis_syndrome"),
                "matched_template_id": row.get("matched_template_id"),
                "matched_template_name": row.get("matched_template_name"),
                "match_ratio": row.get("match_ratio"),
                "record_drug_count": row.get("drug_count", 0),
                "template_drug_count": len(row.get("common_drugs", [])) + len(row.get("removed_drugs", [])),
                "common_drug_count": len(row.get("common_drugs", [])),
                "added_drug_count": len(row.get("added_drugs", [])),
                "removed_drug_count": len(row.get("removed_drugs", [])),
            }
        )
    return matches


def symptom_drug_associations(rows: list[dict[str, Any]], target: str = "drugs", limit: int = 50) -> list[dict[str, Any]]:
    total = len(rows)
    min_co_count = 3 if total >= 100 else 2 if total >= 20 else 1
    symptom_counter = Counter()
    drug_counter = Counter()
    pair_counter = Counter()

    for row in rows:
        symptoms = set(row.get("symptom_tags") or [])
        if target == "drugs":
            drugs = drug_set(row.get("drugs", []))
        else:
            drugs = {clean_drug_name(item) for item in row.get(target, []) if clean_drug_name(item)}
        if not symptoms or not drugs:
            continue
        symptom_counter.update(symptoms)
        drug_counter.update(drugs)
        for symptom in symptoms:
            for drug in drugs:
                pair_counter[(symptom, drug)] += 1

    associations = []
    for (symptom, drug), co_count in pair_counter.items():
        if co_count < min_co_count:
            continue
        symptom_count = symptom_counter[symptom]
        drug_count = drug_counter[drug]
        confidence = co_count / symptom_count if symptom_count else 0
        drug_rate = drug_count / total if total else 0
        lift = confidence / drug_rate if drug_rate else 0
        associations.append(
            {
                "symptom": symptom,
                "drug_name": drug,
                "co_count": co_count,
                "symptom_count": symptom_count,
                "drug_count": drug_count,
                "support": round(co_count / total, 4) if total else 0,
                "confidence": round(confidence, 4),
                "lift": round(lift, 4),
            }
        )

    associations.sort(key=lambda item: (item["lift"], item["co_count"], item["confidence"]), reverse=True)
    return associations[:limit]


def summarize_rows(
    rows: list[dict[str, Any]],
    template_catalog: dict[str, dict[str, Any]],
    diagnosis_name: str | None = None,
    diagnosis_syndrome: str | None = None,
) -> dict[str, Any]:
    prescription_count = len(rows)
    inquiry_count = len({row["inquiry_id"] for row in rows})
    drug_counter = Counter()
    template_counter = Counter()
    added_counter = Counter()
    removed_counter = Counter()
    pair_counter = Counter()
    dose_distribution = defaultdict(Counter)

    for row in rows:
        if row.get("matched_template_name"):
            template_counter[row["matched_template_name"]] += 1
        added_counter.update(row.get("added_drugs", []))
        removed_counter.update(row.get("removed_drugs", []))

        names = sorted(drug_set(row.get("drugs", [])))
        drug_counter.update(names)
        pair_counter.update(" + ".join(pair) for pair in itertools.combinations(names, 2))
        for drug in row.get("drugs", []):
            name = clean_drug_name(drug.get("drug_name"))
            dose = dose_key(drug.get("dose"))
            unit = clean_drug_name(drug.get("unit"))
            if name and dose:
                dose_distribution[name][f"{dose}{unit}"] += 1

    summary = {
        "record_count": inquiry_count,
        "prescription_count": prescription_count,
        "top_templates": top_templates(template_counter, prescription_count, template_catalog, 10),
        "match_ratio_summary": summarize_match_ratios(rows),
        "inquiry_template_matches": summarize_inquiry_template_matches(rows),
        "top_drugs": top_counter(drug_counter, prescription_count, "drug_name", 20),
        "core_drugs": [
            item for item in top_counter(drug_counter, prescription_count, "drug_name", 50) if item["rate"] >= 0.3
        ],
        "top_added_drugs": top_counter(added_counter, prescription_count, "drug_name", 20),
        "top_removed_drugs": top_counter(removed_counter, prescription_count, "drug_name", 20),
        "top_drug_pairs": top_counter(pair_counter, prescription_count, "drug_pair", 20),
        "top_symptom_drug_associations": symptom_drug_associations(rows, "drugs", 50),
        "top_symptom_added_drug_associations": symptom_drug_associations(rows, "added_drugs", 50),
        "top_symptom_removed_drug_associations": symptom_drug_associations(rows, "removed_drugs", 50),
        "dose_distribution": {
            name: top_counter(counter, sum(counter.values()), "dose", 8)
            for name, counter in sorted(dose_distribution.items())
        },
    }
    if diagnosis_name is not None:
        summary["diagnosis_name"] = diagnosis_name
    if diagnosis_syndrome is not None:
        summary["diagnosis_syndrome"] = diagnosis_syndrome
    return summary


def build_analysis(records: list[dict[str, Any]], templates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = flatten_prescriptions(records, templates)
    template_catalog = {template.get("template_name", ""): template for template in templates if template.get("template_name")}
    disease_groups = defaultdict(list)
    disease_syndrome_groups = defaultdict(list)

    for row in rows:
        disease_groups[row["diagnosis_name"]].append(row)
        disease_syndrome_groups[(row["diagnosis_name"], row["diagnosis_syndrome"])].append(row)

    syndrome_summaries_by_disease = defaultdict(list)
    for (diagnosis_name, diagnosis_syndrome), group_rows in disease_syndrome_groups.items():
        syndrome_summaries_by_disease[diagnosis_name].append(
            summarize_rows(group_rows, template_catalog, diagnosis_syndrome=diagnosis_syndrome)
        )

    disease_summary = []
    for diagnosis_name, group_rows in disease_groups.items():
        summary = summarize_rows(group_rows, template_catalog, diagnosis_name=diagnosis_name)
        summary["syndrome_summary"] = sorted(
            syndrome_summaries_by_disease[diagnosis_name],
            key=lambda item: item["prescription_count"],
            reverse=True,
        )
        disease_summary.append(summary)

    disease_summary.sort(key=lambda item: item["prescription_count"], reverse=True)

    return {
        "metadata": {
            "record_count": len({record.get("inquiry_id") for record in records}),
            "prescription_count": len(rows),
            "template_count": len(templates),
            "disease_count": len(disease_summary),
        },
        "disease_summary": disease_summary,
        "template_match_details": rows,
    }


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析医生针对同一疾病的开方规律和用药策略。")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_FILE, help="模板方 JSON 文件")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORD_FILE, help="病历处方提取 JSON 文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="分析结果输出 JSON 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json(args.records)
    templates = load_templates(args.templates)
    analysis = build_analysis(records, templates)
    write_json(analysis, args.output)
    print(
        f"Analyzed {analysis['metadata']['prescription_count']} prescriptions "
        f"across {analysis['metadata']['disease_count']} diseases to {args.output}"
    )


if __name__ == "__main__":
    main()
