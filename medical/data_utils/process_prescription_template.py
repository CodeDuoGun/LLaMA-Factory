#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理吴卫平医生模板处方数据 medical/data/wuweiping医生模板处方.xlsx
将 Excel 数据转换为结构化 JSON，便于与真实处方进行匹配或训练使用。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import openpyxl


DEFAULT_INPUT_FILE = Path("medical/data/wuweiping医生模板处方.xlsx")
DEFAULT_OUTPUT_FILE = Path("medical/data/wuweiping_template_prescription.json")


def read_excel(file_path: Path) -> list[dict]:
    """读取 Excel 文件，返回所有非空行（跳过表头）。"""
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active

    headers = [cell.value for cell in ws[1]]
    expected_headers = ["模板id", "模板名称", "药态", "药品名称", "单位", "数量"]
    if headers[: len(expected_headers)] != expected_headers:
        raise ValueError(f"表头不符合预期，期望: {expected_headers}，实际: {headers}")

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(v is None for v in row):
            continue
        rows.append(
            {
                "template_id": int(row[0]) if row[0] is not None else None,
                "template_name": str(row[1]).strip() if row[1] is not None else "",
                "drug_form": str(row[2]).strip() if row[2] is not None else "",
                "drug_name": str(row[3]).strip() if row[3] is not None else "",
                "unit": str(row[4]).strip() if row[4] is not None else "",
                "quantity": int(row[5]) if row[5] is not None else None,
            }
        )
    return rows


def group_by_template(rows: list[dict]) -> list[dict]:
    """按模板id分组，汇总每个模板下的药品。"""
    templates = defaultdict(lambda: {"template_id": None, "template_name": "", "drugs": []})

    for row in rows:
        tid = row["template_id"]
        if tid is None:
            continue
        templates[tid]["template_id"] = tid
        templates[tid]["template_name"] = row["template_name"]
        templates[tid]["drugs"].append(
            {
                "drug_name": row["drug_name"],
                "drug_form": row["drug_form"],
                "quantity": row["quantity"],
                "unit": row["unit"],
            }
        )

    return list(templates.values())


def summarize_by_usage(templates: list[dict]) -> dict[str, list[dict]]:
    """按药态（内服/外用）分类模板。"""
    internal_templates = []
    external_templates = []
    powder_templates = []

    for t in templates:
        drug_forms = {d["drug_form"] for d in t["drugs"]}

        if "饮片" in drug_forms and len(drug_forms) == 1:
            internal_templates.append(t)
        elif "颗粒" in drug_forms and len(drug_forms) == 1:
            external_templates.append(t)
        elif "粉剂" in drug_forms and len(drug_forms) == 1:
            powder_templates.append(t)
        elif "饮片" in drug_forms:
            internal_templates.append(t)
        else:
            external_templates.append(t)

    return {
        "内服_饮片": internal_templates,
        "外用_颗粒": external_templates,
        "外用_粉剂": powder_templates,
    }


def build_prescription_text(t: dict) -> str:
    """将模板转换为可读处方文本。"""
    lines = [f"【{t['template_name']}】"]
    for drug in t["drugs"]:
        qty = f"{drug['quantity']}{drug['unit']}" if drug["quantity"] else ""
        lines.append(f"  - {drug['drug_name']} {qty}".rstrip())
    return "\n".join(lines)


def find_template_by_name(
    templates: list[dict], name: str
) -> list[dict]:
    """根据模板名称关键词模糊搜索模板。"""
    name_lower = name.lower()
    return [t for t in templates if name_lower in t["template_name"].lower()]


def match_template_with_record(
    record_internal_prescriptions: list[dict], templates: list[dict]
) -> list[dict]:
    """根据真实处方中的药品，匹配最相似的模板处方。"""
    from difflib import SequenceMatcher

    results = []
    for prescription in record_internal_prescriptions:
        drug_names = {d["drug_name"] for d in prescription.get("drugs", []) if d.get("drug_name")}
        best_match = None
        best_ratio = 0.0

        for t in templates:
            t_drug_names = {d["drug_name"] for d in t["drugs"] if d.get("drug_name")}
            if not t_drug_names:
                continue

            common = len(drug_names & t_drug_names)
            total = len(drug_names | t_drug_names)
            ratio = common / total if total > 0 else 0.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = t

        results.append(
            {
                "prescription_order_id": prescription.get("prescription_order_id"),
                "matched_template": best_match,
                "match_ratio": round(best_ratio, 3),
            }
        )

    return results


def write_json(data: list[dict] | dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="处理吴卫平医生模板处方 Excel 文件。")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_FILE, help="输入 Excel 文件路径"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--record", type=Path, default=None, help="（可选）真实处方 JSON，用于模板匹配"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_excel(args.input)
    templates = group_by_template(rows)
    by_usage = summarize_by_usage(templates)

    print(f"共读取 {len(rows)} 条药品记录")
    print(f"共 {len(templates)} 个模板处方")
    print(f"  - 内服（饮片）: {len(by_usage['内服_饮片'])} 个")
    print(f"  - 外用（颗粒）: {len(by_usage['外用_颗粒'])} 个")
    print(f"  - 外用（粉剂）: {len(by_usage['外用_粉剂'])} 个")

    if args.record:
        record_data = json.loads(args.record.read_text(encoding="utf-8"))
        matches = match_template_with_record(
            record_data.get("internal_prescriptions", []), templates
        )
        output = {
            "templates": templates,
            "by_usage": by_usage,
            "matches": matches,
        }
    else:
        output = {
            "templates": templates,
            "by_usage": by_usage,
        }

    write_json(output, args.output)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
