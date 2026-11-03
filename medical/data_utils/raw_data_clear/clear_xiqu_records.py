# Copyright 2026 The LlamaFactory team.
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

"""清洗 medical/data/二期西区 下的医生病历和处方明细，生成逐条 JSONL 数据集。"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "medical/data/二期西区"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "medical/processed_data"
ALLOWED_CATEGORIES = {"中成药费", "中草药费", "中草药颗粒费", "西药费"}
HERBAL_CATEGORIES = {"中草药费", "中草药颗粒费"}
DRUG_PROCESS_NAMES = {
    "中成药费": "中成药",
    "中草药费": "饮片",
    "中草药颗粒费": "颗粒",
    "西药费": "西药",
}
DRUG_PROCESS_NAMES_BY_COOKING_METHOD = {
    "颗粒": "颗粒",
    "代煎": "饮片",
    "自煎": "饮片",
    "打粉": "饮片",
}

RECORD_REQUIRED_COLUMNS = ("门诊号", "时间", "患者姓名", "主诉", "初步诊断")
PRESCRIPTION_REQUIRED_COLUMNS = (
    "门诊号",
    "处方号",
    "证件号",
    "项目名称",
    "单量",
    "单位",
    "剂数",
    "频次",
    "用法",
    "煎法",
    "分类名称",
    "收费标志",
)


def clean_text(value: Any) -> str:
    """将单元格值转为字符串，清理 Excel 空值标记并保留正文换行。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none", "<na>"} else text


def normalize_outpatient_no(value: Any) -> str:
    """规范门诊号，避免浮点格式及空白导致无法匹配。"""
    return re.sub(r"\.0$", "", clean_text(value)).strip()


def normalize_number(value: Any) -> str:
    """把 Excel 数值格式如 14.0 规范为 14。"""
    text = clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(number)) if number.is_integer() else text


def parse_datetime(value: Any) -> pd.Timestamp | None:
    """解析 Excel 日期时间。"""
    if value is None or clean_text(value) == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def datetime_text(value: Any) -> str:
    parsed = parse_datetime(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed is not None else ""


def visit_date(value: Any) -> Any:
    parsed = parse_datetime(value)
    return parsed.date() if parsed is not None else None


def read_excel_table(path: Path) -> pd.DataFrame:
    """读取二期西区导出的表格：前三行是报表标题/筛选条件，第四行是表头。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    frame = pd.read_excel(path, skiprows=3, dtype=str)
    frame = frame.dropna(how="all").copy()
    frame.columns = [clean_text(column) for column in frame.columns]
    return frame


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少必需列: {', '.join(missing)}")


def _is_blank_cell(value: Any) -> bool:
    return clean_text(value) == ""


def has_history_content(value: Any) -> bool:
    """判断病史字段是否包含肯定信息，而不是空值或否定/未记录占位语。"""
    text = clean_text(value)
    if not text:
        return False

    negative_prefixes = (
        "没有",
        "暂无",
        "暂时无",
        "未有",
        "未见",
        "未发现",
        "未记录",
        "未提供",
        "未询问",
        "未说明",
        "未诉",
        "未述",
        "否认",
        "不详",
        "未知",
        "无",
    )
    labels = ("既往史", "过敏史", "个人史", "家族史")
    relatives = ("父母", "父亲", "母亲", "兄弟姐妹", "兄弟", "姐妹", "直系亲属", "家族成员")
    clauses = re.split(r"[\r\n,，;；。]+", text)

    for clause in clauses:
        normalized = re.sub(r"\s+", "", clause).strip("：:")
        for label in labels:
            if normalized.startswith(label):
                normalized = normalized[len(label) :].lstrip("：:")
                break
        for relative in relatives:
            if normalized.startswith(relative):
                normalized = normalized[len(relative) :].lstrip("：:")
                break
        if not normalized:
            continue

        is_negative = normalized.startswith(negative_prefixes) and not normalized.startswith("无花果")
        if is_negative and not re.search(r"但|不过|曾患|曾有|现有|另有|同时有|并有", normalized):
            continue
        return True

    return False


def classify_diagnosis_by_rule(diagnosis: Any) -> dict[str, str]:
    """从“初步诊断”拆分西医诊断、中医诊断和中医证候，不调用外部模型。"""
    text = clean_text(diagnosis)
    result = {"diagnosis_illness": "", "diagnosis_sickness": "", "diagnosis_disease": ""}
    if not text:
        return result

    western = _extract_labeled_diagnosis(text, "西医诊断")
    chinese = _extract_labeled_diagnosis(text, "中医诊断")
    if not western and not chinese:
        result["diagnosis_illness"] = text
        return result

    sicknesses = []
    syndromes = []
    for text in chinese:
        for item in _split_chinese_diagnoses(text):
            match = re.match(r"^(.*?)[（(]([^（）()]*证)[）)]\s*$", item)
            if match:
                sickness, syndrome = match.groups()
                sickness = re.sub(r"^(?:\d+|[一二三四五六七八九十]+)[.、．]\s*", "", sickness).strip()
                syndrome = syndrome.strip()
                if sickness:
                    sicknesses.append(sickness)
                if syndrome:
                    syndromes.extend(piece.strip() for piece in re.split(r"[、，,；;]+", syndrome) if piece.strip())
            elif item.strip():
                sicknesses.append(re.sub(r"^(?:\d+|[一二三四五六七八九十]+)[.、．]\s*", "", item).strip())

    result["diagnosis_illness"] = "、".join(_dedupe(western))
    result["diagnosis_sickness"] = "、".join(_dedupe(sicknesses))
    result["diagnosis_disease"] = "、".join(_dedupe(syndromes))
    return result


def _extract_labeled_diagnosis(text: str, label: str) -> list[str]:
    pattern = rf"{label}\s*[:：]\s*([\s\S]*?)(?=(?:\n\s*)?(?:西医诊断|中医诊断)\s*[:：]|$)"
    matches = re.findall(pattern, text)
    # Some exported notes repeat a clean diagnosis label after a long copied
    # clinical narrative. Prefer the shortest labeled section in that case.
    nonempty = [match.strip() for match in matches if match.strip()]
    if len(nonempty) > 1:
        matches = [min(nonempty, key=len)]
    parts = []
    for match in matches:
        parts.extend(_split_chinese_diagnoses(match))
    return parts


def _split_chinese_diagnoses(text: str) -> list[str]:
    """拆分中文诊断列表，兼容空格分隔的编号项。"""
    text = re.sub(r"\s+(?=(?:\d+|[一二三四五六七八九十]+)[.、．])", "\n", text.strip())
    pieces = []
    current = []
    bracket_depth = 0
    for character in text:
        if character in "（(":
            bracket_depth += 1
        elif character in "）)":
            bracket_depth = max(0, bracket_depth - 1)
        if bracket_depth == 0 and character in "；;，,、\n":
            piece = "".join(current).strip()
            if piece:
                pieces.append(piece)
            current = []
        else:
            current.append(character)
    piece = "".join(current).strip()
    if piece:
        pieces.append(piece)
    return pieces


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _first_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    for value in frame[column]:
        text = clean_text(value)
        if text:
            return text
    return ""


def _chinese_number(value: str) -> int | None:
    numerals = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if value.isdigit():
        return int(value)
    if value in numerals:
        return numerals[value]
    if len(value) == 2 and value[0] == "十" and value[1] in numerals:
        return 10 + numerals[value[1]]
    if len(value) == 2 and value[1] == "十" and value[0] in numerals:
        return numerals[value[0]] * 10
    return None


def _frequency_count(frequency: str) -> str:
    """从常见频次文字中提取每日次数，用于 use_limit。"""
    text = frequency.strip()
    match = re.search(r"(?:每日|每天|一日|1日)?\s*([0-9]+|[一二两三四五六七八九十]+)\s*次", text)
    if match:
        value = _chinese_number(match.group(1))
        return str(value) if value is not None else match.group(1)
    interval = re.search(r"每\s*([0-9]+)\s*小时", text)
    if interval and int(interval.group(1)) > 0:
        hours = int(interval.group(1))
        return str(max(1, 24 // hours))
    if "一次性" in text or text == "立即":
        return "1"
    return ""


def build_usage_desc(dose_count: Any, frequency: Any) -> str:
    """按剂数和频次生成处方用量说明。"""
    count = normalize_number(dose_count)
    freq = clean_text(frequency)
    parts = []
    if count:
        parts.append(f"共{count}剂")
    if freq:
        parts.append(freq)
    return "，".join(parts)


def build_use_limit(usage_desc: str) -> str:
    """按 usage_desc 生成兼容数据集的用量限制：剂数_每日次数。"""
    if not usage_desc:
        return ""
    count_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*剂", usage_desc)
    frequency = usage_desc.split("，", 1)[1] if "，" in usage_desc else usage_desc
    parts = [normalize_number(count_match.group(1))] if count_match else []
    frequency_count = _frequency_count(frequency)
    if frequency_count:
        parts.append(frequency_count)
    elif frequency and not count_match:
        parts.append(frequency)
    return "_".join(part for part in parts if part)


def prepare_record_data(path: Path) -> pd.DataFrame:
    records = read_excel_table(path)
    require_columns(records, RECORD_REQUIRED_COLUMNS, path)
    records["_mzh"] = records["门诊号"].map(normalize_outpatient_no)
    records["_visit_time"] = records["时间"].map(parse_datetime)
    records["_visit_date"] = records["_visit_time"].map(lambda value: value.date() if value is not None else None)
    records = records[
        records["_mzh"].ne("")
        & records["主诉"].map(lambda value: not _is_blank_cell(value))
    ].copy()
    return records.reset_index(drop=True)


def prepare_prescription_data(path: Path) -> pd.DataFrame:
    prescriptions = read_excel_table(path)
    require_columns(prescriptions, PRESCRIPTION_REQUIRED_COLUMNS, path)
    prescriptions["_mzh"] = prescriptions["门诊号"].map(normalize_outpatient_no)
    prescriptions["_rx_no"] = prescriptions["处方号"].map(clean_text)
    prescriptions["_category"] = prescriptions["分类名称"].map(clean_text)
    prescriptions["_cooking_method"] = prescriptions["煎法"].map(clean_text)
    prescriptions["_drug_process_name"] = prescriptions["_cooking_method"].map(
        DRUG_PROCESS_NAMES_BY_COOKING_METHOD
    )
    prescriptions["_cooking_method_fallback"] = prescriptions["_drug_process_name"].isna()
    prescriptions["_drug_process_name"] = prescriptions["_drug_process_name"].fillna(
        prescriptions["_category"].map(DRUG_PROCESS_NAMES)
    )
    order_values = prescriptions.get("开嘱时间", pd.Series(index=prescriptions.index, dtype="object"))
    charge_values = prescriptions.get("收费时间", pd.Series(index=prescriptions.index, dtype="object"))
    order_dates = pd.to_datetime(order_values, errors="coerce")
    charge_dates = pd.to_datetime(charge_values, errors="coerce")
    fee_dates = order_dates.combine_first(charge_dates)
    prescriptions["_fee_date"] = fee_dates.dt.date
    prescriptions = prescriptions[
        prescriptions["_mzh"].ne("")
        & prescriptions["_category"].isin(ALLOWED_CATEGORIES)
        & prescriptions["收费标志"].map(clean_text).eq("正常")
    ].copy()
    return prescriptions.reset_index(drop=True)


def select_prescriptions_for_record(
    record: pd.Series,
    prescriptions_by_mzh: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """按门诊号关联；同一门诊号跨多个日期时，用就诊日期避免串入其他日期处方。"""
    candidates = prescriptions_by_mzh.get(record["_mzh"])
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    record_date = record.get("_visit_date")
    dated = candidates[candidates["_fee_date"].notna()]
    if record_date is not None and not dated.empty:
        same_day = dated[dated["_fee_date"].eq(record_date)]
        if not same_day.empty:
            return same_day.copy()
        if dated["_fee_date"].nunique() > 1:
            return pd.DataFrame()
        # 只有一个处方日期但与病历日期不一致时，不把它误配给本次病历。
        return pd.DataFrame()

    if record_date is None and dated["_fee_date"].nunique() > 1:
        return pd.DataFrame()
    if not dated.empty:
        return dated.copy()
    return candidates.copy()


def build_drug_entry(row: pd.Series) -> dict[str, Any]:
    return OrderedDict(
        drug_name=clean_text(row.get("项目名称")),
        drug_num=normalize_number(row.get("单量")),
        unit_name=clean_text(row.get("单位")),
        decoction_name="",
    )


def aggregate_prescriptions(rows: pd.DataFrame, treatment_plan: Any) -> list[dict[str, Any]]:
    """按处方号和药品加工类型组装 ps；同一处方中的每个药品保留为一条明细。"""
    if rows.empty:
        return []

    frame = rows.copy()
    frame["_group_rx_no"] = frame.apply(
        lambda row: row["_rx_no"] or f"__row_{row.name}", axis=1
    )
    result = []
    for (rx_no, drug_process_name), group in frame.groupby(
        ["_group_rx_no", "_drug_process_name"], sort=False, dropna=False
    ):
        dose_count = _first_value(group, "剂数")
        frequency = _first_value(group, "频次")
        usage = _first_value(group, "用法")
        usage_desc = build_usage_desc(dose_count, frequency)
        drugs = [build_drug_entry(row) for _, row in group.iterrows() if clean_text(row.get("项目名称"))]
        if not drugs:
            continue

        result.append(
            OrderedDict(
                id=None,
                prescription_order_id="" if str(rx_no).startswith("__row_") else str(rx_no),
                order_sn="" if str(rx_no).startswith("__row_") else str(rx_no),
                usage_type=usage,
                drug_process_name=drug_process_name,
                use_limit=build_use_limit(usage_desc),
                doctor_advice=clean_text(treatment_plan),
                usage_desc=usage_desc,
                prescription_items={"drugList": drugs},
            )
        )
    return result


def build_record(
    row: pd.Series,
    prescription_rows: pd.DataFrame,
    doctor_name: str,
    doctor_id: int | None,
) -> dict[str, Any]:
    visit_time = datetime_text(row.get("时间"))
    complaint = clean_text(row.get("主诉"))
    history_parts = []
    for label, column in (("现病史", "现病史"), ("体格检查", "体格检查"), ("辅助检查", "辅助检查")):
        value = clean_text(row.get(column))
        if value:
            history_parts.append(f"{label}：{value}")

    treatment_plan = clean_text(row.get("治疗计划及建议"))
    diagnoses = classify_diagnosis_by_rule(row.get("初步诊断"))
    is_first = _first_value(prescription_rows, "初复诊")
    old_medical_history = clean_text(row.get("既往史"))
    allergic_history = clean_text(row.get("过敏史"))
    personal_history = clean_text(row.get("个人史"))
    family_history = clean_text(row.get("家族史"))

    return OrderedDict(
        order_sn=normalize_outpatient_no(row.get("门诊号")),
        inquiry_method="门诊",
        is_first=is_first,
        created_at=visit_time,
        start_time=visit_time,
        stop_time="",
        user_info_id=None,
        patient_id=None,
        patient_name=clean_text(row.get("患者姓名")),
        patient_sex=clean_text(row.get("患者性别")),
        patient_age=normalize_number(re.sub(r"[^0-9.]", "", clean_text(row.get("年龄")))),
        patient_idcard=_first_value(prescription_rows, "证件号"),
        patient_mobile="",
        patient_height=None,
        patient_weight=None,
        patient_appeal=complaint,
        doc_ass_stu_appeal=complaint,
        new_medical_history="\n".join(history_parts),
        is_old_medical_history="是" if has_history_content(old_medical_history) else "否",
        old_medical_history=old_medical_history,
        is_allergic_history="是" if has_history_content(allergic_history) else "否",
        allergic_history=allergic_history,
        is_personal_history="是" if has_history_content(personal_history) else "否",
        personal_history=personal_history,
        is_special="",
        special_history="",
        is_child_history="",
        birth_detail="",
        is_marriage_history="",
        is_family_history="是" if has_history_content(family_history) else "否",
        family_history=family_history,
        inspection_report_img=[],
        tongue_face_img=[],
        admin_report_img=[],
        admin_face_img=[],
        admin_face_describe="",
        diagnosis_illness=diagnoses["diagnosis_illness"],
        diagnosis_sickness=diagnoses["diagnosis_sickness"],
        diagnosis_disease=diagnoses["diagnosis_disease"],
        disposition=treatment_plan,
        see_doc_time=f"{visit_time[:10]} 00:00:00" if visit_time else "",
        doctor_id=doctor_id,
        doctor_name=doctor_name,
        ps=aggregate_prescriptions(prescription_rows, treatment_plan),
        im_msg=[],
    )


def discover_doctors(data_dir: Path) -> list[str]:
    names = []
    for record_path in sorted(data_dir.glob("* 病历.xlsx")):
        if record_path.name.startswith("~$"):
            continue
        doctor_name = record_path.stem.removesuffix(" 病历")
        if (data_dir / f"{doctor_name}.xlsx").exists():
            names.append(doctor_name)
    return names


def load_doctor_id_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("医生 ID 映射文件必须是 JSON 对象，例如 {\"于振宣\": 123}")
    return {str(name): int(doctor_id) for name, doctor_id in raw.items() if doctor_id is not None}


def process_doctor(
    doctor_name: str,
    record_path: Path,
    prescription_path: Path,
    output_dir: Path,
    doctor_id: int | None = None,
) -> Path:
    print(f"\n[{doctor_name}] 读取病历: {record_path}")
    records = prepare_record_data(record_path)
    print(f"  有效病历行: {len(records)}，门诊号数: {records['_mzh'].nunique()}")

    print(f"[{doctor_name}] 读取收费处方明细: {prescription_path}")
    prescriptions = prepare_prescription_data(prescription_path)
    print(f"  保留药品明细: {len(prescriptions)}，门诊号数: {prescriptions['_mzh'].nunique()}")
    fallback_count = int(
        (
            prescriptions["_cooking_method_fallback"]
            & prescriptions["_category"].isin(HERBAL_CATEGORIES)
        ).sum()
    )
    if fallback_count:
        print(f"  草药类煎法为空或未识别、按分类名称回退: {fallback_count} 条")

    prescriptions_by_mzh = {mzh: group for mzh, group in prescriptions.groupby("_mzh", sort=False)}
    output_records = []
    unmatched = 0
    for _, record in records.iterrows():
        matching_rows = select_prescriptions_for_record(record, prescriptions_by_mzh)
        if matching_rows.empty:
            unmatched += 1
            continue
        output_records.append(build_record(record, matching_rows, doctor_name, doctor_id))

    if doctor_id is None:
        folder_name = f"doctor_二期西区_{doctor_name}"
    else:
        folder_name = f"doctor_{doctor_id}_{doctor_name}"
    target_dir = output_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "medical_records_ai.jsonl"
    with out_path.open("w", encoding="utf-8") as file:
        for record in output_records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")

    duplicate_order_sn = len(output_records) - len({row["order_sn"] for row in output_records})
    print(f"[{doctor_name}] 已写出 {len(output_records)} 条 -> {out_path}")
    print(f"  无匹配处方的病历行: {unmatched}；重复门诊号行: {duplicate_order_sn}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="清洗二期西区医生病历与收费处方明细，生成 medical_records_ai.jsonl")
    parser.add_argument("--doctorname", "--doctor-name", dest="doctor_name", help="只处理指定医生；默认处理目录中的全部医生")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="二期西区源数据目录")
    parser.add_argument("--record-path", "--record_path", type=Path, help="指定病历文件，需要同时指定处方文件和医生姓名")
    parser.add_argument("--prescription-path", "--prescription_path", type=Path, help="指定收费处方明细文件")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="处理结果根目录")
    parser.add_argument("--doctor-id-map", type=Path, help="医生 ID 映射 JSON，键为医生姓名、值为医生 ID")
    return parser


def main() -> list[Path]:
    args = build_parser().parse_args()
    id_map = load_doctor_id_map(args.doctor_id_map)

    if args.record_path or args.prescription_path:
        if not args.doctor_name or not args.record_path or not args.prescription_path:
            raise SystemExit("指定文件时必须同时提供 --doctorname、--record-path 和 --prescription-path")
        doctors = [(args.doctor_name, args.record_path, args.prescription_path)]
    elif args.doctor_name:
        doctors = [
            (
                args.doctor_name,
                args.data_dir / f"{args.doctor_name} 病历.xlsx",
                args.data_dir / f"{args.doctor_name}.xlsx",
            )
        ]
    else:
        doctors = [
            (name, args.data_dir / f"{name} 病历.xlsx", args.data_dir / f"{name}.xlsx")
            for name in discover_doctors(args.data_dir)
        ]

    if not doctors:
        raise SystemExit(f"在 {args.data_dir} 下没有找到“医生名 病历.xlsx”与“医生名.xlsx”配对文件")

    return [
        process_doctor(name, record_path, prescription_path, args.output_dir, id_map.get(name))
        for name, record_path, prescription_path in doctors
    ]


if __name__ == "__main__":
    main()
