"""
史大卓 - 线下诊疗数据清洗脚本
=============================

数据源：
- 线下诊疗数据采集 (medical/data/shidazhuo/线下诊疗数据采集_20260603151959.xlsx)
  每行 = 一次门诊就诊的病历主诉/病史/诊断/治疗建议，按门诊号去重。
- 门诊病人处方明细 (medical/data/shidazhuo/门诊病人(处方明细)_20260608173446.xlsx)
  每行 = 一条收费/处方项目；项目代码首字母表示药品类型：
      C -> 饮片，X -> 西药，K -> 颗粒，Z -> 中成药

输出：
medical/processed_data/shidazhuo_record_YYYYMMDD.json
结构参考 medical/data/wuweiping_record_20260525.json：
    序号 / 就诊科室 / 门诊号 / 姓名 / 性别 / 年龄 /
    主诉 / 现病史 / 过敏史 / 既往史 / 体格检查 / 辅助检查 /
    中医诊断 / 西医诊断 / 处方明细 (ps) / 频次 / 治疗计划及建议
其中 ps 按 (门诊号, 处方号, 药品类别) 分组生成饮片/西药/颗粒/中成药处方。

用法:
    .venv/bin/python medical/data_utils/clear_shidazhuo_record.py
"""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD_XLSX = PROJECT_ROOT / "medical/data/shidazhuo/线下诊疗数据采集_20260603151959.xlsx"
DEFAULT_PRESCRIPTION_XLSX = PROJECT_ROOT / "medical/data/shidazhuo/门诊病人(处方明细)_20260608173446.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "medical/processed_data"

# --------------------------------------------------------------------------- #
# 表1：病历主表 字段映射
# skiprows=3 后第 0 行就是表头
# --------------------------------------------------------------------------- #
RECORD_COLUMNS: list[str] = [
    "idx", "dept", "doctor", "visit_time", "name", "sex", "age", "mzh",
    "chief_complaint", "present_illness", "allergy_history", "past_history",
    "physical_exam", "auxiliary_exam", "diagnosis", "treatment_plan",
]

# --------------------------------------------------------------------------- #
# 表2：处方明细 字段映射
# --------------------------------------------------------------------------- #
PRESCRIPTION_COLUMNS: list[str] = [
    "idx", "dept", "fee_type", "rx_no", "mzh", "name", "id_no", "sex", "age",
    "bill_type", "bill_no", "item_code", "item_name", "spec", "unit_price", "qty", "dose",
    "doc_advice", "freq", "order_note", "total", "dose_count", "usage", "count", "unit",
    "total_amt", "category", "doctor", "exec_dept", "charge_flag", "order_time",
    "charge_time", "operator", "checkout_time", "remark",
]

# 项目代码首字母 -> (处方类别名 / 药品来源类别名)
ITEM_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "C": ("中药饮片", "饮片"),
    "X": ("西药", "西药"),
    "K": ("颗粒", "颗粒"),
    "Z": ("中成药", "中成药"),
}


# --------------------------------------------------------------------------- #
# 读取 + 预处理
# --------------------------------------------------------------------------- #
def read_record_xlsx(path: Path = DEFAULT_RECORD_XLSX) -> pd.DataFrame:
    """读取表1（病历主表），并规范字段。"""
    df = pd.read_excel(path, skiprows=3)
    df.columns = RECORD_COLUMNS
    df["mzh"] = (
        df["mzh"]
        .astype("Int64")  # 容忍 NaN 的整型
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    # 整型字段处理缺失（age 字段带"岁"字，需先剥掉）
    df["idx"] = df["idx"].astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
    df["age"] = (
        df["age"]
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df["visit_time"] = pd.to_datetime(df["visit_time"], errors="coerce")
    df = df.dropna(subset=["mzh"])
    df = df[df["mzh"] != ""]
    return df.reset_index(drop=True)


def read_prescription_xlsx(path: Path = DEFAULT_PRESCRIPTION_XLSX) -> pd.DataFrame:
    """读取表2（处方明细），过滤出药品类条目，并规范字段。"""
    df = pd.read_excel(path, skiprows=3)
    df.columns = PRESCRIPTION_COLUMNS
    df["mzh"] = (
        df["mzh"]
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    df["rx_no"] = df["rx_no"].astype(str).str.strip().replace({"nan": ""})
    df["item_code"] = df["item_code"].astype(str).str.strip()
    # 仅保留药品类：项目代码首字母 ∈ {C, X, K, Z}
    df["item_prefix"] = df["item_code"].str[:1]
    df = df[df["item_prefix"].isin(ITEM_PREFIX_MAP.keys())].copy()
    # 剂量字段转字符串，保留原始单位（"20g" / "47.5mg"）
    df["dose"] = df["dose"].astype(str).str.strip()
    # 剂数 / 频次 / 用法
    for col in ("dose_count", "freq", "usage"):
        df[col] = df[col].astype(str).str.strip().replace({"nan": ""})
    # 剂数整理：浮点 .0 -> 整数字符串
    df["dose_count"] = df["dose_count"].apply(_normalize_int_str)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 处方聚合
# --------------------------------------------------------------------------- #
def build_drug_entry(row: pd.Series) -> dict[str, Any]:
    """把处方明细中一行 C/X/K/Z 转成 drugList 中的一条药物。"""
    prefix = row["item_prefix"]
    _, process_name = ITEM_PREFIX_MAP[prefix]
    return OrderedDict(
        show_name=row["item_name"],
        drug_name=row["item_name"],
        unit_name=str(row.get("unit", "")).strip() or "g",
        spec_name=str(row["spec"]).strip(),
        drug_num=str(row["dose"]).strip(),       # 每次剂量（带单位，例如 "20g"）
        drug_weight=_extract_dose_value(row["dose"]),
        sale_price=str(row["unit_price"]).strip(),
        item_code=row["item_code"],
        category=row["category"],
        drug_process_name=process_name,
    )


def _extract_dose_value(dose: Any) -> float | None:
    """从剂量字符串中提取数值（如 '20g' -> 20.0, '47.5mg' -> 47.5）。"""
    if dose is None:
        return None
    s = str(dose).strip()
    if not s or s.lower() == "nan":
        return None
    import re
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _normalize_int_str(v: Any) -> str:
    """把 '5.0' / '5' / 5.0 -> '5'，非数值字符串原样返回。"""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def aggregate_prescriptions(prescription_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """
    按 (门诊号) 聚合处方，输出 {mzh: [ps, ps, ...]}.
    同一 (门诊号, 处方号, 项目首字母) 归并到一张处方。
    """
    out: dict[str, list[dict[str, Any]]] = {}

    # 按 门诊号 + 处方号 + 药品类别 分组
    grouped = prescription_df.groupby(
        ["mzh", "rx_no", "item_prefix"], dropna=False, sort=False
    )

    for (mzh, rx_no, prefix), group in grouped:
        if not mzh:
            continue
        _, process_name = ITEM_PREFIX_MAP[prefix]

        # 同一组内频次/用法/剂数应当一致；取第一条非空
        freq = next((v for v in group["freq"] if v and v != "nan"), "")
        usage = next((v for v in group["usage"] if v and v != "nan"), "")
        dose_count = next((v for v in group["dose_count"] if v and v != "nan"), "")
        doc_advice = next((v for v in group["doc_advice"] if v and v != "nan"), "")

        drug_list = [build_drug_entry(row) for _, row in group.iterrows()]

        ps_entry = OrderedDict(
            prescription_order_id=rx_no,
            order_sn=rx_no,
            usage_type=_infer_usage_type(prefix, usage),
            drug_process_name=process_name,
            use_limit=_build_use_limit(dose_count, freq),
            doctor_advice=doc_advice,
            usage_desc=_build_usage_desc(dose_count, freq, usage),
            prescription_items={"drugList": drug_list},
            freq=freq,
            usage=usage,
            dose_count=dose_count,
        )
        out.setdefault(mzh, []).append(ps_entry)
    return out


def _infer_usage_type(prefix: str, usage: str) -> str:
    """根据处方类型和用法推断内服/外用等。"""
    u = str(usage).strip()
    if u and u != "nan":
        return u
    return {"C": "内服", "K": "内服", "X": "内服", "Z": "内服"}.get(prefix, "")


def _build_use_limit(dose_count: Any, freq: Any) -> str:
    """构造 use_limit 字段，如 8_1_2。"""
    parts = []
    for v in (dose_count, freq):
        s = str(v).strip()
        if s and s != "nan":
            parts.append(s)
    return "_".join(parts) if parts else ""


def _build_usage_desc(dose_count: Any, freq: Any, usage: Any) -> str:
    """生成 usage_desc 自然语言描述。"""
    pieces = []
    for v in (dose_count, freq, usage):
        s = str(v).strip()
        if s and s != "nan":
            pieces.append(s)
    return "，".join(pieces) if pieces else ""


# --------------------------------------------------------------------------- #
# 主表 -> JSON 记录
# --------------------------------------------------------------------------- #
def build_record_dict(
    row: pd.Series,
    prescription_map: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """构造一条最终的病历 JSON。"""
    mzh = row["mzh"]
    visit_time = row["visit_time"]
    visit_time_str = (
        visit_time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(visit_time, pd.Timestamp) and not pd.isna(visit_time)
        else ""
    )

    # 拆分初步诊断为「中医诊断 / 西医诊断」
    tcm_diag, west_diag = _split_diagnosis(str(row["diagnosis"]))

    ps_list = prescription_map.get(mzh, [])

    return OrderedDict(
        idx=str(row.get("idx", "")).strip(),
        order_sn=f"ZX{visit_time_str.replace('-', '').replace(' ', '').replace(':', '')}"
        if visit_time_str
        else "",
        inquiry_method="",
        is_first="",
        created_at=visit_time_str,
        start_time=visit_time_str,
        stop_time="",
        user_info_id="",
        patient_id="",
        patient_name=str(row["name"]).strip(),
        patient_sex=str(row["sex"]).strip(),
        patient_age=_extract_age(row["age"]),
        patient_idcard="",
        patient_mobile="",
        patient_height="",
        patient_weight="",
        patient_appeal="",
        chief_complaint=str(row["chief_complaint"]).strip(),       # 主诉
        doc_ass_stu_appeal=str(row["chief_complaint"]).strip(),
        new_medical_history=str(row["present_illness"]).strip(),   # 现病史
        is_old_medical_history="",
        old_medical_history=str(row["past_history"]).strip(),      # 既往史
        is_allergic_history="",
        allergic_history=str(row["allergy_history"]).strip(),      # 过敏史
        is_personal_history="",
        personal_history="",
        is_special="",
        special_history="",
        is_child_history="",
        birth_detail="",
        is_marriage_history="",
        is_family_history="",
        family_history="",
        inspection_report_img=[],
        tongue_face_img=[],
        admin_report_img=[],
        admin_face_img=[],
        admin_face_describe="",
        diagnosis_illness=west_diag,                               # 西医诊断
        diagnosis_disease=tcm_diag,                               # 中医诊断
        diagnosis_sickness="",
        physical_exam=str(row["physical_exam"]).strip(),           # 体格检查
        auxiliary_exam=str(row["auxiliary_exam"]).strip(),         # 辅助检查
        treatment_plan=str(row["treatment_plan"]).strip(),         # 治疗计划及建议
        disposition=str(row["treatment_plan"]).strip(),
        im_msg=[],
        ps=ps_list,
    )


def _extract_age(age: Any) -> str:
    """从 '63岁' 或 63 中提取纯数字字符串。"""
    s = str(age).strip()
    if not s or s == "<NA>" or s.lower() == "nan":
        return ""
    import re
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def _split_diagnosis(diag: Any) -> tuple[str, str]:
    """把 '初步诊断' 字段拆成 (中医诊断, 西医诊断)。"""
    s = str(diag or "").strip()
    if not s:
        return "", ""
    tcm, west = "", ""
    import re
    m_west = re.search(r"西医诊断[:：]?\s*([\s\S]+?)(?:\n|$)", s)
    m_tcm = re.search(r"中医诊断[:：]?\s*([\s\S]+?)(?:\n|$)", s)
    if m_west:
        west = m_west.group(1).strip()
    if m_tcm:
        tcm = m_tcm.group(1).strip()
    if not tcm and not west:
        # 没有标记的话，整段放在西医诊断
        west = s
    return tcm, west


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(
    record_path: Path = DEFAULT_RECORD_XLSX,
    prescription_path: Path = DEFAULT_PRESCRIPTION_XLSX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 读取病历主表: {record_path}")
    record_df = read_record_xlsx(record_path)
    print(f"      共 {len(record_df)} 条记录，唯一门诊号 {record_df['mzh'].nunique()}")

    print(f"[2/4] 读取处方明细表: {prescription_path}")
    prescription_df = read_prescription_xlsx(prescription_path)
    print(f"      共 {len(prescription_df)} 条药品条目")
    print("      类别分布:")
    for prefix, (cat_name, _) in ITEM_PREFIX_MAP.items():
        n = (prescription_df["item_prefix"] == prefix).sum()
        if n:
            print(f"        {prefix} -> {cat_name}: {n}")

    print("[3/4] 按门诊号聚合处方...")
    prescription_map = aggregate_prescriptions(prescription_df)
    print(f"      涉及 {len(prescription_map)} 个门诊号")

    print("[4/4] 生成最终 JSON...")
    records: list[dict[str, Any]] = []
    matched = 0
    for _, row in record_df.iterrows():
        mzh = row["mzh"]
        if mzh in prescription_map:
            matched += 1
        records.append(build_record_dict(row, prescription_map))

    today = datetime.now().strftime("%Y%m%d")
    out_path = output_dir / f"shidazhuo_record_{today}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已写出 {len(records)} 条记录 -> {out_path}")
    print(f"   其中 {matched} 条匹配到处方明细")

    # 简单打印一条样例
    if records:
        sample = records[0]
        print("\n=== 样例 (第一条) ===")
        slim = {
            "idx": sample["idx"],
            "chief_complaint": sample["chief_complaint"],
            "new_medical_history": sample["new_medical_history"],
            "diagnosis_illness": sample["diagnosis_illness"],
            "diagnosis_disease": sample["diagnosis_disease"],
            "treatment_plan": sample["treatment_plan"],
            "ps_count": len(sample["ps"]),
            "ps_process": [p["drug_process_name"] for p in sample["ps"]],
        }
        print(json.dumps(slim, ensure_ascii=False, indent=2))

    return out_path


if __name__ == "__main__":
    main()