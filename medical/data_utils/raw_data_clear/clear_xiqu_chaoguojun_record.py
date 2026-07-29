# ruff: noqa: D205, D212, D415
"""
史大卓 - 线下诊疗数据清洗脚本
=============================

数据源：
- 线下诊疗数据采集 (medical/data/xiqu_shidazhuo/线下诊疗数据采集_20260603151959(1).xlsx)
  每行 = 一次门诊就诊的病历主诉/病史/诊断/治疗建议，按门诊号去重。
- 门诊病人处方明细 (medical/data/xiqu_shidazhuo/史大卓处方明细.xlsx)
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
    python medical/data_utils/raw_data_clear/clear_xiqu_record.py \
        --doctorname shidazhuo \
        --record-path "medical/data/xiqu_shidazhuo/线下诊疗数据采集_20260603151959(1).xlsx" \
        --prescription-path "medical/data/xiqu_shidazhuo/史大卓处方明细.xlsx"
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------- #
# 路径配置
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCTORNAME = "shidazhuo"
DEFAULT_RECORD_XLSX = PROJECT_ROOT / "medical/data/xiqu_shidazhuo/线下诊疗数据采集_20260603151959(1).xlsx"
DEFAULT_PRESCRIPTION_XLSX = PROJECT_ROOT / "medical/data/xiqu_shidazhuo/史大卓处方明细.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "medical/processed_data"

# --------------------------------------------------------------------------- #
# 表1：病历主表 字段映射
# skiprows=3 后第 0 行就是表头
# --------------------------------------------------------------------------- #
RECORD_COLUMNS: list[str] = [
    "idx",
    "dept",
    "doctor",
    "visit_time",
    "name",
    "sex",
    "age",
    "mzh",
    "chief_complaint",
    "present_illness",
    "allergy_history",
    "past_history",
    "physical_exam",
    "auxiliary_exam",
    "diagnosis",
    "treatment_plan",
]

# --------------------------------------------------------------------------- #
# 表2：处方明细 字段映射
# --------------------------------------------------------------------------- #
PRESCRIPTION_COLUMNS: list[str] = [
    "idx",
    "dept",
    "visit_type",
    "fee_type",
    "rx_no",
    "mzh",
    "name",
    "id_no",
    "sex",
    "age",
    "bill_type",
    "bill_no",
    "item_code",
    "item_name",
    "spec",
    "unit_price",
    "qty",
    "dose",
    "doc_advice",
    "freq",
    "order_note",
    "total",
    "dose_count",
    "usage",
    "count",
    "unit",
    "total_amt",
    "category",
    "doctor",
    "exec_dept",
    "charge_flag",
    "order_time",
    "charge_time",
    "operator",
    "checkout_time",
    "remark",
]

# 项目代码首字母 -> (处方类别名 / 药品来源类别名)
ITEM_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "C": ("中药饮片", "饮片"),
    "X": ("西药", "西药"),
    "K": ("颗粒", "颗粒"),
    "Z": ("中成药", "中成药"),
}

DIAGNOSIS_CLASSIFICATION_SYSTEM_PROMPT = """
你是严谨的中西医门诊诊断结构化助手。
请只根据输入的“初步诊断”原文做字段分类，不要补充、推测或改写诊断。
输出必须是 JSON 对象，字段固定为：
- diagnosis_illness: 西医诊断
- diagnosis_sickness: 中医诊断
- diagnosis_disease: 中医证候

规则：
1. “西医诊断”只放现代医学诊断，如干眼症、结膜炎、黄斑变性。
2. “中医诊断”只放中医病名，如白涩病、视瞻昏渺、暴盲。
3. “中医证候”只放证候/证型，如肝经湿热证、肝肾不足证。
4. 如果中医诊断写作“白涩病(肝经湿热证)”，应拆为中医诊断“白涩病”、中医证候“肝经湿热证”。
5. 多个结果用中文顿号“、”连接。
6. 原文没有对应信息时填空字符串。
7. 只输出 JSON，不要 Markdown，不要解释。
""".strip()

DIAGNOSIS_KEYS = ("diagnosis_illness", "diagnosis_sickness", "diagnosis_disease")


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
        df["age"].astype(str).str.replace("<NA>", "", regex=False).str.replace(r"\.0$", "", regex=True).str.strip()
    )

    df["visit_time"] = pd.to_datetime(df["visit_time"], errors="coerce")
    df = df.dropna(subset=["mzh"])
    df = df[df["mzh"] != ""]
    # 主诉缺失的行不能构成有效病历。
    df = df[~df["chief_complaint"].apply(_is_blank_cell)]
    return df.reset_index(drop=True)


def read_prescription_xlsx(path: Path = DEFAULT_PRESCRIPTION_XLSX) -> pd.DataFrame:
    """读取表2（处方明细），仅保留正常收费条目，并规范字段。"""
    df = pd.read_excel(path, skiprows=3)
    df.columns = PRESCRIPTION_COLUMNS
    df["charge_flag"] = df["charge_flag"].fillna("").astype(str).str.strip()
    df = df[df["charge_flag"] == "正常"].copy()
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
    # 项目代码首字母 ∈ {C, X, K, Z} 的条目会在处方聚合时保留。
    df["item_prefix"] = df["item_code"].str[:1]
    # 剂量字段转字符串，保留原始单位（"20g" / "47.5mg"）
    df["dose"] = df["dose"].astype(str).str.strip()
    # 剂数 / 频次 / 用法
    for col in ("dose_count", "freq", "usage"):
        df[col] = df[col].astype(str).str.strip().replace({"nan": ""})
    df["visit_type"] = df["visit_type"].fillna("").astype(str).str.strip().replace({"nan": ""})
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
        drug_num=str(row["dose"]).strip(),  # 每次剂量（带单位，例如 "20g"）
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
    drug_df = prescription_df[prescription_df["item_prefix"].isin(ITEM_PREFIX_MAP)].copy()

    # 按 门诊号 + 处方号 + 药品类别 分组
    grouped = drug_df.groupby(["mzh", "rx_no", "item_prefix"], dropna=False, sort=False)

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


def build_visit_type_map(prescription_df: pd.DataFrame) -> dict[str, str]:
    """按门诊号获取“初复诊”信息。"""
    visit_type_map: dict[str, str] = {}
    for mzh, group in prescription_df.groupby("mzh", sort=False):
        visit_type = next((value for value in group["visit_type"] if value), "")
        visit_type_map[mzh] = visit_type
    return visit_type_map


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
def clean_output_value(value: Any) -> Any:
    """递归将输出结构中的缺失值转为空字符串。"""
    if isinstance(value, dict):
        return {key: clean_output_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_output_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clean_output_value(item) for item in value)
    if isinstance(value, str):
        return "" if value.strip().lower() in {"nan", "nat", "none", "<na>"} else value
    try:
        return "" if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _is_blank_cell(value: Any) -> bool:
    """判断 Excel 单元格是否为空，包括常见空值字符串。"""
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "nat", "none", "<na>"}


def build_record_dict(
    row: pd.Series,
    prescription_map: dict[str, list[dict[str, Any]]],
    visit_type_map: dict[str, str] | None = None,
    diagnosis_result: dict[str, str] | None = None,
) -> dict[str, Any]:
    """构造一条最终的病历 JSON。"""
    mzh = row["mzh"]
    visit_time = row["visit_time"]
    visit_time_str = (
        visit_time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(visit_time, pd.Timestamp) and not pd.isna(visit_time)
        else ""
    )

    diagnosis_result = diagnosis_result or classify_diagnosis_by_rule(row["diagnosis"])

    treatment_plan = clean_output_value(row.get("treatment_plan", ""))
    ps_list = copy.deepcopy(prescription_map.get(mzh, []))
    for prescription in ps_list:
        prescription["doctor_advice"] = treatment_plan

    record = OrderedDict(
        idx=str(row.get("idx", "")).strip(),
        order_sn=f"ZX{visit_time_str.replace('-', '').replace(' ', '').replace(':', '')}" if visit_time_str else "",
        inquiry_method="",
        is_first=(visit_type_map or {}).get(mzh, ""),
        created_at=visit_time_str,
        start_time=visit_time_str,
        see_doc_time=visit_time_str,
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
        chief_complaint=str(row["chief_complaint"]).strip(),  # 主诉
        doc_ass_stu_appeal=str(row["chief_complaint"]).strip(),
        new_medical_history=str(row["present_illness"]).strip(),  # 现病史
        is_old_medical_history="",
        old_medical_history=str(row["past_history"]).strip(),  # 既往史
        is_allergic_history="",
        allergic_history=str(row["allergy_history"]).strip(),  # 过敏史
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
        diagnosis_illness=diagnosis_result["diagnosis_illness"],  # 西医诊断
        diagnosis_sickness=diagnosis_result["diagnosis_sickness"],  # 中医诊断
        diagnosis_disease=diagnosis_result["diagnosis_disease"],  # 中医证候
        physical_exam=str(row["physical_exam"]).strip(),  # 体格检查
        auxiliary_exam=str(row["auxiliary_exam"]).strip(),  # 辅助检查
        treatment_plan=treatment_plan,  # 治疗计划及建议
        disposition=treatment_plan,
        im_msg=[],
        ps=ps_list,
    )
    return clean_output_value(record)


def _extract_age(age: Any) -> str:
    """从 '63岁' 或 63 中提取纯数字字符串。"""
    s = str(age).strip()
    if not s or s == "<NA>" or s.lower() == "nan":
        return ""
    import re

    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def classify_diagnosis_by_rule(diag: Any) -> dict[str, str]:
    """用规则从“初步诊断”中兜底拆分西医诊断、中医诊断、中医证候。"""
    s = "" if _is_blank_cell(diag) else str(diag).strip()
    result = dict.fromkeys(DIAGNOSIS_KEYS, "")
    if not s:
        return result

    west_parts = _extract_labeled_diagnosis(s, "西医诊断")
    tcm_parts = _extract_labeled_diagnosis(s, "中医诊断")
    if not west_parts and not tcm_parts:
        result["diagnosis_illness"] = s
        return result

    tcm_names: list[str] = []
    syndromes: list[str] = []
    for item in tcm_parts:
        tcm_name, syndrome = _split_tcm_diagnosis_and_syndrome(item)
        if tcm_name:
            tcm_names.append(tcm_name)
        if syndrome:
            syndromes.append(syndrome)

    syndromes = _dedupe_keep_order(syndromes)
    result["diagnosis_illness"] = "、".join(
        _dedupe_keep_order(_clean_west_diagnosis(item, syndromes) for item in west_parts)
    )
    result["diagnosis_sickness"] = "、".join(_dedupe_keep_order(tcm_names))
    result["diagnosis_disease"] = "、".join(syndromes)
    return result


def classify_diagnosis_with_llm(diag: Any, model: str) -> dict[str, str]:
    """调用 OpenAI 兼容 LLM 对“初步诊断”字段分类，失败时回退到规则拆分。"""
    rule_result = classify_diagnosis_by_rule(diag)
    if _is_blank_cell(diag):
        return rule_result

    try:
        from medical.config import config
        from openai import OpenAI

        base_url = (
            os.getenv("DASHSCOPE_BASE_URL")
            or str(getattr(config, "DASHSCOPE_BASE_URL", "") or "")
            or str(getattr(config, "QWEN_API_URL", "") or "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if base_url.rstrip("/").endswith("/chat/completions"):
            base_url = base_url.rsplit("/chat/completions", 1)[0]

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY") or str(getattr(config, "DASHSCOPE_API_KEY", "") or ""),
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DIAGNOSIS_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"初步诊断：\n{str(diag).strip()}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json_object(content)
        return _validate_diagnosis_result(parsed, fallback=rule_result)
    except Exception as exc:
        print(f"      ⚠️ LLM 诊断分类失败，回退到规则拆分。初步诊断={diag}: {exc}")
        return rule_result


def classify_diagnoses_with_llm(diagnoses: list[str], model: str) -> dict[str, dict[str, str]]:
    """批量调用 LLM 分类诊断，返回 {diagnosis_text: diagnosis_result}。"""
    fallback_map = {diagnosis: classify_diagnosis_by_rule(diagnosis) for diagnosis in diagnoses}
    if not diagnoses:
        return fallback_map

    try:
        from medical.config import config
        from openai import OpenAI

        diagnosis_payload = [
            {"idx": idx, "text": text} for idx, text in enumerate(diagnoses, start=1)
        ]
        base_url = (
            os.getenv("DASHSCOPE_BASE_URL")
            or str(getattr(config, "DASHSCOPE_BASE_URL", "") or "")
            or str(getattr(config, "QWEN_API_URL", "") or "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if base_url.rstrip("/").endswith("/chat/completions"):
            base_url = base_url.rsplit("/chat/completions", 1)[0]

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY") or str(getattr(config, "DASHSCOPE_API_KEY", "") or ""),
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DIAGNOSIS_CLASSIFICATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "请批量分类以下初步诊断，返回 JSON：\n"
                        '{"diagnoses":[{"idx":1,"diagnosis_illness":"",'
                        '"diagnosis_sickness":"","diagnosis_disease":""}]}\n'
                        "idx 必须和输入一致。\n\n"
                        f"{json.dumps(diagnosis_payload, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json_object(content)
        items = parsed.get("diagnoses", [])
        if not isinstance(items, list):
            return fallback_map

        out = fallback_map.copy()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                diagnosis = diagnoses[int(item.get("idx")) - 1]
            except (TypeError, ValueError, IndexError):
                continue
            out[diagnosis] = _validate_diagnosis_result(item, fallback=fallback_map[diagnosis])
        return out
    except Exception as exc:
        print(f"      ⚠️ LLM 批量诊断分类失败，回退到规则拆分。批量大小={len(diagnoses)}: {exc}")
        return fallback_map


def build_diagnosis_map(
    record_df: pd.DataFrame,
    use_llm: bool,
    llm_model: str,
    batch_size: int = 20,
) -> dict[str, dict[str, str]]:
    """按诊断原文分类并缓存，返回 {diagnosis_text: diagnosis_result}。"""
    diagnosis_map: dict[str, dict[str, str]] = {}
    diagnosis_values = [str(value).strip() for value in record_df["diagnosis"] if not _is_blank_cell(value)]
    unique_values = _dedupe_keep_order(diagnosis_values)
    if not unique_values:
        return diagnosis_map

    print(f"      需要分类的唯一初步诊断 {len(unique_values)} 条，LLM={'开启' if use_llm else '关闭'}")
    if not use_llm:
        return {diagnosis: classify_diagnosis_by_rule(diagnosis) for diagnosis in unique_values}

    batch_size = max(1, batch_size)
    for start in range(0, len(unique_values), batch_size):
        batch = unique_values[start : start + batch_size]
        end = start + len(batch)
        print(f"      [{end}/{len(unique_values)}] 批量分类初步诊断")
        diagnosis_map.update(classify_diagnoses_with_llm(batch, model=llm_model))
    return diagnosis_map


def _extract_labeled_diagnosis(text: str, label: str) -> list[str]:
    """抽取形如“西医诊断：...”或“中医诊断：...”的诊断片段。"""
    pattern = rf"{label}\s*[:：]\s*([\s\S]*?)(?=(?:\n\s*)?(?:西医诊断|中医诊断)\s*[:：]|$)"
    matches = re.findall(pattern, text)
    out: list[str] = []
    for match in matches:
        pieces = re.split(r"[；;，,\n]+", match)
        out.extend(piece.strip() for piece in pieces if piece.strip())
    return out


def _split_tcm_diagnosis_and_syndrome(text: str) -> tuple[str, str]:
    """把“中医病名(证候)”拆成中医诊断和中医证候。"""
    text = text.strip()
    match = re.match(r"^(.*?)[（(]([^（）()]*证)[）)]\s*$", text)
    if not match:
        return text, ""
    return match.group(1).strip(), match.group(2).strip()


def _clean_west_diagnosis(text: str, known_syndromes: list[str]) -> str:
    """清理误拼在西医诊断末尾的中医证候。"""
    text = text.strip()
    for syndrome in known_syndromes:
        if text.endswith(syndrome):
            return text[: -len(syndrome)].strip(" ，,；;")

    return re.sub(r"[^，、；;\s（）()]{2,12}证$", "", text).strip(" ，,；;")


def _extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_diagnosis_result(data: dict[str, Any], fallback: dict[str, str]) -> dict[str, str]:
    """校验并补齐 LLM 诊断分类结果。"""
    if not isinstance(data, dict):
        return fallback
    result = {}
    for key in DIAGNOSIS_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            value = "、".join(str(item).strip() for item in value if str(item).strip())
        value = "" if _is_blank_cell(value) else str(value).strip()
        result[key] = value
    return {key: result[key] or fallback[key] for key in DIAGNOSIS_KEYS}


def _dedupe_keep_order(items) -> list[str]:
    """去重并保留原始顺序。"""
    seen = set()
    out = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(
    doctorname: str = DEFAULT_DOCTORNAME,
    record_path: Path = DEFAULT_RECORD_XLSX,
    prescription_path: Path = DEFAULT_PRESCRIPTION_XLSX,
    use_llm_diagnosis: bool = True,
    llm_model: str = "qwen-plus",
    llm_batch_size: int = 20,
) -> Path:
    output_dir = DEFAULT_OUTPUT_DIR / doctorname
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 读取病历主表: {record_path}")
    record_df = read_record_xlsx(record_path)
    print(f"      共 {len(record_df)} 条记录，唯一门诊号 {record_df['mzh'].nunique()}")

    print(f"[2/5] 读取处方明细表: {prescription_path}")
    prescription_df = read_prescription_xlsx(prescription_path)
    drug_count = prescription_df["item_prefix"].isin(ITEM_PREFIX_MAP).sum()
    print(f"      正常收费 {len(prescription_df)} 条，其中药品条目 {drug_count} 条")
    print("      类别分布:")
    for prefix, (cat_name, _) in ITEM_PREFIX_MAP.items():
        n = (prescription_df["item_prefix"] == prefix).sum()
        if n:
            print(f"        {prefix} -> {cat_name}: {n}")

    prescription_mzhs = set(prescription_df["mzh"])
    before_join_count = len(record_df)
    record_df = record_df[record_df["mzh"].isin(prescription_mzhs)].reset_index(drop=True)
    skipped_without_prescription = before_join_count - len(record_df)
    print(f"      跳过 {skipped_without_prescription} 条未在处方明细表出现门诊号的病历记录")
    print(f"      保留 {len(record_df)} 条两个文件均存在门诊号的病历记录")

    print("[3/5] 按门诊号聚合处方...")
    prescription_map = aggregate_prescriptions(prescription_df)
    visit_type_map = build_visit_type_map(prescription_df)
    print(f"      涉及 {len(prescription_map)} 个门诊号")

    print("[4/5] 分类初步诊断...")
    diagnosis_map = build_diagnosis_map(
        record_df,
        use_llm=use_llm_diagnosis,
        llm_model=llm_model,
        batch_size=llm_batch_size,
    )

    print("[5/5] 生成最终 JSON...")
    records: list[dict[str, Any]] = []
    drug_matched = 0
    for _, row in record_df.iterrows():
        mzh = row["mzh"]
        if mzh in prescription_map:
            drug_matched += 1
        diagnosis_text = "" if _is_blank_cell(row["diagnosis"]) else str(row["diagnosis"]).strip()
        records.append(
            build_record_dict(
                row,
                prescription_map,
                visit_type_map,
                diagnosis_result=diagnosis_map.get(diagnosis_text),
            )
        )

    today = datetime.now().strftime("%Y%m%d")
    out_path = output_dir / f"{doctorname}_record_{today}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已写出 {len(records)} 条记录 -> {out_path}")
    print(f"   其中 {drug_matched} 条匹配到药品处方聚合")

    # 简单打印一条样例
    if records:
        sample = records[0]
        print("\n=== 样例 (第一条) ===")
        slim = {
            "idx": sample["idx"],
            "chief_complaint": sample["chief_complaint"],
            "new_medical_history": sample["new_medical_history"],
            "diagnosis_illness": sample["diagnosis_illness"],
            "diagnosis_sickness": sample["diagnosis_sickness"],
            "diagnosis_disease": sample["diagnosis_disease"],
            "treatment_plan": sample["treatment_plan"],
            "ps_count": len(sample["ps"]),
            "ps_process": [p["drug_process_name"] for p in sample["ps"]],
        }
        print(json.dumps(slim, ensure_ascii=False, indent=2))

    return out_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="清洗西区医生病历和处方明细。")
    parser.add_argument("--doctorname", default=DEFAULT_DOCTORNAME, help="医生标识，用于生成输出文件名。")
    parser.add_argument(
        "--record-path", "--record_path", type=Path, default=DEFAULT_RECORD_XLSX, help="病历主表 Excel 路径。"
    )
    parser.add_argument(
        "--prescription-path",
        "--prescription_path",
        type=Path,
        default=DEFAULT_PRESCRIPTION_XLSX,
        help="处方明细 Excel 路径。",
    )
    parser.add_argument(
        "--llm-diagnosis-model",
        default=os.getenv("XIQ_RECORD_LLM_DIAGNOSIS_MODEL", "qwen-plus"),
        help="用于分类“初步诊断”的 LLM 模型名。",
    )

    parser.add_argument(
        "--llm_diagnosis_batch_size",
        type=int,
        default=20,
        help="LLM 批量分类“初步诊断”的每批条数。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        doctorname=args.doctorname,
        record_path=args.record_path,
        prescription_path=args.prescription_path,
        use_llm_diagnosis=True,
        llm_model=args.llm_diagnosis_model,
        llm_batch_size=args.llm_diagnosis_batch_size,
    )

"""
python medical/data_utils/raw_data_clear/clear_xiqu_record.py \
  --doctorname 巢国俊 \
  --record-path "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/西区医生分身数据/西区巢国俊/医院病历数据_20260706164235(1).xlsx" \
  --prescription-path "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/西区医生分身数据/西区巢国俊/巢国俊处方明细.xlsx"

"""
