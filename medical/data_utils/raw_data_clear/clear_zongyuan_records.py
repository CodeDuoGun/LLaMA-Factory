# ruff: noqa: D205, D212, D301, D415, W605
"""
处理总院线下医生的病历数据为结构化数据
门诊病历表：medical/data/zongyuan_hukaiwen/胡凯文门诊病历-20240101-20260712.xlsx
    问诊单ID、费用类别【中药费、中成药费、西药费】、数量、剂数、计价单位、剂量【数量/剂数】
处方明细表：medical/data/zongyuan_hukaiwen/胡凯文门诊收费明细20250101-20260630-2.xlsx
    问诊单ID\患者性别、挂号年龄、正文【需要调用llm，提取患者主诉、现病史、过敏史、既往史、家族史、个人史、婚育史】、中医主诊断、西医主诊断、中医症候名称
处理步骤：
1、分别读取两个文件，获取数据 raw_data
2、根据病历表的【问诊单ID】,获取相同问诊单ID的处方明细表中的数据
3、将处方明细表中的数据，按照病历表的【问诊单ID】，进行合并
4、将两个表格中合并后的数据，保存为一个最新的表。脚本的参数支持输入doctoname。用于后续处理不同的doctor数据
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "medical/processed_data"
DEFAULT_PHASE2_DATA_DIR = PROJECT_ROOT / "medical/data/二期总院"
PHASE2_DOCTORS = {"冯春祥", "崔霞", "杨宗艳", "王素梅", "钱彦方"}
PHASE2_PROMPT_PATH = PROJECT_ROOT / "medical/prompts/zongyuan_cleaner.txt"
PHASE2_LLM_LOCAL = threading.local()
PHASE2_RECORD_REQUIRED_COLUMNS = (
    "就诊ID",
    "病历ID",
    "病人ID",
    "患者姓名",
    "患者性别",
    "挂号年龄",
    "身份证号",
    "移动电话",
    "正文",
    "病历书写时间",
)
PHASE2_PRESCRIPTION_REQUIRED_COLUMNS = (
    "就诊ID",
    "处方号",
    "结算时间",
    "费用类别",
    "医院项目名称",
    "数量",
    "剂数",
    "计价单位",
    "退费标识",
)
PHASE2_OUTPUT_FIELDS = (
    "patient_appeal",
    "new_medical_history",
    "old_medical_history",
    "allergic_history",
    "personal_history",
    "family_history",
    "special_history",
    "marriage_history",
    "birth_detail",
    "diagnosis_illness",
    "diagnosis_sickness",
    "diagnosis_disease",
    "diagnosis_illness_reason",
    "diagnosis_disease_reason",
    "diagnosis_sickness_reason",
)
PHASE2_MEDICAL_CATEGORIES = {"中药费", "中成药费", "西药费"}
PHASE2_SOURCE_FIELDS = {
    "patient_appeal": ("患者主诉", "主诉"),
    "new_medical_history": ("现病史",),
    "old_medical_history": ("既往史",),
    "allergic_history": ("过敏史", "药物过敏史"),
    "family_history": ("家族史",),
    "personal_history": ("个人史",),
    "special_history": ("特殊时期", "特殊史"),
    "marriage_history": ("婚姻史", "婚姻状态", "婚姻状况"),
    "birth_detail": ("生育史", "孕产史", "婚育史"),
    "diagnosis_illness": ("西医主诊断", "西医诊断", "diagnosis_illness"),
    "diagnosis_sickness": ("中医病名", "中医主诊断", "diagnosis_sickness"),
    "diagnosis_disease": ("中医症候名称", "中医证候", "证候", "diagnosis_disease"),
}
PHASE2_SECTION_STOP_LABELS = tuple(
    alias for aliases in PHASE2_SOURCE_FIELDS.values() for alias in aliases
) + ("体格检查", "辅助检查", "初步诊断", "处理措施", "治疗方案", "医师签名")

DOCTOR_ALIASES = {
    "hukaiwen": "hukaiwen",
    "胡凯文": "hukaiwen",
    "chuyuping": "chuyuping",
    "初玉平": "chuyuping",
    "liugenshang": "liugenshang",
    "刘根尚": "liugenshang",
    "liuluming": "liuluming",
    "刘鲁明": "liuluming",
    "wangsumei": "wangsumei",
    "王素梅": "wangsumei",
}

DOCTOR_CONFIGS = {
    "hukaiwen": {
        "doctor_display_name": "胡凯文",
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_hukaiwen/胡凯文门诊病历-20240101-20260712.xlsx",
        "prescription_file": PROJECT_ROOT / "medical/data/zongyuan_hukaiwen/胡凯文门诊收费明细20250101-20260630-2(1).xlsx",
    },
    "chuyuping": {
        "doctor_display_name": "初玉平",
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_chuyuping/初玉平-门诊病历-20240101-20260706.xlsx",
        "prescription_file": PROJECT_ROOT / "medical/data/zongyuan_chuyuping/初玉平-门诊收费明细-20240101-20260706.xlsx",
    },
    "liugenshang": {
        "doctor_display_name": "刘根尚",
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_liugenshang/刘根尚-门诊病历-20240101-20260701.xlsx",
        "prescription_file": PROJECT_ROOT / "medical/data/zongyuan_liugenshang/刘根尚-门诊收费明细-20240101-20260706.xlsx",
    },
    "liuluming": {
        "doctor_display_name": "刘鲁明",
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_liuluming/刘鲁明-门诊病历-20240101-20260701.xlsx",
        "prescription_file": PROJECT_ROOT / "medical/data/zongyuan_liuluming/刘鲁明-门诊收费明细-20240101-20260706.xlsx",
    },
    "wangsumei": {
        "doctor_display_name": "王素梅",
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_wangsumei/王素梅-门诊病历-20240101-20260701.xlsx",
        "prescription_file": PROJECT_ROOT / "medical/data/zongyuan_wangsumei/王素梅-门诊收费明细-20240101-20260706.xlsx",
    },
}

ID_CANDIDATES = ("就诊ID", "门诊号", "病历ID")
RECORD_DATE_CANDIDATES = ("就诊时间", "病历书写时间")
PRESCRIPTION_DATE_CANDIDATES = ("项目开单时间", "结算时间")
FEE_CATEGORIES = ("中药费", "中成药费", "西药费")
CLINICAL_FIELDS = ("患者主诉", "现病史", "过敏史", "既往史", "家族史", "个人史", "婚育史")
DIAGNOSIS_FIELDS = ("中医主诊断", "西医主诊断", "中医症候名称")

SECTION_ALIASES = {
    "患者主诉": ("患者主诉", "主诉"),
    "现病史": ("现病史",),
    "过敏史": ("药物过敏史", "过敏史"),
    "既往史": ("既往史",),
    "家族史": ("家族史",),
    "个人史": ("个人史",),
    "婚育史": ("婚育史",),
}

SECTION_STOP_LABELS = (
    "患者主诉",
    "主诉",
    "现病史",
    "过敏史",
    "药物过敏史",
    "既往史",
    "家族史",
    "个人史",
    "婚育史",
    "体格检查",
    "辅助检查",
    "初步诊断",
    "处理措施",
    "医师签名",
)


def clean_cell(value: Any) -> str:
    """把 Excel 单元格值转成干净字符串。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none", "<na>"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_id(value: Any) -> str:
    """规范问诊单 ID，避免 559733.0 这类 Excel 数字格式影响关联。"""
    text = clean_cell(value)
    if not text:
        return ""
    return re.sub(r"\.0$", "", text).strip()


def clean_dose(value: Any) -> str:
    """
    规范剂量字段。
    - 如果 "单量" 字段包含小括号（如 "10g(每日3次)"），优先取括号内的内容作为剂量。
    - 否则直接清理后返回。
    """
    text = clean_cell(value)
    if not text:
        return ""
    # 去掉小括号再使用里面的内容：优先取括号内，否则取括号外
    inner = re.findall(r"\(([^)]+)\)", text)
    if inner:
        return clean_cell(inner[0])
    return text


def normalize_doctorname(doctorname: str) -> str:
    """统一 doctorname 参数。"""
    key = (doctorname or "").strip()
    return DOCTOR_ALIASES.get(key, key)


def read_excel_all_sheets(path: Path) -> pd.DataFrame:
    """读取 Excel 中所有非空 sheet，并合并为一个 DataFrame。"""
    if not path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue
        frame = frame.dropna(how="all").copy()
        if frame.empty:
            continue
        frame["来源Sheet"] = sheet_name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_phase2_excel_all_sheets(path: Path) -> pd.DataFrame:
    """以文本方式读取二期总院 Excel，避免病历ID和证件号被转成浮点数。"""
    if not path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {path}")
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    frames = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue
        frame = frame.dropna(how="all").copy()
        if frame.empty:
            continue
        frame["来源Sheet"] = sheet_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pick_id_column(record_df: pd.DataFrame, prescription_df: pd.DataFrame) -> str:
    """从两张表中选择共同的问诊单 ID 字段。"""
    for column in ID_CANDIDATES:
        if column in record_df.columns and column in prescription_df.columns:
            return column
    raise ValueError(f"无法找到共同 ID 字段，候选字段: {', '.join(ID_CANDIDATES)}")


def pick_date_column(df: pd.DataFrame, candidates: tuple[str, ...], table_name: str) -> str:
    """从表格中选择日期字段。"""
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"{table_name}无法找到日期字段，候选字段: {', '.join(candidates)}")


def normalize_date_series(series: pd.Series) -> pd.Series:
    """把日期时间字段标准化为 date，用于按天匹配。"""
    return pd.to_datetime(series, errors="coerce").dt.date


def filter_complete_same_day_records(
    record_df: pd.DataFrame,
    prescription_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """保留截止日期内且同一问诊单 ID 同一天同时存在病历和处方的数据。"""
    record_date_column = pick_date_column(record_df, RECORD_DATE_CANDIDATES, "病历表")
    prescription_date_column = pick_date_column(prescription_df, PRESCRIPTION_DATE_CANDIDATES, "处方明细表")

    record_df = record_df.copy()
    prescription_df = prescription_df.copy()
    record_df["_join_date"] = normalize_date_series(record_df[record_date_column])
    prescription_df["_join_date"] = normalize_date_series(prescription_df[prescription_date_column])

    record_dates = record_df["_join_date"].dropna()
    prescription_dates = prescription_df["_join_date"].dropna()
    if record_dates.empty:
        raise ValueError(f"病历表日期字段无有效日期: {record_date_column}")
    if prescription_dates.empty:
        raise ValueError(f"处方明细表日期字段无有效日期: {prescription_date_column}")

    record_max_date = record_dates.max()
    prescription_max_date = prescription_dates.max()
    cutoff_date = min(record_max_date, prescription_max_date)

    before_record_count = len(record_df)
    before_prescription_count = len(prescription_df)
    record_df = record_df[record_df["_join_date"].notna() & (record_df["_join_date"] <= cutoff_date)].copy()
    prescription_df = prescription_df[
        prescription_df["_join_date"].notna() & (prescription_df["_join_date"] <= cutoff_date)
    ].copy()

    record_pairs = record_df[["问诊单ID", "_join_date"]].drop_duplicates()
    prescription_df = prescription_df.merge(record_pairs, on=["问诊单ID", "_join_date"], how="inner")
    matched_pairs = prescription_df[["问诊单ID", "_join_date"]].drop_duplicates()
    record_df = record_df.merge(matched_pairs, on=["问诊单ID", "_join_date"], how="inner")

    stats = {
        "record_date_column": record_date_column,
        "prescription_date_column": prescription_date_column,
        "record_max_date": record_max_date,
        "prescription_max_date": prescription_max_date,
        "cutoff_date": cutoff_date,
        "before_record_count": before_record_count,
        "before_prescription_count": before_prescription_count,
        "after_record_count": len(record_df),
        "after_prescription_count": len(prescription_df),
    }
    return record_df, prescription_df, stats


def resolve_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    """根据 doctorname 和显式参数解析输入/输出路径。"""
    doctorname = normalize_doctorname(args.doctorname)
    config = DOCTOR_CONFIGS.get(doctorname, {})

    record_file = args.record_file or config.get("record_file")
    prescription_file = args.prescription_file or config.get("prescription_file")
    if record_file is None or prescription_file is None:
        raise ValueError("未知 doctorname 时必须同时传入 --record-file 和 --prescription-file")

    today = datetime.now().strftime("%Y%m%d")
    output = args.output or args.output_dir / f"zongyuan_{doctorname}" / f"{doctorname}_zongyuan_records_{today}.json"
    return doctorname, Path(record_file), Path(prescription_file), Path(output)


def to_number(value: Any) -> float | None:
    """把单元格转成数字，失败返回 None。"""
    text = clean_cell(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float) -> str:
    """格式化数字，整数不保留 .0。"""
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"



def build_drug_item(row: pd.Series) -> OrderedDict[str, Any]:
    """把收费明细中的一行转换为一个 drug 条目。"""
    return OrderedDict(
        drug_name=clean_cell(row.get("医院项目名称")) or clean_cell(row.get("医保项目名称")),
        dose=clean_cell(row.get("单量")),
        unit=clean_cell(row.get("计价单位")),
        decoction_name="",
    )


def build_prescription_item(
    fee_category: str,
    rows: list[pd.Series],
) -> OrderedDict[str, Any]:
    """把同一费用类别下的一组行转换为一个完整处方。"""
    drugs = [build_drug_item(row) for row in rows if clean_cell(row.get("退费标识")) == "否"]
    prescription_order_id = clean_cell(rows[0].get("处方号")) if rows else ""
    return OrderedDict(
        prescription_order_id=prescription_order_id,
        usage_type=fee_category,
        drug_process_name="饮片" if fee_category == "中药费" else fee_category,
        doctor_advice="",
        drugs=drugs,
        prescription_detail_count=len(drugs),
        prescription_summary=build_prescription_summary(drugs),
    )


def build_prescription_summary(drugs: list[dict[str, Any]]) -> str:
    """生成便于人工查看的处方摘要。"""
    pieces = []
    for drug in drugs:
        name = drug.get("drug_name")
        if not name:
            continue
        dose = drug.get("dose") or ""
        unit = drug.get("unit") or ""
        if dose:
            pieces.append(f"{name} {dose}{unit}")
        else:
            pieces.append(f"{name}")
    return "；".join(pieces)


def get_medical_code_prefix(item: dict[str, Any]) -> str:
    """获取医保编码首字母，用于中药费再次聚合。"""
    code = clean_cell(item.get("医保项目编码"))
    if not code:
        return "未知编码"
    return code[0].upper()


def aggregate_prescriptions(
    prescription_df: pd.DataFrame,
    fee_categories: set[str],
) -> dict[str, dict[str, Any]]:
    """按问诊单ID聚合处方明细。

    规则：
    1. 先经过问诊单ID筛选；
    2. 再只保留 中药费/中成药费/西药费；
    3. 同一个问诊单ID下，不同费用类别表示不同处方；
    4. 中药费按医保项目编码首字母再次拆分处方。
    """
    if prescription_df.empty:
        return {}

    filtered_df = prescription_df.copy()

    if fee_categories:
        filtered_df = filtered_df[
            filtered_df["费用类别"].map(clean_cell).isin(fee_categories)
        ].copy()

    prescription_map: dict[str, dict[str, Any]] = {}

    for inquiry_id, group in filtered_df.groupby("问诊单ID", sort=False, dropna=False):
        inquiry_id = clean_cell(inquiry_id)
        if not inquiry_id:
            continue

        prescriptions = []

        for fee_category, category_group in group.groupby("费用类别", sort=False):
            fee_category = clean_cell(fee_category)
            rows = [row for _, row in category_group.iterrows()]

            if fee_category == "中药费":
                herb_groups: dict[str, list[pd.Series]] = OrderedDict()

                for row in rows:
                    prefix = get_medical_code_prefix(
                        {"医保项目编码": row.get("医保项目编码")}
                    )
                    herb_groups.setdefault(prefix, []).append(row)

                for prefix, prefix_rows in herb_groups.items():
                    prescriptions.append(build_prescription_item(fee_category, prefix_rows))
            else:
                prescriptions.append(build_prescription_item(fee_category, rows))

        prescription_map[inquiry_id] = {
            "prescriptions": prescriptions,
            "categories": list(
                OrderedDict.fromkeys(
                    clean_cell(v) for v in group["费用类别"].tolist() if clean_cell(v)
                )
            ),
        }

    return prescription_map

def extract_section(text: str, aliases: tuple[str, ...]) -> str:
    """从正文中按标签抽取一个病历段落。"""
    if not text:
        return ""

    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    match = re.search(rf"(?:^|\s)(?:{alias_pattern})\s*[：:]\s*", text)
    if not match:
        return ""

    start = match.end()
    stop_pattern = "|".join(re.escape(label) for label in SECTION_STOP_LABELS)
    stop_match = re.search(rf"\s(?:{stop_pattern})\s*[：:]", text[start:])
    end = start + stop_match.start() if stop_match else len(text)
    value = text[start:end].strip(" ，,。；;\n\t")
    value = re.sub(rf"^(?:{alias_pattern})\s*[：:]\s*", "", value).strip()
    return value


def regex_extract_clinical_fields(record: pd.Series) -> dict[str, str]:
    """不用 LLM 时，从正文中按常见病历标签抽取字段。"""
    text = clean_cell(record.get("正文"))
    extracted = {}
    for field in CLINICAL_FIELDS:
        structured_value = clean_cell(record.get(field))
        extracted[field] = structured_value or extract_section(text, SECTION_ALIASES[field])
    return extracted


def extract_json_payload(response_text: str) -> str:
    """从 LLM 回复中提取 JSON 对象。"""
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    if start < 0:
        return text

    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text[start:])
    return text[start : start + end]


def build_llm_prompt(text: str) -> list[dict[str, str]]:
    """构造正文病史字段抽取提示词。"""
    fields = "、".join(CLINICAL_FIELDS)
    result_msg = [
        {
            "role": "system",
            "content": (
                "你是严谨的中医门诊病历结构化助手。"
                "只根据输入的“正文”中出现的字段抽取信息，不要编造。"
                "字段值必须在‘正文’中显示出现，才可提取，否则认定为'无'。不得从描述中推测字段，如正文中未提及“既往史”，不得根据“现病史”进行推测和提取"
                "未提取到的字段必须返回“无”，不要返回空字符串。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请从下面“正文”字段中抽取字段：{fields}。\n"
                "要求：\n"
                "1. 只返回 JSON 对象，不要输出解释。\n"
                "2. JSON key 只能包含上述 7 个中文字段名，不能新增其他字段。\n"
                "3. value 必须为字符串；未提取到的内容统一写“无”。\n\n"
                "4. 禁止出现冗余字符，如果字段内容出现多余符号，一律删除。如：既往史：发现乙肝病史10余年，膀胱切除] , 结果应为：‘既往史：发现乙肝病史10余年，膀胱切除’ "
                f"正文：\n{text}"
            ),
        },
    ]
    # print(result_msg)
    return result_msg



def llm_extract_clinical_fields(text: str, model: str) -> dict[str, str]:
    """使用 Qwen 从正文字段抽取病史字段。"""
    from openai import OpenAI

    from medical.config import config

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=build_llm_prompt(text),
        response_format={"type": "json_object"},
        # max_tokens=1200,
        temperature=0.1,
    )
    response_text = response.choices[0].message.content or "{}"
    parsed = json.loads(extract_json_payload(response_text))
    print(parsed)
    return {field: clean_cell(parsed.get(field)) or "无" for field in CLINICAL_FIELDS}


def extract_clinical_fields(
    record: pd.Series,
    use_llm: bool,
    llm_model: str,
) -> dict[str, str]:
    """优先使用表格结构化字段，必要时从正文抽取。"""
    regex_result = regex_extract_clinical_fields(record)
    if not use_llm:
        return regex_result

    text = clean_cell(record.get("正文"))
    if not text:
        return regex_result

    try:
        return llm_extract_clinical_fields(text, model=llm_model)
    except Exception as exc:
        print(f"      ⚠️ LLM 抽取失败，回退到正则抽取。问诊单ID={record.get('问诊单ID')}: {exc}")
        return {field: regex_result[field] or "无" for field in CLINICAL_FIELDS}


def build_output_record(
    record: pd.Series,
    prescription_map: dict[str, dict[str, Any]],
    use_llm: bool,
    llm_model: str,
) -> OrderedDict[str, Any]:
    """构造最终输出的一条 JSON 记录。"""
    inquiry_id = clean_cell(record.get("问诊单ID"))
    prescription = prescription_map.get(inquiry_id, {})
    prescriptions = prescription.get("prescriptions", [])
    clinical_fields = extract_clinical_fields(record, use_llm, llm_model)

    output = OrderedDict(
        inquiry_id=inquiry_id,
        order_sn=clean_cell(record.get("门诊号")),
        inquiry_method="",
        is_first=clean_cell(record.get("初复诊")),
        patient_name=clean_cell(record.get("患者姓名")),
        patient_sex=clean_cell(record.get("患者性别")),
        patient_age=clean_cell(record.get("挂号年龄")),
        patient_appeal=clinical_fields.get("患者主诉", "无"),
        doc_ass_stu_appeal="",
        new_medical_history=clinical_fields.get("现病史", "无"),
        old_medical_history=clinical_fields.get("既往史", "无"),
        allergic_history=clinical_fields.get("过敏史", "无"),
        family_history=clinical_fields.get("家族史", "无"),
        personal_history=clinical_fields.get("个人史", "无"),
        birth_detail=clinical_fields.get("婚育史", "无"),
        admin_report_describe="",
        admin_face_describe="",
        inspection_report_img=[],
        tongue_face_img=[],
        admin_report_img=[],
        admin_face_img=[],
        diagnosis_sickness=clean_cell(record.get("中医症候名称")),
        diagnosis_illness=clean_cell(record.get("西医主诊断")),
        diagnosis_disease=clean_cell(record.get("中医主诊断")),
        internal_prescriptions=prescriptions,
    )

    return output


def write_output(records: list[dict[str, Any]], output_path: Path) -> None:
    """原子写出 JSON 文件，避免处理中断留下不完整结果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(".json")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(records, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def build_output_records(
    record_df: pd.DataFrame,
    prescription_map: dict[str, dict[str, Any]],
    use_llm: bool,
    llm_model: str,
    batchsize: int,
) -> list[OrderedDict[str, Any]]:
    """分批并发处理病历，按输入顺序收集结果。"""
    if batchsize < 1:
        raise ValueError("batchsize 必须是大于 0 的整数")

    record_count = len(record_df)
    records: list[OrderedDict[str, Any] | None] = [None] * record_count

    # 输入表和处方映射只在主线程读取；worker 各自构造结果，主线程按原始行号回填。
    # 这样相同问诊单 ID 的多行也不会互相覆盖，输出顺序与输入顺序一致。
    with ThreadPoolExecutor(max_workers=batchsize, thread_name_prefix="zongyuan-record") as executor:
        for start in range(0, record_count, batchsize):
            batch = record_df.iloc[start : start + batchsize]
            future_to_record = {}
            for offset, (_, row) in enumerate(batch.iterrows()):
                index = start + offset
                future = executor.submit(build_output_record, row, prescription_map, use_llm, llm_model)
                future_to_record[future] = (index, row)

            for future in as_completed(future_to_record):
                index, row = future_to_record[future]
                try:
                    records[index] = future.result()
                except Exception as exc:
                    inquiry_id = clean_cell(row.get("问诊单ID"))
                    raise RuntimeError(f"第 {index + 1} 条病历处理失败，问诊单ID={inquiry_id}") from exc

    if any(record is None for record in records):
        raise RuntimeError("病历处理结果不完整，已取消写出")

    return [record for record in records if record is not None]


def should_use_phase2_layout(args: argparse.Namespace) -> bool:
    """识别二期总院新导出格式，未指定时兼容原有总院清洗流程。"""
    layout = getattr(args, "source_layout", "auto")
    if layout == "phase2":
        return True
    if layout == "legacy":
        return False

    doctorname = clean_cell(getattr(args, "doctorname", ""))
    record_file = getattr(args, "record_file", None)
    return doctorname in PHASE2_DOCTORS or (record_file is not None and "二期总院" in str(record_file))


def resolve_phase2_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    """定位二期总院医生的门诊病历、收费明细和 JSONL 输出路径。"""
    doctorname = clean_cell(args.doctorname)
    source_dir = Path(getattr(args, "source_dir", DEFAULT_PHASE2_DATA_DIR))

    def resolve_input(explicit_path: Path | None, pattern: str, label: str) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)
        matches = sorted(
            path for path in source_dir.glob(pattern) if path.is_file() and not path.name.startswith("~$")
        )
        if len(matches) != 1:
            if not matches:
                raise FileNotFoundError(f"{source_dir} 下未找到{doctorname}的{label}文件（{pattern}）")
            raise ValueError(
                f"{source_dir} 下找到多个{doctorname}的{label}文件，请通过命令行显式指定："
                + ", ".join(path.name for path in matches)
            )
        return matches[0]

    record_file = resolve_input(args.record_file, f"{doctorname}-门诊病历-*.xlsx", "门诊病历")
    prescription_file = resolve_input(
        args.prescription_file,
        f"{doctorname}-门诊收费明细-*.xlsx",
        "门诊收费明细",
    )
    output_path = args.output or args.output_dir / f"doctor_二期总院_{doctorname}" / "medical_records_ai.jsonl"
    if output_path.suffix.lower() != ".jsonl":
        output_path = output_path.with_suffix(".jsonl")
    return doctorname, record_file, prescription_file, Path(output_path)


def clean_body_text(value: Any) -> str:
    """清理正文空值，但保留换行，方便模型识别病历分段。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return "" if text.lower() in {"nan", "nat", "none", "<na>"} else text


def phase2_extract_section(text: str, aliases: tuple[str, ...]) -> str:
    """LLM 失败时，按病历标签从正文中做保守兜底抽取。"""
    if not text:
        return ""
    labels = "|".join(re.escape(label) for label in sorted(aliases, key=len, reverse=True))
    match = re.search(rf"(?:^|[\s\n])(?:{labels})\s*[：:]\s*", text)
    if not match:
        return ""
    start = match.end()
    stops = "|".join(re.escape(label) for label in sorted(PHASE2_SECTION_STOP_LABELS, key=len, reverse=True))
    stop_match = re.search(rf"(?:[\s\n])(?:{stops})\s*[：:]", text[start:])
    end = start + stop_match.start() if stop_match else len(text)
    return text[start:end].strip(" ，,。；;\n\t")


def phase2_structured_fallback(record: pd.Series) -> dict[str, str]:
    """从门诊病历表已有字段和正文标签提供提示词目标结构的兜底值。"""
    body = clean_body_text(record.get("正文"))
    result = {field: "" for field in PHASE2_OUTPUT_FIELDS}
    for field, aliases in PHASE2_SOURCE_FIELDS.items():
        result[field] = next(
            (clean_cell(record.get(alias)) for alias in aliases if clean_cell(record.get(alias))),
            "",
        )
        if not result[field]:
            result[field] = phase2_extract_section(body, aliases)

    return result


def phase2_normalize_prompt_fields(fields: dict[str, str]) -> dict[str, str]:
    """将婚姻和特殊时期字段约束到提示词规定的枚举值。"""
    result = {field: fields.get(field, "") for field in PHASE2_OUTPUT_FIELDS}
    if result["marriage_history"] not in {"", "已婚", "未婚"}:
        result["marriage_history"] = ""
    if result["special_history"] not in {"妊娠状态", "备孕", "哺乳期", "月经期", "无"}:
        result["special_history"] = "无"
    result["special_history"] = result["special_history"] or "无"
    return result


def load_phase2_prompt() -> str:
    """读取二期总院正文抽取提示词。"""
    if not PHASE2_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"二期总院正文抽取提示词不存在: {PHASE2_PROMPT_PATH}")
    prompt = PHASE2_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"二期总院正文抽取提示词为空: {PHASE2_PROMPT_PATH}")
    return prompt


def llm_extract_phase2_fields(
    record: pd.Series,
    text: str,
    model: str,
    system_prompt: str,
) -> dict[str, str]:
    """调用 LLM 并解析提示词定义的目标 JSON Schema。"""
    from openai import OpenAI

    from medical.config import config

    client = getattr(PHASE2_LLM_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
        PHASE2_LLM_LOCAL.client = client
    fallback = phase2_structured_fallback(record)
    see_doc_time = clean_cell(record.get("就诊时间")) or clean_cell(record.get("病历书写时间"))
    input_payload = {
        "see_doc_time": see_doc_time,
        "visit_type": clean_cell(record.get("初复诊")),
        "patient_appeal": fallback["patient_appeal"],
        "doctor_appeal": next(
            (
                clean_cell(record.get(field))
                for field in ("医生主诉", "医师主诉", "接诊主诉")
                if clean_cell(record.get(field))
            ),
            "",
        ),
        "new_medical_history": text,
        "allergic_history": fallback["allergic_history"],
        "old_medical_history": fallback["old_medical_history"],
        "personal_history": fallback["personal_history"],
        "family_history": fallback["family_history"],
        "marriage_history": fallback["marriage_history"],
        "birth_detail": fallback["birth_detail"],
        "source_marriage_and_birth_history": clean_cell(record.get("婚育史")),
        "special_history": fallback["special_history"],
        "original_diagnoses": {
            "diagnosis_illness": fallback["diagnosis_illness"],
            "diagnosis_sickness": fallback["diagnosis_sickness"],
            "diagnosis_disease": fallback["diagnosis_disease"],
            "初步诊断": clean_cell(record.get("初步诊断")),
        },
    }
    for input_field, source_field in (
        ("previous_visit_time", "previous_visit_time"),
        ("previous_visit_diagnoses", "previous_visit_diagnoses"),
    ):
        value = clean_cell(record.get(source_field))
        if value:
            input_payload[input_field] = value
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请根据系统提示清洗以下本次门诊病历，并且只返回规定 Schema 中的 JSON 对象。"
                    "输入数据：\n" + json.dumps(input_payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    response_text = response.choices[0].message.content or "{}"
    parsed = json.loads(extract_json_payload(response_text))
    if not isinstance(parsed, dict):
        raise ValueError("LLM 返回内容不是 JSON 对象")
    return {field: clean_cell(parsed[field]) for field in PHASE2_OUTPUT_FIELDS if field in parsed}


def extract_phase2_fields(
    record: pd.Series,
    use_llm: bool,
    llm_model: str,
    system_prompt: str,
) -> dict[str, str]:
    """调用 LLM 清洗提示词输出字段；缺失字段回退到原表明确内容。"""
    fallback = phase2_structured_fallback(record)
    if not use_llm:
        return phase2_normalize_prompt_fields(fallback)

    text = clean_body_text(record.get("正文"))
    if not text:
        return phase2_normalize_prompt_fields(fallback)

    try:
        extracted = llm_extract_phase2_fields(record, text, llm_model, system_prompt)
    except Exception as exc:
        print(f"      ⚠️ LLM 抽取失败，回退到表格/正文标签。order_sn={normalize_id(record.get('病历ID'))}: {exc}")
        extracted = {}
    result = {
        field: extracted[field] if field in extracted else fallback.get(field, "")
        for field in PHASE2_OUTPUT_FIELDS
    }
    return phase2_normalize_prompt_fields(result)


def phase2_history_is_positive(value: Any) -> bool:
    """判断病史内容是否为肯定信息，处理常见的未记录/否定占位值。"""
    text = re.sub(r"\s+", "", clean_cell(value)).strip("：:")
    if not text:
        return False
    negative_prefixes = (
        "无",
        "没有",
        "暂无",
        "未见",
        "未发现",
        "未记录",
        "未询问",
        "未说明",
        "否认",
        "不详",
        "未知",
    )
    clauses = re.split(r"[\r\n,，;；。]+", text)
    return any(
        clause and (not clause.startswith(negative_prefixes) or re.search(r"但|不过|曾有|现有|另有", clause))
        for clause in clauses
    )


def phase2_child_history_flag(value: Any) -> str:
    """仅在清洗后的孕产/生育信息明确时生成已育状态。"""
    text = re.sub(r"\s+", "", clean_cell(value))
    if not text:
        return ""
    if re.search(r"未育|未生育|无子女|无生育史", text):
        return "否"
    if re.search(r"已育|育有|生育\d|孕\d+产|产\d+|一男|一女|儿女|子女", text):
        return "是"
    return ""


def phase2_format_datetime(value: Any) -> str:
    """将 Excel 日期时间格式化为 JSONL 使用的时间字符串。"""
    return datetime_text(value)


def phase2_format_patient_id(value: Any) -> int | None:
    """将病人ID转成数据库使用的整数，空值或非整数返回 None。"""
    text = normalize_id(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def phase2_format_drug_num(row: pd.Series) -> str:
    """按数量/剂数计算每剂药量；缺字段时保留现有数量或剂数。"""
    quantity = to_number(row.get("数量"))
    dose_count = to_number(row.get("剂数"))
    if quantity is not None and dose_count is not None and dose_count > 0:
        return format_number(quantity / dose_count)
    if quantity is not None:
        return format_number(quantity)
    if dose_count is not None:
        return format_number(dose_count)
    return ""


def phase2_frequency_count(value: Any) -> str:
    """从处方频次中抽取每日次数，供 use_limit 使用。"""
    text = clean_cell(value)
    match = re.search(r"(?:每日|每天|一日|1日)?\s*([0-9]+|[一二两三四五六七八九十]+)\s*次", text)
    if match:
        count = match.group(1)
        if count.isdigit():
            return count
        chinese_numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        return str(chinese_numbers.get(count, ""))
    if "每天一剂" in text or "每天一付" in text or "每日一剂" in text:
        return "1"
    if "一次性" in text or text == "立即":
        return "1"
    return ""


def phase2_build_usage_fields(rows: list[pd.Series]) -> tuple[str, str]:
    """按处方明细中的剂数和频次生成用量说明及 use_limit。"""
    dose_count = next((clean_cell(row.get("剂数")) for row in rows if clean_cell(row.get("剂数"))), "")
    frequency = next((clean_cell(row.get("频次")) for row in rows if clean_cell(row.get("频次"))), "")
    parts = []
    if dose_count:
        parts.append(f"共{normalize_number(dose_count)}剂")
    if frequency:
        parts.append(frequency)
    usage_desc = "，".join(parts)
    use_limit_parts = [normalize_number(dose_count)] if dose_count else []
    frequency_count = phase2_frequency_count(frequency)
    if frequency_count:
        use_limit_parts.append(frequency_count)
    return usage_desc, "_".join(use_limit_parts)


def phase2_prescription_type(rows: list[pd.Series]) -> str:
    """把收费类别映射为既有数据集使用的药品加工类型。"""
    categories = [clean_cell(row.get("费用类别")) for row in rows]
    categories = [category for category in categories if category]
    if not categories:
        return ""
    category = Counter(categories).most_common(1)[0][0]
    if category == "中药费":
        dosage_forms = {clean_cell(row.get("剂型")) for row in rows if clean_cell(row.get("剂型"))}
        return "颗粒" if any("颗粒" in dosage for dosage in dosage_forms) else "饮片"
    return {"中成药费": "中成药", "西药费": "西药"}.get(category, category)


def phase2_build_prescriptions(rows: pd.DataFrame) -> list[OrderedDict[str, Any]]:
    """按处方号和结算日归并收费明细，保留一个处方中的全部药品行。"""
    if rows.empty:
        return []

    frame = rows.copy()
    frame["_rx_group_id"] = frame.apply(
        lambda row: (
            clean_cell(row.get("处方号"))
            if clean_cell(row.get("处方号")) and pd.notna(row.get("_settlement_date"))
            else f"__row_{row.name}"
        ),
        axis=1,
    )
    prescriptions = []
    for (rx_group_id, _settlement_date), group in frame.groupby(
        ["_rx_group_id", "_settlement_date"], sort=False, dropna=False
    ):
        prescription_rows = [row for _, row in group.iterrows()]
        drugs = [
            OrderedDict(
                drug_name=clean_cell(row.get("医院项目名称")),
                drug_num=phase2_format_drug_num(row),
                unit_name=clean_cell(row.get("计价单位")),
                decoction_name="",
            )
            for row in prescription_rows
            if clean_cell(row.get("医院项目名称"))
        ]
        if not drugs:
            continue
        prescription_order_id = "" if str(rx_group_id).startswith("__row_") else clean_cell(rx_group_id)
        usage_desc, use_limit = phase2_build_usage_fields(prescription_rows)
        advice_values = list(
            OrderedDict.fromkeys(
                clean_cell(row.get("医嘱备注"))
                for row in prescription_rows
                if clean_cell(row.get("医嘱备注"))
            )
        )
        prescriptions.append(
            OrderedDict(
                id=None,
                prescription_order_id=prescription_order_id,
                order_sn=prescription_order_id,
                usage_type="",
                drug_process_name=phase2_prescription_type(prescription_rows),
                use_limit=use_limit,
                doctor_advice="；".join(advice_values),
                usage_desc=usage_desc,
                prescription_items={"drugList": drugs},
            )
        )
    return prescriptions


def build_phase2_output_record(
    record: pd.Series,
    prescription_rows: pd.DataFrame,
    doctor_name: str,
    use_llm: bool,
    llm_model: str,
    system_prompt: str,
) -> OrderedDict[str, Any]:
    """构造一条符合 medical_records_ai.jsonl 格式的二期总院病历。"""
    order_sn = normalize_id(record.get("病历ID"))
    print(f"处理中 order_sn={order_sn}")
    clinical = extract_phase2_fields(record, use_llm, llm_model, system_prompt)
    prescriptions = phase2_build_prescriptions(prescription_rows)
    visit_time = phase2_format_datetime(record.get("病历书写时间"))
    encounter_time = phase2_format_datetime(record.get("就诊时间")) or visit_time
    stop_time = ""
    if not prescription_rows.empty:
        settlement_times = pd.to_datetime(prescription_rows["结算时间"], errors="coerce").dropna()
        if not settlement_times.empty:
            stop_time = settlement_times.max().strftime("%Y-%m-%d %H:%M:%S")

    old_history = clinical.get("old_medical_history", "")
    allergy_history = clinical.get("allergic_history", "")
    family_history = clinical.get("family_history", "")
    personal_history = clinical.get("personal_history", "")
    special_history = clinical.get("special_history", "无")
    marriage_history = clinical.get("marriage_history", "")
    birth_detail = clinical.get("birth_detail", "")
    doctor_name = clean_cell(record.get("就诊医师")) or doctor_name

    return OrderedDict(
        order_sn=order_sn,
        inquiry_method="门诊",
        is_first=clean_cell(record.get("初复诊")),
        created_at=encounter_time,
        start_time=encounter_time,
        stop_time=stop_time,
        user_info_id=None,
        patient_id=phase2_format_patient_id(record.get("病人ID")),
        patient_name=clean_cell(record.get("患者姓名")),
        patient_sex=clean_cell(record.get("患者性别")),
        patient_age=clean_cell(record.get("挂号年龄")),
        patient_idcard=clean_cell(record.get("身份证号")),
        patient_mobile=clean_cell(record.get("移动电话")),
        patient_height=to_number(record.get("身高")),
        patient_weight=to_number(record.get("体重")),
        patient_appeal=clinical.get("patient_appeal", ""),
        doc_ass_stu_appeal=clinical.get("patient_appeal", ""),
        new_medical_history=clinical.get("new_medical_history", ""),
        is_old_medical_history="是" if phase2_history_is_positive(old_history) else "否",
        old_medical_history=old_history,
        is_allergic_history="是" if phase2_history_is_positive(allergy_history) else "否",
        allergic_history=allergy_history,
        is_personal_history="是" if phase2_history_is_positive(personal_history) else "否",
        personal_history=personal_history,
        is_special="否" if special_history == "无" else ("是" if special_history else ""),
        special_history=special_history,
        is_child_history=phase2_child_history_flag(birth_detail),
        birth_detail=birth_detail,
        is_marriage_history=marriage_history,
        is_family_history="是" if phase2_history_is_positive(family_history) else "否",
        family_history=family_history,
        inspection_report_img=[],
        tongue_face_img=[],
        admin_report_img=[],
        admin_face_img=[],
        admin_report_describe=None,
        admin_face_describe="",
        diagnosis_illness=clinical.get("diagnosis_illness", ""),
        diagnosis_sickness=clinical.get("diagnosis_sickness", ""),
        diagnosis_disease=clinical.get("diagnosis_disease", ""),
        ai_diagnosis_illness_reason=clinical.get("diagnosis_illness_reason", ""),
        ai_diagnosis_disease_reason=clinical.get("diagnosis_disease_reason", ""),
        ai_diagnosis_sickness_reason=clinical.get("diagnosis_sickness_reason", ""),
        disposition=clean_cell(record.get("治疗方案")),
        see_doc_time=visit_time,
        doctor_id=None,
        doctor_name=doctor_name,
        ps=prescriptions,
        im_msg=[],
    )


def build_phase2_output_records(
    record_df: pd.DataFrame,
    prescription_map: dict[str, pd.DataFrame],
    doctor_name: str,
    use_llm: bool,
    llm_model: str,
    system_prompt: str,
    batchsize: int,
) -> list[OrderedDict[str, Any]]:
    """按 batch 并发调用 LLM 处理病历，最终按源表顺序输出。"""
    if batchsize < 1:
        raise ValueError("batchsize 必须是大于 0 的整数")
    records: list[OrderedDict[str, Any] | None] = [None] * len(record_df)
    with ThreadPoolExecutor(max_workers=batchsize, thread_name_prefix="phase2-zongyuan") as executor:
        for start in range(0, len(record_df), batchsize):
            batch = record_df.iloc[start : start + batchsize]
            futures = {}
            for offset, (_, row) in enumerate(batch.iterrows()):
                join_id = row["_join_id"]
                prescription_rows = prescription_map.get(join_id, pd.DataFrame())
                index = start + offset
                future = executor.submit(
                    build_phase2_output_record,
                    row,
                    prescription_rows,
                    doctor_name,
                    use_llm,
                    llm_model,
                    system_prompt,
                )
                futures[future] = (index, row)
            for future in as_completed(futures):
                index, row = futures[future]
                try:
                    records[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"第 {index + 1} 条病历处理失败，order_sn={normalize_id(row.get('病历ID'))}"
                    ) from exc
    if any(record is None for record in records):
        raise RuntimeError("二期总院病历处理结果不完整，已取消写出")
    return [record for record in records if record is not None]


def write_jsonl_output(records: list[dict[str, Any]], output_path: Path) -> None:
    """原子写出 JSONL 文件。"""
    if output_path.suffix.lower() != ".jsonl":
        output_path = output_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            for record in records:
                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def process_phase2_records(args: argparse.Namespace) -> Path:
    """处理二期总院新格式病历表和门诊收费明细。"""
    if args.batchsize < 1:
        raise ValueError("batchsize 必须是大于 0 的整数")
    doctor_name, record_file, prescription_file, output_path = resolve_phase2_paths(args)

    print(f"[1/4] 读取病历表: {record_file}")
    record_df = read_phase2_excel_all_sheets(record_file)
    require_columns(record_df, PHASE2_RECORD_REQUIRED_COLUMNS, record_file)
    before_records = len(record_df)
    record_df["_body_text"] = record_df["正文"].map(clean_body_text)
    record_df = record_df[record_df["_body_text"].ne("")].copy()
    print(f"      病历 {before_records} 行，正文非空 {len(record_df)} 行")

    print(f"[2/4] 读取门诊收费明细: {prescription_file}")
    prescription_df = read_phase2_excel_all_sheets(prescription_file)
    require_columns(prescription_df, PHASE2_PRESCRIPTION_REQUIRED_COLUMNS, prescription_file)
    before_prescriptions = len(prescription_df)
    prescription_df["_join_id"] = prescription_df["就诊ID"].map(normalize_id)
    prescription_df["_settlement_date"] = normalize_date_series(prescription_df["结算时间"])
    prescription_df = prescription_df[
        prescription_df["_join_id"].ne("")
        & prescription_df["退费标识"].map(clean_cell).ne("是")
        & prescription_df["费用类别"].map(clean_cell).isin(PHASE2_MEDICAL_CATEGORIES)
        & prescription_df["医院项目名称"].map(clean_cell).ne("")
    ].copy()
    record_df["_join_id"] = record_df["就诊ID"].map(normalize_id)
    record_df = record_df[record_df["_join_id"].ne("")].copy()
    valid_visit_ids = set(record_df["_join_id"])
    prescription_df = prescription_df[prescription_df["_join_id"].isin(valid_visit_ids)].copy()
    prescription_map = {
        normalize_id(join_id): group.reset_index(drop=True)
        for join_id, group in prescription_df.groupby("_join_id", sort=False)
    }
    record_df = record_df[record_df["_join_id"].isin(prescription_map)].copy().reset_index(drop=True)
    print(
        f"      处方明细 {before_prescriptions} 行，去退费/非药品费用并关联后 {len(prescription_df)} 行；"
        f"保留有处方的正文病历 {len(record_df)} 行"
    )

    if args.limit:
        record_df = record_df.head(args.limit).copy().reset_index(drop=True)
        needed_ids = set(record_df["_join_id"])
        prescription_map = {key: value for key, value in prescription_map.items() if key in needed_ids}

    use_llm = not getattr(args, "no_llm", False)
    system_prompt = load_phase2_prompt() if use_llm and not record_df.empty else ""
    print(f"[3/4] 使用就诊ID关联，batch 并发数={args.batchsize}，LLM={'开启' if use_llm else '关闭'}")
    records = build_phase2_output_records(
        record_df,
        prescription_map,
        doctor_name,
        use_llm,
        args.llm_model,
        system_prompt,
        args.batchsize,
    )
    print("[4/4] 原子写出 JSONL...")
    write_jsonl_output(records, output_path)
    print(f"✅ doctorname={doctor_name}，写出 {len(records)} 条记录 -> {output_path}")
    return output_path


def process_records(args: argparse.Namespace) -> Path:
    """读取两张表，按问诊单 ID 合并并写出最新结构化表。"""
    if should_use_phase2_layout(args):
        return process_phase2_records(args)

    if args.batchsize < 1:
        raise ValueError("batchsize 必须是大于 0 的整数")

    doctorname, record_file, prescription_file, output_path = resolve_paths(args)
    fee_categories = {clean_cell(category) for category in args.fee_categories if clean_cell(category)}

    print(f"[1/4] 读取病历表: {record_file}")
    raw_record_df = read_excel_all_sheets(record_file)
    print(f"      raw_data: {len(raw_record_df)} 行")

    print(f"[2/4] 读取处方明细表: {prescription_file}")
    raw_prescription_df = read_excel_all_sheets(prescription_file)
    print(f"      raw_data: {len(raw_prescription_df)} 行")

    # 过滤掉病历表中医案ID为空的行
    raw_record_df = raw_record_df.dropna(subset=ID_CANDIDATES, how="all").copy()
    print(f"      过滤空ID后: 病历 {len(raw_record_df)} 行")

    id_column = pick_id_column(raw_record_df, raw_prescription_df)
    print(f"[3/4] 使用关联字段: {id_column}")
    raw_record_df["问诊单ID"] = raw_record_df[id_column].map(normalize_id)
    raw_prescription_df["问诊单ID"] = raw_prescription_df[id_column].map(normalize_id)
    raw_record_df = raw_record_df[raw_record_df["问诊单ID"] != ""].copy()
    raw_prescription_df = raw_prescription_df[raw_prescription_df["问诊单ID"] != ""].copy()
    raw_prescription_df["剂量"] = raw_prescription_df["单量"].map(clean_dose)

    raw_record_df, raw_prescription_df, date_stats = filter_complete_same_day_records(
        raw_record_df,
        raw_prescription_df,
    )
    print(
        "      日期对齐: "
        f"病历日期字段={date_stats['record_date_column']}，处方日期字段={date_stats['prescription_date_column']}；"
        f"病历最大日期={date_stats['record_max_date']}，处方最大日期={date_stats['prescription_max_date']}；"
        f"截止日期={date_stats['cutoff_date']}"
    )
    print(
        "      完整性过滤后: "
        f"病历 {date_stats['before_record_count']} -> {date_stats['after_record_count']} 行；"
        f"处方 {date_stats['before_prescription_count']} -> {date_stats['after_prescription_count']} 行"
    )

    prescription_map = aggregate_prescriptions(raw_prescription_df, fee_categories=fee_categories)
    before_prescription_match_count = len(raw_record_df)
    raw_record_df = raw_record_df[raw_record_df["问诊单ID"].isin(prescription_map)].copy()
    print(
        "      有效处方过滤后: "
        f"病历 {before_prescription_match_count} -> {len(raw_record_df)} 行；"
        f"舍弃无处方/非目标费用类别记录 {before_prescription_match_count - len(raw_record_df)} 条"
    )

    if args.limit:
        raw_record_df = raw_record_df.head(args.limit).copy()

    matched_count = raw_record_df["问诊单ID"].isin(prescription_map).sum()

    print("[4/4] 合并并写出结构化表...")
    records = build_output_records(
        raw_record_df,
        prescription_map,
        args.use_llm,
        args.llm_model,
        args.batchsize,
    )
    write_output(records, output_path)

    print(f"✅ doctorname={doctorname}，写出 {len(records)} 条记录 -> {output_path}")
    print(f"   匹配到处方明细: {matched_count} 条；未匹配: {len(records) - matched_count} 条")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="处理总院线下医生病历和收费明细为结构化合并表。")
    parser.add_argument("--doctorname", "--doctoname", default="hukaiwen", help="医生数据标识，如 hukaiwen。")
    parser.add_argument("--record-file", type=Path, help="门诊病历表路径。")
    parser.add_argument("--prescription-file", type=Path, help="门诊收费/处方明细表路径。")
    parser.add_argument("--output", type=Path, help="输出文件路径；二期总院模式写为 JSONL。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="默认输出目录。")
    parser.add_argument(
        "--source-layout",
        choices=("auto", "legacy", "phase2"),
        default="auto",
        help="总院数据格式；auto 根据医生名或数据目录选择，phase2 用于 medical/data/二期总院。",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_PHASE2_DATA_DIR, help="二期总院 Excel 所在目录。")
    parser.add_argument("--fee-categories", nargs="*", default=list(FEE_CATEGORIES), help="需要合并的费用类别。")
    parser.add_argument("--use-llm", action="store_true", help="调用 LLM 从正文抽取病史字段。")
    parser.add_argument("--no-llm", action="store_true", help="二期总院模式下跳过 LLM，使用表格字段和正文标签兜底。")
    parser.add_argument("--llm-model", default="qwen3.6-flash", help="LLM 模型名。")
    parser.add_argument(
        "--batchsize",
        "--batch-size",
        "--batch_size",
        dest="batchsize",
        type=int,
        default=1,
        help="每批并发处理的病历数，默认 1（顺序处理）。",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条病历，0 表示处理全部。")
    return parser.parse_args()


if __name__ == "__main__":
    process_records(parse_args())
