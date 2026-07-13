# ruff: noqa: D205, D212, D301, D415, W605
"""
处理总院线下医生的病历数据为结构化数据
门诊病历表：medical/data/zongyuan_hukaiwen/胡凯文门诊病历20250101-20260630(1).xlsx
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
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "medical/processed_data"

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
        "record_file": PROJECT_ROOT / "medical/data/zongyuan_hukaiwen/胡凯文门诊病历20250101-20260630(1).xlsx",
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


def pick_id_column(record_df: pd.DataFrame, prescription_df: pd.DataFrame) -> str:
    """从两张表中选择共同的问诊单 ID 字段。"""
    for column in ID_CANDIDATES:
        if column in record_df.columns and column in prescription_df.columns:
            return column
    raise ValueError(f"无法找到共同 ID 字段，候选字段: {', '.join(ID_CANDIDATES)}")


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
    drugs = [build_drug_item(row) for row in rows]
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
        is_first="初诊",
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
    """写出 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(".json")

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_records(args: argparse.Namespace) -> Path:
    """读取两张表，按问诊单 ID 合并并写出最新结构化表。"""
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

    if args.limit:
        raw_record_df = raw_record_df.head(args.limit).copy()

    prescription_map = aggregate_prescriptions(raw_prescription_df, fee_categories=fee_categories)
    matched_count = raw_record_df["问诊单ID"].isin(prescription_map).sum()

    print("[4/4] 合并并写出结构化表...")
    records = [
        build_output_record(row, prescription_map, args.use_llm, args.llm_model)
        for _, row in raw_record_df.iterrows()
    ]
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
    parser.add_argument("--output", type=Path, help="输出文件路径，支持 .xlsx/.csv/.json。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="默认输出目录。")
    parser.add_argument("--fee-categories", nargs="*", default=list(FEE_CATEGORIES), help="需要合并的费用类别。")
    parser.add_argument("--use-llm", action="store_true", help="调用 LLM 从正文抽取病史字段。")
    parser.add_argument("--llm-model", default="qwen3.6-flash", help="LLM 模型名。")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条病历，0 表示处理全部。")
    return parser.parse_args()


if __name__ == "__main__":
    process_records(parse_args())
