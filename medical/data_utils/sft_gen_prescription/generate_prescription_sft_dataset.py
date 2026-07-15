import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict
from medical.config import config
from openai import OpenAI
from medical.data_utils.knowledge import (
    TCMKnowledgeRetriever,
    format_kg_context_for_llm,
    format_relations_as_text,
    _PROPERTY_LABELS,
    _format_similar_cases,
)
from medical.data_utils.sft_generate_prescription.template import get_most_similar_template


DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")
DEFAULT_RECORD_FILE = Path("medical/data/sft_generate_prescription/extract_valid_record.json")
DEFAULT_OUTPUT_DIR = Path("medical/processed_data")
DEFAULT_DOCTOR_ID = os.getenv("KG_DOCTOR_ID", "wuweiping")
retriever = TCMKnowledgeRetriever(
    uri="bolt://127.0.0.1:7687",
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD,
    database=config.NEO4J_DATABASE,
    doctor_id=DEFAULT_DOCTOR_ID,
)


SFT_PRESCRIPTION_SYSTEM_PROMPT = """
你是一名资深中医皮肤科医生，擅长根据患者病历信息、知识图谱知识及模板方信息进行辨证论治并开具处方、给出调理建议。
你的任务是根据输入信息完成辨证分析、治则治法、配伍组方、调理建议及最终处方生成。
请先在 <think> 中完成辨证分析，再输出固定格式的 JSON。
【<think> 生成要求】
请严格依据输入信息进行辨证，不得凭空推断。
可使用的信息包括：
- 患者基本信息
- 主诉
- 现病史
- 症状变化
- 既往史
- 过敏史
- 舌面分析（如有）
- 患处分析（如有）
- 检查报告（如有）
- 中西医诊断（如有）
- 模板方匹配信息（如有）
- 知识图谱知识（如有）

辨证逻辑必须做到有据可循，每一个判断都必须对应具体依据。
<think> 要求：
- 需要依次分析：
1. 病位：结合主诉和病名，说明病位在面部肌肤，可关联肺胃、肝脾等，但必须说明依据。
2. 病性：依据患者已提供症状判断寒热虚实、湿热毒瘀等；信息不足时说明“证据不足”。
3. 病机演变：结合病程长短、症状变化、治疗反应分析，但不得过度推断。
4. 证候判断：优先依据患者当前症状；知识图谱候选证型仅作参考。信息不足时输出“倾向某证”。
5. 组方思路：说明内服、外洗、外涂等不同剂型的分工；若患者信息不足，应避免过重处方。
6. 君臣佐使：必须围绕实际处方药物说明，不得虚构药物作用。
7. 调理依据：饮食、运动建议必须与证候、病机、诱因控制一致。
- 有明确推理依据。
- 不超过300字。
- 不得输出任何没有依据的信息。


模板方使用规则：
1. 若最高模板方匹配度小于0.55，不得参考模板方，不得说明模板方内容，应完全依据病历、辨证及治法完成组方。
2. 若最高模板方匹配度大于等于0.55，可以参考模板方，但必须说明：
- 保留了哪些药物及依据；
- 去除了哪些药物及依据；
- 新增了哪些药物及依据；
- 调整了哪些药物剂量及依据。

【JSON 输出要求】

1. <think> 后仅输出 JSON。
2. 不要 Markdown,不要解释。
3. 知识图谱中的[相似病例参考]是按本患者刻下症检索到的相似历史病例及其真实处方（含剂量），仅供参考：须结合本患者刻下症辨证后进行组方，不得照搬；剂量须参考常见区间，不得臆造。
4. JSON 字段名称必须完全一致。
5. 不得新增、删除或修改字段名称。
6. 不得输出病历中不存在的信息。
7. 所有字符串不能为空，未知信息统一填写"未明确"。
"""
LLM_SYSTEM_PROMPT = """
你是一名资深中医皮肤科医生助手，负责结合病史信息、知识图谱信息、最高相似模板方信息，生成结构化辨证分析结果。
你的核心任务是根据病史信息和知识图谱信息中的【已知的诊断结果、辨证结果、处方内容】，反推辨证逻辑、反推治则治法、反推配伍逻辑，并保持输出字段不变。

你必须严格遵守以下要求：

【辨证逻辑】
1. 必须先依据患者当前病史、主诉、现病史、舌象、患处、检查报告、既往史进行辨证。
2. 知识图谱、三元组事实、相似病例、模板方只作为参考锚点，不得替代患者当前病情。
3. 不得出现“需与已知诊断结果/辨证结果保持一致”“根据既定证型”“已知辨证为”等倒推式表达。
4. 若病历信息不足，必须明确说明“不足以完全定证”，只能输出“倾向证候”或“待补充信息后确认”。
5. 不得根据相似病例或模板方反推患者存在未提供的症状、舌象、脉象、二便、睡眠、丘疹、脓疱、瘙痒等信息。
6. 辨证内容不超过300字。

【临床症状】
1. 必须根据主诉和现病史提取中医四诊相关内容，返回症状 list。
2. 每个症状必须是中文字符串，必须来自病历原文或对原文症状变化的概括，不得编造。
3. 可提取内容包括但不限于：舌象症状、患处症状、二便情况、女性患者月经情况、饮食睡眠情况、异常指标情况。
4. 如果现病史中包含多次问诊或复诊记录，应优先总结最新一次问诊的刻下症状，并保留明确的症状变化，如“较前减轻”“新发”“加重”“仍有”“消失”。
5. 不要把诊断名、证候名、处方药名、治疗建议当作临床症状。
6. 未提取到症状时返回空数组 []。

【治则治法】
1. 必须根据已知辨证结果、辨证逻辑和知识图谱知识反推治则治法。
2. 治则治法必须服务于已知诊断结果、辨证结果，不得生成与其冲突的治法。
3. 若知识图谱中治则治法为空，应基于已知辨证结果和辨证逻辑合理生成，并说明“知识图谱未提供明确治法”。

【模板匹配】
1. 如果最高模板匹配度低于0.55，不得参考模板方。
2. 如果最高模板匹配度大于等于0.55，可以参考模板方。
3. 需要说明保留、去除、新增、剂量调整的依据。
4. 模板匹配内容不超过300字。

【配伍逻辑】
1. 必须根据知识图谱信息和病史信息中的处方内容反推配伍逻辑。
2. 必须从君、臣、佐、使角度说明。
3. 必须结合处方药物、已知辨证结果、治则治法进行分析。
4. 配伍解释必须与已知诊断结果、辨证结果、处方内容严格一致。
5. 不得虚构病历中不存在的症状或检查结果。

【调理建议】
1. 必须结合病史信息和知识图谱知识。
2. 必须包含“饮食”和“运动”两个字段。
3. 饮食建议应包含宜忌，可包含具体食材、食疗方与禁忌。
4. 运动建议应包含运动类型、频次、强度与注意事项。

【输出要求】
1. 只输出 JSON。
2. 不要 Markdown。
3. 不要解释。
4. JSON 字段必须固定。
5. 不得输出病历中没有依据的检查结果、舌象或症状。
6. 所有字符串必须是中文。
7. JSON 必须可以被 json.loads 直接解析。

输出 JSON 格式如下：
{
  "辨证逻辑": "",
  "治则治法": "",
  "临床症状": [],
  "模板匹配": {
    "是否参考模板方": false,
    "候选模板方": "",
    "模板匹配度": 0.0,
    "判断依据": "",
    "加减逻辑": ""
  },
  "配伍逻辑": "",
  "调理建议": {
    "饮食": "",
    "运动": ""
  }
}
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    从 LLM 输出中提取 JSON，兼容模型偶尔输出 ```json 的情况。
    """
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"未找到 JSON 内容：{text}")
    return json.loads(match.group(0))


def validate_llm_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    兜底补齐固定字段，避免后续程序 KeyError。
    """
    default = {
        "辨证逻辑": "",
        "治则治法": "",
        "临床症状": [],
        "模板匹配": {
            "是否参考模板方": False,
            "候选模板方": "",
            "模板匹配度": 0.0,
            "判断依据": "",
            "加减逻辑": "",
        },
        "配伍逻辑": "",
        "调理建议": {
            "饮食": "",
            "运动": "",
        },
    }

    if not isinstance(data, dict):
        return default

    for key in ["辨证逻辑", "治则治法", "配伍逻辑"]:
        if key not in data or not isinstance(data[key], str):
            data[key] = default[key]

    if not isinstance(data.get("临床症状"), list):
        data["临床症状"] = default["临床症状"]
    data["临床症状"] = [str(symptom).strip() for symptom in data["临床症状"] if str(symptom).strip()]

    if not isinstance(data.get("模板匹配"), dict):
        data["模板匹配"] = default["模板匹配"]

    for key, value in default["模板匹配"].items():
        if key not in data["模板匹配"]:
            data["模板匹配"][key] = value

    data["模板匹配"]["是否参考模板方"] = bool(data["模板匹配"]["是否参考模板方"])

    try:
        data["模板匹配"]["模板匹配度"] = float(data["模板匹配"]["模板匹配度"])
    except Exception:
        data["模板匹配"]["模板匹配度"] = 0.0
    if not isinstance(data.get("调理建议"), dict):
        data["调理建议"] = default["调理建议"]
    for key, value in default["调理建议"].items():
        if key not in data["调理建议"]:
            data["调理建议"][key] = value
    return data


def build_user_prompt(
    patient_context: str,
    knowledge_context: str,
    most_similar_template_info: str,
) -> str:
    return f"""
请根据以下信息生成结构化辨证结果。
【患者病史信息】
{patient_context}

【知识图谱知识】
{knowledge_context}

【最高相似模板方信息】
{most_similar_template_info}


请严格输出 JSON，字段固定为：
辨证逻辑、治则治法、临床症状、模板匹配、配伍逻辑、调理建议。

注意：
1. 如果最高模板匹配度 < 0.55，则“是否参考模板方”为 false，“加减逻辑”为空字符串。
2. 如果最高模板匹配度 >= 0.55，则“是否参考模板方”为 true，并说明保留、去除、新增、剂量调整依据。
3. 不得编造患者未提供的检查报告、舌象、患处表现。
4. 调理建议必须包含“饮食”和“运动”。
5. 临床症状必须为 list，根据主诉和现病史总结最新刻下症状及明确症状变化。
6. 辩证逻辑必须和诊断结果一致，不得编造。
""".strip()


def format_known_prescription_details(prescriptions: list[Dict[str, Any]]) -> str:
    prescription_parts = []
    for idx, prescription in enumerate(prescriptions, start=1):
        part_items = [f"处方{idx}：{prescription.get('usage_type') or '未明确'}"]
        for drug in prescription.get("drugs", []) or []:
            if isinstance(drug, str):
                part_items.append(drug)
                continue

            drug_name = drug.get("drug_name") or drug.get("name") or ""
            dose = drug.get("dose") or drug.get("dosage") or ""
            unit = drug.get("unit") or ""
            drug_text = f"{drug_name}{dose}{unit}".strip()
            if drug_text:
                part_items.append(drug_text)
        prescription_parts.append("，".join(part_items))

    if not prescription_parts:
        return "【未明确】"
    return f"【{'，'.join(prescription_parts)}】"


def build_known_result_context(record_info: Dict[str, Any]) -> str:
    return f"""
【已知结果】
诊断结果：{record_info.get("diagnosis_illness") or "未明确"}
辨证结果：{record_info.get("diagnosis_disease") or "未明确"}
处方明细：{format_known_prescription_details(record_info.get("internal_prescriptions", []) or [])}
""".strip()


def append_known_result_context(user_prompt: str, known_result_context: str = "") -> str:
    if not known_result_context:
        return user_prompt
    return f"{user_prompt}\n\n{known_result_context}".strip()


def call_llm(
    patient_context: str,
    knowledge_context: str,
    most_similar_template_info: str,
    known_result_context: str = "",
    model: str = "qwen3.6-flash",
) -> Dict[str, Any]:
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    user_prompt = build_user_prompt(
        patient_context=patient_context,
        knowledge_context=knowledge_context,
        most_similar_template_info=most_similar_template_info,
    )
    user_prompt = append_known_result_context(user_prompt, known_result_context)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        top_p=0.8,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    data = extract_json(content)
    return validate_llm_result(data)


def get_template_prescription(template_file: Path | None = None):
    """读取文件 medical/data/wuweiping_template_prescription.json 并获取templates字段值"""
    template_file = template_file or DEFAULT_TEMPLATE_FILE
    with template_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("templates", [])


def get_records(record_file: Path | None = None):
    """
    读取文件medical/data/sft_generate_prescription/extract_valid_record.json
    """
    record_file = record_file or DEFAULT_RECORD_FILE
    with record_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_patient_context(record_info):
    """"""
    context = f"""患者性别{record_info["patient_sex"]},年龄{record_info["patient_age"]},病史信息：主诉{record_info.get("doc_ass_stu_appeal", "无")},现病史(可能包含中医四诊结果、检查报告结果):{record_info["new_medical_history"]},
既往史:{record_info["old_medical_history"] if record_info.get("old_medical_history") else "无"},过敏史:{record_info["allergic_history"] if record_info.get("allergic_history") else "无"},
家族史:{record_info["family_history"] if record_info.get("family_history") else "无"},个人史:{record_info["personal_history"] if record_info.get("personal_history") else "无"},
婚育史:{record_info["birth_detail"] if record_info.get("birth_detail") else "无"}\n"""
    return context

def _normalize_symptoms(symptoms):
    """Normalize symptoms from LLM output to a clean list."""
    if not symptoms:
        return []
    if isinstance(symptoms, str):
        symptoms = re.split(r"[、,，；;\n]+", symptoms)
    if not isinstance(symptoms, list):
        return []
    return [str(symptom).strip() for symptom in symptoms if str(symptom).strip()]


def _extract_internal_herbs(record_info):
    """Extract internal prescription herb names from a record."""
    herbs = []
    for prescription in record_info.get("internal_prescriptions", []) or []:
        if prescription.get("usage_type") != "内服":
            continue
        for herb_item in prescription.get("drugs", []) or []:
            if isinstance(herb_item, dict) and herb_item.get("drug_name"):
                herbs.append(herb_item["drug_name"])
        break
    return herbs


def _format_property_section(title, properties):
    """Format node properties as a text section."""
    if not properties:
        return f"{title}\n- 知识图谱中暂无相关属性。"

    lines = [title]
    for key, value in properties.items():
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = "、".join(str(item) for item in value if item)
        lines.append(f"- {key}：{value}")
    if len(lines) == 1:
        lines.append("- 知识图谱中暂无相关属性。")
    return "\n".join(lines)


def _format_node_properties(properties, max_items=20):
    """Format node properties with Chinese labels."""
    lines = []
    for key, value in (properties or {}).items():
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = "、".join(str(item) for item in value if item)
        label = _PROPERTY_LABELS.get(key, key)
        lines.append(f"- {label}：{value}")
        if len(lines) >= max_items:
            break
    return "\n".join(lines) if lines else "- 知识图谱中暂无相关属性。"


def _format_syndrome_relations(title, relations, max_relations=20):
    """Format syndrome relations from KG as readable text."""
    return format_relations_as_text(
        relations,
        title=title,
        max_relations=max_relations,
        empty_text="- 知识图谱中暂无相关证候关系。",
    )


def build_knowledge_sft_context(record_info, symptoms: list | str | None = None, doctor_id: str ="wuweiping"):
    """根据症状检索相似案例、同疾病知识、同证候知识，并格式化为文本。"""
    symptoms = _normalize_symptoms(symptoms)

    disease = record_info.get("diagnosis_illness", "")
    syndrome = record_info.get("diagnosis_disease", "")

    # 先根据症状list检索相似病例
    symptom_similar_cases = retriever.get_similar_cases(
        disease=disease,
        syndrome=syndrome,
        symptoms=symptoms,
        limit=3,
        doctor_id=doctor_id,
    )
    # 再检索同疾病知识、同证候知识
    disease_properties = retriever.get_disease_properties(disease, doctor_id=doctor_id)
    disease_relations = retriever.get_disease_relations(disease, doctor_id=doctor_id)
    syndrome_relations = retriever.get_syndrome_relations(syndrome, doctor_id=doctor_id)

    sections = [
        f"症状检索词：{'、'.join(symptoms) if symptoms else '无'}",
        "【同疾病知识】",
        _format_node_properties(disease_properties),
        "",
        format_relations_as_text(disease_relations, "【疾病知识参考】", 10),
        format_relations_as_text(syndrome_relations, "【同证候关系知识】", 10),
        "",
        _format_similar_cases("【症状相似病例】", symptom_similar_cases, max_cases=3),
    ]
    return "\n".join(sections)


def build_knowledge_context(record_info, doctor_id="wuweiping"):
    """"""
    prescriptions = []
    for prescription in record_info.get("internal_prescriptions", []):
        if prescription.get("usage_type") == "内服":
            prescriptions = prescription.get("drugs", [])
            break

    disease = record_info.get("diagnosis_illness", "")
    syndrome = record_info.get("diagnosis_disease", "")
    symptoms = []
    herbs = [herb_item["drug_name"] for herb_item in prescriptions if herb_item.get("drug_name")]
    print(f"doctor_id: {doctor_id}, disease: {disease}, syndrome: {syndrome}, symptoms: {symptoms}, herbs: {herbs}")
    knowledge_context = retriever.get_syndrome_diagnosis_context(
        disease,
        syndrome,
        symptoms,
        herbs,
        doctor_id=doctor_id,
    )

    return knowledge_context


def format_prescription_for_sft(prescriptions, llm_response):
    """"""
    result = []
    for idx, prescription in enumerate(prescriptions):
        # 内服处方给出配伍逻辑
        prescription_json = {
            "类型": prescription.get("drug_process_name", ""),
            "用法": prescription.get("usage_type", ""),
            "处方": prescription.get("drugs", []) or [],
            "医嘱": prescription.get("doctor_advice", ""),
            "用法用量": prescription.get("usage_desc", "")
        }
        if prescription.get("usage_type") == "内服":
            prescription_json["配伍逻辑"] = llm_response["配伍逻辑"]
        result.append({f"处方{idx+1}": prescription_json})
    return result

def build_sft_data(record_info, response, patient_context, knowledge_sft_context):
    """
    根据response生成assistant_content
    """
    print(f"sft 知识图谱信息 {knowledge_sft_context}")
    user_prompt = f"""请根据患者病史信息:{patient_context}, 知识图谱信息:{knowledge_sft_context}，先进行中医辨证，再输出诊断结果、处方结果、调理建议"""
    sft_system_prompt = SFT_PRESCRIPTION_SYSTEM_PROMPT 
    llm_response = validate_llm_result(response)
    # 模板匹配中新增模板明细信息，用于
    sft_json_out = {
        "中医证型": record_info["diagnosis_disease"],
        "诊断结果": record_info["diagnosis_illness"],
        "临床症状": llm_response["临床症状"],
        "处方建议": format_prescription_for_sft(record_info["internal_prescriptions"], llm_response),
        "治则治法": llm_response["治则治法"],
        "调理建议": llm_response["调理建议"],
    }
    assistant_content = (
        f"<think>{llm_response['辨证逻辑']}</think>"
        f"<json_start>{json.dumps(sft_json_out, ensure_ascii=False)}</json_end>"
    )
    messages = []
    messages.append({"role": "system", "content": sft_system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_content})
    return {
        "record_id": record_info.get("inquiry_id") or record_info.get("order_sn") or "",
        "messages": messages,
    }


def build_fallback_item(
    record_info: Dict[str, Any],
    error: Exception,
    index: int,
    patient_context: str = "",
    knowledge_context: str = "",
    most_similar_template_info: str = "",
    user_prompt: str = "",
) -> Dict[str, Any]:
    return {
        "index": index,
        "record_id": record_info.get("inquiry_id") or record_info.get("order_sn") or "",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "patient_context": patient_context,
        "knowledge_context": knowledge_context,
        "most_similar_template_info": most_similar_template_info,
        "user_prompt": user_prompt,
        "record": record_info,
    }


def save_json(data: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def gen_datasets(
    limit: int | None = None,
    record_file: Path | None = None,
    template_file: Path | None = None,
    records: list[Dict[str, Any]] | None = None,
    templates: list[Dict[str, Any]] | None = None,
    knowledge_builder: Callable[[Dict[str, Any]], str] = build_knowledge_context,
    llm_caller: Callable[[str, str, str], Dict[str, Any]] = call_llm,
    fallback_items: list[Dict[str, Any]] | None = None,
):
    if limit is not None and limit < 0:
        raise ValueError("limit 必须大于等于 0")

    records = list(get_records(record_file) if records is None else records)
    templates = get_template_prescription(template_file) if templates is None else templates
    if limit is not None:
        records = records[:limit]

    # 遍历records 生成sft数据
    datasets = []
    fallback_items = [] if fallback_items is None else fallback_items
    for index, record in enumerate(records):
        patient_context = ""
        knowledge_context = ""
        most_similar_template_info = ""
        user_prompt = ""
        try:
            patient_context = build_patient_context(record)
            knowledge_context = knowledge_builder(record)
            llm_knowledge_context = format_kg_context_for_llm(knowledge_context)
            most_similar_template_info = get_most_similar_template(record, templates)
            # print(f"*********llm_knowledge_context {llm_knowledge_context}")
            known_result_context = build_known_result_context(record)
            print(f"known_result_context: {known_result_context}")
            user_prompt = append_known_result_context(
                build_user_prompt(patient_context, llm_knowledge_context, most_similar_template_info),
                known_result_context,
            )
            if llm_caller is call_llm:
                response = llm_caller(
                    patient_context,
                    llm_knowledge_context,
                    most_similar_template_info,
                    known_result_context=known_result_context,
                )
            else:
                response = llm_caller(patient_context, llm_knowledge_context, most_similar_template_info)
            print(f"llm response: {response}")
            llm_response = validate_llm_result(response)
            knowledge_sft_context = build_knowledge_sft_context(record, llm_response.get("临床症状", record["doc_ass_stu_appeal"]))
            sft_data = build_sft_data(record, llm_response, patient_context, knowledge_sft_context)
            datasets.append(sft_data)
        except Exception as error:
            fallback_items.append(
                build_fallback_item(
                    record_info=record,
                    error=error,
                    index=index,
                    patient_context=patient_context,
                    knowledge_context=knowledge_context,
                    most_similar_template_info=most_similar_template_info,
                    user_prompt=user_prompt,
                )
            )
    return datasets



def main():
    parser = argparse.ArgumentParser(description="生成处方 SFT 数据集。")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的数据条数，默认处理全部数据。")
    parser.add_argument("--record-file", type=Path, default=None, help=f"有效病历输入路径，默认 {DEFAULT_RECORD_FILE}")
    parser.add_argument("--template-file", type=Path, default=None, help=f"模板方输入路径，默认 {DEFAULT_TEMPLATE_FILE}")
    parser.add_argument("--output", type=Path, default=None, help="最终 SFT 数据集输出路径。")
    parser.add_argument(
        "--fallback-output",
        type=Path,
        default=None,
        help="失败 fallback 数据输出路径。",
    )
    parser.add_argument(
        "--doctor-id",
        choices=["wuweiping", "hukaiwen"],
        default=DEFAULT_DOCTOR_ID,
        help="知识图谱检索的医生 ID，默认读取 KG_DOCTOR_ID 或 wuweiping。",
    )
    args = parser.parse_args()

    retriever.doctor_id = args.doctor_id
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.doctor_id}_prescription_sft_dataset.json"
    fallback_output = args.fallback_output or DEFAULT_OUTPUT_DIR / f"{args.doctor_id}_prescription_sft_fallback.json"
    fallback_items = []
    datasets = gen_datasets(
        limit=args.limit,
        record_file=args.record_file,
        template_file=args.template_file,
        fallback_items=fallback_items,
    )
    save_json(datasets, output)
    save_json(fallback_items, fallback_output)
    print(f"saved dataset: {output} ({len(datasets)} items)")
    print(f"saved fallback: {fallback_output} ({len(fallback_items)} items)")

if __name__ == "__main__":
    main()
