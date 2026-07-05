import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict
from medical.config import config
from openai import OpenAI
from medical.data_utils.knowledge import TCMKnowledgeRetriever, format_kg_context_for_llm
from medical.data_utils.sft_generate_prescription.template import get_most_similar_template


DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")
DEFAULT_RECORD_FILE = Path("medical/data/sft_generate_prescription/extract_valid_record.json")
DEFAULT_OUTPUT_FILE = Path("medical/data/sft_generate_prescription/prescription_sft_dataset.json")
DEFAULT_FALLBACK_FILE = Path("medical/data/sft_generate_prescription/prescription_sft_fallback.json")
retriever = TCMKnowledgeRetriever(
    uri="bolt://127.0.0.1:7687",
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD,
    database=config.NEO4J_DATABASE,
)


SFT_PRESCRIPTION_SYSTEM_PROMPT = """
你是一名资深中医皮肤科医生，擅长根据患者病历、舌面分析、患处分析、检查报告、知识图谱知识及模板方信息进行辨证论治并开具处方。
你的任务是根据输入信息完成辨证分析、治则治法、配伍组方及最终处方生成。
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

需要依次分析：
1. 主症依据。
2. 兼症依据。
3. 舌象依据（未提供请说明"未提供"，不得推断）。
4. 患处依据（未提供请说明"未提供"，不得推断）。
5. 检查依据（未提供请说明"未提供"，不得推断）。
6. 结合上述依据分析病机。
7. 根据病机确定证候。
8. 根据证候确定治则治法。
9. 根据治则治法分析最终组方思路。

模板方使用规则：
1. 若最高模板方匹配度小于0.55，不得参考模板方，不得说明模板方内容，应完全依据病历、辨证及治法完成组方。
2. 若最高模板方匹配度大于等于0.55，可以参考模板方，但必须说明：
- 保留了哪些药物及依据；
- 去除了哪些药物及依据；
- 新增了哪些药物及依据；
- 调整了哪些药物剂量及依据。

<think> 要求：
- 简洁专业。
- 有明确推理依据。
- 不超过300字。
- 不得输出任何没有依据的信息。

【JSON 输出要求】

1. <think> 后仅输出 JSON。
2. 不要 Markdown。
3. 不要解释。
4. JSON 字段名称必须完全一致。
5. 不得新增、删除或修改字段名称。
6. 不得输出病历中不存在的信息。
7. 所有字符串不能为空，未知信息统一填写"未明确"。
"""
LLM_SYSTEM_PROMPT = """
你是一名资深中医皮肤科医生助手，负责结合病史信息、知识图谱知识、最高相似模板方信息，生成结构化辨证分析结果。

你必须严格遵守以下要求：

【辨证逻辑】
1. 必须结合患者主诉、症状变化、既往病史、检查报告结果【如有】、舌面结果【如有】、患处分析结果【如有】进行辨证。
2. 每个辨证判断都必须绑定具体依据，不能凭空推断。
3. 未提供的信息必须说明“未提供”，不得作为依据。
4. 需要区分主症、兼症、舌面/患处/检查报告依据，并说明这些依据如何支持证型、病机、治则治法。
5. 辨证内容不超过300字。

【治则治法】
1. 根据辨证逻辑和知识图谱知识生成。
2. 若知识图谱中治则治法为空，应基于辨证逻辑合理生成，并说明“知识图谱未提供明确治法”。

【模板匹配】
1. 如果最高模板匹配度低于0.55，不得参考模板方。
2. 如果最高模板匹配度大于等于0.55，可以参考模板方。
3. 需要说明保留、去除、新增、剂量调整的依据。
4. 模板匹配内容不超过300字。

【配伍逻辑】
1. 必须从君、臣、佐、使角度说明。
2. 必须结合处方药物、证候、治法进行分析。
3. 不得虚构病历中不存在的症状或检查结果。

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
辨证逻辑、治则治法、模板匹配、配伍逻辑、调理建议。

注意：
1. 如果最高模板匹配度 < 0.55，则“是否参考模板方”为 false，“加减逻辑”为空字符串。
2. 如果最高模板匹配度 >= 0.55，则“是否参考模板方”为 true，并说明保留、去除、新增、剂量调整依据。
3. 不得编造患者未提供的检查报告、舌象、患处表现。
4. 调理建议必须包含“饮食”和“运动”。
4. 辩证逻辑必须和诊断结果一致，不得编造。
""".strip()


def call_llm(
    patient_context: str,
    knowledge_context: str,
    most_similar_template_info: str,
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


def get_template_prescription():
    """读取文件 medical/data/wuweiping_template_prescription.json 并获取templates字段值"""
    with DEFAULT_TEMPLATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("templates", [])


def get_records():
    """
    读取文件medical/data/sft_generate_prescription/extract_valid_record.json
    """ 
    with DEFAULT_RECORD_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_patient_context(record_info):
    """"""
    context = f"""患者性别{record_info["patient_sex"]},年龄{record_info["patient_age"]},病史信息：主诉{record_info.get("doc_ass_stu_appeal", "无")},现病史(可能包含中医四诊结果、检查报告结果):{record_info["new_medical_history"]},
既往史:{record_info.get("old_medical_history", "无")},过敏史:{record_info.get("allergic_history", "无")},
家族史:{record_info.get("family_history", "无")},个人史:{record_info.get("personal_history", "无")},
婚育史:{record_info.get("birth_detail", "无")}\n
    """
    return context

def build_knowledge_symptoms_context(record_info):
    """根据诊断结果获取知识图谱信息，获取相似病历信息"""
    pass


def build_knowledge_context(record_info):
    """"""
    prescriptions = []
    for prescription in record_info.get("internal_prescriptions", []):
        if prescription.get("usage_type") == "内服":
            prescriptions = prescription.get("drugs", [])
            break

    disease = record_info.get("diagnosis_illness", "")
    syndrome = record_info.get("diagnosis_disease", "")
    # TODO 症状未确定
    # symptoms = [record_info.get("doc_ass_stu_appeal", "")]
    symptoms = []
    herbs = [herb_item["drug_name"] for herb_item in prescriptions if herb_item.get("drug_name")]
    print(f"disease: {disease}, syndrome: {syndrome}, symptoms: {symptoms}, herbs: {herbs}")
    knowledge_context = retriever.get_syndrome_diagnosis_context(disease, syndrome, symptoms, herbs)
    knowledge_context = format_kg_context_for_llm(knowledge_context)
    print(f"knowledge context: {knowledge_context}")
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

def build_sft_data(record_info, response, patient_context, knowledge_context):
    """
    根据response生成assistant_content
    """
    user_prompt = f"""请根据患者病史信息:{patient_context}, 知识图谱信息:{knowledge_context}，相似病例信息，先进行中医辨证，再输出诊断结果、处方结果、调理建议、模版匹配结果"""
    sft_system_prompt = SFT_PRESCRIPTION_SYSTEM_PROMPT 
    llm_response = validate_llm_result(response)
    sft_json_out = {
        "中医诊断": record_info["diagnosis_sickness"],
        "中医证型": record_info["diagnosis_disease"],
        "西医诊断": record_info["diagnosis_illness"],
        "处方建议": format_prescription_for_sft(record_info["internal_prescriptions"], llm_response),
        "治则治法": llm_response["治则治法"],
        "调理建议": llm_response["调理建议"],
        "模板匹配": llm_response["模板匹配"],
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
    records: list[Dict[str, Any]] | None = None,
    templates: list[Dict[str, Any]] | None = None,
    knowledge_builder: Callable[[Dict[str, Any]], str] = build_knowledge_context,
    llm_caller: Callable[[str, str, str], Dict[str, Any]] = call_llm,
    fallback_items: list[Dict[str, Any]] | None = None,
):
    if limit is not None and limit < 0:
        raise ValueError("limit 必须大于等于 0")

    records = list(get_records() if records is None else records)
    templates = get_template_prescription() if templates is None else templates
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
            most_similar_template_info = get_most_similar_template(record, templates)
            user_prompt = build_user_prompt(patient_context, knowledge_context, most_similar_template_info)
            response = llm_caller(patient_context, knowledge_context, most_similar_template_info)
            print(f"llm response: {response}")
            knowledge_symptoms_context = build_knowledge_symptoms_context(record)
            sft_data = build_sft_data(record, response, patient_context, knowledge_context)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="最终 SFT 数据集输出路径。")
    parser.add_argument(
        "--fallback-output",
        type=Path,
        default=DEFAULT_FALLBACK_FILE,
        help="失败 fallback 数据输出路径。",
    )
    args = parser.parse_args()

    fallback_items = []
    datasets = gen_datasets(limit=args.limit, fallback_items=fallback_items)
    save_json(datasets, args.output)
    save_json(fallback_items, args.fallback_output)
    print(f"saved dataset: {args.output} ({len(datasets)} items)")
    print(f"saved fallback: {args.fallback_output} ({len(fallback_items)} items)")

if __name__ == "__main__":
    main()
