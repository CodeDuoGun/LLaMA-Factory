"""
将后清洗阶段的医患对话转换为 ShareGPT 格式的多轮对话数据集。

当前适配两类输入：
1. 结构化记录列表：[{record_id, is_first, record_text, cleared_data: {dialogue: [...]}}]
2. 纯对话列表：[[{speaker, content}, ...], ...]

输出：
  medical/processed_data/medical_consult_sharegpt.json
  medical/processed_data/medical_consult.sft.jsonl
"""

import json
import os
import re
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. System Prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """【角色】你是一位资深中医皮肤科医生，擅长通过视频问诊采集玫瑰痤疮患者的症状信息。

【任务】进行结构化多轮问诊，主动、循序渐进地采集症状，不给出最终诊断结论。

【问诊原则】
1. 每次只问一个问题，避免一次抛出多个问题
2. 初诊：先采集基本信息（年龄、性别、病程时长、主要症状及部位），再深入追问
3. 复诊：先询问上次治疗后的用药反应和症状变化，再针对性追问当前症状
4. 根据患者回答灵活调整追问方向，避免机械地走流程
5. 涉及专业术语时用通俗易懂的语言解释
6. 主动引导患者展示面部舌象（手机镜头指导）
7. 问诊结束时不给出辨证结论或治疗方案，只表达"继续观察/下次复诊"等

【沟通风格】亲切、专业、有耐心，问诊逻辑清晰。"""

# ──────────────────────────────────────────────────────────────────────────────
# 2. 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def parse_record_text(record_text: str) -> dict:
    """
    解析 record_text 字段，提取患者基本信息，组装成结构化摘要
    返回 dict: {sex, age, height, weight, appeal, history, current_symptoms, visit_records}
    """
    result = {}

    # 匹配 key:value 对
    pairs = re.findall(r'([^:,]+?)_([a-z_]+?):([^,]+?)(?:,|$)', record_text)
    for raw_key, key, val in pairs:
        key = key.strip()
        val = val.strip()
        if 'sex' in key:
            result['sex'] = val
        elif 'age' in key:
            result['age'] = val
        elif 'height' in key:
            result['height'] = val
        elif 'weight' in key:
            result['weight'] = val

    # doc_ass_stu_appeal
    m = re.search(r'doc_ass_stu_appeal[:：]([^,，\n]+)', record_text)
    if m:
        result['appeal'] = m.group(1).strip()

    # new_medical_history
    m = re.search(r'new_medical_history[:：](.+?)(?=\n[^取]|\Z)', record_text, re.DOTALL)
    if m:
        result['history'] = m.group(1).strip().replace('\n', ' ')

    # 提取各次就诊记录段落（以"YYYY.M/D第N诊"为分隔点）
    visit_sections = re.findall(
        r'(20\d{2}[./]\d{1,2}[./]\d{1,2})(第?\d+诊)[:：](.+?)(?=(?:20\d{2}[./]\d{1,2}[./]\d{1,2})(第?\d+诊)|$)',
        record_text, re.DOTALL)
    if visit_sections:
        # 最后一个就诊段落的刻下症
        last_section = visit_sections[-1][2]
        m2 = re.search(r'刻下症[:：]([^。.]+)', last_section)
        if m2:
            result['last_visit_symptoms'] = m2.group(1).strip()

    return result


def build_system(record: dict) -> str:
    """
    根据 record 构建完整的 system prompt。
    对于缺少病历元数据的纯对话输入，仅保留通用 system prompt 和默认就诊类型。
    """
    is_first = record.get("is_first", "")
    record_text = record.get("record_text", "")

    if is_first in (True, "初诊", "first_visit"):
        visit_type_cn = "初诊"
        visit_type_en = "first_visit"
    else:
        visit_type_cn = "复诊"
        visit_type_en = "return_visit"

    parts = [SYSTEM_PROMPT]
    parts.append(f"\n## 就诊类型：{visit_type_cn}（{visit_type_en}）")

    if not record_text:
        return "".join(parts)

    parsed = parse_record_text(record_text)

    # 基本信息
    info_lines = []
    if parsed.get("age"):
        info_lines.append(f"性别：{parsed.get('sex', '未知')}，年龄：{parsed.get('age', '')}岁")
    if parsed.get("height"):
        info_lines.append(f"身高：{parsed.get('height', '')}cm，体重：{parsed.get('weight', '')}kg")
    if parsed.get("appeal"):
        info_lines.append(f"主诉：{parsed.get('appeal', '')}")
    if parsed.get("history"):
        info_lines.append(f"病史：{parsed.get('history', '')}")

    if info_lines:
        parts.append("\n## 患者基本信息")
        for line in info_lines:
            parts.append(f"- {line}")

    if visit_type_cn == "复诊" and parsed.get("last_visit_symptoms"):
        parts.append(f"\n## 上次复诊时症状（参考）：{parsed['last_visit_symptoms']}")

    return "".join(parts)


def normalize_record(record, index: int):
    """
    将不同输入结构标准化为统一 record：
    - dict 结构：直接提取 cleared_data.dialogue
    - list 结构：视为一整段对话
    """
    if isinstance(record, dict):
        cleared_data = record.get("cleared_data", {})
        if isinstance(cleared_data, dict):
            dialogue = cleared_data.get("dialogue") or cleared_data.get("cleaned_dialogue") or []
        elif isinstance(cleared_data, list):
            dialogue = cleared_data
        else:
            dialogue = []

        return {
            "record_id": record.get("record_id") or f"dialogue_{index:06d}",
            "is_first": record.get("is_first", ""),
            "record_text": record.get("record_text", ""),
            "dialogue": dialogue,
        }

    if isinstance(record, list):
        return {
            "record_id": f"dialogue_{index:06d}",
            "is_first": "",
            "record_text": "",
            "dialogue": record,
        }

    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def build_conversations(dialogue_data, visit_type: str = "first_visit"):
    """
    从对话列表构建 conversations。

    LLaMA-Factory 要求：
      - 第 1,3,5... 轮 (even idx) = user_tag = "human"  → 患者输入
      - 第 2,4,6... 轮 (odd idx)  = assistant_tag = "gpt" → 医生输出

    原始数据以医生开头时，补一条患者触发语，使对话以 human → gpt 开始。
    """
    if isinstance(dialogue_data, dict):
        dialogue = dialogue_data.get("dialogue") or dialogue_data.get("cleaned_dialogue") or []
    elif isinstance(dialogue_data, list):
        dialogue = dialogue_data
    else:
        return []

    if not isinstance(dialogue, list):
        return []

    raw = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue

        speaker = str(turn.get("speaker", "")).strip()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if "医生" in speaker or "doctor" in speaker.lower():
            role = "gpt"
        else:
            role = "human"
        raw.append({"from": role, "value": content})

    if len(raw) < 2:
        return []

    if raw[0]["from"] == "gpt":
        trigger = "医生您好，我来看诊。" if visit_type == "first_visit" else "医生您好，我来进行复诊。"
        conversations = [{"from": "human", "value": trigger}] + raw
    else:
        conversations = raw

    validated = []
    for i, msg in enumerate(conversations):
        expected = "human" if i % 2 == 0 else "gpt"
        if msg["from"] == expected:
            validated.append(msg)

    return validated


def filter_conversations(conversations: list, min_turns: int = 4) -> list:
    """
    过滤：
    - 至少保留 min_turns 轮（一个完整问答至少需要）
    - human/gpt 交替至少出现过一次
    """
    if len(conversations) < min_turns:
        return []

    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not (has_human and has_gpt):
        return []

    # 额外检查：首尾不能连续同类角色（头尾各有一次 human/gpt 交替即可）
    # 修复：参考原始数据的角色（prev_role_from），避免级联翻转导致前几条全部错乱
    i = 0
    while i < len(conversations) - 1:
        if conversations[i]["from"] == conversations[i + 1]["from"]:
            # 把后一条翻转，翻成与前一条不同的角色
            conversations[i + 1]["from"] = "human" if conversations[i]["from"] == "gpt" else "gpt"
            # 如果翻转后仍与再下一条相同，继续处理（不推进 i），直到该位置角色不再连续重复
        else:
            i += 1

    return conversations


# ──────────────────────────────────────────────────────────────────────────────
# 3. 主转换逻辑
# ──────────────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_dir: str):
    """
    读取输入数据，输出 ShareGPT JSON + SFT JSONL。
    支持结构化 record 列表和纯对话列表两种输入。
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"读取数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sharegpt_data = []
    skipped_short = 0
    skipped_invalid = 0

    for index, record in enumerate(tqdm(data, desc="转换 ShareGPT 格式")):
        record_id = f"dialogue_{index:06d}"
        try:
            normalized = normalize_record(record, index)
            record_id = normalized["record_id"]
            is_first = normalized.get("is_first", "")
            visit_type = "first_visit" if is_first in (True, "初诊", "first_visit") else "return_visit"
            conversations = build_conversations(normalized["dialogue"], visit_type)
            conversations = filter_conversations(conversations, min_turns=4)

            if not conversations:
                skipped_short += 1
                continue

            system = build_system(normalized)

            item = {
                "conversations": conversations,
                "system": system,
                "record_id": record_id,
                "visit_type": visit_type,
            }

            sharegpt_data.append(item)
        except Exception as e:
            skipped_invalid += 1
            tqdm.write(f"跳过 record_id={record_id}, error={e}")

    # ── 保存 ShareGPT JSON ──────────────────────────────────────────────────
    sharegpt_path = os.path.join(output_dir, "medical_consult_sharegpt.json")
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

    # 同时拷贝到 data/ 目录（LLaMA-Factory 读取 data/ 下的文件）
    data_dir_sharegpt = os.path.join(os.path.dirname(output_dir), "..", "data", "medical_consult_sharegpt.json")
    data_dir_sharegpt = os.path.normpath(data_dir_sharegpt)
    with open(data_dir_sharegpt, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ ShareGPT 格式保存: {sharegpt_path}  ({len(sharegpt_data)} 条)")
    print(f"✓ 同步到: {data_dir_sharegpt}")

    # ── 保存 SFT JSONL（LLaMA-Factory 直接使用）────────────────────────────
    sft_path = os.path.join(output_dir, "medical_consult.sft.jsonl")
    sft_count = 0
    with open(sft_path, "w", encoding="utf-8") as f:
        for item in sharegpt_data:
            system = item.get("system", "")
            conversations = item.get("conversations", [])

            text_parts = []
            if system:
                text_parts.append(f"<|im_start|>system\n{system}<|im_end|>")
            for conv in conversations:
                role = conv.get("from", "")
                value = conv.get("value", "")
                if role == "human":
                    text_parts.append(f"<|im_start|>user\n{value}<|im_end|>")
                elif role == "gpt":
                    text_parts.append(f"<|im_start|>assistant\n{value}<|im_end|>")

            text = "\n".join(text_parts)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            sft_count += 1

    print(f"✓ SFT JSONL 保存: {sft_path}  ({sft_count} 条)")

    # ── 统计 ────────────────────────────────────────────────────────────────
    first_count = sum(1 for item in sharegpt_data if item.get("visit_type") == "first_visit")
    return_count = sum(1 for item in sharegpt_data if item.get("visit_type") == "return_visit")

    print(f"\n========== 转换统计 ==========")
    print(f"原始记录数  : {len(data)}")
    print(f"有效转换数  : {len(sharegpt_data)}")
    print(f"  - 初诊    : {first_count}")
    print(f"  - 复诊    : {return_count}")
    print(f"跳过（对话过短）: {skipped_short}")
    print(f"跳过（结构异常）: {skipped_invalid}")

    return sharegpt_data


# ──────────────────────────────────────────────────────────────────────────────
# 4. 注册数据集到 dataset_info.json
# ──────────────────────────────────────────────────────────────────────────────

def register_dataset(first_count: int, return_count: int):
    """
    将 medical_consult 添加到 data/dataset_info.json
    """
    info_path = "data/dataset_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info["medical_consult"] = {
        "file_name": "medical_consult_sharegpt.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        },
        "description": f"中医玫瑰痤疮多轮问诊数据集，含初诊{first_count}条、复诊{return_count}条，医生只问诊不给诊断。"
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✓ 已注册到 data/dataset_info.json -> medical_consult")


# ──────────────────────────────────────────────────────────────────────────────
# 5. 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT = "medical/processed_data/postasr_speaker_content.json"
    OUTPUT_DIR = "medical/processed_data"

    sharegpt_data = convert(INPUT, OUTPUT_DIR)
    first_c = sum(1 for item in sharegpt_data if item.get("visit_type") == "first_visit")
    return_c = sum(1 for item in sharegpt_data if item.get("visit_type") == "return_visit")
    register_dataset(first_c, return_c)

    print("\n=== 全部完成 ===")
    print(f"下一步：将 medical/processed_data/medical_consult.sft.jsonl 用于微调训练")
