"""
将后清洗阶段的医患对话转换为 ShareGPT 格式的多轮对话数据集。

当前适配两类输入：
1. JSON/JSONL 结构化记录：[{record_id, is_first, record_text, cleared_data: {dialogue: [...]}}]
2. JSON/JSONL 纯对话：[[{speaker, content}, ...], ...]

输出：
  medical/processed_data/medical_consult_sharegpt.json
  medical/processed_data/medical_consult.sft.jsonl
"""

import json
import os
import re

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

    tqdm.write = print


EXAM_KEYWORDS = (
    "肝肾功能",
    "血尿常规",
    "尿常规",
    "血常规",
    "过敏原",
    "皮肤镜",
    "真菌",
    "螨虫",
    "B超",
    "彩超",
    "甲状腺",
)

EXAM_STOP_MARKERS = (
    "刻下症",
    "舌象",
    "末次月经",
    "剖腹产",
    "顺产",
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. System Prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """【角色】你是一位资深中医皮肤科医生，擅长根据患者主诉、既往病史、舌面分析结果和检查报告结果，对玫瑰痤疮患者进行多轮问诊。

【总任务】根据就诊类型进行结构化多轮问诊。你需要主动追问、循序渐进地收集病情信息，并在信息充分后给出安全、谨慎、可执行的医嘱或随访安排。

【通用问诊原则】
1. 每轮只围绕一个重点提问，避免一次抛出多个无关问题。
2. 先听患者当前最困扰的问题，再结合已提供的舌面分析结果、检查报告结果、病史和用药情况逐步追问。
3. 舌面情况和检查报告已经提前获取，不要在问诊中重复采集影像或舌面资料。
4. 围绕红斑、丘疹/脓疱、发烫、瘙痒、疼痛、肿胀、干紧、破溃流水、色沉等表现追问程度、频率、诱因和变化。
5. 结合中医四诊追问食纳、大小便、睡眠、口干口苦、胃胀反酸等情况；舌面相关判断以已给出的舌面分析结果为准。
6. 追问既往诊疗、外用/口服药、光电或美容项目、过敏史、肝肾功能、血尿常规、牙齿处理、鼻炎/咽炎、妇科和月经情况等与玫瑰痤疮相关的线索。
7. 根据患者回答灵活调整追问方向，避免机械照读清单。
8. 涉及专业术语时用通俗语言解释，表达要亲切、专业、有耐心。

【初诊问诊流程】
1. 开始先确认主诉：患者最想解决的问题、病程多久、最近是否加重。
2. 根据患者描述明确皮损部位、范围、颜色、是否对称、是否有丘疹脓疱或肿胀破溃；不进行额外影像采集。
3. 追问当前症状：红、烫、痒、痛、干紧、夜间发烫、遇热/情绪/饮食/口罩/护肤品后的变化。
4. 追问既往治疗：在哪些医院诊治过，诊断为什么，用过哪些口服药、外用药、光电美容或护肤修复，疗效和不良反应如何。
5. 补充基础病史：饮食二便睡眠、胃肠情况、牙齿情况、鼻炎咽炎、过敏史、肝肾功能和血尿常规；女性患者询问月经周期、经量、血块、经期前后皮肤变化和生育史。
6. 信息充分后给出初诊医嘱：皮肤保护、饮食作息、检查建议、用药观察要点和复诊安排；避免武断保证疗效。

【复诊问诊流程】
1. 开始先复盘上次治疗：口服药、外洗/外敷/外涂药是否按医嘱使用，有无腹泻、胃痛、口干、瘙痒、红烫加重、过敏等异常反应。
2. 对比上次和现在的症状变化：红斑面积和颜色、发烫频率、丘疹脓疱、瘙痒、肿胀、夜烫、干紧、睡眠和二便是否改善。
3. 确认剩余药量和患者实际用法，区分中药、西药、抗过敏药、外用药、护肤品或自行加用药物。
4. 追问复诊期间新增情况：检查复查结果、肝肾功能变化、其他医院处理、牙齿/鼻炎/胃肠/月经变化、饮食作息和环境诱因。
5. 对女性患者重点追问近期月经：周期、经量、颜色、血块、经期前后面部红烫痒痛变化。
6. 信息充分后给出复诊医嘱：是否继续、暂停或调整既有方案，外用药如何试用，何时复查，出现哪些情况需要停药或线下就诊。

【输出边界】
1. 可以进行病情解释、风险提醒、用药观察和生活方式医嘱，但不要给出绝对化诊断或保证疗效。
2. 如果信息不足，应继续追问；不要在缺少关键病史时直接下结论。
3. 涉及肝肾功能异常、明显过敏、破溃流脓、严重肿胀或全身不适时，要建议复查或线下就医。"""

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

    tongue_matches = re.findall(r'舌(?:象)?[:：]?[淡红红胖嫩瘦紫暗苔薄白厚腻黄少齿痕裂纹点刺润燥，、；, ]+', record_text)
    if tongue_matches:
        result["tongue_analysis"] = "；".join(dict.fromkeys(item.strip(" ，,；") for item in tongue_matches if item.strip()))

    reports = []
    for sentence in re.split(r'[。\n]', record_text):
        sentence = sentence.strip(" ，,；;")
        if not sentence:
            continue

        keyword_positions = [
            sentence.find(keyword)
            for keyword in EXAM_KEYWORDS
            if keyword in sentence
        ]
        if not keyword_positions:
            continue

        start = max(sentence.rfind(delimiter, 0, min(keyword_positions)) for delimiter in "，,；;。")
        report = sentence[start + 1:].strip(" ，,；;")
        for marker in EXAM_STOP_MARKERS:
            marker_index = report.find(marker)
            if marker_index > 0:
                report = report[:marker_index].strip(" ，,；;")
        reports.append(report)

    if reports:
        result["examination_report"] = "；".join(dict.fromkeys(reports))

    return result


def build_system(record: dict) -> str:
    """
    根据 record 构建完整的 system prompt。
    对于缺少病历元数据的纯对话输入，仅保留通用 system prompt 和默认就诊类型。
    """
    is_first = record.get("is_first", "")
    record_text = record.get("record_text", "")
    pre_collected_info = record.get("pre_collected_info", {})

    if is_first in (True, "初诊", "first_visit"):
        visit_type_cn = "初诊"
        visit_type_en = "first_visit"
    else:
        visit_type_cn = "复诊"
        visit_type_en = "return_visit"

    parts = [SYSTEM_PROMPT]
    parts.append(f"\n## 就诊类型：{visit_type_cn}（{visit_type_en}）")

    parsed = parse_record_text(record_text) if record_text else {}

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
            parts.append(f"\n- {line}")

    if visit_type_cn == "复诊" and parsed.get("last_visit_symptoms"):
        parts.append(f"\n## 上次复诊时症状（参考）：{parsed['last_visit_symptoms']}")

    tongue_analysis = pre_collected_info.get("tongue_analysis") or parsed.get("tongue_analysis")
    examination_report = pre_collected_info.get("examination_report") or parsed.get("examination_report")

    parts.append("\n## 已提前获取的信息")
    parts.append(f"\n- 舌面分析结果：{tongue_analysis or '未提供'}")
    parts.append(f"\n- 检查报告结果：{examination_report or '未提供'}")

    return "".join(parts)


def load_records(input_path: str):
    """
    读取 JSON 数组或 JSONL 数据。
    JSONL 每行可以是结构化 record，也可以是一段纯对话列表。
    """
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.endswith(".jsonl"):
            records = []
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            return records

        return json.load(f)


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
            "pre_collected_info": record.get("pre_collected_info", {}),
            "dialogue": dialogue,
        }

    if isinstance(record, list):
        return {
            "record_id": f"dialogue_{index:06d}",
            "is_first": "",
            "record_text": "",
            "pre_collected_info": {},
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
    data = load_records(input_path)

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
        "description": f"中医玫瑰痤疮多轮问诊数据集，含初诊{first_count}条、复诊{return_count}条，按初诊和复诊流程进行多轮问诊并给出谨慎医嘱。"
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✓ 已注册到 data/dataset_info.json -> medical_consult")


# ──────────────────────────────────────────────────────────────────────────────
# 5. 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT = "medical/processed_data/postasr_prompt_context.jsonl"
    OUTPUT_DIR = "medical/processed_data"

    sharegpt_data = convert(INPUT, OUTPUT_DIR)
    first_c = sum(1 for item in sharegpt_data if item.get("visit_type") == "first_visit")
    return_c = sum(1 for item in sharegpt_data if item.get("visit_type") == "return_visit")
    register_dataset(first_c, return_c)

    print("\n=== 全部完成 ===")
    print(f"下一步：将 medical/processed_data/medical_consult.sft.jsonl 用于微调训练")
