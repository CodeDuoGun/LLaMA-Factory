"""
将清洗后的医患对话数据转换为 sharegpt 格式，用于 LLaMA-Factory 多轮对话微调
"""
import json
import os
from tqdm import tqdm


def convert_to_sharegpt(cleared_data_file, output_file):
    """
    将 cleared_audio_asr.jsonl 转换为 sharegpt 格式

    输出格式：
    [
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
                ...
            ],
            "system": "你是一位资深中医...",
            "tools": [...]
        }
    ]
    """
    with open(cleared_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sharegpt_data = []
    system_prompt = """你是一位资深中医，擅长通过多轮问诊采集患者的症状信息，
进行中医辨证论治。请严格按照以下规则进行问诊：
1. 初诊患者需先采集基本信息（年龄、性别、主要症状、既往病史等）
2. 复诊患者需先评估上次治疗效果，再采集当前症状
3. 每次只问一个问题，循序渐进
4. 根据患者回答灵活调整问诊策略
5. 问诊过程中如涉及专业术语，需用通俗语言解释
6. 问诊结束时给出辨证结论和治疗建议"""

    for record in tqdm(data, desc="转换数据格式"):
        record_id = record.get("record_id", "")
        is_first = record.get("is_first")
        record_text = record.get("record_text", "")
        cleared_data = record.get("cleared_data", {})
        dialogue = cleared_data.get("cleaned_dialogue", [])

        if not dialogue:
            continue

        conversations = []
        for turn in dialogue:
            speaker = turn.get("speaker", "")
            content = turn.get("content", "")

            # 跳过空内容
            if not content or not content.strip():
                continue

            if "医" in speaker or "doctor" in speaker.lower():
                role = "gpt"
            else:
                role = "human"

            conversations.append({"from": role, "value": content})

        if len(conversations) < 2:
            continue

        # 构建 system prompt，融入患者基本信息
        system_with_info = system_prompt
        if record_text:
            system_with_info = f"{system_prompt}\n\n## 患者病历信息\n{record_text}"
        if is_first is not None:
            visit_type = "初诊" if is_first else "复诊"
            system_with_info = f"{system_with_info}\n## 就诊类型：{visit_type}"

        sharegpt_data.append({
            "conversations": conversations,
            "system": system_with_info
        })

    # 保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(sharegpt_data)} 条样本，保存至 {output_file}")
    return sharegpt_data


def convert_to_llamafactory_format(sharegpt_data, output_file):
    """
    转换为 LLaMA-Factory 支持的 dataset_info.jsonl 格式

    LLaMA-Factory 要求数据集注册在 data/dataset_info.json 中，
    然后将数据保存为 dataset_name.sft.jsonl 格式
    """
    output_lines = []
    for item in sharegpt_data:
        # 拼接对话为单条文本
        conversations = item.get("conversations", [])
        system = item.get("system", "")

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
        output_lines.append(json.dumps({"text": text}, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"LLaMA-Factory 格式转换完成，保存至 {output_file}")
    return output_lines


if __name__ == "__main__":
    os.makedirs("medical/processed_data", exist_ok=True)

    # 输入文件
    input_file = "medical/processed_data/cleared_audio_asr.jsonl"
    # 输出文件
    sharegpt_file = "medical/processed_data/medical_dialogue_sharegpt.json"
    llm_factory_file = "medical/processed_data/medical_dialogue.sft.jsonl"

    # 转换为 sharegpt 格式
    sharegpt_data = convert_to_sharegpt(input_file, sharegpt_file)

    # 转换为 LLaMA-Factory 格式
    convert_to_llamafactory_format(sharegpt_data, llm_factory_file)

    print("\n=== 转换完成 ===")
    print(f"sharegpt 格式: {sharegpt_file}")
    print(f"LLaMA-Factory 格式: {llm_factory_file}")
