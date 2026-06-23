"""
Prepare post-ASR medical dialogue data for prompt-based consultation training.

Input:
  medical/processed_data/postasr_speaker_content.jsonl

Output:
  medical/processed_data/postasr_prompt_context.jsonl

This script keeps the original record structure, adds pre_collected_info, and
removes dialogue fragments that only ask the patient to adjust camera, show face,
or show tongue. The target training setting assumes tongue analysis and exam
reports are already available before the conversation starts.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("medical/processed_data/postasr_speaker_content.jsonl")
DEFAULT_OUTPUT = Path("medical/processed_data/postasr_prompt_context.jsonl")

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
    "检查报告",
)

EXAM_STOP_MARKERS = (
    "刻下症",
    "舌象",
    "末次月经",
    "剖腹产",
    "顺产",
)

VIDEO_KEYWORDS = (
    "镜头",
    "前置",
    "后置",
    "正面",
    "左脸",
    "右脸",
    "左右脸",
    "转转",
    "转过来",
    "拉近",
    "舌头",
    "伸出来",
    "光线",
    "拍照",
)

VIDEO_VIEW_PATTERNS = (
    "让我看",
    "给我看",
    "我看看",
    "看一下",
)

SHORT_VIDEO_REPLIES = {
    "好",
    "好的",
    "嗯",
    "嗯嗯",
    "是",
    "可以",
    "前置",
    "后置",
    "看到了",
    "能看到",
}

INTENT_RE = re.compile(r"\s*\[意图:[^\]]+\]\s*")

PATIENT_VIDEO_KEYWORDS = (
    "镜头",
    "前置",
    "后置",
    "正面",
    "左右脸",
    "左脸",
    "右脸",
    "拉近",
    "伸出来",
    "美颜",
)

HIGH_CONFIDENCE_PATIENT_VIDEO_PHRASES = (
    "镜头拉近",
    "拉近一点",
    "再拉近",
    "舌头我看看",
    "左右脸我看看",
    "正面正面",
)

CLINICAL_SYMPTOM_KEYWORDS = (
    "红",
    "烫",
    "痒",
    "痛",
    "疼",
    "肿",
    "疹",
    "痘",
    "脓",
    "干",
    "紧",
    "药",
    "吃",
    "睡",
    "便",
    "月经",
    "报告",
    "检查",
)


def read_jsonl(path: Path) -> list[Any]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return records


def write_jsonl(records: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique_join(items: list[str]) -> str:
    seen = []
    for item in items:
        cleaned = item.strip(" ，,；;。")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return "；".join(seen)


def extract_tongue_analysis(record_text: str) -> str:
    matches = re.findall(
        r"舌(?:象)?[:：]?[淡红红胖嫩瘦紫暗苔薄白厚腻黄少齿痕裂纹点刺润燥，、；, ]+",
        record_text,
    )
    return unique_join(matches)


def extract_examination_report(record_text: str) -> str:
    reports = []
    for sentence in re.split(r"[。\n]", record_text):
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

    return unique_join(reports)


def split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[。！？?])", content)
    return [part.strip() for part in parts if part.strip()]


def strip_video_collection_sentences(content: str) -> str:
    kept = []
    for sentence in split_sentences(content):
        sentence_without_intent = INTENT_RE.sub("", sentence).strip()
        if not sentence_without_intent:
            continue
        if any(keyword in sentence_without_intent for keyword in VIDEO_KEYWORDS):
            continue
        if any(pattern in sentence_without_intent for pattern in VIDEO_VIEW_PATTERNS) and any(
            target in sentence_without_intent for target in ("脸", "面部", "舌", "患处", "皮肤")
        ):
            continue
        kept.append(sentence)
    return "".join(kept).strip()


def is_short_video_reply(content: str) -> bool:
    normalized = re.sub(r"[\s。！？?，,；;、]", "", content)
    return normalized in SHORT_VIDEO_REPLIES or len(normalized) <= 2


def is_patient_video_operation(content: str) -> bool:
    if any(phrase in content for phrase in HIGH_CONFIDENCE_PATIENT_VIDEO_PHRASES):
        return True
    if not any(keyword in content for keyword in PATIENT_VIDEO_KEYWORDS):
        return False
    if any(keyword in content for keyword in CLINICAL_SYMPTOM_KEYWORDS):
        return False
    return len(content) <= 40


def clean_dialogue(dialogue: list[dict[str, Any]]) -> tuple[list[dict[str, str]], int]:
    cleaned_dialogue = []
    removed_count = 0
    previous_removed_video_doctor = False

    for turn in dialogue:
        speaker = str(turn.get("speaker", "")).strip()
        content = str(turn.get("content", "")).strip()
        if not speaker or not content:
            continue

        if "医生" in speaker:
            stripped = strip_video_collection_sentences(content)
            if not stripped:
                removed_count += 1
                previous_removed_video_doctor = True
                continue

            cleaned_dialogue.append({"speaker": speaker, "content": stripped})
            previous_removed_video_doctor = False
            continue

        if previous_removed_video_doctor and is_short_video_reply(content):
            removed_count += 1
            previous_removed_video_doctor = False
            continue

        if is_patient_video_operation(content):
            removed_count += 1
            previous_removed_video_doctor = False
            continue

        cleaned_dialogue.append({"speaker": speaker, "content": content})
        previous_removed_video_doctor = False

    return cleaned_dialogue, removed_count


def process_record(record: Any) -> tuple[Any, int]:
    if not isinstance(record, dict):
        return record, 0

    processed = dict(record)
    record_text = str(processed.get("record_text", ""))
    pre_collected_info = dict(processed.get("pre_collected_info") or {})
    pre_collected_info["tongue_analysis"] = (
        pre_collected_info.get("tongue_analysis") or extract_tongue_analysis(record_text)
    )
    pre_collected_info["examination_report"] = (
        pre_collected_info.get("examination_report") or extract_examination_report(record_text)
    )
    processed["pre_collected_info"] = pre_collected_info

    cleared_data = processed.get("cleared_data")
    if isinstance(cleared_data, dict) and isinstance(cleared_data.get("dialogue"), list):
        cleaned_dialogue, removed_count = clean_dialogue(cleared_data["dialogue"])
        processed["cleared_data"] = dict(cleared_data)
        processed["cleared_data"]["dialogue"] = cleaned_dialogue
        return processed, removed_count

    if isinstance(processed.get("dialogue"), list):
        cleaned_dialogue, removed_count = clean_dialogue(processed["dialogue"])
        processed["dialogue"] = cleaned_dialogue
        return processed, removed_count

    return processed, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    processed_records = []
    removed_turns = 0

    for record in records:
        processed, removed_count = process_record(record)
        processed_records.append(processed)
        removed_turns += removed_count

    write_jsonl(processed_records, args.output)
    print(f"Processed {len(processed_records)} records -> {args.output}")
    print(f"Removed video-only dialogue turns: {removed_turns}")


if __name__ == "__main__":
    main()
