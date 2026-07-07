"""
按照下面的要求 处理 medical/processed_data/postclear_audio_asr.json 并将新的处理结果存储为 medical/processed_data/postasr_speaker_content.json

1、只保留 speker 和 content 两个字段，去掉每组中的其他字段
2、保证每组医患对话为偶数，如果是奇数，且最后一个speaker为患者，删掉最后一组speaker,content. 如果是奇数且最后一个speaker为医生，新增一组speaker为患者，content为“好的”。
3、如果存在多个连续的speaker为患者，合并为一个。
4、如果存在多个连续的speaker为医生，在每一个医生speaker下，增加一个新的speaker为患者，content为“嗯嗯”。
"""

"""Post-process ASR dialogue data into normalized speaker-content pairs."""

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("medical/processed_data/postclear_audio_asr.json")
OUTPUT_PATH = Path("medical/processed_data/postasr_speaker_content.json")
PATIENT_SPEAKER = "患者"
DOCTOR_SPEAKER = "医生"
DEFAULT_PATIENT_REPLY = "好的"
BRIDGE_PATIENT_REPLY = "嗯嗯"


def merge_same_speaker(dialogue: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge consecutive turns from the same speaker."""
    merged_dialogue: list[dict[str, str]] = []

    for turn in dialogue:
        speaker = turn.get("speaker")
        content = (turn.get("content") or "").strip()
        if speaker not in {PATIENT_SPEAKER, DOCTOR_SPEAKER} or not content:
            continue

        if merged_dialogue and merged_dialogue[-1]["speaker"] == speaker:
            merged_dialogue[-1]["content"] = f"{merged_dialogue[-1]['content']} {content}".strip()
        else:
            merged_dialogue.append({"speaker": speaker, "content": content})

    return merged_dialogue


def normalize_doctor_runs(dialogue: list[dict[str, str]]) -> list[dict[str, str]]:
    """Insert placeholder patient replies between consecutive doctor turns."""
    normalized_dialogue: list[dict[str, str]] = []

    for turn in dialogue:
        if normalized_dialogue and normalized_dialogue[-1]["speaker"] == DOCTOR_SPEAKER and turn["speaker"] == DOCTOR_SPEAKER:
            normalized_dialogue.append({"speaker": PATIENT_SPEAKER, "content": BRIDGE_PATIENT_REPLY})
        normalized_dialogue.append(turn)
    return normalized_dialogue


def ensure_even_dialogue(dialogue: list[dict[str, str]]) -> list[dict[str, str]]:
    """Make the dialogue length even by trimming or padding the last patient turn."""
    if len(dialogue) % 2 == 0 or not dialogue:
        return dialogue

    if dialogue[-1]["speaker"] == PATIENT_SPEAKER:
        return dialogue[:-1]
    return dialogue + [{"speaker": PATIENT_SPEAKER, "content": DEFAULT_PATIENT_REPLY}]


def process_dialogue(dialogue: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply all normalization rules to one dialogue."""
    normalized = merge_same_speaker(dialogue)
    normalized = normalize_doctor_runs(normalized)
    normalized = ensure_even_dialogue(normalized)
    return normalized


def extract_dialogue(record: Any) -> list[dict[str, str]]:
    """Extract dialogue turns from mixed record structures."""
    if isinstance(record, dict):
        cleared_data = record.get("cleared_data")
        if isinstance(cleared_data, dict):
            dialogue = cleared_data.get("dialogue")
            if isinstance(dialogue, list):
                return dialogue

        dialogue = record.get("dialogue")
        if isinstance(dialogue, list):
            return dialogue

        if {"speaker", "content"}.issubset(record):
            return [record]

    if isinstance(record, list):
        return [turn for turn in record if isinstance(turn, dict)]

    return []


def main() -> None:
    """Read the source file, transform each dialogue, and write the result."""
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    processed_records = []
    for record in records:
        dialogue = extract_dialogue(record)
        processed_records.append(process_dialogue(dialogue))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(processed_records, output_file, ensure_ascii=False, indent=2)

    print(f"Processed {len(processed_records)} dialogues -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
