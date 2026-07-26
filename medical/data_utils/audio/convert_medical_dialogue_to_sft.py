#!/usr/bin/env python3
"""Convert doctor-patient dialogue exports to LLaMA-Factory SFT JSON.

Input format expected:
[
  {
    "record_id": "...",
    "is_first": "初诊|复诊",
    "record_text": "...",
    "cleared_data": {
      "dialogue": [
        {"speaker": "医生", "content": "..."},
        {"speaker": "患者", "content": "..."}
      ]
    }
  }
]

Output format:
[
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "record_id": "...",
    "metadata": {...}
  }
]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INTENT_RE = re.compile(r"\s*\[意图:\s*[^\]]+\]\s*")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")
PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9})(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")


def clean_text(text: Any, *, remove_intent: bool = True, desensitize: bool = True) -> str:
    if text is None:
        return ""
    text = str(text)
    if remove_intent:
        text = INTENT_RE.sub("", text)
    if desensitize:
        text = PHONE_RE.sub("[手机号]", text)
        text = ID_CARD_RE.sub("[身份证号]", text)
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def speaker_to_role(speaker: str) -> Optional[str]:
    speaker = (speaker or "").strip()
    if speaker in {"患者", "病人", "用户", "家属"}:
        return "user"
    if speaker in {"医生", "医师", "大夫", "助手"}:
        return "assistant"
    return None


def merge_same_role(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] = f'{merged[-1]["content"]}\n{content}'
        else:
            merged.append({"role": role, "content": content})
    return merged


def make_case_context(item: Dict[str, Any], max_record_chars: int) -> str:
    parts = []
    record_id = clean_text(item.get("record_id"), remove_intent=False)
    is_first = clean_text(item.get("is_first"), remove_intent=False)
    record_text = clean_text(item.get("record_text"))
    if max_record_chars > 0 and len(record_text) > max_record_chars:
        record_text = record_text[:max_record_chars].rstrip() + "..."

    if record_id:
        parts.append(f"病例编号：{record_id}")
    if is_first:
        parts.append(f"就诊类型：{is_first}")
    if record_text:
        parts.append(f"病历背景：{record_text}")

    if not parts:
        return ""
    return "以下是患者病历背景，请基于背景进行真实、谨慎的问诊对话。\n" + "\n".join(parts)


def normalize_dialogue(
    item: Dict[str, Any],
    *,
    include_record_context: bool,
    max_record_chars: int,
    remove_intent: bool,
    desensitize: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    dialogue = item.get("cleared_data", {}).get("dialogue", [])
    if not isinstance(dialogue, list) or not dialogue:
        return None, "empty_dialogue"

    raw_messages: List[Dict[str, str]] = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        role = speaker_to_role(str(turn.get("speaker", "")))
        if role is None:
            continue
        content = clean_text(
            turn.get("content", ""),
            remove_intent=remove_intent,
            desensitize=desensitize,
        )
        if content:
            raw_messages.append({"role": role, "content": content})

    messages = merge_same_role(raw_messages)
    if len(messages) < 2:
        return None, "too_few_messages"

    # LLaMA-Factory chat SFT is most useful when training assistant responses.
    # If a transcript starts with a doctor instruction, keep it by prepending a
    # synthetic patient request instead of dropping clinically useful dialogue.
    if messages[0]["role"] == "assistant":
        context = make_case_context(item, max_record_chars) if include_record_context else ""
        synthetic = "请根据当前就诊信息开始问诊。"
        if context:
            synthetic = context + "\n\n" + synthetic
        messages.insert(0, {"role": "user", "content": synthetic})
    elif include_record_context:
        context = make_case_context(item, max_record_chars)
        if context:
            messages[0]["content"] = context + "\n\n患者：" + messages[0]["content"]

    # Training samples should end on an assistant response.
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant_target"

    record_id = clean_text(item.get("record_id"), remove_intent=False)
    sample = {
        "messages": messages,
        "record_id": record_id,
        "metadata": {
            "is_first": item.get("is_first"),
            "source_turns": len(dialogue),
            "sft_turns": len(messages),
        },
    }
    return sample, None


def split_samples(
    samples: List[Dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    test = shuffled[:test_n]
    val = shuffled[test_n : test_n + val_n]
    train = shuffled[test_n + val_n :]
    return train, val, test


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert doctor-patient dialogue JSON to LLaMA-Factory messages SFT JSON."
    )
    parser.add_argument("--input", required=True, help="Path to raw dialogue JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for converted JSON files.")
    parser.add_argument("--doctor", type=str, required=True, help="doctor name")
    parser.add_argument("--dataset-name", default="medical_consult_sft", help="Output file prefix.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-record-chars", type=int, default=1200)
    parser.add_argument("--no-record-context", action="store_true", help="Do not prepend record_text context.")
    parser.add_argument("--keep-intent", action="store_true", help="Keep [意图: xxx] tags in outputs.")
    parser.add_argument("--no-desensitize", action="store_true", help="Do not mask phone/id-card patterns.")
    parser.add_argument("--min-assistant-chars", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise ValueError("Input JSON must be a list of records.")

    samples: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for item in raw_data:
        if not isinstance(item, dict):
            skipped["non_dict_record"] = skipped.get("non_dict_record", 0) + 1
            continue
        sample, reason = normalize_dialogue(
            item,
            include_record_context=not args.no_record_context,
            max_record_chars=args.max_record_chars,
            remove_intent=not args.keep_intent,
            desensitize=not args.no_desensitize,
        )
        if sample is None:
            skipped[reason or "unknown"] = skipped.get(reason or "unknown", 0) + 1
            continue
        assistant_chars = sum(
            len(m["content"]) for m in sample["messages"] if m["role"] == "assistant"
        )
        if assistant_chars < args.min_assistant_chars:
            skipped["assistant_too_short"] = skipped.get("assistant_too_short", 0) + 1
            continue
        samples.append(sample)

    train, val, test = split_samples(
        samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    prefix = f"{args.doctor}_{args.dataset_name}"
    write_json(output_dir / f"{prefix}.json", samples)
    write_json(output_dir / f"{prefix}_train.json", train)
    write_json(output_dir / f"{prefix}_val.json", val)
    write_json(output_dir / f"{prefix}_test.json", test)

    report = {
        "input": str(input_path),
        "total_records": len(raw_data),
        "usable_samples": len(samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "skipped": skipped,
        "options": {
            "include_record_context": not args.no_record_context,
            "max_record_chars": args.max_record_chars,
            "remove_intent": not args.keep_intent,
            "desensitize": not args.no_desensitize,
        },
    }
    write_json(output_dir / f"{prefix}_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
