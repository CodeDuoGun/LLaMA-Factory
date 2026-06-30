"""
Remove the separate `intent` field from dialogue entries by appending it to `content`.

Input format (separate intent field):
{
    "speaker": "医生",
    "content": "先给我看个正面。舌头我看看。",
    "intent": "面部症状询问"
}

Output format (intent merged, field removed):
{
    "speaker": "医生",
    "content": "先给我看个正面。舌头我看看。[意图：面部症状询问]"
}

If `content` already ends with a `[意图：...]` pattern, the intent is merged without duplication.
"""

import json
import os
import re


def process_dialogue(dialogue: list[dict]) -> list[dict]:
    """Merge `intent` field into `content` for all entries, then delete `intent`."""
    intent_pattern = re.compile(r"\[意图[：:]\s*[^]]*\]$")

    for entry in dialogue:
        if "intent" not in entry:
            continue

        intent_value = entry.pop("intent")
        content = entry.get("content", "")

        if intent_value is None:
            continue

        if intent_pattern.search(content):
            content = intent_pattern.sub(f"[意图：{intent_value}]", content)
        else:
            content = f"{content}[意图：{intent_value}]"

        entry["content"] = content

    return dialogue


def main():
    input_path = "medical/postclear_audio_asr.json"
    output_path = "medical/postclear_audio_asr_without_intent.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        cleared = record.get("cleared_data", {})
        if "dialogue" in cleared:
            cleared["dialogue"] = process_dialogue(cleared["dialogue"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
