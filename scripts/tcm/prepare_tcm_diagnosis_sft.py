# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare Alpaca-format SFT data for structured TCM diagnosis generation.

The input is a JSONL export containing medical records. The generated examples
use de-identified clinical text as the input and prefer the doctor's annotated
``ai_*`` diagnosis fields as a JSON response. The original diagnosis fields
are used only as a fallback when an annotated field is empty:

* ``diagnosis_illness``
* ``diagnosis_disease``
* ``diagnosis_sickness``

Records are split by ``patient_id`` so that different visits from one patient
cannot leak across the train, validation, and test sets.

Example:
    python scripts/tcm/prepare_tcm_diagnosis_sft.py \
        --input /path/to/medical_records_labeled.jsonl \
        --output-dir data/wuweiping_diagnosis_sft \
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/labeled_records/doctor_1314_吴卫平/medical_records_labeled.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/wuweiping_diagnosis_sft")

DIAGNOSIS_FIELD_SOURCES = (
    ("diagnosis_illness", "ai_diagnosis_illness", "diagnosis_illness"),
    ("diagnosis_disease", "ai_diagnosis_disease", "diagnosis_disease"),
    ("diagnosis_sickness", "ai_diagnosis_sickness", "diagnosis_sickness"),
)

INPUT_FIELDS = (
    ("就诊类型", ("is_first",)),
    ("性别", ("patient_sex",)),
    ("年龄", ("patient_age",)),
    ("身高", ("patient_height",)),
    ("体重", ("patient_weight",)),
    ("主诉", ("ai_patient_appeal", "patient_appeal")),
    ("现病史", ("ai_new_medical_history", "new_medical_history")),
    ("既往史", ("ai_old_medical_history", "old_medical_history")),
    ("过敏史", ("ai_allergic_history", "allergic_history")),
    ("个人史", ("ai_personal_history", "personal_history")),
    ("特殊病史", ("ai_special_history", "special_history")),
    ("家族史", ("ai_family_history", "family_history")),
    ("舌面与面诊信息", ("ai_tongue_face_img", "admin_face_describe")),
    ("检查结果", ("ai_inspection_report_img",)),
)

SYSTEM_PROMPT = (
    "你是一名中医临床辨病辨证助手。你的核心任务不是复述或总结病历，"
    "而是依据当前病例中的主诉、现病史、既往史、体征和其他四诊信息完成辨病辨证。"
    "请先辨病，判断疾病所属的中医病名；再辨证，结合症状、病程、病位和病机判断中医证候。"
    "辨病辨证必须以当前病例证据为依据，不得仅凭单个症状、历史标签或常见病名机械套用，"
    "也不得补充病例中没有提供的信息。资料不足时，相关诊断字段输出空字符串。"
    "最终只输出指定的合法 JSON，不输出病历摘要、分析过程、治疗建议或其他解释。"
)

INSTRUCTION = (
    "请完成当前病例的辨病辨证，而不是对病例进行信息总结。"
    "请先根据病例证据判断病名，再根据病位、病性、病机和主要症状判断证候。"
    "最后输出三个结构化诊断字段：diagnosis_illness（西医诊断）、"
    "diagnosis_disease（中医病名）、diagnosis_sickness（中医证候）。"
    "输出必须是合法 JSON，并且只能包含这三个字段。"
)


def _text(value: Any) -> str:
    """Convert a JSON value to a compact, human-readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        if not value:
            return ""
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _preferred_text(record: dict[str, Any], field_names: tuple[str, ...]) -> str:
    """Use the first non-empty value, preferring doctor-annotated ``ai_*`` fields."""
    for field_name in field_names:
        value = _text(record.get(field_name))
        if value:
            return value
    return ""


def _build_input(record: dict[str, Any]) -> str:
    """Build the text input while excluding identifiers and timestamps."""
    parts = []
    for label, field_names in INPUT_FIELDS:
        value = _preferred_text(record, field_names)
        if value:
            parts.append(f"{label}：{value}")
    return "\n".join(parts)


def _patient_key(record: dict[str, Any], row_number: int) -> str:
    """Return an internal grouping key that is never written to the output."""
    patient_id = _text(record.get("patient_id"))
    if patient_id:
        return f"patient:{patient_id}"

    source_record_id = _text(record.get("source_record_id"))
    return f"record:{source_record_id or row_number}"


def _build_example(record: dict[str, Any]) -> dict[str, str] | None:
    """Convert one source record to an Alpaca example."""
    input_text = _build_input(record)
    if not input_text:
        return None

    output = _build_diagnosis_output(record)
    if not any(output.values()):
        return None

    return {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
    }


def _build_diagnosis_output(record: dict[str, Any]) -> dict[str, str]:
    """Build output labels with annotated fields taking priority over source fields."""
    return {
        output_field: _preferred_text(record, (annotated_field, source_field))
        for output_field, annotated_field, source_field in DIAGNOSIS_FIELD_SOURCES
    }


def _assign_split(patient_key: str, seed: int, validation_ratio: float, test_ratio: float) -> str:
    """Assign a patient to a deterministic split using a stable hash."""
    digest = hashlib.sha256(f"{seed}:{patient_key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "validation"
    return "train"


def _write_json(path: Path, examples: list[dict[str, str]], overwrite: bool) -> None:
    """Write a UTF-8 JSON array without silently overwriting existing files."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(examples, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_dataset_info(output_dir: Path, overwrite: bool) -> None:
    """Write a registry that can be merged into LLaMA-Factory's dataset_info.json."""
    path = output_dir / "dataset_info.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "wuweiping_diagnosis_train": {
            "file_name": "train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "wuweiping_diagnosis_validation": {
            "file_name": "validation.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "wuweiping_diagnosis_test": {
            "file_name": "test.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source medical-record JSONL file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.json, validation.json, test.json, and dataset_info.json.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Patient-level validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Patient-level test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic patient-level splits.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit valid examples for a smoke test.")
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Keep records where one or more diagnosis fields are empty.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing generated output files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL file does not exist: {args.input}")
    if args.validation_ratio < 0 or args.test_ratio < 0 or args.validation_ratio + args.test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be non-negative and sum to less than 1.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive when specified.")

    examples: list[tuple[str, dict[str, str]]] = []
    stats = Counter()
    with args.input.open(encoding="utf-8") as file:
        for row_number, line in enumerate(file, start=1):
            if not line.strip():
                stats["empty_lines"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {row_number}: {error}") from error

            if not isinstance(record, dict):
                stats["non_object_rows"] += 1
                continue

            diagnosis_output = _build_diagnosis_output(record)
            for output_field, annotated_field, source_field in DIAGNOSIS_FIELD_SOURCES:
                if _text(record.get(annotated_field)):
                    stats[f"{annotated_field}_used"] += 1
                elif _text(record.get(source_field)):
                    stats[f"{source_field}_fallback"] += 1

            labels = list(diagnosis_output.values())
            if not args.allow_missing_labels and not all(labels):
                stats["missing_labels"] += 1
                continue

            example = _build_example(record)
            if example is None:
                stats["empty_input_or_output"] += 1
                continue

            examples.append((_patient_key(record, row_number), example))

    if args.max_samples is not None:
        examples = examples[: args.max_samples]

    split_examples: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    patient_counts: dict[str, set[str]] = {split: set() for split in split_examples}
    for patient_key, example in examples:
        split = _assign_split(patient_key, args.seed, args.validation_ratio, args.test_ratio)
        split_examples[split].append(example)
        patient_counts[split].add(patient_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_data in split_examples.items():
        _write_json(args.output_dir / f"{split}.json", split_data, args.overwrite)
    _write_dataset_info(args.output_dir, args.overwrite)

    print(f"Input: {args.input}")
    print(f"Valid examples: {len(examples)}")
    for split in ("train", "validation", "test"):
        print(f"{split}: {len(split_examples[split])} examples, {len(patient_counts[split])} patients")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
