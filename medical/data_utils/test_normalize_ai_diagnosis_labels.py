# Copyright 2025 the LlamaFactory team.
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

import json

import pytest

from medical.analysis.engine import normalize_diagnosis_value, normalize_sickness_value, normalize_syndrome_value
from medical.data_utils.normalize_ai_diagnosis_labels import (
    DEFAULT_TCM_DISEASE_MAPPING,
    DEFAULT_TCM_SYNDROME_MAPPING,
    DEFAULT_WESTERN_DIAGNOSIS_MAPPING,
    LabelMapping,
    MappingSpec,
    load_label_mapping,
    normalize_jsonl,
    normalize_label_value,
)


def _mapping(
    field_name: str,
    display_name: str,
    normalizer,
    normalized_to_label: dict[str, str],
) -> LabelMapping:
    spec = MappingSpec(field_name, display_name, DEFAULT_TCM_DISEASE_MAPPING, "测试标签", normalizer, True)
    return LabelMapping(
        spec=spec,
        alias_to_standard=dict(normalized_to_label),
        normalized_to_standard=normalized_to_label,
        standard_names=tuple(dict.fromkeys(normalized_to_label.values())),
        ambiguous_aliases={},
        ambiguous_normalized_names={},
    )


def test_normalize_jsonl_only_changes_three_ai_diagnosis_fields(tmp_path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "records_normalized.jsonl"
    record = {
        "diagnosis_sickness": "原始中医疾病",
        "diagnosis_disease": "原始中医证候",
        "diagnosis_illness": "原始西医诊断",
        "ai_diagnosis_sickness": "胃痞",
        "ai_diagnosis_sickness_reason": "中医疾病理由",
        "ai_diagnosis_disease": "实热证",
        "ai_diagnosis_disease_reason": "中医证候理由",
        "ai_diagnosis_illness": "肺结节",
        "ai_diagnosis_illness_reason": "西医诊断理由",
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    mappings = [
        _mapping("ai_diagnosis_sickness", "中医疾病", normalize_sickness_value, {"胃痞": "胃痞病"}),
        _mapping("ai_diagnosis_disease", "中医证候", normalize_syndrome_value, {"实热证": "里热证"}),
        _mapping("ai_diagnosis_illness", "西医诊断", normalize_diagnosis_value, {"肺结节": "肺结节病"}),
    ]

    report = normalize_jsonl(input_path, mappings, output_path=output_path)
    normalized = json.loads(output_path.read_text(encoding="utf-8"))

    assert normalized["ai_diagnosis_sickness"] == "胃痞病"
    assert normalized["ai_diagnosis_disease"] == "里热证"
    assert normalized["ai_diagnosis_illness"] == "肺结节病"
    assert normalized["diagnosis_sickness"] == record["diagnosis_sickness"]
    assert normalized["diagnosis_disease"] == record["diagnosis_disease"]
    assert normalized["diagnosis_illness"] == record["diagnosis_illness"]
    assert normalized["ai_diagnosis_sickness_reason"] == record["ai_diagnosis_sickness_reason"]
    assert normalized["ai_diagnosis_disease_reason"] == record["ai_diagnosis_disease_reason"]
    assert normalized["ai_diagnosis_illness_reason"] == record["ai_diagnosis_illness_reason"]
    assert report["record_count"] == 1
    assert all(stats["fully_mapped_records"] == 1 for stats in report["fields"].values())


def test_default_workbooks_build_three_mappings() -> None:
    paths = [DEFAULT_TCM_DISEASE_MAPPING, DEFAULT_TCM_SYNDROME_MAPPING, DEFAULT_WESTERN_DIAGNOSIS_MAPPING]
    if not all(path.is_file() for path in paths):
        pytest.skip("本地诊断词表未提供")
    specs = [
        MappingSpec("ai_diagnosis_sickness", "中医疾病", paths[0], "辨病", normalize_sickness_value, True),
        MappingSpec("ai_diagnosis_disease", "中医证候", paths[1], "辩证", normalize_syndrome_value, True),
        MappingSpec("ai_diagnosis_illness", "西医诊断", paths[2], "ICD10", normalize_diagnosis_value, False),
    ]

    mappings = [load_label_mapping(spec) for spec in specs]

    assert all(
        len(mapping.alias_to_standard) >= minimum
        for mapping, minimum in zip(mappings, [2000, 3000, 27000], strict=True)
    )
    assert mappings[0].alias_to_standard["胃痞"] == "胃痞病"
    assert mappings[0].alias_to_standard["时邪感冒"] == "时行感冒"
    assert mappings[0].alias_to_standard["咳嗽"] == "咳嗽"
    assert "咳嗽" not in mappings[0].normalized_to_standard
    assert mappings[0].ambiguous_normalized_names["咳嗽"] == ("咳嗽", "咳嗽病")
    assert mappings[1].alias_to_standard["实热证"] == "里热证"
    assert mappings[1].normalized_to_standard["胃阴虚证"] == "胃阴虚证"
    assert mappings[2].normalized_to_standard["肺结节"] == "肺结节病"
    assert [len(mapping.standard_names) for mapping in mappings] == [1369, 2060, 27819]


def test_unmatched_structured_text_is_preserved_as_one_value() -> None:
    mapping = _mapping("ai_diagnosis_disease", "中医证候", normalize_syndrome_value, {"里热证": "里热证"})
    value = "{'code': '', 'name': '', 'type': '2'}"

    normalized, matched_count, unmatched = normalize_label_value(value, mapping)

    assert normalized == value
    assert matched_count == 0
    assert unmatched == [value]
