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

from medical.data_utils.normalize_annotation_diagnosis_fields import normalize_record


def test_normalize_record_uses_the_three_engine_normalizers() -> None:
    changes = normalize_record(
        {
            "ai_diagnosis_illness": "酒渣鼻",
            "ai_diagnosis_disease": "脾肺两虚",
            "ai_diagnosis_sickness": "肝气郁结",
        }
    )

    assert changes == {
        "ai_diagnosis_illness": "玫瑰痤疮",
        "ai_diagnosis_disease": "脾肺两虚证",
        "ai_diagnosis_sickness": "肝气郁结证",
    }


def test_normalize_record_only_returns_changed_fields() -> None:
    record = {
        "ai_diagnosis_illness": "玫瑰痤疮",
        "ai_diagnosis_disease": "脾肺两虚证",
        "ai_diagnosis_sickness": "肝气郁结证",
    }

    assert normalize_record(record) == {}
