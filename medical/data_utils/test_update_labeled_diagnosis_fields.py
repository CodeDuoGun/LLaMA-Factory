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

from medical.data_utils.update_labeled_diagnosis_fields import load_updates


def test_load_updates_uses_the_three_engine_normalizers(tmp_path) -> None:
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    path = labeled_dir / "medical_records_labeled.jsonl"
    path.write_text(
        json.dumps(
            {
                "doctor_id": 97,
                "order_sn": "ORDER-1",
                "ai_diagnosis_illness": "干眼",
                "ai_diagnosis_disease": "肺脾两虚",
                "ai_diagnosis_sickness": "耳鸣",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    updates = load_updates(labeled_dir)

    assert updates[0].values == {
        "ai_diagnosis_illness": "干眼症",
        "ai_diagnosis_disease": "脾肺两虚证",
        "ai_diagnosis_sickness": "耳鸣病",
    }
