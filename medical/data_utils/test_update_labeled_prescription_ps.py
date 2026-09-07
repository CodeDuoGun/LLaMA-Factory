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

from medical.data_utils.update_labeled_prescription_ps import (
    build_update_plan,
    load_labeled_records,
    load_raw_prescriptions,
)


def test_build_update_plan_uses_raw_ps_instead_of_labeled_ps(tmp_path) -> None:
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    (labeled_dir / "doctor_97_巢国俊").mkdir()
    labeled_path = labeled_dir / "doctor_97_巢国俊" / "medical_records_labeled.jsonl"
    labeled_path.write_text(
        json.dumps(
            {
                "doctor_id": 97,
                "order_sn": "ORDER-1",
                "ps": [{"old": "value"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doctor_97.json").write_text(
        json.dumps(
            [
                {
                    "doctor_id": 97,
                    "order_sn": "ORDER-1",
                    "ps": [
                        {
                            "usage_type": "内服",
                            "drug_process_name": "饮片",
                            "dosage": "",
                            "doctor_advice": "早晚服用",
                            "usage_desc": "每日1剂",
                            "prescription_items": {
                                "drugList": [
                                    {
                                        "drug_name": "黄芩",
                                        "drug_num": "10",
                                        "unit_name": "g",
                                        "use_limit_text": "",
                                        "produce_merchant": "药房",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    labeled = load_labeled_records(labeled_dir)
    raw = load_raw_prescriptions(raw_dir, set(labeled))
    updates = build_update_plan(labeled, raw)

    assert updates[0].ps == [
        {
            "usage_type": "内服",
            "drug_process_name": "饮片",
            "dosage": "",
            "doctor_advice": "早晚服用",
            "produce_merchant": "药房",
            "usage_desc": "每日1剂",
            "drugs": [
                {
                    "drug_name": "黄芩",
                    "drug_num": "10",
                    "unit": "g",
                    "max_use": "",
                    "min_use": "",
                }
            ],
        }
    ]


def test_load_labeled_records_can_filter_doctors(tmp_path) -> None:
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()
    (labeled_dir / "records.jsonl").write_text(
        "\n".join(json.dumps({"doctor_id": doctor_id, "order_sn": f"ORDER-{doctor_id}"}) for doctor_id in (43, 97))
        + "\n",
        encoding="utf-8",
    )

    records = load_labeled_records(labeled_dir, {"97"})

    assert list(records) == [("97", "ORDER-97")]
