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

# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from medical.analysis.llm import LLMSettings, PrescriptionLLMService


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        content = json.dumps(
            {
                "prescription_reason": "针对现有病情进行回顾性解释。",
                "compatibility_logic": "扶正与祛邪药物配伍。",
                "key_drug_roles": ["黄芪：扶正"],
                "evidence_limits": "缺少舌脉信息。",
            },
            ensure_ascii=False,
        )
        return {"choices": [{"message": {"content": content}}]}


def test_llm_prompt_uses_required_fields_and_cache() -> None:
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    service = PrescriptionLLMService(
        LLMSettings("test", "key", "https://example.test/v1", "test-model"),
        request_post=fake_post,
    )
    case = {
        "visit_id": 1,
        "patient_id": "10086",
        "diagnosis_illness": "肺癌",
        "sex": "女",
        "age": "56",
        "new_medical_history": "乏力",
        "allergic_history": "青霉素过敏",
        "family_history": "无",
        "personal_history": "无吸烟史",
        "old_medical_history": "高血压",
        "drugs": [{"drug_name": "黄芪", "dose": 20, "unit": "g"}],
    }

    result = service.explain("doctor", case)
    cached = service.explain("doctor", case)
    prompt = calls[0][1]["json"]["messages"][1]["content"]

    assert len(calls) == 1
    assert all(value in prompt for value in ("女", "56", "乏力", "青霉素过敏", "无吸烟史", "高血压", "黄芪20g"))
    assert "10086" not in prompt
    assert result["compatibility_logic"] == "扶正与祛邪药物配伍。"
    assert cached["cached"] is True
