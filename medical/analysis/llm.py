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

"""LLM-backed prescription reasoning for a selected consultation case."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests


def _config_value(name: str, default: str = "") -> str:
    try:
        from medical.config import config

        return str(getattr(config, name, default) or default)
    except Exception:
        return default


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: int = 90

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @classmethod
    def from_environment(cls) -> LLMSettings:
        provider = os.getenv("MEDICAL_ANALYSIS_LLM_PROVIDER") or _config_value("LLM_PROVIDER", "doubao")
        provider = provider.lower().strip()
        if provider == "openai":
            fallback_key = _config_value("OPENIAI_API_KEY")
            fallback_url = _config_value("OPENIAI_BASE_URL")
            fallback_model = "gpt-4o-mini"
        elif provider in ("qwen", "dashscope"):
            fallback_key = _config_value("DASHSCOPE_API_KEY")
            fallback_url = _config_value("DASHSCOPE_BASE_URL") or _config_value("QWEN_API_URL")
            fallback_model = _config_value("QWEN_TEXT_MODEL", "qwen-plus")
        elif provider == "deepseek":
            fallback_key = _config_value("DEEPSEEK_API_KEY")
            fallback_url = _config_value("DEEPSEEK_BASE_URL")
            fallback_model = _config_value("DEEPSEEK_MODEL", "deepseek-v3")
        elif provider == "baichuan":
            fallback_key = _config_value("BAICHUAN_API_KEY")
            fallback_url = _config_value("BAICHUAN_BASE_URL")
            fallback_model = _config_value("BAICHUAN_TEXT_MODEL")
        else:
            fallback_key = _config_value("ARK_API_KEY")
            fallback_url = _config_value("ARK_API_URL") or _config_value("ARK_BASE_URL")
            fallback_model = _config_value("DOUBAO_TEXT_MODEL", "doubao-seed-1-6-250615")
        return cls(
            provider=provider,
            api_key=os.getenv("MEDICAL_ANALYSIS_LLM_API_KEY", fallback_key),
            base_url=os.getenv("MEDICAL_ANALYSIS_LLM_BASE_URL", fallback_url),
            model=os.getenv("MEDICAL_ANALYSIS_LLM_MODEL", fallback_model),
            timeout=int(os.getenv("MEDICAL_ANALYSIS_LLM_TIMEOUT", "90")),
        )


class PrescriptionLLMService:
    """Call an OpenAI-compatible chat API and cache explanations per visit."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        request_post: Callable[..., Any] = requests.post,
    ) -> None:
        self.settings = settings or LLMSettings.from_environment()
        self.request_post = request_post
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.settings.configured,
            "provider": self.settings.provider,
            "model": self.settings.model,
        }

    @staticmethod
    def _prescription_text(case: dict[str, Any]) -> str:
        parts = []
        for drug in case.get("drugs", []):
            dose = "" if drug.get("dose") is None else str(drug["dose"])
            parts.append(f"{drug.get('drug_name', '')}{dose}{drug.get('unit', '')}")
        return "、".join(parts) or "未记录内服处方"

    def _messages(self, case: dict[str, Any]) -> list[dict[str, str]]:
        clinical = {
            "性别": case.get("sex") or "未记录",
            "年龄": case.get("age") or "未记录",
            "new_medical_history": case.get("new_medical_history") or "未记录",
            "过敏史": case.get("allergic_history") or "未记录",
            "家族史": case.get("family_history") or "未记录",
            "个人史": case.get("personal_history") or "未记录",
            "既往史": case.get("old_medical_history") or "未记录",
        }
        user_content = {
            "疾病诊断_diagnosis_illness": case.get("diagnosis_illness") or "未明确",
            "患者病情信息": clinical,
            "本次处方明细": self._prescription_text(case),
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是一名中医临床处方分析专家。请仅依据给定病情和处方，解释医生可能的开方原因与配伍逻辑；"
                    "不得补充输入中不存在的症状、舌脉、检查结果或疗效结论。输出严格JSON，字段为"
                    "prescription_reason（字符串）、compatibility_logic（字符串）、key_drug_roles（字符串数组）、"
                    "evidence_limits（字符串）。说明这是回顾性解释，不作为诊疗建议。"
                ),
            },
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ]

    @staticmethod
    def _parse_content(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "prescription_reason": cleaned,
                "compatibility_logic": "模型未返回结构化配伍逻辑。",
                "key_drug_roles": [],
                "evidence_limits": "模型返回格式非JSON，请人工复核。",
            }
        return {
            "prescription_reason": str(parsed.get("prescription_reason") or "未返回"),
            "compatibility_logic": str(parsed.get("compatibility_logic") or "未返回"),
            "key_drug_roles": [str(item) for item in (parsed.get("key_drug_roles") or [])],
            "evidence_limits": str(parsed.get("evidence_limits") or "未返回"),
        }

    def explain(self, doctor: str, case: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
        if not self.settings.configured:
            raise RuntimeError(
                "LLM未配置，请设置 MEDICAL_ANALYSIS_LLM_API_KEY、MEDICAL_ANALYSIS_LLM_BASE_URL 和 "
                "MEDICAL_ANALYSIS_LLM_MODEL。"
            )
        cache_key = (doctor, int(case["visit_id"]))
        if not refresh and cache_key in self.cache:
            return {**self.cache[cache_key], "cached": True}

        url = self.settings.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        response = self.request_post(
            url,
            headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.settings.model,
                "messages": self._messages(case),
                "temperature": 0.2,
                "max_tokens": 1200,
                "stream": False,
            },
            timeout=self.settings.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = {
            "visit_id": case["visit_id"],
            "provider": self.settings.provider,
            "model": self.settings.model,
            **self._parse_content(content),
            "cached": False,
        }
        self.cache[cache_key] = result
        return result
