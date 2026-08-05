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

"""Structured disease-differentiation and prescription agents for RAG evaluation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import BaseModel, Field

from medical.analysis.llm import LLMSettings


class DiseaseDifferentiationResult(BaseModel):
    """辨病 Agent 的结构化结果，不包含处方答案."""

    western_diagnoses: list[str] = Field(default_factory=list)
    tcm_diseases: list[str] = Field(default_factory=list)
    tcm_syndromes: list[str] = Field(default_factory=list)
    etiologies: list[str] = Field(default_factory=list)
    pathogenesis: list[str] = Field(default_factory=list)
    disease_locations: list[str] = Field(default_factory=list)
    disease_stage: str = ""
    disease_course: str = ""
    treatment_principles: list[str] = Field(default_factory=list)
    treatment_methods: list[str] = Field(default_factory=list)
    key_symptoms: list[str] = Field(default_factory=list)
    evidence_limits: list[str] = Field(default_factory=list)


class PrescriptionDrug(BaseModel):
    """开方 Agent 输出的单味药及可追溯证据."""

    name: str
    dose: float | None = None
    unit: str = "g"
    role: str = ""
    evidence_case_ids: list[str] = Field(default_factory=list)


class PrescriptionProposal(BaseModel):
    """开方 Agent 的结构化建议；允许证据不足时拒绝生成."""

    refused: bool = False
    refusal_reason: str = ""
    base_formula: str = ""
    drugs: list[PrescriptionDrug] = Field(default_factory=list)
    core_drugs: list[str] = Field(default_factory=list)
    added_drugs: list[str] = Field(default_factory=list)
    removed_drugs: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    safety_notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PrescriptionRAGAgentSettings:
    """OpenAI-compatible settings shared by both evaluation agents."""

    api_key: str
    base_url: str
    model: str
    timeout: int = 120

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @classmethod
    def from_environment(cls) -> PrescriptionRAGAgentSettings:
        fallback = LLMSettings.from_environment()
        return cls(
            api_key=os.getenv("MEDICAL_RAG_LLM_API_KEY", fallback.api_key),
            base_url=os.getenv("MEDICAL_RAG_LLM_BASE_URL", fallback.base_url),
            model=os.getenv("MEDICAL_RAG_LLM_MODEL", fallback.model),
            timeout=int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", str(fallback.timeout))),
        )


class PrescriptionRAGAgents:
    """Run disease differentiation first, then generate a traceable RAG prescription proposal."""

    def __init__(
        self,
        settings: PrescriptionRAGAgentSettings | None = None,
        request_post: Callable[..., Any] = requests.post,
    ) -> None:
        self.settings = settings or PrescriptionRAGAgentSettings.from_environment()
        self.request_post = request_post

    def status(self) -> dict[str, Any]:
        return {"configured": self.settings.configured, "model": self.settings.model}

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)

    def _invoke(self, system_prompt: str, payload: dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        if not self.settings.configured:
            raise RuntimeError(
                "开方RAG评测LLM未配置，请设置 MEDICAL_RAG_LLM_API_KEY、MEDICAL_RAG_LLM_BASE_URL、"
                "MEDICAL_RAG_LLM_MODEL，或复用 MEDICAL_ANALYSIS_LLM_* 配置。"
            )
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        url = self.settings.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        response = self.request_post(
            url,
            headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.settings.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n必须仅返回一个合法JSON对象，不要返回Markdown。"
                            f"输出必须符合JSON Schema：{schema_json}"
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0.0,
                "max_tokens": 3000,
                "stream": False,
            },
            timeout=self.settings.timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return schema.model_validate_json(self._strip_json_fence(content))

    def differentiate(self, case: dict[str, Any]) -> DiseaseDifferentiationResult:
        """仅根据本次病情和此前病历辨病，不读取金标准诊断或真实处方."""
        clinical_input = {
            "patient_sex": case.get("patient_sex") or case.get("sex") or "",
            "patient_age": case.get("patient_age") or case.get("age"),
            "chief_complaint": case.get("chief_complaint") or case.get("patient_appeal") or "",
            "present_history": case.get("present_history") or case.get("new_medical_history") or "",
            "past_history": case.get("past_history") or case.get("old_medical_history") or "",
            "allergic_history": case.get("allergic_history") or "",
            "personal_history": case.get("personal_history") or "",
            "family_history": case.get("family_history") or "",
            "inspection_findings": case.get("inspection_findings") or "",
            "tongue_pulse": case.get("tongue_pulse") or "",
            "prior_records": case.get("prior_records") or case.get("history_records") or [],
        }
        result = self._invoke(
            (
                "你是中医辨病辨证评测Agent。只根据给出的主诉、现病史、既往史和当前就诊之前的历史病历，"
                "提取西医诊断候选、中医疾病、中医证候、病因、病机、病位、病期、病程、治则治法和关键症状。"
                "不得假设未提供的舌脉、检查或诊断，不得生成处方。信息不足的内容保持空值并写入evidence_limits。"
            ),
            clinical_input,
            DiseaseDifferentiationResult,
        )
        return result  # type: ignore[return-value]

    def prescribe(
        self,
        case: dict[str, Any],
        differentiation: DiseaseDifferentiationResult,
        retrieved_cases: list[dict[str, Any]],
    ) -> PrescriptionProposal:
        """依据辨病结果和召回证据生成候选处方，证据不足时必须拒绝."""
        allowed_case_ids = [str(item.get("case_id") or item.get("_id") or "") for item in retrieved_cases]
        payload = {
            "clinical_input": {
                "patient_sex": case.get("patient_sex") or case.get("sex") or "",
                "patient_age": case.get("patient_age") or case.get("age"),
                "chief_complaint": case.get("chief_complaint") or case.get("patient_appeal") or "",
                "present_history": case.get("present_history") or case.get("new_medical_history") or "",
                "allergic_history": case.get("allergic_history") or "",
                "special_population": case.get("special_population") or [],
                "previous_prescription": case.get("previous_prescription") or [],
            },
            "differentiation": differentiation.model_dump(),
            "retrieved_cases": retrieved_cases,
            "allowed_evidence_case_ids": allowed_case_ids,
        }
        result = self._invoke(
            (
                "你是面向执业医生的中医开方评测Agent。只能依据辨病结果和retrieved_cases给出候选内服方，"
                "不能查看或推测真实处方。每味药必须列出来自allowed_evidence_case_ids的证据病例ID；"
                "无病例支持的药不得生成。必须结合过敏史和特殊人群信息。若有效证据少于3例、辨病信息不足、"
                "证据互相冲突或存在无法排除的安全风险，设置refused=true、drugs为空并说明原因。"
                "该输出仅供医生审核，不得宣称替代临床决策。"
            ),
            payload,
            PrescriptionProposal,
        )
        return result  # type: ignore[return-value]
