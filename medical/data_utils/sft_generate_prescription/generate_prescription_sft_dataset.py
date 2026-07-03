#!/usr/bin/env python3
#
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
"""生成辩证开方任务的 SFT 数据集.

默认输入:
    medical/data/sft_generate_prescription/extract_valid_record.json
    medical/data/wuweiping_template_prescription.json

默认输出:
    medical/data/sft_generate_prescription/prescription_sft_sharegpt.json
    medical/data/sft_generate_prescription/prescription_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_RECORD_FILE = Path("medical/data/sft_generate_prescription/extract_valid_record.json")
DEFAULT_TEMPLATE_FILE = Path("medical/data/wuweiping_template_prescription.json")
DEFAULT_OUTPUT_FILE = Path("medical/data/sft_generate_prescription/prescription_sft_sharegpt.json")
DEFAULT_JSONL_FILE = Path("medical/data/sft_generate_prescription/prescription_sft.jsonl")


SYSTEM_PROMPT = """你是中医皮肤科医生助手，擅长根据皮肤病病历进行辨证开方。请先在 <think> 中给出有据可循的辨证逻辑，再输出严格 JSON。

<think> 要求：
1. 必须结合患者主诉、症状变化、既往病史、检查报告结果【如有】、舌面结果【如有】、患处分析结果【如有】进行辨证。
2. 每个辨证判断都必须绑定具体依据，不能凭空推断；未提供的信息要说明“未提供”，不得作为依据。
3. 需要区分主症、兼症、舌面/患处/检查报告证据，并说明这些证据如何支持证型、病机、治则治法。
4. 如果最高模板匹配度低于0.55，不得参考模板方，应根据病历、诊断、证候、知识图谱参考和治法直接生成最终处方。
5. 如果最高模板匹配度大于等于0.55，可以参考模板方，并说明保留、去除、新增、剂量调整的依据。
6. 必须结合知识图谱中的证候知识、十八反十九畏检查、相似病历参考，生成配伍逻辑、君臣佐使、加减逻辑。
7. 思考内容应简洁、专业、可追溯，不要泛泛而谈。

JSON 要求：
1. <think> 结束后只输出 JSON，不要 Markdown。
2. JSON 字段必须固定，字段为：中医诊断、西医诊断、证型、病机分析、治则、治法、模板匹配、加减逻辑、配伍逻辑、君臣佐使、最终处方、医嘱。
3. 最终处方必须为处方数组，允许包含多个处方；每个处方必须包含处方名字、服用类型、服用频次、医嘱、明细；明细中每味药必须包含名称、剂量、单位。
4. 不得输出病历中没有依据的检查结果、舌象或症状。
5. 模板匹配必须包含：是否参考模板方、候选模板方、模板匹配度、判断依据。
6. 加减逻辑必须包含：保留药物、新增药物、去除药物、剂量调整。
7. 君臣佐使必须包含：君药、臣药、佐药、使药。"""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(data: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_drug_name(name: Any) -> str:
    return re.sub(r"\s+", "", clean_text(name))


def record_drug_names(record: dict[str, Any]) -> list[str]:
    names = []
    for prescription in record.get("internal_prescriptions") or []:
        for drug in prescription.get("drugs") or []:
            name = normalize_drug_name(drug.get("drug_name"))
            if name:
                names.append(name)
    return sorted(set(names))


def format_drugs(drugs: Iterable[dict[str, Any]]) -> list[str]:
    items = []
    for drug in drugs:
        name = clean_text(drug.get("drug_name"))
        dose = clean_text(drug.get("dose") or drug.get("quantity"))
        unit = clean_text(drug.get("unit"))
        decoction = clean_text(drug.get("decoction_name"))
        if not name:
            continue
        amount = f"{dose}{unit}" if dose or unit else ""
        suffix = f"（{decoction}）" if decoction else ""
        items.append(f"{name}{amount}{suffix}")
    return items


def format_prescriptions(record: dict[str, Any]) -> list[dict[str, Any]]:
    prescriptions = []
    for prescription in record.get("internal_prescriptions") or []:
        prescriptions.append(
            {
                "usage_type": clean_text(prescription.get("usage_type")),
                "drug_process_name": clean_text(prescription.get("drug_process_name")),
                "doctor_advice": clean_text(prescription.get("doctor_advice")),
                "drugs": format_drugs(prescription.get("drugs") or []),
            }
        )
    return prescriptions


def find_template_candidates(record: dict[str, Any], templates: dict[str, Any] | list[dict[str, Any]], top_k: int = 5):
    template_list = templates.get("templates", []) if isinstance(templates, dict) else templates
    record_names = set(record_drug_names(record))
    candidates = []
    for template in template_list:
        template_drugs = template.get("drugs") or []
        template_names = {normalize_drug_name(drug.get("drug_name")) for drug in template_drugs}
        template_names.discard("")
        if not template_names:
            continue
        matched = sorted(record_names & template_names)
        union_size = len(record_names | template_names) or 1
        overlap_score = len(matched) / union_size
        candidates.append(
            {
                "template_id": template.get("template_id"),
                "template_name": clean_text(template.get("template_name")),
                "overlap_score": round(overlap_score, 4),
                "matched_drugs": matched,
                "drugs": [
                    {
                        "drug_name": clean_text(drug.get("drug_name")),
                        "quantity": drug.get("quantity"),
                        "unit": clean_text(drug.get("unit")),
                    }
                    for drug in template_drugs
                    if clean_text(drug.get("drug_name"))
                ],
            }
        )
    candidates.sort(key=lambda item: (item["overlap_score"], len(item["matched_drugs"])), reverse=True)
    return candidates[:top_k]


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class EmptyKnowledgeProvider:
    def build_context(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "syndrome_knowledge": {},
            "incompatibilities": {"warnings": []},
            "similar_cases": {"cases": []},
        }


class GraphKnowledgeProvider:
    def __init__(self, top_k_cases: int = 5):
        self.manager = self._make_manager()
        self.top_k_cases = top_k_cases

    def _make_manager(self) -> Any:
        try:
            from medical.data_utils.knowledge import KnowledgeGraphManager

            return KnowledgeGraphManager()
        except ModuleNotFoundError as exc:
            if exc.name != "app":
                raise
            return MinimalGraphKnowledgeManager()

    def build_context(self, record: dict[str, Any]) -> dict[str, Any]:
        disease = clean_text(record.get("diagnosis_sickness") or record.get("diagnosis_illness"))
        syndrome = clean_text(record.get("diagnosis_disease"))
        herbs = record_drug_names(record)
        context = {
            "syndrome_knowledge": {},
            "incompatibilities": {"warnings": []},
            "similar_cases": {"cases": []},
        }
        if syndrome:
            context["syndrome_knowledge"] = _json_ready(self.manager.get_syndrome_knowledge(syndrome)).get("data") or {}
        if herbs:
            context["incompatibilities"] = _json_ready(self.manager.check_incompatibilities(herbs)).get("data") or {}
        if disease or syndrome:
            context["similar_cases"] = (
                _json_ready(self.manager.search_similar_cases(disease=disease, syndrome=syndrome, limit=self.top_k_cases)).get(
                    "data"
                )
                or {}
            )
        return context

    def close(self) -> None:
        close = getattr(self.manager, "close", None)
        if callable(close):
            close()


class MinimalGraphKnowledgeManager:
    """Fallback reader used when medical.data_utils.knowledge depends on the app package."""

    def __init__(self):
        from neo4j import GraphDatabase

        from medical.config import config

        password = getattr(config, "NEO4J_PASSWORD", "")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD 未配置，无法连接知识图谱；可用 --disable-kg 跳过图谱上下文。")
        self.database = getattr(config, "NEO4J_DATABASE", None) or "neo4j"
        self._driver = GraphDatabase.driver(
            getattr(config, "NEO4J_URI", "bolt://localhost:7687"),
            auth=(getattr(config, "NEO4J_USER", "neo4j"), password),
        )

    def _exec(self, cypher: str, **params) -> list[dict[str, Any]]:
        with self._driver.session(database=self.database) as session:
            return [dict(row) for row in session.run(cypher, **params)]

    def get_syndrome_knowledge(self, syndrome: str) -> dict[str, Any]:
        herb_rows = self._exec(
            "MATCH (s:Syndrome {name:$s})-[r:COMMONLY_USES]->(h:Herb) "
            "RETURN h.name AS herb, r.count AS count, r.avg_dose AS avg_dose "
            "ORDER BY r.count DESC LIMIT 12",
            s=syndrome,
        )
        symptom_rows = self._exec(
            "MATCH (c:Case)-[:HAS_SYNDROME]->(s:Syndrome {name:$s}) "
            "MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom) "
            "RETURN sym.name AS symptom, count(c) AS cases "
            "ORDER BY cases DESC LIMIT 20",
            s=syndrome,
        )
        disease_rows = self._exec(
            "MATCH (d:Disease)-[r:MANIFESTS_AS]->(s:Syndrome {name:$s}) "
            "RETURN d.name AS disease, r.count AS cases "
            "ORDER BY r.count DESC LIMIT 10",
            s=syndrome,
        )
        return {
            "ok": True,
            "data": {
                "syndrome": syndrome,
                "common_herbs": herb_rows,
                "symptoms": symptom_rows,
                "diseases": disease_rows,
            },
        }

    def check_incompatibilities(self, herbs: Iterable[str]) -> dict[str, Any]:
        names = sorted({clean_text(herb) for herb in herbs if clean_text(herb)})
        if len(names) < 2:
            return {"ok": True, "data": {"herbs": names, "warnings": []}}
        rows = self._exec(
            "MATCH (a:Herb)-[r:INCOMPATIBLE_WITH|ANTAGONIZES]-(b:Herb) "
            "WHERE a.name IN $names AND b.name IN $names AND a.name < b.name "
            "RETURN a.name AS herb_a, b.name AS herb_b, type(r) AS rel, "
            "r.rule AS rule, r.description AS description",
            names=names,
        )
        warnings = []
        for row in rows:
            warnings.append(
                {
                    "herb_a": row["herb_a"],
                    "herb_b": row["herb_b"],
                    "kind": "十八反" if row["rel"] == "INCOMPATIBLE_WITH" else "十九畏",
                    "rule": row.get("rule"),
                    "description": row.get("description"),
                }
            )
        return {"ok": True, "data": {"herbs": names, "warnings": warnings}}

    def search_similar_cases(self, disease: str = "", syndrome: str = "", limit: int = 5) -> dict[str, Any]:
        conditions, params = [], {"limit": int(limit)}
        if disease:
            conditions.append("EXISTS { (c)-[:HAS_DISEASE]->(:Disease {name:$d}) }")
            params["d"] = disease
        if syndrome:
            conditions.append("EXISTS { (c)-[:HAS_SYNDROME]->(:Syndrome {name:$s}) }")
            params["s"] = syndrome
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._exec(
            "MATCH (c:Case) "
            f"{where} "
            "OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease) "
            "OPTIONAL MATCH (c)-[:HAS_SYNDROME]->(s:Syndrome) "
            "OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom) "
            "RETURN c.id AS id, c.gender AS gender, c.age AS age, "
            "collect(DISTINCT d.name) AS diseases, collect(DISTINCT s.name) AS syndromes, "
            "collect(DISTINCT sym.name) AS symptoms "
            "ORDER BY c.created_at DESC LIMIT $limit",
            **params,
        )
        return {"ok": True, "data": {"cases": rows}}

    def close(self) -> None:
        self._driver.close()


def build_case_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "inquiry_id": record.get("inquiry_id"),
        "visit_type": clean_text(record.get("is_first")),
        "patient": {
            "sex": clean_text(record.get("patient_sex")),
            "age": clean_text(record.get("patient_age")),
        },
        "chief_complaint": clean_text(record.get("patient_appeal") or record.get("doc_ass_stu_appeal")),
        "current_history": clean_text(record.get("new_medical_history")),
        "past_history": clean_text(record.get("old_medical_history")),
        "allergic_history": clean_text(record.get("allergic_history")),
        "family_history": clean_text(record.get("family_history")),
        "personal_history": clean_text(record.get("personal_history")),
        "inspection_report": clean_text(record.get("admin_report_describe")),
        "tongue_face_findings": clean_text(record.get("admin_face_describe")),
        "lesion_findings": clean_text(record.get("lesion_findings") or record.get("admin_face_describe")),
        "diagnosis": {
            "disease_tcm": clean_text(record.get("diagnosis_sickness")),
            "disease_western": clean_text(record.get("diagnosis_illness")),
            "syndrome": clean_text(record.get("diagnosis_disease")),
        },
        "reference_prescription_from_record": format_prescriptions(record),
    }


def build_user_prompt(
    record: dict[str, Any],
    template_candidates: list[dict[str, Any]],
    kg_context: dict[str, Any],
) -> str:
    case_context = build_case_context(record)
    top_template = template_candidates[0] if template_candidates else {}
    template_name = clean_text(top_template.get("template_name")) or "无"
    template_score = top_template.get("overlap_score", 0)
    template_drugs = "、".join(
        clean_text(drug.get("drug_name")) for drug in top_template.get("drugs", []) if clean_text(drug.get("drug_name"))
    ) or "无"
    kg_payload = {
        "证候知识": kg_context.get("syndrome_knowledge") or {},
        "十八反十九畏检查": kg_context.get("incompatibilities") or {},
        "相似病历": kg_context.get("similar_cases") or {},
    }

    return (
        "请根据以下病历辨证并开方。\n\n"
        "【基础信息】\n"
        f"就诊类型：{case_context['visit_type'] or '未提供'}\n"
        f"性别：{case_context['patient']['sex'] or '未提供'}\n"
        f"年龄：{case_context['patient']['age'] or '未提供'}岁\n\n"
        "【主诉】\n"
        f"{case_context['chief_complaint'] or '未提供'}\n\n"
        "【现病史与症状变化】\n"
        f"{case_context['current_history'] or '未提供'}\n\n"
        "【既往史】\n"
        f"{case_context['past_history'] or '未提供'}\n\n"
        "【过敏史】\n"
        f"{case_context['allergic_history'] or '未提供'}\n\n"
        "【检查报告结果】\n"
        f"{case_context['inspection_report'] or '未提供'}\n\n"
        "【舌面结果】\n"
        f"{case_context['tongue_face_findings'] or '未提供'}\n\n"
        "【患处分析结果】\n"
        f"{case_context['lesion_findings'] or '未提供'}\n\n"
        "【诊断信息】\n"
        f"中医诊断：{case_context['diagnosis']['disease_tcm'] or '未提供'}\n"
        f"西医诊断：{case_context['diagnosis']['disease_western'] or '未提供'}\n"
        f"原始证候：{case_context['diagnosis']['syndrome'] or '未提供'}\n\n"
        "【候选模板方】\n"
        f"模板方名称：{template_name}\n"
        f"模板匹配度：{template_score}\n"
        f"模板方药物：{template_drugs}\n"
        f"候选模板详情：{json.dumps(template_candidates, ensure_ascii=False)}\n\n"
        "【知识图谱参考】\n"
        f"{json.dumps(kg_payload, ensure_ascii=False, indent=2)}\n\n"
        "【相似处方参考】\n"
        f"{json.dumps(case_context['reference_prescription_from_record'], ensure_ascii=False, indent=2)}\n\n"
        "请输出辨证逻辑、治则治法、模板方判断、加减逻辑、配伍逻辑、君臣佐使、最终处方和医嘱。"
    )


def parse_llm_json(text: str) -> Any:
    content = clean_text(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start_positions = [pos for pos in (content.find("{"), content.find("[")) if pos >= 0]
    if not start_positions:
        raise ValueError("LLM response does not contain JSON")
    start = min(start_positions)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(content[start:])
    return parsed


def build_dry_run_answer(record: dict[str, Any]) -> str:
    diagnosis = build_case_context(record)["diagnosis"]
    payload = {
        "中医诊断": diagnosis["disease_tcm"],
        "西医诊断": diagnosis["disease_western"],
        "证型": diagnosis["syndrome"],
        "病机分析": "dry-run 样本：未调用 LLM，仅用于检查 messages 与 JSONL 格式。",
        "治则": "",
        "治法": "",
        "模板匹配": {
            "是否参考模板方": False,
            "候选模板方": "",
            "模板匹配度": 0,
            "判断依据": "dry-run 未执行 LLM 辨证判断。",
        },
        "加减逻辑": {"保留药物": [], "新增药物": [], "去除药物": [], "剂量调整": []},
        "配伍逻辑": "",
        "君臣佐使": {"君药": [], "臣药": [], "佐药": [], "使药": []},
        "最终处方": [
            {
                "处方名字": f"原始处方{index + 1}",
                "服用类型": clean_text(raw_prescription.get("usage_type")),
                "服用频次": "",
                "医嘱": prescription["doctor_advice"],
                "明细": [
                    {
                        "名称": clean_text(drug.get("drug_name")),
                        "剂量": clean_text(drug.get("dose")),
                        "单位": clean_text(drug.get("unit")),
                    }
                    for drug in raw_prescription.get("drugs", [])
                    if clean_text(drug.get("drug_name"))
                ],
            }
            for index, (prescription, raw_prescription) in enumerate(
                zip(format_prescriptions(record), record.get("internal_prescriptions") or [])
            )
        ],
        "医嘱": [],
    }
    return "<think>dry-run 样本：未调用 LLM，仅根据原始病历处方回填，用于验证 SFT 数据格式。</think>\n" + json.dumps(
        payload, ensure_ascii=False
    )


def build_dataset_item(
    record: dict[str, Any],
    template_candidates: list[dict[str, Any]],
    kg_context: dict[str, Any],
    assistant_content: str,
) -> dict[str, Any]:
    record_id = clean_text(record.get("inquiry_id") or record.get("order_sn"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record, template_candidates, kg_context)},
            {"role": "assistant", "content": assistant_content},
        ],
        "record_id": record_id,
        "task": "辨证开方",
        "diagnosis_sickness": clean_text(record.get("diagnosis_sickness")),
        "diagnosis_illness": clean_text(record.get("diagnosis_illness")),
        "diagnosis_disease": clean_text(record.get("diagnosis_disease")),
    }


def sharegpt_to_text(item: dict[str, Any]) -> dict[str, str]:
    parts = []
    for message in item.get("messages") or []:
        role = clean_text(message.get("role"))
        if role not in {"system", "user", "assistant"}:
            continue
        parts.append(f"<|im_start|>{role}\n{message.get('content', '')}<|im_end|>")
    return {"text": "\n".join(parts)}


def generate_dataset(
    records: list[dict[str, Any]],
    templates: dict[str, Any] | list[dict[str, Any]],
    kg_provider: Any,
    llm_call: Callable[[list[dict[str, str]]], str] | None,
    *,
    limit: int | None = None,
    offset: int = 0,
    template_top_k: int = 5,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    selected = records[offset : offset + limit if limit is not None else None]
    dataset = []
    for record in tqdm(selected, desc="生成辩证开方 SFT"):
        template_candidates = find_template_candidates(record, templates, top_k=template_top_k)
        kg_context = kg_provider.build_context(record)
        if dry_run:
            answer = build_dry_run_answer(record)
        else:
            if llm_call is None:
                raise ValueError("llm_call is required unless dry_run=True")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(record, template_candidates, kg_context)}]
            answer = clean_text(llm_call(messages))
        dataset.append(build_dataset_item(record, template_candidates, kg_context, answer))
    return dataset


def make_llm_call(provider: str, model: str, max_tokens: int, temperature: float) -> Callable[[list[dict[str, str]]], str]:
    if provider != "doubao":
        raise ValueError("当前脚本内置 provider 仅支持 doubao；其他 provider 可通过 generate_dataset 注入 llm_call。")

    from medical.config import config
    from medical.llm_v2.doubao import DoubaoAIClient

    client = DoubaoAIClient(base_url=config.ARK_API_URL, api_key=config.ARK_API_KEY)

    def call(messages: list[dict[str, str]]) -> str:
        return client.chat(messages=messages, stream=False, llm_model=model, max_tokens=max_tokens, temperature=temperature)

    return call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于病历、模板方、知识图谱和 LLM 生成辩证开方 SFT 数据集。")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORD_FILE)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_FILE)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--template-top-k", type=int, default=5)
    parser.add_argument("--case-top-k", type=int, default=5)
    parser.add_argument("--provider", default="doubao", choices=["doubao"])
    parser.add_argument("--model", default="doubao-seed-1-6-250615")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--disable-kg", action="store_true", help="不连接知识图谱，仅生成空参考上下文。")
    parser.add_argument("--dry-run", action="store_true", help="不调用 LLM，用原始处方构造答案，便于检查格式。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json(args.records)
    templates = load_json(args.templates)
    if not isinstance(records, list):
        raise ValueError(f"records must be a JSON list: {args.records}")

    kg_provider = EmptyKnowledgeProvider() if args.disable_kg else GraphKnowledgeProvider(top_k_cases=args.case_top_k)
    try:
        llm_call = None if args.dry_run else make_llm_call(args.provider, args.model, args.max_tokens, args.temperature)
        dataset = generate_dataset(
            records=records,
            templates=templates,
            kg_provider=kg_provider,
            llm_call=llm_call,
            limit=args.limit,
            offset=args.offset,
            template_top_k=args.template_top_k,
            dry_run=args.dry_run,
        )
    finally:
        close = getattr(kg_provider, "close", None)
        if callable(close):
            close()

    save_json(dataset, args.output)
    save_jsonl((sharegpt_to_text(item) for item in dataset), args.jsonl_output)
    print(json.dumps({"items": len(dataset), "output": str(args.output), "jsonl_output": str(args.jsonl_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
