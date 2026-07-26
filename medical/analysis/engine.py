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

"""Analysis engine for disease, clinical condition, and prescription relationships."""

from __future__ import annotations

import heapq
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DISEASE_RULES = (
    ("胰腺癌", ("胰腺", "胰癌")),
    ("肺癌", ("肺癌", "肺腺癌", "小细胞肺", "肺小细胞癌", "肺鳞癌", "肺恶性肿瘤")),
    ("肝癌", ("肝癌", "肝细胞癌", "肝恶性肿瘤")),
    ("胃癌", ("胃癌", "胃腺癌", "胃恶性肿瘤", "贲门癌")),
    ("结直肠癌", ("结肠癌", "直肠癌", "结直肠癌", "肠癌", "结肠恶性肿瘤", "直肠恶性肿瘤")),
    ("乳腺癌", ("乳腺癌", "乳癌", "乳岩")),
    ("卵巢癌", ("卵巢癌", "卵巢恶性肿瘤")),
    ("宫颈癌", ("宫颈癌", "宫颈恶性肿瘤")),
    ("食管癌", ("食管癌", "食道癌")),
    ("胆道癌", ("胆管癌", "胆囊癌", "胆道癌", "胆管细胞癌", "胆管恶性肿瘤")),
    ("前列腺癌", ("前列腺癌",)),
    ("肾癌", ("肾癌", "肾恶性肿瘤", "肾透明细胞癌", "肾盂癌")),
    ("膀胱癌", ("膀胱癌",)),
    ("鼻咽癌", ("鼻咽癌",)),
    ("甲状腺癌", ("甲状腺癌",)),
    ("淋巴瘤", ("淋巴瘤",)),
    ("白血病", ("白血病",)),
    ("骨髓瘤", ("骨髓瘤",)),
    ("脑肿瘤", ("脑胶质瘤", "脑瘤", "颅内肿瘤")),
    ("肉瘤", ("肉瘤",)),
)

SYMPTOM_ALIASES = {
    "乏力": ("乏力", "疲乏", "疲倦", "没力气"),
    "纳差": ("纳差", "食欲差", "食欲不振", "不思饮食", "胃口差"),
    "腹痛": ("腹痛", "肚子痛", "腹部疼痛"),
    "腹胀": ("腹胀", "腹部胀", "肚胀"),
    "胸闷": ("胸闷", "胸部憋闷"),
    "气短": ("气短", "呼吸困难", "喘憋"),
    "咳嗽": ("咳嗽", "咳"),
    "咳痰": ("咳痰", "痰多"),
    "疼痛": ("疼痛",),
    "恶心": ("恶心",),
    "呕吐": ("呕吐", "吐"),
    "反酸": ("反酸", "烧心"),
    "嗳气": ("嗳气", "打嗝"),
    "口干": ("口干", "口渴"),
    "口苦": ("口苦",),
    "便秘": ("便秘", "大便干", "大便偏干", "排便困难"),
    "腹泻": ("腹泻", "便溏", "大便稀", "拉肚子"),
    "睡眠差": ("睡眠差", "失眠", "入睡困难", "眠差", "多梦", "易醒"),
    "水肿": ("水肿", "浮肿"),
    "腹水": ("腹水", "腹腔积液"),
    "胸腔积液": ("胸腔积液", "胸水"),
    "发热": ("发热", "低热", "高热"),
    "出汗": ("出汗", "盗汗", "自汗"),
    "头晕": ("头晕", "眩晕"),
    "心悸": ("心悸", "心慌"),
    "尿频": ("尿频",),
    "尿少": ("尿少", "小便少"),
    "出血": ("出血", "流血", "便血", "咯血", "黑便"),
    "麻木": ("麻木", "手麻", "脚麻"),
}

NEGATION_PREFIXES = ("无", "未见", "没有", "否认", "不伴", "未", "不")
IMPROVED_WORDS = ("好转", "改善", "减轻", "缓解", "消失", "恢复", "感觉良好", "较前好")
WORSE_WORDS = ("加重", "恶化", "进展", "增多", "明显", "较前差", "新发")
STABLE_WORDS = ("稳定", "同前", "无明显变化", "控制尚可", "一般")
DIAGNOSIS_TERM_ALIASES = {
    "咳嗽病": "咳嗽",
    "肝气郁结": "肝气郁结证",
    "肝气不舒": "肝气不舒证",
    "肺结节" + "病": "肺结节",
    "间质性肺纤维化": "间质肺纤维化",
    "痰瘀阻肺": "痰瘀阻肺证",
    "瘿瘤病": "瘿瘤",
}
COMPOUND_DIAGNOSIS_TERMS = (
    "痰瘀阻肺证",
    "肝气郁结证",
    "肝气郁结",
    "肝气不舒证",
    "肝脾不和证",
    "痰瘀阻肺",
    "肝气不舒",
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_diagnosis_part(value: str) -> str:
    part = re.sub(r"\s+", "", value).strip()
    return DIAGNOSIS_TERM_ALIASES.get(part, part)


def _split_compound_diagnosis_part(value: str) -> list[str]:
    part = _normalize_diagnosis_part(value)
    if not part:
        return []
    terms = sorted(COMPOUND_DIAGNOSIS_TERMS, key=len, reverse=True)
    items = []
    index = 0
    while index < len(part):
        matched = next((term for term in terms if part.startswith(term, index)), "")
        if not matched:
            return [part]
        items.append(_normalize_diagnosis_part(matched))
        index += len(matched)
    return items


def normalize_diagnosis_value(value: Any) -> str:
    """Normalize multi-item diagnosis fields without merging distinct concepts."""
    text = _clean(value)
    if not text:
        return ""
    parts = []
    for part in re.split(r"[、，,；;：:／/|｜·+＋&＆\s]+", text):
        parts.extend(_split_compound_diagnosis_part(part))
    if not parts:
        return ""
    return "、".join(sorted(set(parts)))


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _quantile(values: list[float], index: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * index / 4)], 3)


def _date(record: dict[str, Any]) -> str:
    return _clean(record.get("start_time") or record.get("created_at"))


def canonical_disease(value: Any) -> str:
    """Collapse fragmented diagnosis strings into an analysis-level primary disease."""
    text = re.sub(r"[\s，、；;/]+", ",", _clean(value))
    for canonical, aliases in DISEASE_RULES:
        if any(alias in text for alias in aliases):
            return canonical
    first = text.split(",", 1)[0].strip()
    first = re.sub(r"(术后|晚期|早期|中期|复发|多发转移|转移.*)$", "", first)
    return first or "未明确"


def diagnosis_attributes(value: Any) -> list[str]:
    text = _clean(value)
    attrs = []
    for name, pattern in (
        ("术后", r"术后|切除术"),
        ("复发", r"复发"),
        ("转移", r"转移"),
        ("晚期", r"晚期|Ⅳ|IV期|4期"),
        ("化疗相关", r"化疗|化疗后"),
        ("放疗相关", r"放疗|放疗后"),
        ("靶向治疗相关", r"靶向"),
        ("免疫治疗相关", r"免疫治疗"),
        ("肝硬化", r"肝硬化"),
        ("腹水", r"腹水|腹腔积液"),
    ):
        if re.search(pattern, text):
            attrs.append(name)
    return attrs


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 4) : start]
    return any(prefix.endswith(word) for word in NEGATION_PREFIXES)


def extract_symptoms(value: Any) -> list[str]:
    text = _clean(value)
    found = []
    for canonical, aliases in SYMPTOM_ALIASES.items():
        present = False
        for alias in aliases:
            for match in re.finditer(re.escape(alias), text):
                if not _is_negated(text, match.start()):
                    present = True
                    break
            if present:
                break
        if present:
            found.append(canonical)
    return found


def classify_outcome(value: Any) -> str:
    text = _clean(value)
    scores = {
        "改善": sum(text.count(word) for word in IMPROVED_WORDS),
        "加重": sum(text.count(word) for word in WORSE_WORDS),
        "稳定": sum(text.count(word) for word in STABLE_WORDS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "不明确"


def _parse_days(value: Any) -> int | None:
    match = re.match(r"(\d+)_", _clean(value))
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class Drug:
    drug_id: str
    name: str
    dose: float | None
    unit: str


@dataclass
class Visit:
    visit_id: int
    patient_id: str
    date: str
    is_first: str
    disease: str
    tcm_disease: str
    raw_diagnosis: str
    attributes: list[str]
    syndrome: str
    symptoms: list[str]
    new_medical_history: str
    sex: str
    age: str
    allergic_history: str
    family_history: str
    personal_history: str
    old_medical_history: str
    outcome: str
    internal_drugs: dict[str, Drug]
    internal_rx_count: int
    external_rx_count: int
    process_types: list[str]
    treatment_days: int | None
    prescriptions: list[dict[str, Any]]


class MedicalAnalysis:
    """Precompute anonymized visit, disease, drug, and longitudinal indexes."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        source_name: str = "",
        doctor_name: str = "",
        doctor_id: str = "",
    ) -> None:
        self.source_name = source_name
        self.doctor_name = doctor_name
        self.doctor_id = doctor_id
        self.raw_record_count = len(records)
        self.visits = [self._build_visit(record) for record in records]
        self.visits.sort(key=lambda visit: (visit.date, visit.visit_id))
        self.visit_by_id = {visit.visit_id: visit for visit in self.visits}
        self.patient_visits: dict[str, list[Visit]] = defaultdict(list)
        for visit in self.visits:
            self.patient_visits[visit.patient_id].append(visit)
        for visits in self.patient_visits.values():
            visits.sort(key=lambda visit: (visit.date, visit.visit_id))
        self.revisit_pairs = self._build_revisit_pairs()
        self._drug_global_visits = Counter()
        for visit in self.visits:
            self._drug_global_visits.update(visit.internal_drugs.keys())

    @classmethod
    def from_json(cls, path: str | Path) -> MedicalAnalysis:
        source = Path(path)
        with source.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("The input JSON must contain a list of consultation records.")
        return cls(records, source.name)

    @staticmethod
    def _patient_id(record: dict[str, Any]) -> str:
        raw_value = record.get("patient_id")
        if raw_value in (None, ""):
            raw_value = record.get("user_info_id")
        if raw_value in (None, ""):
            raw_value = record.get("id")
        return str(raw_value)

    @staticmethod
    def _main_internal_prescription(record: dict[str, Any]) -> tuple[dict[str, Drug], int, int, list[str], int | None]:
        internal = []
        external_count = 0
        processes = []
        days = []
        for prescription in record.get("ps") or []:
            if prescription.get("usage_type") != "内服":
                external_count += 1
                continue
            drug_list = (prescription.get("prescription_items") or {}).get("drugList") or []
            internal.append((prescription, drug_list))
            process = _clean(prescription.get("drug_process_name"))
            if process:
                processes.append(process)
            parsed_days = _parse_days(prescription.get("use_limit"))
            if parsed_days is not None:
                days.append(parsed_days)

        # Multiple oral prescriptions can be pharmacy splits or separate courses. The largest complete formula is
        # the least ambiguous representative for visit-to-visit composition comparisons.
        selected = max(internal, key=lambda item: len(item[1]), default=(None, []))[1]
        drugs = {}
        for item in selected:
            drug_id = str(item.get("drug_id") or _clean(item.get("drug_name") or item.get("show_name")))
            if not drug_id:
                continue
            name = _clean(item.get("drug_name") or item.get("show_name") or item.get("sub_drug_name"))
            dose = _safe_float(item.get("drug_weight", item.get("drug_num")))
            drugs[drug_id] = Drug(drug_id, name, dose, _clean(item.get("unit_name")))
        return drugs, len(internal), external_count, sorted(set(processes)), max(days) if days else None

    @staticmethod
    def _timeline_prescriptions(record: dict[str, Any]) -> list[dict[str, Any]]:
        prescriptions = []
        for prescription in record.get("ps") or []:
            drugs = []
            for item in (prescription.get("prescription_items") or {}).get("drugList") or []:
                drug_id = str(item.get("drug_id") or _clean(item.get("drug_name") or item.get("show_name")))
                name = _clean(item.get("drug_name") or item.get("show_name") or item.get("sub_drug_name"))
                if not drug_id and not name:
                    continue
                drugs.append(
                    {
                        "drug_id": drug_id,
                        "drug_name": name,
                        "dose": _safe_float(item.get("drug_weight", item.get("drug_num"))),
                        "unit": _clean(item.get("unit_name")),
                    }
                )
            prescriptions.append(
                {
                    "usage_type": _clean(prescription.get("usage_type")),
                    "process_type": _clean(prescription.get("drug_process_name")),
                    "drug_count": len(drugs),
                    "drugs": drugs,
                }
            )
        return prescriptions

    def _build_visit(self, record: dict[str, Any]) -> Visit:
        history = str(record.get("new_medical_history") or "").strip()
        appeal = _clean(record.get("patient_appeal"))
        summary = _clean(record.get("doc_ass_stu_appeal"))
        diagnosis = normalize_diagnosis_value(record.get("diagnosis_illness", ""))
        tcm_disease = normalize_diagnosis_value(record.get("diagnosis_sickness"))
        syndrome = normalize_diagnosis_value(record.get("diagnosis_disease"))
        drugs, internal_count, external_count, processes, days = self._main_internal_prescription(record)
        return Visit(
            visit_id=int(record.get("id") or 0),
            patient_id=self._patient_id(record),
            date=_date(record),
            is_first=_clean(record.get("is_first")),
            disease=diagnosis or "未明确",
            tcm_disease=tcm_disease,
            raw_diagnosis=diagnosis,
            attributes=diagnosis_attributes("。".join((diagnosis, history))),
            # diagnosis_disease stores the structured syndrome/pattern; diagnosis_sickness is a TCM disease name
            # and must not be used as a fallback when grouping disease-syndrome strata.
            syndrome=syndrome,
            symptoms=extract_symptoms("。".join((appeal, summary, history))),
            new_medical_history=history,
            sex=_clean(record.get("patient_sex")),
            age=_clean(record.get("patient_age")),
            allergic_history=str(record.get("allergic_history") or "").strip(),
            family_history=str(record.get("family_history") or "").strip(),
            personal_history=str(record.get("personal_history") or "").strip(),
            old_medical_history=str(record.get("old_medical_history") or "").strip(),
            outcome=classify_outcome(history),
            internal_drugs=drugs,
            internal_rx_count=internal_count,
            external_rx_count=external_count,
            process_types=processes,
            treatment_days=days,
            prescriptions=self._timeline_prescriptions(record),
        )

    @staticmethod
    def _difference(previous: Visit, current: Visit) -> dict[str, Any]:
        previous_ids = set(previous.internal_drugs)
        current_ids = set(current.internal_drugs)
        retained = previous_ids & current_ids
        added = current_ids - previous_ids
        removed = previous_ids - current_ids
        dose_changes = []
        for drug_id in retained:
            old = previous.internal_drugs[drug_id]
            new = current.internal_drugs[drug_id]
            if old.dose is None or new.dose is None or old.unit != new.unit or math.isclose(old.dose, new.dose):
                continue
            dose_changes.append(
                {
                    "drug_id": drug_id,
                    "drug_name": new.name,
                    "before": old.dose,
                    "after": new.dose,
                    "unit": new.unit,
                    "direction": "增加" if new.dose > old.dose else "减少",
                }
            )
        union = previous_ids | current_ids
        return {
            "jaccard": round(len(retained) / len(union), 3) if union else 1.0,
            "retention": round(len(retained) / len(previous_ids), 3) if previous_ids else None,
            "added": [current.internal_drugs[key].name for key in sorted(added)],
            "removed": [previous.internal_drugs[key].name for key in sorted(removed)],
            "dose_changes": sorted(dose_changes, key=lambda item: item["drug_name"]),
            "previous_count": len(previous_ids),
            "current_count": len(current_ids),
        }

    def _build_revisit_pairs(self) -> list[dict[str, Any]]:
        pairs = []
        for patient_id, visits in self.patient_visits.items():
            for previous, current in zip(visits, visits[1:]):
                if current.is_first != "复诊" or not previous.internal_drugs or not current.internal_drugs:
                    continue
                difference = self._difference(previous, current)
                try:
                    interval = (datetime.fromisoformat(current.date) - datetime.fromisoformat(previous.date)).days
                except (TypeError, ValueError):
                    interval = None
                pairs.append(
                    {
                        "patient_id": patient_id,
                        "previous_visit_id": previous.visit_id,
                        "current_visit_id": current.visit_id,
                        "previous_date": previous.date[:10],
                        "current_date": current.date[:10],
                        "interval_days": interval,
                        "disease": current.disease,
                        "raw_diagnosis": current.raw_diagnosis,
                        "outcome": current.outcome,
                        "symptoms": current.symptoms,
                        **difference,
                    }
                )
        return pairs

    def overview(self) -> dict[str, Any]:
        internal_visits = [visit for visit in self.visits if visit.internal_drugs]
        multi_visit_patients = sum(len(visits) > 1 for visits in self.patient_visits.values())
        revisit_count = sum(visit.is_first == "复诊" for visit in self.visits)
        no_prior = 0
        for visits in self.patient_visits.values():
            if visits and visits[0].is_first == "复诊":
                no_prior += 1
        pair_summary = self._summarize_pairs(self.revisit_pairs)
        return {
            "source": self.source_name,
            "doctor_name": self.doctor_name,
            "doctor_id": self.doctor_id,
            "records": len(self.visits),
            "patients": len(self.patient_visits),
            "multi_visit_patients": multi_visit_patients,
            "first_visits": sum(visit.is_first == "初诊" for visit in self.visits),
            "revisits": revisit_count,
            "internal_visits": len(internal_visits),
            "external_only_visits": sum(not visit.internal_drugs and visit.external_rx_count for visit in self.visits),
            "disease_count": len({visit.disease for visit in self.visits}),
            "comparable_revisit_pairs": len(self.revisit_pairs),
            "quality": {
                "revisit_patients_without_prior_record": no_prior,
                "missing_syndrome": sum(not visit.syndrome for visit in self.visits),
                "multiple_internal_prescriptions": sum(visit.internal_rx_count > 1 for visit in self.visits),
                "unclassified_disease": sum(visit.disease == "未明确" for visit in self.visits),
            },
            "revisit_summary": pair_summary,
            "top_diseases": self.diseases(limit=10)["items"],
            "diagnosis_distributions": {
                "diagnosis_illness": self._diagnosis_distribution("raw_diagnosis"),
                "diagnosis_sickness": self._diagnosis_distribution("tcm_disease"),
                "diagnosis_disease": self._diagnosis_distribution("syndrome"),
            },
            "top_drugs": self.top_drugs(limit=12),
        }

    def _diagnosis_distribution(self, attribute: str) -> list[dict[str, Any]]:
        visits = Counter()
        patients: dict[str, set[str]] = defaultdict(set)
        for visit in self.visits:
            value = str(getattr(visit, attribute, "") or "").strip()
            if not value:
                continue
            visits[value] += 1
            patients[value].add(visit.patient_id)
        return [
            {"value": value, "visit_count": count, "patient_count": len(patients[value])}
            for value, count in sorted(visits.items(), key=lambda item: (-item[1], item[0]))
        ]

    def top_drugs(self, limit: int = 20) -> list[dict[str, Any]]:
        names = {}
        doses: dict[str, list[float]] = defaultdict(list)
        units: dict[str, Counter[str]] = defaultdict(Counter)
        for visit in self.visits:
            for drug_id, drug in visit.internal_drugs.items():
                names[drug_id] = drug.name
                if drug.dose is not None:
                    doses[drug_id].append(drug.dose)
                if drug.unit:
                    units[drug_id][drug.unit] += 1
        denominator = sum(bool(visit.internal_drugs) for visit in self.visits)
        return [
            {
                "drug_id": drug_id,
                "drug_name": names.get(drug_id, drug_id),
                "visit_count": count,
                "visit_rate": round(count / denominator, 4) if denominator else 0,
                "median_dose": _median(doses[drug_id]),
                "unit": units[drug_id].most_common(1)[0][0] if units[drug_id] else "",
            }
            for drug_id, count in self._drug_global_visits.most_common(limit)
        ]

    def diseases(self, limit: int = 50, minimum_patients: int = 1) -> dict[str, Any]:
        grouped: dict[str, list[Visit]] = defaultdict(list)
        for visit in self.visits:
            grouped[visit.disease].append(visit)
        items = []
        for disease, visits in grouped.items():
            patients = {visit.patient_id for visit in visits}
            if len(patients) < minimum_patients:
                continue
            internal = [visit for visit in visits if visit.internal_drugs]
            pairs = [pair for pair in self.revisit_pairs if pair["disease"] == disease]
            items.append(
                {
                    "disease": disease,
                    "visit_count": len(visits),
                    "patient_count": len(patients),
                    "revisit_count": sum(visit.is_first == "复诊" for visit in visits),
                    "internal_visit_count": len(internal),
                    "comparable_pairs": len(pairs),
                    "median_drug_count": _median([float(len(visit.internal_drugs)) for visit in internal]),
                }
            )
        items.sort(key=lambda item: (-item["visit_count"], item["disease"]))
        return {"total": len(items), "items": items[:limit]}

    def disease_detail(self, disease: str, limit: int = 30) -> dict[str, Any]:
        visits = [visit for visit in self.visits if visit.disease == disease]
        if not visits:
            raise KeyError(disease)
        internal = [visit for visit in visits if visit.internal_drugs]
        patient_count = len({visit.patient_id for visit in visits})
        drug_visits = Counter()
        drug_patients: dict[str, set[str]] = defaultdict(set)
        drug_doses: dict[str, list[float]] = defaultdict(list)
        drug_units: dict[str, Counter[str]] = defaultdict(Counter)
        names = {}
        symptoms = Counter()
        attributes = Counter()
        for visit in visits:
            symptoms.update(visit.symptoms)
            attributes.update(visit.attributes)
            for drug_id, drug in visit.internal_drugs.items():
                drug_visits[drug_id] += 1
                drug_patients[drug_id].add(visit.patient_id)
                names[drug_id] = drug.name
                if drug.dose is not None:
                    drug_doses[drug_id].append(drug.dose)
                if drug.unit:
                    drug_units[drug_id][drug.unit] += 1
        global_denominator = sum(bool(visit.internal_drugs) for visit in self.visits)
        drug_rows = []
        for drug_id, count in drug_visits.most_common(limit):
            disease_rate = count / len(internal) if internal else 0
            global_rate = self._drug_global_visits[drug_id] / global_denominator if global_denominator else 0
            drug_rows.append(
                {
                    "drug_id": drug_id,
                    "drug_name": names[drug_id],
                    "visit_count": count,
                    "patient_count": len(drug_patients[drug_id]),
                    "disease_rate": round(disease_rate, 4),
                    "global_rate": round(global_rate, 4),
                    "lift": round(disease_rate / global_rate, 3) if global_rate else None,
                    "median_dose": _median(drug_doses[drug_id]),
                    "unit": drug_units[drug_id].most_common(1)[0][0] if drug_units[drug_id] else "",
                }
            )
        pairs = [pair for pair in self.revisit_pairs if pair["disease"] == disease]
        return {
            "disease": disease,
            "visit_count": len(visits),
            "patient_count": patient_count,
            "internal_visit_count": len(internal),
            "revisit_summary": self._summarize_pairs(pairs),
            "top_drugs": drug_rows,
            "top_symptoms": [
                {"symptom": name, "count": count, "rate": round(count / len(visits), 4)}
                for name, count in symptoms.most_common(20)
            ],
            "attributes": [{"attribute": name, "count": count} for name, count in attributes.most_common()],
            "symptom_drug_associations": self.symptom_drug_associations(disease=disease, limit=20)["items"],
        }

    @staticmethod
    def _prescription_signature(visit: Visit) -> tuple[tuple[str, float | None, str], ...]:
        return tuple(sorted((drug.drug_id, drug.dose, drug.unit) for drug in visit.internal_drugs.values()))

    @staticmethod
    def _prescription_similarity(left: Visit, right: Visit) -> float:
        left_ids = set(left.internal_drugs)
        right_ids = set(right.internal_drugs)
        union = left_ids | right_ids
        return len(left_ids & right_ids) / len(union) if union else 1.0

    @staticmethod
    def _case_payload(visit: Visit) -> dict[str, Any]:
        drugs = [
            {"drug_id": drug.drug_id, "drug_name": drug.name, "dose": drug.dose, "unit": drug.unit}
            for drug in sorted(visit.internal_drugs.values(), key=lambda item: item.name)
        ]
        return {
            "visit_id": visit.visit_id,
            "patient_id": visit.patient_id,
            "date": visit.date[:10],
            "diagnosis_illness": visit.raw_diagnosis,
            "sex": visit.sex,
            "age": visit.age,
            "new_medical_history": visit.new_medical_history,
            "allergic_history": visit.allergic_history,
            "family_history": visit.family_history,
            "personal_history": visit.personal_history,
            "old_medical_history": visit.old_medical_history,
            "drug_count": len(drugs),
            "drugs": drugs,
        }

    def disease_cases(self, disease: str, limit: int = 10) -> dict[str, Any]:
        visits = [visit for visit in self.visits if visit.disease == disease and visit.internal_drugs]
        groups: dict[tuple[tuple[str, float | None, str], ...], list[Visit]] = defaultdict(list)
        for visit in visits:
            groups[self._prescription_signature(visit)].append(visit)
        ranked_groups = sorted(groups.values(), key=lambda group: (-len(group), group[-1].date), reverse=False)
        items = []
        for group in ranked_groups[:limit]:
            representative = max(group, key=lambda visit: (visit.date, visit.visit_id))
            item = self._case_payload(representative)
            item["prescription_occurrences"] = len(group)
            items.append(item)
        return {
            "disease": disease,
            "visit_count": len(visits),
            "distinct_prescriptions": len(groups),
            "items": items,
        }

    def prescription_comparisons(
        self,
        disease: str,
        case_limit: int = 8,
        pair_limit: int = 8,
        minimum_similarity: float = 0.8,
    ) -> dict[str, Any]:
        same_disease = self.disease_cases(disease, limit=case_limit)
        anchors = [visit for visit in self.visits if visit.disease == disease and visit.internal_drugs]
        others = [visit for visit in self.visits if visit.disease != disease and visit.internal_drugs]
        heap: list[tuple[float, int, Visit, Visit]] = []
        sequence = 0
        heap_size = max(pair_limit * 30, 100)
        for anchor in anchors:
            for other in others:
                score = self._prescription_similarity(anchor, other)
                if score < minimum_similarity:
                    continue
                candidate = (score, sequence, anchor, other)
                sequence += 1
                if len(heap) < heap_size:
                    heapq.heappush(heap, candidate)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, candidate)

        pairs = []
        seen = set()
        for score, _, anchor, other in sorted(heap, key=lambda item: (-item[0], item[1])):
            key = (self._prescription_signature(anchor), other.disease, self._prescription_signature(other))
            if key in seen:
                continue
            seen.add(key)
            anchor_ids = set(anchor.internal_drugs)
            other_ids = set(other.internal_drugs)
            pairs.append(
                {
                    "similarity": round(score, 3),
                    "common_drugs": [anchor.internal_drugs[key].name for key in sorted(anchor_ids & other_ids)],
                    "left_only": [anchor.internal_drugs[key].name for key in sorted(anchor_ids - other_ids)],
                    "right_only": [other.internal_drugs[key].name for key in sorted(other_ids - anchor_ids)],
                    "left_case": self._case_payload(anchor),
                    "right_case": self._case_payload(other),
                }
            )
            if len(pairs) >= pair_limit:
                break
        return {
            "disease": disease,
            "same_disease_different_prescriptions": same_disease,
            "same_prescription_different_diseases": {
                "minimum_similarity": minimum_similarity,
                "pair_count": len(pairs),
                "items": pairs,
            },
        }

    def visit_case(self, visit_id: int) -> dict[str, Any]:
        visit = self.visit_by_id.get(visit_id)
        if visit is None:
            raise KeyError(visit_id)
        return self._case_payload(visit)

    def metadata(self) -> dict[str, Any]:
        symptoms = Counter(symptom for visit in self.visits for symptom in visit.symptoms)
        return {
            "symptoms": [{"symptom": name, "count": count} for name, count in symptoms.most_common()],
            "outcomes": ["改善", "稳定", "加重", "不明确"],
        }

    def symptom_drug_associations(
        self,
        disease: str | None = None,
        symptom: str | None = None,
        minimum_support: int = 3,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Measure exploratory associations between current symptoms and newly added drugs."""
        pairs = [pair for pair in self.revisit_pairs if not disease or pair["disease"] == disease]
        symptoms = [symptom] if symptom else sorted({name for pair in pairs for name in pair["symptoms"]})
        rows = []
        for symptom_name in symptoms:
            exposed = [pair for pair in pairs if symptom_name in pair["symptoms"]]
            unexposed = [pair for pair in pairs if symptom_name not in pair["symptoms"]]
            if not exposed:
                continue
            exposed_drugs = Counter(name for pair in exposed for name in set(pair["added"]))
            unexposed_drugs = Counter(name for pair in unexposed for name in set(pair["added"]))
            for drug_name, support in exposed_drugs.items():
                if support < minimum_support:
                    continue
                exposed_rate = support / len(exposed)
                baseline_rate = unexposed_drugs[drug_name] / len(unexposed) if unexposed else 0
                rows.append(
                    {
                        "symptom": symptom_name,
                        "drug_name": drug_name,
                        "support": support,
                        "exposed_pairs": len(exposed),
                        "exposed_rate": round(exposed_rate, 4),
                        "baseline_rate": round(baseline_rate, 4),
                        "lift": round(exposed_rate / baseline_rate, 3) if baseline_rate else None,
                        "patient_count": len({pair["patient_id"] for pair in exposed if drug_name in pair["added"]}),
                    }
                )
        rows.sort(
            key=lambda row: (
                -(row["lift"] if row["lift"] is not None else 999),
                -row["support"],
                row["symptom"],
                row["drug_name"],
            )
        )
        return {"pair_count": len(pairs), "total": len(rows), "items": rows[:limit]}

    @staticmethod
    def _summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        if not pairs:
            return {
                "pair_count": 0,
                "median_jaccard": None,
                "median_retention": None,
                "median_added": None,
                "median_removed": None,
                "median_dose_changes": None,
                "unchanged_rate": None,
            }
        exact = [not pair["added"] and not pair["removed"] and not pair["dose_changes"] for pair in pairs]
        return {
            "pair_count": len(pairs),
            "median_jaccard": _median([pair["jaccard"] for pair in pairs]),
            "jaccard_q1": _quantile([pair["jaccard"] for pair in pairs], 1),
            "jaccard_q3": _quantile([pair["jaccard"] for pair in pairs], 3),
            "median_retention": _median([pair["retention"] for pair in pairs if pair["retention"] is not None]),
            "median_added": _median([float(len(pair["added"])) for pair in pairs]),
            "median_removed": _median([float(len(pair["removed"])) for pair in pairs]),
            "median_dose_changes": _median([float(len(pair["dose_changes"])) for pair in pairs]),
            "unchanged_rate": round(sum(exact) / len(exact), 4),
            "outcomes": dict(Counter(pair["outcome"] for pair in pairs)),
        }

    def revisits(
        self, disease: str | None = None, outcome: str | None = None, symptom: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        pairs = self.revisit_pairs
        if disease:
            pairs = [pair for pair in pairs if pair["disease"] == disease]
        if outcome:
            pairs = [pair for pair in pairs if pair["outcome"] == outcome]
        if symptom:
            pairs = [pair for pair in pairs if symptom in pair["symptoms"]]
        added = Counter(name for pair in pairs for name in pair["added"])
        removed = Counter(name for pair in pairs for name in pair["removed"])
        dose = Counter(change["drug_name"] for pair in pairs for change in pair["dose_changes"])
        return {
            "total": len(pairs),
            "summary": self._summarize_pairs(pairs),
            "top_added": [{"drug_name": name, "count": count} for name, count in added.most_common(15)],
            "top_removed": [{"drug_name": name, "count": count} for name, count in removed.most_common(15)],
            "top_dose_changed": [{"drug_name": name, "count": count} for name, count in dose.most_common(15)],
            "items": pairs[:limit],
        }

    def patients(self, query: str = "", limit: int = 50) -> dict[str, Any]:
        query = query.strip().upper()
        items = []
        for patient_id, visits in self.patient_visits.items():
            if query and query not in patient_id.upper():
                continue
            items.append(
                {
                    "patient_id": patient_id,
                    "visit_count": len(visits),
                    "first_date": visits[0].date[:10],
                    "last_date": visits[-1].date[:10],
                    "diseases": sorted({visit.disease for visit in visits}),
                }
            )
        items.sort(key=lambda item: (-item["visit_count"], item["patient_id"]))
        return {"total": len(items), "items": items[:limit]}

    def patient_timeline(self, patient_id: str) -> dict[str, Any]:
        patient_id = str(patient_id)
        visits = self.patient_visits.get(patient_id)
        if not visits:
            raise KeyError(patient_id)
        timeline = []
        previous = None
        for visit in visits:
            change = (
                self._difference(previous, visit)
                if previous and previous.internal_drugs and visit.internal_drugs
                else None
            )
            timeline.append(
                {
                    "visit_id": visit.visit_id,
                    "date": visit.date[:10],
                    "is_first": visit.is_first,
                    "disease": visit.disease,
                    "raw_diagnosis": visit.raw_diagnosis,
                    "attributes": visit.attributes,
                    "syndrome": visit.syndrome,
                    "symptoms": visit.symptoms,
                    "new_medical_history": visit.new_medical_history,
                    "outcome": visit.outcome,
                    "process_types": visit.process_types,
                    "treatment_days": visit.treatment_days,
                    "drug_count": len(visit.internal_drugs),
                    "drugs": [
                        {"drug_id": drug.drug_id, "drug_name": drug.name, "dose": drug.dose, "unit": drug.unit}
                        for drug in sorted(visit.internal_drugs.values(), key=lambda item: item.name)
                    ],
                    "prescriptions": visit.prescriptions,
                    "change": change,
                }
            )
            previous = visit
        return {"patient_id": patient_id, "visit_count": len(timeline), "timeline": timeline}
