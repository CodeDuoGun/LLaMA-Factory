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

"""Patient-weighted discovery of empirical base formulas from consultation records.

The output contains aggregate observational patterns only. It is not a treatment
recommendation and intentionally excludes visit and patient identifiers.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from medical.analysis.engine import MedicalAnalysis, Visit


EPSILON = 1e-12

FORMULA_DIMENSIONS: dict[str, dict[str, Any]] = {
    "diagnosis_illness": {"label": "西医诊断", "visit_attribute": "disease", "supported": True},
    "diagnosis_sickness": {"label": "中医诊断", "visit_attribute": "tcm_disease", "supported": True},
    "diagnosis_disease": {"label": "中医证候", "visit_attribute": "syndrome", "supported": True},
    "is_first": {"label": "初诊/复诊", "visit_attribute": "is_first", "supported": True},
    "disease_course": {"label": "病程", "supported": False, "reason": "暂无结构化数据支持"},
    "etiology_pathogenesis": {"label": "病因病机", "supported": False, "reason": "暂无结构化数据支持"},
}


@dataclass
class _FPNode:
    item: str | None
    count: float
    parent: _FPNode | None
    children: dict[str, _FPNode] = field(default_factory=dict)
    link: _FPNode | None = None


@dataclass(frozen=True)
class WeightedTransaction:
    items: frozenset[str]
    weight: float


def _build_fp_tree(
    transactions: Iterable[WeightedTransaction], minimum_weight: float
) -> tuple[_FPNode | None, dict[str, tuple[float, _FPNode | None]]]:
    materialized = [transaction for transaction in transactions if transaction.items and transaction.weight > 0]
    support = Counter()
    for transaction in materialized:
        for item in transaction.items:
            support[item] += transaction.weight
    frequent = {item: count for item, count in support.items() if count + EPSILON >= minimum_weight}
    if not frequent:
        return None, {}

    root = _FPNode(None, 0.0, None)
    headers: dict[str, tuple[float, _FPNode | None]] = {item: (count, None) for item, count in frequent.items()}
    tails: dict[str, _FPNode] = {}
    order_key = lambda item: (-frequent[item], item)
    for transaction in materialized:
        ordered = sorted((item for item in transaction.items if item in frequent), key=order_key)
        node = root
        for item in ordered:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, 0.0, node)
                node.children[item] = child
                if item in tails:
                    tails[item].link = child
                else:
                    headers[item] = (frequent[item], child)
                tails[item] = child
            child.count += transaction.weight
            node = child
    return root, headers


def weighted_fp_growth(
    transactions: Iterable[WeightedTransaction],
    minimum_weight: float,
    maximum_length: int = 10,
) -> dict[frozenset[str], float]:
    """Mine weighted frequent itemsets with an FP-tree and no candidate generation."""
    _, headers = _build_fp_tree(transactions, minimum_weight)
    patterns: dict[frozenset[str], float] = {}

    def mine(current_headers: dict[str, tuple[float, _FPNode | None]], suffix: frozenset[str]) -> None:
        for item, (support, head) in sorted(current_headers.items(), key=lambda entry: (entry[1][0], entry[0])):
            pattern = suffix | {item}
            patterns[pattern] = support
            if len(pattern) >= maximum_length:
                continue
            conditional = []
            node = head
            while node is not None:
                path = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional.append(WeightedTransaction(frozenset(path), node.count))
                node = node.link
            _, conditional_headers = _build_fp_tree(conditional, minimum_weight)
            if conditional_headers:
                mine(conditional_headers, pattern)

    mine(headers, frozenset())
    return patterns


def _visit_matches(
    visit: Visit,
    disease: str,
    syndrome: str | None,
    visit_type: str | None,
    process_type: str | None,
) -> bool:
    return bool(
        visit.internal_drugs
        and visit.disease == disease
        and (syndrome is None or visit.syndrome == syndrome)
        and (not visit_type or visit.is_first == visit_type)
        # Exclude mixed-process visits because the shared analysis engine retains one representative formula but
        # does not retain which of several oral prescriptions supplied it.
        and (not process_type or visit.process_types == [process_type])
    )


def formula_strata(
    analysis: MedicalAnalysis,
    minimum_patients: int = 30,
    include_unspecified_syndrome: bool = False,
    visit_type: str | None = None,
    process_type: str | None = None,
) -> dict[str, Any]:
    """List exact disease-syndrome groups and their independent-patient eligibility."""
    if minimum_patients < 1:
        raise ValueError("minimum_patients must be positive.")
    grouped: dict[tuple[str, str], list[Visit]] = defaultdict(list)
    for visit in analysis.visits:
        if (
            not visit.internal_drugs
            or (visit_type and visit.is_first != visit_type)
            or (process_type and visit.process_types != [process_type])
            or (not visit.syndrome and not include_unspecified_syndrome)
        ):
            continue
        grouped[(visit.disease, visit.syndrome)].append(visit)
    items = []
    for (disease, syndrome), visits in grouped.items():
        patient_count = len({visit.patient_id for visit in visits})
        items.append(
            {
                "disease": disease,
                "syndrome": syndrome or "未明确",
                "syndrome_value": syndrome,
                "visit_count": len(visits),
                "patient_count": patient_count,
                "status": "eligible" if patient_count >= minimum_patients else "insufficient_sample",
            }
        )
    items.sort(key=lambda row: (-row["patient_count"], -row["visit_count"], row["disease"], row["syndrome"]))
    return {
        "source": analysis.source_name,
        "minimum_patients": minimum_patients,
        "visit_type": visit_type,
        "process_type": process_type,
        "stratum_count": len(items),
        "eligible_count": sum(item["status"] == "eligible" for item in items),
        "items": items,
    }


def _validate_dimensions(dimensions: Iterable[str]) -> list[str]:
    selected = list(dimensions)
    if not selected:
        raise ValueError("At least one analysis dimension is required.")
    if len(selected) != len(set(selected)):
        raise ValueError("Analysis dimensions must not contain duplicates.")
    unknown = [dimension for dimension in selected if dimension not in FORMULA_DIMENSIONS]
    if unknown:
        raise ValueError(f"Unknown analysis dimensions: {', '.join(unknown)}.")
    unsupported = [dimension for dimension in selected if not FORMULA_DIMENSIONS[dimension]["supported"]]
    if unsupported:
        raise ValueError(f"Unsupported analysis dimensions: {', '.join(unsupported)}.")
    return selected


def _dimension_value(visit: Visit, dimension: str) -> str:
    return str(getattr(visit, FORMULA_DIMENSIONS[dimension]["visit_attribute"], "") or "").strip()


def formula_dimension_metadata(analysis: MedicalAnalysis) -> dict[str, Any]:
    """Describe selectable dimensions and their observed, privacy-safe value counts."""
    dimensions = []
    for name, definition in FORMULA_DIMENSIONS.items():
        row = {"name": name, "label": definition["label"], "supported": definition["supported"]}
        if definition["supported"]:
            counts = Counter(_dimension_value(visit, name) for visit in analysis.visits if visit.internal_drugs)
            counts.pop("", None)
            row.update(
                {
                    "value_count": len(counts),
                    "nonempty_visit_count": sum(counts.values()),
                    "values": [
                        {"value": value, "visit_count": count}
                        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                    ],
                }
            )
        else:
            row["reason"] = definition["reason"]
        dimensions.append(row)
    return {"source": analysis.source_name, "dimensions": dimensions}


def multidimensional_formula_strata(
    analysis: MedicalAnalysis,
    dimensions: Iterable[str],
    minimum_patients: int = 30,
    include_unspecified: bool = False,
    process_type: str | None = None,
) -> dict[str, Any]:
    """List exact-value groups for any supported dimension combination."""
    selected = _validate_dimensions(dimensions)
    if minimum_patients < 1:
        raise ValueError("minimum_patients must be positive.")
    grouped: dict[tuple[str, ...], list[Visit]] = defaultdict(list)
    for visit in analysis.visits:
        if not visit.internal_drugs or (process_type and visit.process_types != [process_type]):
            continue
        values = tuple(_dimension_value(visit, dimension) for dimension in selected)
        if not include_unspecified and any(not value for value in values):
            continue
        grouped[values].append(visit)

    items = []
    for values, visits in grouped.items():
        raw_values = dict(zip(selected, values))
        display_values = {dimension: value or "未明确" for dimension, value in raw_values.items()}
        patient_count = len({visit.patient_id for visit in visits})
        items.append(
            {
                "dimension_values": raw_values,
                "display_dimension_values": display_values,
                "label": " · ".join(
                    f"{FORMULA_DIMENSIONS[dimension]['label']}：{display_values[dimension]}" for dimension in selected
                ),
                "visit_count": len(visits),
                "patient_count": patient_count,
                "status": "eligible" if patient_count >= minimum_patients else "insufficient_sample",
            }
        )
    items.sort(key=lambda row: (-row["patient_count"], -row["visit_count"], row["label"]))
    return {
        "source": analysis.source_name,
        "dimensions": selected,
        "dimension_labels": {dimension: FORMULA_DIMENSIONS[dimension]["label"] for dimension in selected},
        "minimum_patients": minimum_patients,
        "process_type": process_type,
        "stratum_count": len(items),
        "eligible_count": sum(item["status"] == "eligible" for item in items),
        "items": items,
    }


def _weighted_transactions(visits: list[Visit]) -> tuple[list[WeightedTransaction], dict[str, list[Visit]]]:
    by_patient: dict[str, list[Visit]] = defaultdict(list)
    for visit in visits:
        by_patient[visit.patient_id].append(visit)
    transactions = []
    for patient_visits in by_patient.values():
        weight = 1.0 / len(patient_visits)
        for visit in patient_visits:
            names = frozenset(drug.name for drug in visit.internal_drugs.values() if drug.name)
            if names:
                transactions.append(WeightedTransaction(names, weight))
    return transactions, by_patient


def _patient_pattern_rates(
    patterns: Iterable[frozenset[str]], by_patient: dict[str, list[Visit]]
) -> dict[frozenset[str], np.ndarray]:
    pattern_list = list(patterns)
    patient_rates = {pattern: [] for pattern in pattern_list}
    for visits in by_patient.values():
        prescriptions = [{drug.name for drug in visit.internal_drugs.values()} for visit in visits]
        for pattern in pattern_list:
            patient_rates[pattern].append(sum(pattern <= drugs for drugs in prescriptions) / len(prescriptions))
    return {pattern: np.asarray(rates, dtype=float) for pattern, rates in patient_rates.items()}


def _bootstrap_patterns(
    patterns: list[frozenset[str]],
    by_patient: dict[str, list[Visit]],
    minimum_support: float,
    rounds: int,
    seed: int,
) -> dict[frozenset[str], dict[str, float]]:
    if not patterns or rounds <= 0:
        return {}
    rates = _patient_pattern_rates(patterns, by_patient)
    patient_count = len(by_patient)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, patient_count, size=(rounds, patient_count))
    result = {}
    for pattern, values in rates.items():
        samples = values[sample_indices].mean(axis=1)
        result[pattern] = {
            "support_ci_low": round(float(np.quantile(samples, 0.025)), 4),
            "support_ci_high": round(float(np.quantile(samples, 0.975)), 4),
            "bootstrap_support_pass_rate": round(float(np.mean(samples + EPSILON >= minimum_support)), 4),
        }
    return result


def _rank_patterns(
    patterns: dict[frozenset[str], float],
    total_weight: float,
    limit: int,
    minimum_length: int = 2,
) -> list[tuple[frozenset[str], float]]:
    by_length: dict[int, list[tuple[frozenset[str], float]]] = defaultdict(list)
    for pattern, weight in patterns.items():
        if len(pattern) >= minimum_length:
            by_length[len(pattern)].append((pattern, weight / total_weight))
    if not by_length:
        return []
    for candidates in by_length.values():
        candidates.sort(key=lambda entry: (-entry[1], tuple(sorted(entry[0]))))

    # A global support ranking is dominated by pairs and hides formula-sized patterns. Preserve the strongest
    # alternatives at every observed length, then rank longer formula candidates before their shorter subsets.
    quota = max(1, limit // len(by_length))
    selected = [candidate for length in sorted(by_length, reverse=True) for candidate in by_length[length][:quota]]
    if len(selected) < limit:
        selected_patterns = {pattern for pattern, _ in selected}
        remaining = [
            candidate
            for candidates in by_length.values()
            for candidate in candidates
            if candidate[0] not in selected_patterns
        ]
        remaining.sort(key=lambda entry: (-entry[1], -len(entry[0]), tuple(sorted(entry[0]))))
        selected.extend(remaining[: limit - len(selected)])
    selected.sort(key=lambda entry: (-len(entry[0]), -entry[1], tuple(sorted(entry[0]))))
    return selected[:limit]


def _network(
    patterns: dict[frozenset[str], float], total_weight: float, edge_limit: int
) -> dict[str, list[dict[str, Any]]]:
    singles = {next(iter(pattern)): weight / total_weight for pattern, weight in patterns.items() if len(pattern) == 1}
    edges = []
    centrality = Counter()
    for pattern, weight in patterns.items():
        if len(pattern) != 2:
            continue
        left, right = sorted(pattern)
        support = weight / total_weight
        lift = support / (singles[left] * singles[right]) if singles[left] and singles[right] else 0.0
        pmi = math.log2(lift) if lift > 0 else None
        centrality[left] += support
        centrality[right] += support
        edges.append(
            {
                "left": left,
                "right": right,
                "support": round(support, 4),
                "lift": round(lift, 3),
                "pmi": round(pmi, 3) if pmi is not None else None,
            }
        )
    edges.sort(key=lambda row: (-row["support"], -row["lift"], row["left"], row["right"]))
    nodes = [
        {"drug": drug, "weighted_degree": round(value, 4), "support": round(singles.get(drug, 0.0), 4)}
        for drug, value in centrality.most_common(edge_limit)
    ]
    return {"nodes": nodes, "edges": edges[:edge_limit]}


def _nmf_components(
    by_patient: dict[str, list[Visit]],
    component_count: int,
    drugs_per_component: int,
    seed: int,
    iterations: int = 300,
) -> dict[str, Any]:
    drug_names = sorted(
        {drug.name for visits in by_patient.values() for visit in visits for drug in visit.internal_drugs.values()}
    )
    if not drug_names or not by_patient:
        return {"components": [], "reconstruction_error": None}
    index = {drug: position for position, drug in enumerate(drug_names)}
    matrix = np.zeros((len(by_patient), len(drug_names)), dtype=float)
    for row, visits in enumerate(by_patient.values()):
        for visit in visits:
            for drug in visit.internal_drugs.values():
                matrix[row, index[drug.name]] += 1.0 / len(visits)

    rank = min(component_count, matrix.shape[0], matrix.shape[1])
    if rank <= 0:
        return {"components": [], "reconstruction_error": None}
    rng = np.random.default_rng(seed)
    w = rng.random((matrix.shape[0], rank)) + 0.1
    h = rng.random((rank, matrix.shape[1])) + 0.1
    for _ in range(iterations):
        h *= (w.T @ matrix) / (w.T @ w @ h + EPSILON)
        w *= (matrix @ h.T) / (w @ h @ h.T + EPSILON)

    scale = h.sum(axis=1)
    h = h / np.maximum(scale[:, None], EPSILON)
    w = w * scale[None, :]
    components = []
    patient_strength = w / np.maximum(w.sum(axis=1, keepdims=True), EPSILON)
    for component in range(rank):
        order = np.argsort(-h[component])[:drugs_per_component]
        components.append(
            {
                "component": component + 1,
                "patient_prevalence": round(float(np.mean(patient_strength[:, component] >= 0.2)), 4),
                "mean_patient_weight": round(float(np.mean(patient_strength[:, component])), 4),
                "drugs": [
                    {"drug": drug_names[position], "weight": round(float(h[component, position]), 4)}
                    for position in order
                    if h[component, position] > EPSILON
                ],
            }
        )
    error = np.linalg.norm(matrix - w @ h) / max(np.linalg.norm(matrix), EPSILON)
    components.sort(key=lambda row: -row["mean_patient_weight"])
    return {"components": components, "relative_reconstruction_error": round(float(error), 4)}


def _dose_summary(visits: list[Visit], core_drugs: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for visit in visits:
        process = "+".join(visit.process_types) or "未标注"
        for drug in visit.internal_drugs.values():
            if drug.name in core_drugs and drug.dose is not None:
                grouped[(drug.name, process, drug.unit or "未标注")].append(drug.dose)
    rows = []
    for (drug, process, unit), values in grouped.items():
        ordered = sorted(values)
        rows.append(
            {
                "drug": drug,
                "process_type": process,
                "unit": unit,
                "observations": len(values),
                "median": round(float(statistics.median(values)), 3),
                "q1": round(float(np.quantile(ordered, 0.25)), 3),
                "q3": round(float(np.quantile(ordered, 0.75)), 3),
            }
        )
    rows.sort(key=lambda row: (row["drug"], row["process_type"], row["unit"]))
    return rows


def _discover_for_visits(
    analysis: MedicalAnalysis,
    visits: list[Visit],
    scope: dict[str, Any],
    minimum_support: float = 0.3,
    maximum_pattern_length: int = 10,
    pattern_limit: int = 30,
    component_count: int = 3,
    drugs_per_component: int = 12,
    bootstrap_rounds: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the shared mining pipeline over an already selected visit stratum."""
    if not 0 < minimum_support <= 1:
        raise ValueError("minimum_support must be in (0, 1].")
    if maximum_pattern_length < 1 or pattern_limit < 1:
        raise ValueError("maximum_pattern_length and pattern_limit must be positive.")
    transactions, by_patient = _weighted_transactions(visits)
    if not by_patient:
        raise ValueError("No internal prescriptions matched the requested stratum.")
    total_weight = float(len(by_patient))
    patterns = weighted_fp_growth(
        transactions,
        minimum_weight=minimum_support * total_weight,
        maximum_length=maximum_pattern_length,
    )
    ranked = _rank_patterns(patterns, total_weight, pattern_limit)
    bootstrap = _bootstrap_patterns(
        [pattern for pattern, _ in ranked], by_patient, minimum_support, bootstrap_rounds, seed
    )
    pattern_rows = []
    for pattern, support in ranked:
        row = {
            "drugs": sorted(pattern),
            "length": len(pattern),
            "support": round(support, 4),
            "coverage_score": round(len(pattern) * support, 4),
        }
        row.update(bootstrap.get(pattern, {}))
        pattern_rows.append(row)
    representative = (
        max(pattern_rows, key=lambda row: (row["coverage_score"], row["support"], row["length"]))
        if pattern_rows
        else None
    )

    singleton_rows = []
    for pattern, weight in patterns.items():
        if len(pattern) == 1:
            singleton_rows.append({"drug": next(iter(pattern)), "support": round(weight / total_weight, 4)})
    singleton_rows.sort(key=lambda row: (-row["support"], row["drug"]))
    core_drugs = {row["drug"] for row in singleton_rows[: max(drugs_per_component, 20)]}
    maximum_discovered_length = max((len(pattern) for pattern in patterns), default=0)
    search_was_capped = maximum_discovered_length == maximum_pattern_length
    warnings = []
    if search_was_capped:
        warnings.append(
            "Frequent itemsets reached maximum_pattern_length; rerun with a larger cap to check for truncation."
        )
    if len(by_patient) < 30:
        warnings.append("Fewer than 30 patients matched; population-level base-formula inference is unstable.")
    return {
        "scope": {
            "source": analysis.source_name,
            **scope,
            "visit_count": len(visits),
            "patient_count": len(by_patient),
            "minimum_patient_weighted_support": minimum_support,
            "weighting": "Each patient's included visits have total weight 1.",
        },
        "mining_summary": {
            "frequent_itemset_count": len(patterns),
            "maximum_discovered_length": maximum_discovered_length,
            "search_was_capped": search_was_capped,
            "reported_candidate_count": len(pattern_rows),
            "bootstrap_rounds": bootstrap_rounds,
        },
        "warnings": warnings,
        "frequent_drugs": singleton_rows,
        "representative_base_formula": (
            {
                **representative,
                "selection_rule": "Maximum drug-count × patient-weighted-support coverage score.",
                "length_is_capped": representative["length"] == maximum_pattern_length,
            }
            if representative
            else None
        ),
        "candidate_base_formulas": pattern_rows,
        "latent_base_formulas": _nmf_components(by_patient, component_count, drugs_per_component, seed),
        "cooccurrence_network": _network(patterns, total_weight, edge_limit=max(pattern_limit, 50)),
        "dose_summary": _dose_summary(visits, core_drugs),
        "limitations": [
            "Observational single-doctor patterns do not establish efficacy or causality.",
            "Candidate formulas require clinician review and must not be used for autonomous prescribing.",
            "Dose summaries keep process type and unit separate and are descriptive only.",
        ],
    }


def discover_base_formulas(
    analysis: MedicalAnalysis,
    disease: str,
    syndrome: str | None = None,
    visit_type: str | None = None,
    process_type: str | None = None,
    minimum_support: float = 0.3,
    maximum_pattern_length: int = 10,
    pattern_limit: int = 30,
    component_count: int = 3,
    drugs_per_component: int = 12,
    bootstrap_rounds: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Discover aggregate empirical base-formula candidates for one legacy disease-syndrome stratum."""
    visits = [visit for visit in analysis.visits if _visit_matches(visit, disease, syndrome, visit_type, process_type)]
    return _discover_for_visits(
        analysis,
        visits,
        {"disease": disease, "syndrome": syndrome, "visit_type": visit_type, "process_type": process_type},
        minimum_support,
        maximum_pattern_length,
        pattern_limit,
        component_count,
        drugs_per_component,
        bootstrap_rounds,
        seed,
    )


def discover_base_formulas_by_dimensions(
    analysis: MedicalAnalysis,
    dimension_values: dict[str, str],
    process_type: str | None = None,
    minimum_support: float = 0.3,
    maximum_pattern_length: int = 10,
    pattern_limit: int = 30,
    component_count: int = 3,
    drugs_per_component: int = 12,
    bootstrap_rounds: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Discover a base formula for one exact multidimensional value combination."""
    dimensions = _validate_dimensions(dimension_values)
    normalized = {dimension: str(dimension_values[dimension]).strip() for dimension in dimensions}
    visits = [
        visit
        for visit in analysis.visits
        if visit.internal_drugs
        and all(_dimension_value(visit, dimension) == value for dimension, value in normalized.items())
        and (not process_type or visit.process_types == [process_type])
    ]
    return _discover_for_visits(
        analysis,
        visits,
        {
            "dimensions": dimensions,
            "dimension_values": normalized,
            "dimension_labels": {dimension: FORMULA_DIMENSIONS[dimension]["label"] for dimension in dimensions},
            "process_type": process_type,
        },
        minimum_support,
        maximum_pattern_length,
        pattern_limit,
        component_count,
        drugs_per_component,
        bootstrap_rounds,
        seed,
    )


def build_base_formula_catalog(
    analysis: MedicalAnalysis,
    minimum_patients: int = 30,
    include_unspecified_syndrome: bool = False,
    visit_type: str | None = None,
    process_type: str | None = None,
    minimum_support: float = 0.3,
    maximum_pattern_length: int = 10,
    pattern_limit: int = 30,
    component_count: int = 3,
    drugs_per_component: int = 12,
    bootstrap_rounds: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a privacy-safe catalog for every observed disease-syndrome stratum."""
    strata = formula_strata(
        analysis,
        minimum_patients=minimum_patients,
        include_unspecified_syndrome=include_unspecified_syndrome,
        visit_type=visit_type,
        process_type=process_type,
    )
    items = []
    for stratum in strata["items"]:
        item = dict(stratum)
        syndrome_value = item.pop("syndrome_value")
        if item["status"] == "eligible":
            item["base_formula"] = discover_base_formulas(
                analysis,
                disease=item["disease"],
                syndrome=syndrome_value,
                visit_type=visit_type,
                process_type=process_type,
                minimum_support=minimum_support,
                maximum_pattern_length=maximum_pattern_length,
                pattern_limit=pattern_limit,
                component_count=component_count,
                drugs_per_component=drugs_per_component,
                bootstrap_rounds=bootstrap_rounds,
                seed=seed,
            )
        else:
            item["base_formula"] = None
            item["reason"] = f"Independent patient count is below the configured minimum of {minimum_patients}."
        items.append(item)
    return {
        "schema_version": "1.0",
        "source": analysis.source_name,
        "configuration": {
            "minimum_patients": minimum_patients,
            "include_unspecified_syndrome": include_unspecified_syndrome,
            "visit_type": visit_type,
            "process_type": process_type,
            "minimum_support": minimum_support,
            "maximum_pattern_length": maximum_pattern_length,
            "pattern_limit": pattern_limit,
            "component_count": component_count,
            "drugs_per_component": drugs_per_component,
            "bootstrap_rounds": bootstrap_rounds,
            "seed": seed,
        },
        "summary": {
            "stratum_count": strata["stratum_count"],
            "eligible_count": strata["eligible_count"],
            "insufficient_sample_count": strata["stratum_count"] - strata["eligible_count"],
        },
        "items": items,
        "limitations": [
            "The catalog contains single-doctor observational patterns, not validated standard prescriptions.",
            "Groups below minimum_patients are listed but intentionally not mined.",
            "All outputs are aggregate and exclude direct patient and visit identifiers.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover patient-weighted empirical base formulas.")
    parser.add_argument("data", type=Path, help="Consultation JSON file containing a list of records.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--disease", help="Exact diagnosis_illness value to analyze.")
    mode.add_argument("--all-strata", action="store_true", help="Build a catalog for all disease-syndrome groups.")
    parser.add_argument("--syndrome", help="Exact diagnosis_disease syndrome value.")
    parser.add_argument("--visit-type", choices=("初诊", "复诊"))
    parser.add_argument("--process-type", help="Exact drug_process_name value, such as 饮片 or 颗粒.")
    parser.add_argument("--min-support", type=float, default=0.3)
    parser.add_argument("--max-pattern-length", type=int, default=10)
    parser.add_argument("--pattern-limit", type=int, default=30)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--component-drugs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-patients", type=int, default=30, help="Minimum independent patients per group.")
    parser.add_argument("--include-unspecified-syndrome", action="store_true")
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this path; otherwise print to stdout.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = MedicalAnalysis.from_json(args.data)
    common = {
        "visit_type": args.visit_type,
        "process_type": args.process_type,
        "minimum_support": args.min_support,
        "maximum_pattern_length": args.max_pattern_length,
        "pattern_limit": args.pattern_limit,
        "component_count": args.components,
        "drugs_per_component": args.component_drugs,
        "bootstrap_rounds": args.bootstrap,
        "seed": args.seed,
    }
    if args.all_strata:
        report = build_base_formula_catalog(
            analysis,
            minimum_patients=args.min_patients,
            include_unspecified_syndrome=args.include_unspecified_syndrome,
            **common,
        )
    else:
        report = discover_base_formulas(analysis, disease=args.disease, syndrome=args.syndrome, **common)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
