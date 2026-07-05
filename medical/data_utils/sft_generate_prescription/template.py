from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_record_drugs(record_drugs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for item in  record_drugs:
        name = str(item.get("drug_name", "")).strip()
        if not name:
            continue

        result[name] = {
            "name": name,
            "dose": _to_float(item.get("dose")),
            "unit": item.get("unit", "g"),
            "source": item,
        }
    return result


def _extract_template_drugs(template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}

    for item in template.get("drugs", []) or []:
        name = str(item.get("drug_name", "")).strip()
        if not name:
            continue

        result[name] = {
            "name": name,
            "dose": _to_float(item.get("quantity")),
            "unit": item.get("unit", "g"),
            "source": item,
        }

    return result


def _dose_similarity(record_dose: float, template_dose: float) -> float:
    if record_dose <= 0 or template_dose <= 0:
        return 0.0

    diff_ratio = abs(record_dose - template_dose) / max(record_dose, template_dose)
    return max(0.0, 1.0 - diff_ratio)


def _format_drug(name: str, drug_map: Dict[str, Dict[str, Any]]) -> str:
    item = drug_map.get(name, {})
    dose = item.get("dose")
    unit = item.get("unit", "g")

    if dose:
        return f"{name}{dose:g}{unit}"
    return name


def _format_drug_list(names: List[str], drug_map: Dict[str, Dict[str, Any]]) -> str:
    if not names:
        return "无"
    return "、".join(_format_drug(name, drug_map) for name in names)


def get_most_similar_template(
    record: Dict[str, Any],
    templates: List[Dict[str, Any]],
    threshold: float = 0.55,
    drug_weight: float = 0.75,
    dose_weight: float = 0.25,
) -> str:
    """
    根据当前处方和模板方，返回最相似模板方的文本描述。
    相似度 = 药物种类相似度 * drug_weight + 剂量相似度 * dose_weight
    药物种类相似度：
        Jaccard = 交集药物数 / 并集药物数
    剂量相似度：
        只计算重合药物的剂量相似度平均值。
    """
    # 获取内服处方药品名字
    prescriptions = record.get("internal_prescriptions", [])
    for prescription in prescriptions:
        if prescription.get("usage_type") == "内服":
            record_drugs = prescription.get("drugs", [])
            break
    record_drugs = _extract_record_drugs(record_drugs)
    # print(f"record_drugs: {record_drugs}")

    if not record_drugs:
        return "当前处方中未解析到有效药物，无法进行模板方匹配。"

    if not templates:
        return "模板方列表为空，无法进行模板方匹配。"

    best_result = None

    for template in templates:
        template_drugs = _extract_template_drugs(template)

        record_names = set(record_drugs.keys())
        template_names = set(template_drugs.keys())

        matched_names = sorted(record_names & template_names)
        added_names = sorted(record_names - template_names)
        missing_names = sorted(template_names - record_names)

        union_names = record_names | template_names
        drug_similarity = len(matched_names) / len(union_names) if union_names else 0.0
        dose_scores = [
            _dose_similarity(
                record_drugs[name]["dose"],
                template_drugs[name]["dose"],
            )
            for name in matched_names
        ]

        dose_similarity = sum(dose_scores) / len(dose_scores) if dose_scores else 0.0
        final_score = drug_weight * drug_similarity + dose_weight * dose_similarity
        current_result = {
            "template": template,
            "template_drugs": template_drugs,
            "matched_names": matched_names,
            "added_names": added_names,
            "missing_names": missing_names,
            "drug_similarity": drug_similarity,
            "dose_similarity": dose_similarity,
            "final_score": final_score,
        }

        if best_result is None or final_score > best_result["final_score"]:
            best_result = current_result

    best = best_result
    template = best["template"]
    template_drugs = best["template_drugs"]

    template_name = (
        template.get("template_name")
        or template.get("name")
        or f"模板方{template.get('template_id', '')}"
    )

    is_reference = best["final_score"] >= threshold

    dose_adjustments = []
    for name in best["matched_names"]:
        record_dose = record_drugs[name]["dose"]
        template_dose = template_drugs[name]["dose"]
        unit = record_drugs[name].get("unit", "g")

        if record_dose != template_dose:
            dose_adjustments.append(
                f"{name}：模板{template_dose:g}{unit}，当前{record_dose:g}{unit}"
            )

    lines = []
    lines.append("【模板方匹配结果】")
    lines.append(f"是否参考模板方：{'是' if is_reference else '否'}")
    # lines.append(f"候选模板方：{template_name}")
    # lines.append(f"模板ID：{template.get('template_id', '未知')}")
    lines.append(f"模板匹配度：{best['final_score']:.2f}")
    lines.append("")

    # lines.append("【匹配依据】")
    # lines.append(
    #     f"当前处方共 {len(record_drugs)} 味药，模板方共 {len(template_drugs)} 味药，"
    #     f"重合 {len(best['matched_names'])} 味。"
    # )
    # lines.append(
    #     f"药物种类相似度：{best['drug_similarity']:.2f}，"
    #     f"剂量相似度：{best['dose_similarity']:.2f}。"
    # )
    # lines.append(
    #     f"综合相似度 = 药物种类相似度 × {drug_weight:.2f} "
    #     f"+ 剂量相似度 × {dose_weight:.2f}"
    # )
    # lines.append("")

    lines.append("【保留药物】")
    lines.append(_format_drug_list(best["matched_names"], record_drugs))
    lines.append("")

    lines.append("【新增药物】")
    lines.append(_format_drug_list(best["added_names"], record_drugs))
    lines.append("")

    lines.append("【模板中未使用药物】")
    lines.append(_format_drug_list(best["missing_names"], template_drugs))
    lines.append("")

    lines.append("【剂量调整】")
    lines.append("；".join(dose_adjustments) if dose_adjustments else "无明显剂量调整。")
    lines.append("")

    lines.append("【模板方明细】")
    lines.append(_format_drug_list(sorted(template_drugs.keys()), template_drugs))
    lines.append("")

    lines.append("【判断结论】")
    if is_reference:
        lines.append(
            f"当前处方与模板方“{template_name}”相似度较高，可认为是在该模板方基础上进行随证加减。"
        )
    else:
        lines.append(
            f"当前处方与最相近模板方“{template_name}”相似度不足，不建议强行判定为模板方加减。"
        )

    return "\n".join(lines)