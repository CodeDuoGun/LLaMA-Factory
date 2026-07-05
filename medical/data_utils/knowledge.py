from typing import Any, Optional


class TCMKnowledgeRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: Optional[dict[str, Any]] = None):
        with self.driver.session(database=self.database) as session:
            rows = [r.data() for r in session.run(cypher, params or {})]
        return json_safe(rows)

    def get_syndrome_diagnosis_context(
        self,
        disease: str,
        syndrome: str,
        symptoms: Optional[list[str]] = None,
        herbs: Optional[list[str]] = None,
        limit: int = 2,
    ) -> dict[str, Any]:
        symptoms = symptoms or []
        herbs = herbs or []

        return {
            "诊断": disease,
            "证候": syndrome,
            "病机信息": self.get_pathogenesis(disease, syndrome),
            "诊断证候相关关系": self.get_disease_syndrome_relations(disease, syndrome),
            "证候相关关系": self.get_syndrome_relations(syndrome),
            "诊断相关关系": self.get_disease_relations(disease),
            "临床症状": self.get_clinical_symptoms(disease, syndrome, symptoms),
            "相关药物知识": self.get_herb_knowledge(herbs),
            "相似病例": self.get_similar_cases(disease, syndrome, symptoms, limit),
            "证候相似病例": self.get_syndrome_similar_cases(syndrome, symptoms, limit),
        }

    def get_disease_syndrome_relations(self, disease: str, syndrome: str, limit: int = 50):
        if not disease or not syndrome:
            return []

        cypher = """
        MATCH (d:Disease {name:$disease})
        MATCH (s:Syndrome {name:$syndrome})
        CALL (d, s) {
            MATCH (d)-[r]->(s)
            RETURN
                labels(d) AS source_labels,
                d.name AS source_name,
                properties(d) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(s) AS target_labels,
                s.name AS target_name,
                properties(s) AS target_properties,
                "disease_to_syndrome" AS direction
            UNION ALL
            MATCH (s)-[r]->(d)
            RETURN
                labels(s) AS source_labels,
                s.name AS source_name,
                properties(s) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(d) AS target_labels,
                d.name AS target_name,
                properties(d) AS target_properties,
                "syndrome_to_disease" AS direction
        }
        RETURN
            source_labels,
            source_name,
            source_properties,
            relation_type,
            relation_properties,
            target_labels,
            target_name,
            target_properties,
            direction
        ORDER BY relation_type ASC, source_name ASC, target_name ASC
        LIMIT $limit
        """
        return self._format_relations(
            self.query(cypher, {"disease": disease, "syndrome": syndrome, "limit": limit})
        )

    def get_disease_properties(self, disease: str):
        if not disease:
            return {}

        cypher = """
        MATCH (d:Disease {name:$disease})
        RETURN properties(d) AS properties
        LIMIT 1
        """
        rows = self.query(cypher, {"disease": disease})
        if not rows:
            return {}
        return rows[0].get("properties") or {}

    def get_syndrome_properties(self, syndrome: str):
        if not syndrome:
            return {}

        cypher = """
        MATCH (s:Syndrome {name:$syndrome})
        RETURN properties(s) AS properties
        LIMIT 1
        """
        rows = self.query(cypher, {"syndrome": syndrome})
        if not rows:
            return {}
        return rows[0].get("properties") or {}

    def get_disease_syndrome_properties(self, disease: str, syndrome: str):
        if not disease and not syndrome:
            return {"诊断属性": {}, "证候属性": {}}

        cypher = """
        OPTIONAL MATCH (d:Disease {name:$disease})
        OPTIONAL MATCH (s:Syndrome {name:$syndrome})
        RETURN
            properties(d) AS disease_properties,
            properties(s) AS syndrome_properties
        LIMIT 1
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome})
        if not rows:
            return {"诊断属性": {}, "证候属性": {}}
        return {
            "诊断属性": rows[0].get("disease_properties") or {},
            "证候属性": rows[0].get("syndrome_properties") or {},
        }


    def get_syndrome_relations(self, syndrome: str, limit: Optional[int] = None):
        """查询与某个证候节点直接相连的全部关系。

        默认返回该证候所有一跳入边和出边关系；如果担心结果过多，可传入 limit 截断返回数量。
        COMMONLY_USES 关系通常较多，会在查询后最多保留前 5 条。
        HAS_SYNDROME 关系通常对应相似病历，会在查询后最多保留前 2 条。
        MANIFESTS_AS 关系通常对应相关疾病，会在查询后最多保留前 5 条。
        返回结果会保留关系方向、关系类型、关系属性、相邻节点标签与属性。
        """
        if not syndrome:
            return []

        limit_clause = "\n        LIMIT $limit" if limit is not None else ""
        cypher = f"""
        MATCH (s:Syndrome {{name:$syndrome}})
        CALL (s) {{
            MATCH (source)-[r]->(s)
            RETURN
                labels(source) AS source_labels,
                source.name AS source_name,
                properties(source) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(s) AS target_labels,
                s.name AS target_name,
                properties(s) AS target_properties,
                "incoming" AS direction
            UNION ALL
            MATCH (s)-[r]->(target)
            RETURN
                labels(s) AS source_labels,
                s.name AS source_name,
                properties(s) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(target) AS target_labels,
                target.name AS target_name,
                properties(target) AS target_properties,
                "outgoing" AS direction
        }}
        RETURN
            source_labels,
            source_name,
            source_properties,
            relation_type,
            relation_properties,
            target_labels,
            target_name,
            target_properties,
            direction
        ORDER BY relation_type ASC, source_name ASC, target_name ASC
        {limit_clause}
        """
        params = {"syndrome": syndrome}
        if limit is not None:
            params["limit"] = limit
        rows = self._limit_syndrome_relation_types(
            self.query(cypher, params),
            max_counts={
                "COMMONLY_USES": 5,
                "HAS_SYNDROME": 2,
                "MANIFESTS_AS": 5,
            },
        )
        return self._format_relations(rows)

    def get_disease_relations(self, disease: str, limit: Optional[int] = None):
        """查询与某个疾病节点直接相连的全部关系。

        默认返回该疾病所有一跳入边和出边关系；如果担心结果过多，可传入 limit 截断返回数量。
        查询后会按关系类型限制数量：默认每种关系最多 5 条，HAS_DISEASE 和 MANIFESTS_AS 最多 2 条。
        返回结果会保留关系方向、关系类型、关系属性、相邻节点标签与属性。
        """
        if not disease:
            return []

        limit_clause = "\n        LIMIT $limit" if limit is not None else ""
        cypher = f"""
        MATCH (d:Disease {{name:$disease}})
        CALL (d) {{
            MATCH (source)-[r]->(d)
            RETURN
                labels(source) AS source_labels,
                source.name AS source_name,
                properties(source) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(d) AS target_labels,
                d.name AS target_name,
                properties(d) AS target_properties,
                "incoming" AS direction
            UNION ALL
            MATCH (d)-[r]->(target)
            RETURN
                labels(d) AS source_labels,
                d.name AS source_name,
                properties(d) AS source_properties,
                type(r) AS relation_type,
                properties(r) AS relation_properties,
                labels(target) AS target_labels,
                target.name AS target_name,
                properties(target) AS target_properties,
                "outgoing" AS direction
        }}
        RETURN
            source_labels,
            source_name,
            source_properties,
            relation_type,
            relation_properties,
            target_labels,
            target_name,
            target_properties,
            direction
        ORDER BY relation_type ASC, source_name ASC, target_name ASC
        {limit_clause}
        """
        params = {"disease": disease}
        if limit is not None:
            params["limit"] = limit
        rows = self._limit_syndrome_relation_types(
            self.query(cypher, params),
            default_max_count=5,
            max_counts={"HAS_DISEASE": 2, "MANIFESTS_AS": 2},
        )
        return self._format_relations(rows)

    def _format_relations(self, rows: list[dict[str, Any]]):
        relations = []
        for row in rows:
            relations.append({
                "source": {
                    "name": row.get("source_name"),
                    "labels": row.get("source_labels") or [],
                    "properties": row.get("source_properties") or {},
                },
                "relation": row.get("relation_type"),
                "target": {
                    "name": row.get("target_name"),
                    "labels": row.get("target_labels") or [],
                    "properties": row.get("target_properties") or {},
                },
                "direction": row.get("direction"),
                "properties": row.get("relation_properties") or {},
            })
        return relations

    def _limit_syndrome_relation_types(
        self,
        rows: list[dict[str, Any]],
        max_counts: dict[str, int],
        default_max_count: Optional[int] = None,
    ):
        relation_counts = {relation_type: 0 for relation_type in max_counts}
        result = []
        for row in rows:
            relation_type = row.get("relation_type")
            max_count = max_counts.get(relation_type, default_max_count)
            if max_count is None:
                result.append(row)
                continue

            relation_count = relation_counts.get(relation_type, 0)
            if relation_count >= max_count:
                continue

            result.append(row)
            relation_counts[relation_type] = relation_count + 1
        return result

    def get_treatment_principles(self, disease: str, syndrome: str):
        """查询指定诊断-证候组合在 MANIFESTS_AS 关系上的治则治法。"""
        cypher = """
        MATCH (d:Disease {name:$disease})-[r:MANIFESTS_AS]->(s:Syndrome {name:$syndrome})
        RETURN
            s.name AS syndrome,
            s.description AS syndrome_description,
            properties(r) AS relation_properties
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome})

        result = []
        for row in rows:
            props = row.get("relation_properties") or {}
            treatment = props.get("treatment") or props.get("治则治法") or props.get("治法")
            if treatment:
                result.append({
                    "name": treatment,
                    "source": "Disease-MANIFESTS_AS-Syndrome",
                    "properties": props,
                })
        return result

    def get_pathogenesis(self, disease: str, syndrome: str):
        """查询指定诊断-证候组合对应的病机，优先使用关系属性，其次使用证候属性。"""
        cypher = """
        MATCH (d:Disease {name:$disease})-[r:MANIFESTS_AS]->(s:Syndrome {name:$syndrome})
        RETURN
            s.name AS syndrome,
            s.description AS syndrome_description,
            properties(r) AS relation_properties,
            properties(s) AS syndrome_properties
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome})

        result = []
        for row in rows:
            rel_props = row.get("relation_properties") or {}
            syn_props = row.get("syndrome_properties") or {}

            pathogenesis = (
                rel_props.get("pathogenesis")
                or rel_props.get("病机")
                or syn_props.get("pathogenesis")
                or syn_props.get("病机")
                or row.get("syndrome_description")
            )

            if pathogenesis:
                result.append({
                    "name": pathogenesis,
                    "source": "Syndrome.description / MANIFESTS_AS.properties",
                    "properties": {
                        "relation": rel_props,
                        "syndrome": syn_props,
                    },
                })
        return result

    def get_syndrome_pathogenesis(self, syndrome: str):
        """查询某个证候相关的病机信息。

        检索范围包括：
        1. 证候节点自身属性，如 pathogenesis、病机、description；
        2. 所有关联疾病通过 MANIFESTS_AS 指向该证候时，关系上的 pathogenesis、病机属性。
        """
        if not syndrome:
            return []

        cypher = """
        MATCH (s:Syndrome {name:$syndrome})
        OPTIONAL MATCH (d:Disease)-[r:MANIFESTS_AS]->(s)
        RETURN
            d.name AS disease,
            s.name AS syndrome,
            s.description AS syndrome_description,
            properties(s) AS syndrome_properties,
            CASE WHEN r IS NULL THEN {} ELSE properties(r) END AS relation_properties
        ORDER BY disease ASC
        """
        rows = self.query(cypher, {"syndrome": syndrome})

        result = []
        seen = set()
        for row in rows:
            rel_props = row.get("relation_properties") or {}
            syn_props = row.get("syndrome_properties") or {}
            pathogenesis = (
                rel_props.get("pathogenesis")
                or rel_props.get("病机")
                or syn_props.get("pathogenesis")
                or syn_props.get("病机")
                or row.get("syndrome_description")
            )
            if not pathogenesis or pathogenesis in seen:
                continue

            result.append({
                "name": pathogenesis,
                "source": "Syndrome.properties / Disease-MANIFESTS_AS-Syndrome",
                "disease": row.get("disease"),
                "syndrome": row.get("syndrome"),
                "properties": {
                    "relation": rel_props,
                    "syndrome": syn_props,
                },
            })
            seen.add(pathogenesis)
        return result

    def get_syndrome_treatments(self, syndrome: str):
        """查询某个证候相关的治则治法。

        检索范围包括：
        1. 证候节点自身属性，如 treatment、治则治法、治法；
        2. 所有关联疾病通过 MANIFESTS_AS 指向该证候时，关系上的 treatment、治则治法、治法属性。
        """
        if not syndrome:
            return []

        cypher = """
        MATCH (s:Syndrome {name:$syndrome})
        OPTIONAL MATCH (d:Disease)-[r:MANIFESTS_AS]->(s)
        RETURN
            d.name AS disease,
            s.name AS syndrome,
            properties(s) AS syndrome_properties,
            CASE WHEN r IS NULL THEN {} ELSE properties(r) END AS relation_properties
        ORDER BY disease ASC
        """
        rows = self.query(cypher, {"syndrome": syndrome})

        result = []
        seen = set()
        for row in rows:
            rel_props = row.get("relation_properties") or {}
            syn_props = row.get("syndrome_properties") or {}
            treatment = (
                rel_props.get("treatment")
                or rel_props.get("治则治法")
                or rel_props.get("治法")
                or syn_props.get("treatment")
                or syn_props.get("治则治法")
                or syn_props.get("治法")
            )
            if not treatment or treatment in seen:
                continue

            result.append({
                "name": treatment,
                "source": "Syndrome.properties / Disease-MANIFESTS_AS-Syndrome",
                "disease": row.get("disease"),
                "syndrome": row.get("syndrome"),
                "properties": {
                    "relation": rel_props,
                    "syndrome": syn_props,
                },
            })
            seen.add(treatment)
        return result

    def get_clinical_symptoms(
        self,
        disease: str,
        syndrome: str,
        input_symptoms: Optional[list[str]] = None,
    ):
        input_symptoms = input_symptoms or []

        cypher = """
        MATCH (c:Case)-[:HAS_DISEASE]->(:Disease {name:$disease})
        MATCH (c)-[:HAS_SYNDROME]->(:Syndrome {name:$syndrome})
        MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)
        RETURN
            sym.name AS symptom,
            count(DISTINCT c) AS cases
        ORDER BY cases DESC, symptom ASC
        LIMIT 50
        """

        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome})

        kg_symptoms = [
            {
                "name": r["symptom"],
                "labels": ["Symptom"],
                "properties": {"cases": r["cases"]},
            }
            for r in rows
            if r.get("symptom")
        ]

        kg_names = {x["name"] for x in kg_symptoms}
        return {
            "知识图谱症状": kg_symptoms,
            "输入症状": input_symptoms,
            "已匹配症状": [s for s in input_symptoms if s in kg_names],
            "未匹配症状": [s for s in input_symptoms if s not in kg_names],
        }

    def get_herb_knowledge(self, herbs: list[str]):
        if not herbs:
            return []

        cypher = """
        MATCH (h:Herb)
        WHERE h.name IN $herbs

        OPTIONAL MATCH (h)-[:HAS_FUNCTION]->(f:Function)
        OPTIONAL MATCH (h)-[:INDICATED_FOR]->(i:Indication)
        OPTIONAL MATCH (h)-[:ENTERS_MERIDIAN]->(m:Meridian)
        OPTIONAL MATCH (h)-[:CONTRAINDICATED_FOR]->(c:Contraindication)
        OPTIONAL MATCH (h)-[:HAS_ADVERSE_EFFECT]->(ae:AdverseEffect)
        OPTIONAL MATCH (h)-[:HAS_PREPARATION]->(p:Preparation)

        RETURN
            h.name AS herb,
            properties(h) AS herb_properties,
            collect(DISTINCT f.name) AS functions,
            collect(DISTINCT i.name) AS indications,
            collect(DISTINCT m.name) AS meridians,
            collect(DISTINCT c.name) AS contraindications,
            collect(DISTINCT ae.name) AS adverse_effects,
            collect(DISTINCT p.name) AS preparations
        ORDER BY herb ASC
        """

        return self.query(cypher, {"herbs": herbs})

    def get_similar_cases(
        self,
        disease: str,
        syndrome: str,
        symptoms: Optional[list[str]] = None,
        limit: int = 2,
    ):
        """查询相似病例，默认返回 2 条。"""
        symptoms = symptoms or []

        cypher = """
        MATCH (c:Case)

        OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease)
        OPTIONAL MATCH (c)-[:HAS_SYNDROME]->(syn:Syndrome)
        OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)

        WITH
            c,
            collect(DISTINCT d.name) AS diseases,
            collect(DISTINCT syn.name) AS syndromes,
            collect(DISTINCT sym.name) AS case_symptoms

        WITH
            c, diseases, syndromes, case_symptoms,
            CASE WHEN $disease IN diseases THEN 3 ELSE 0 END AS disease_score,
            CASE WHEN $syndrome IN syndromes THEN 4 ELSE 0 END AS syndrome_score,
            size([x IN case_symptoms WHERE x IN $symptoms]) AS symptom_score

        WHERE disease_score + syndrome_score + symptom_score > 0

        OPTIONAL MATCH (c)-[pr:PRESCRIBED]->(h:Herb)

        RETURN
            c.id AS id,
            properties(c) AS case_properties,
            diseases,
            syndromes,
            case_symptoms AS symptoms,
            disease_score + syndrome_score + symptom_score AS score,
            collect(DISTINCT {
                name: h.name,
                prescription_relation: properties(pr)
            }) AS herbs
        ORDER BY score DESC, id DESC
        LIMIT $limit
        """

        return self.query(
            cypher,
            {
                "disease": disease,
                "syndrome": syndrome,
                "symptoms": symptoms,
                "limit": limit,
            },
        )

    def get_syndrome_similar_cases(
        self,
        syndrome: str,
        symptoms: Optional[list[str]] = None,
        limit: int = 2,
    ):
        """查询同证候下的相似病例，并按输入症状重合度排序。

        评分规则：
        - 命中同一证候记 4 分；
        - 每个输入症状与病例症状重合记 1 分；
        - 未传入症状时，返回同证候病例并按病例 id 倒序排列。
        """
        if not syndrome:
            return []

        symptoms = symptoms or []

        cypher = """
        MATCH (c:Case)-[:HAS_SYNDROME]->(syn:Syndrome {name:$syndrome})

        OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease)
        OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)

        WITH
            c,
            collect(DISTINCT d.name) AS diseases,
            collect(DISTINCT syn.name) AS syndromes,
            collect(DISTINCT sym.name) AS case_symptoms

        WITH
            c, diseases, syndromes, case_symptoms,
            4 AS syndrome_score,
            size([x IN case_symptoms WHERE x IN $symptoms]) AS symptom_score

        OPTIONAL MATCH (c)-[pr:PRESCRIBED]->(h:Herb)

        RETURN
            c.id AS id,
            properties(c) AS case_properties,
            diseases,
            syndromes,
            case_symptoms AS symptoms,
            syndrome_score + symptom_score AS score,
            collect(DISTINCT {
                name: h.name,
                prescription_relation: properties(pr)
            }) AS herbs
        ORDER BY score DESC, id DESC
        LIMIT $limit
        """

        return self.query(
            cypher,
            {
                "syndrome": syndrome,
                "symptoms": symptoms,
                "limit": limit,
            },
        )

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "iso_format"):   # Neo4j Date / DateTime
        return obj.iso_format()
    if hasattr(obj, "isoformat"):    # Python datetime/date
        return obj.isoformat()
    return obj

def _join(items: Optional[list[Any]], sep: str = "、", default: str = "无") -> str:
    items = [str(x) for x in (items or []) if x]
    return sep.join(items) if items else default


def _get_case_herbs(case: dict[str, Any], max_herbs: int = 20) -> str:
    herbs = []
    for item in case.get("herbs", []) or []:
        name = item.get("name")
        dose = (item.get("prescription_relation") or {}).get("dose")
        if name and dose:
            herbs.append(f"{name}{dose:g}g")
        elif name:
            herbs.append(name)

    return _join(herbs[:max_herbs])


def _format_relation_properties(properties: Optional[dict[str, Any]], max_items: int = 5) -> str:
    items = []
    for key, value in (properties or {}).items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            value = _join(value)
        items.append(f"{key}：{value}")
        if len(items) >= max_items:
            break
    return "；".join(items)


_PROPERTY_LABELS = {
    "name": "名称",
    "tcm_name": "中医病名",
    "affected_population": "好发人群",
    "pathogenesis_tcm": "中医病机",
    "pathogenesis_western": "西医病因病机",
    "prognosis": "预后",
    "description": "描述",
    "pathogenesis": "病机",
    "treatment": "治法",
    "治则治法": "治则治法",
    "治法": "治法",
    "病机": "病机",
}


def _format_node_properties(properties: Optional[dict[str, Any]], max_items: int = 20) -> list[str]:
    items = []
    for key, value in (properties or {}).items():
        if value is None or value == "":
            continue
        if isinstance(value, list):
            value = _join(value)
        label = _PROPERTY_LABELS.get(key, key)
        items.append(f"{label}：{value}")
        if len(items) >= max_items:
            break
    return items


def _append_property_section(lines: list[str], title: str, properties: dict[str, Any]) -> None:
    lines.append(title)
    property_items = _format_node_properties(properties)
    if property_items:
        for item in property_items:
            lines.append(f"- {item}")
    else:
        lines.append("- 知识图谱中暂无属性信息。")
    lines.append("")


def _get_relation_property(properties: Optional[dict[str, Any]], keys: list[str]) -> Any:
    for key in keys:
        value = (properties or {}).get(key)
        if value:
            return value
    return None


def _format_relation_sentence(item: dict[str, Any]) -> str | None:
    source = (item.get("source") or {}).get("name") or ""
    target = (item.get("target") or {}).get("name") or ""
    relation = item.get("relation") or "UNKNOWN_RELATION"
    properties = item.get("properties") or {}
    source_known = bool(source and source != "未知节点")
    target_known = bool(target and target != "未知节点")
    property_text = _format_relation_properties(properties)

    if relation == "COMPLICATES_WITH" and source_known and target_known:
        return f"{source}可能并发或累及{target}。"

    if relation == "DIFFERENTIATE_FROM" and source_known and target_known:
        key_points = _get_relation_property(properties, ["key_points", "鉴别要点", "points"])
        if key_points:
            return f"鉴别诊断：{source}与{target}需鉴别，要点为：{key_points}"
        return f"鉴别诊断：{source}需与{target}鉴别。"

    if relation == "HAS_ADVICE":
        advice = _get_relation_property(properties, ["advice", "建议", "content", "text", "name"])
        if advice:
            return f"诊疗建议：{advice}"
        if target_known:
            return f"诊疗建议：{target}。"
        return None

    if relation == "MANIFESTS_AS" and source_known and target_known:
        if property_text:
            return f"疾病证候关系：{source}可表现为{target}；{property_text}"
        return f"疾病证候关系：{source}可表现为{target}。"

    if not source_known or not target_known:
        return None

    if property_text:
        return f"{source}与{target}存在{relation}关系；{property_text}"
    return f"{source}与{target}存在{relation}关系。"


def _get_node_id_or_name(node: dict[str, Any]) -> str:
    name = node.get("name")
    properties = node.get("properties") or {}
    return name or properties.get("id") or "未知节点"


def _format_disease_relation_item(item: dict[str, Any]) -> str | None:
    source = item.get("source") or {}
    target = item.get("target") or {}
    relation = item.get("relation")
    properties = item.get("properties") or {}
    source_name = _get_node_id_or_name(source)
    target_name = _get_node_id_or_name(target)

    if relation == "COMPLICATES_WITH":
        return f"{source_name}可能并发或累及{target_name}。"

    if relation == "DIFFERENTIATE_FROM":
        key_points = _get_relation_property(properties, ["key_points", "鉴别要点", "points"])
        if key_points:
            return f"鉴别诊断：{source_name}与{target_name}需鉴别；鉴别要点：{key_points}"
        return f"鉴别诊断：{source_name}与{target_name}需鉴别。"

    if relation == "HAS_ADVICE":
        advice = _get_relation_property(properties, ["advice", "建议", "content", "text", "name"])
        advice = advice or _get_relation_property(target.get("properties") or {}, ["advice", "建议", "content", "text", "name"])
        if advice:
            return f"{advice}"
        return f"关联生活建议：{target_name}。"

    if relation == "HAS_DISEASE":
        case_props = source.get("properties") or {}
        case_id = case_props.get("id") or source_name
        chief_complaint = case_props.get("chief_complaint")
        gender = case_props.get("gender")
        age = case_props.get("age")
        patient = "，".join(str(x) for x in [gender, f"{age}岁" if age else None] if x)
        parts = [f"病例 {case_id}"]
        if patient:
            parts.append(patient)
        if chief_complaint:
            parts.append(f"主诉：{chief_complaint}")
        return "；".join(parts)

    if relation == "HAS_STAGE":
        return f"疾病分期：{target_name}。"

    if relation == "MANIFESTS_AS":
        count = properties.get("count")
        if count is not None:
            return f"可表现为证候：{target_name}（{count}例）。"
        return f"可表现为证候：{target_name}。"

    return None


def _append_disease_relation_section(lines: list[str], relations: list[dict[str, Any]]) -> None:
    lines.append("【诊断关系参考】")

    relation_groups = [
        ("疾病可能的并发症", "COMPLICATES_WITH"),
        ("疾病鉴别诊断关系", "DIFFERENTIATE_FROM"),
        ("疾病关联生活建议", "HAS_ADVICE"),
        ("疾病关联的病例", "HAS_DISEASE"),
        ("疾病分期", "HAS_STAGE"),
        ("疾病表现的证候", "MANIFESTS_AS"),
    ]

    for title, relation_type in relation_groups:
        lines.append(f"{title}：")
        relation_texts = []
        seen = set()
        for item in relations:
            if item.get("relation") != relation_type:
                continue
            relation_text = _format_disease_relation_item(item)
            if not relation_text or relation_text in seen:
                continue
            relation_texts.append(relation_text)
            seen.add(relation_text)

        if relation_texts:
            for relation_text in relation_texts:
                lines.append(f"- {relation_text}")
        else:
            lines.append("- 知识图谱中暂无相关信息。")
    lines.append("")


def _append_relation_section(
    lines: list[str], title: str, relations: list[dict[str, Any]], max_relations: int
) -> None:
    lines.append(title)
    relation_texts = []
    seen = set()
    for item in relations:
        relation_text = _format_relation_sentence(item)
        if not relation_text or relation_text in seen:
            continue
        relation_texts.append(relation_text)
        seen.add(relation_text)
        if len(relation_texts) >= max_relations:
            break

    if relation_texts:
        for relation_text in relation_texts:
            lines.append(f"- {relation_text}")
    else:
        lines.append("- 知识图谱中暂无相关关系。")
    lines.append("")


def format_kg_context_for_llm(
    kg: dict[str, Any],
    max_symptoms: int = 20,
    max_cases: int = 2,
    max_herbs_per_case: int = 20,
    max_relations: int = 20,
) -> str:
    disease = kg.get("诊断") or ""
    syndrome = kg.get("证候") or ""

    lines = []
    lines.append("【知识图谱参考】")
    lines.append(f"疾病：{disease}")
    lines.append(f"证候：{syndrome}")
    lines.append("")

    # 病机
    pathogenesis = kg.get("病机信息") or []
    lines.append("【病机参考】")
    if pathogenesis:
        for item in pathogenesis:
            lines.append(f"- {item.get('name')}")
    else:
        lines.append("- 知识图谱中暂无明确病机描述。")
    lines.append("")

    _append_relation_section(lines, "【诊断-证候关系参考】", kg.get("诊断证候相关关系") or [], max_relations)
    _append_relation_section(lines, "【证候关系参考】", kg.get("证候相关关系") or [], max_relations)
    _append_disease_relation_section(lines, kg.get("诊断相关关系") or [])

    # 临床症状
    clinical = kg.get("临床症状") or {}
    kg_symptoms = clinical.get("知识图谱症状") or []
    input_symptoms = clinical.get("输入症状") or []
    matched = clinical.get("已匹配症状") or []
    unmatched = clinical.get("未匹配症状") or []

    lines.append("【临床症状参考】")
    if kg_symptoms:
        lines.append("该疾病-证候下常见症状：")
        for item in kg_symptoms[:max_symptoms]:
            name = item.get("name")
            cases = (item.get("properties") or {}).get("cases")
            if cases is not None:
                lines.append(f"- {name}（{cases}例）")
            else:
                lines.append(f"- {name}")
    else:
        lines.append("- 知识图谱中暂无该证候的症状统计。")
    if input_symptoms:
        lines.append(f"本次输入症状：{_join(input_symptoms)}")
    if matched:
        lines.append(f"已匹配症状：{_join(matched)}")
    if unmatched:
        lines.append(f"未匹配症状：{_join(unmatched)}")
    lines.append("")

    # 药物知识
    herb_knowledge = kg.get("相关药物知识") or []
    lines.append("【药物知识参考】")
    if herb_knowledge:
        for herb in herb_knowledge:
            name = herb.get("herb")
            props = herb.get("herb_properties") or {}
            functions = herb.get("functions") or []
            indications = herb.get("indications") or []
            meridians = herb.get("meridians") or []
            contraindications = herb.get("contraindications") or []

            lines.append(f"{name}：")
            if props.get("category"):
                lines.append(f"- 类别：{props.get('category')}")
            if props.get("nature") or props.get("flavors"):
                lines.append(f"- 性味：{props.get('nature', '')}，{_join(props.get('flavors'))}")
            if meridians:
                lines.append(f"- 归经：{_join(meridians)}")
            if functions:
                lines.append(f"- 功效：{_join(functions)}")
            if indications:
                lines.append(f"- 主治：{_join(indications[:10])}")
            if props.get("textbook_dose"):
                lines.append(f"- 常用剂量：{props.get('textbook_dose')}")
            if contraindications:
                lines.append(f"- 禁忌：{_join(contraindications)}")
    else:
        lines.append("- 知识图谱中暂无相关药物知识。")
    lines.append("")

    # 相似病例
    cases = kg.get("相似病例") or []
    lines.append("【相似病例参考】")
    if cases:
        for idx, case in enumerate(cases[:max_cases], start=1):
            props = case.get("case_properties") or {}
            gender = props.get("gender") or "未知性别"
            age = props.get("age") or "未知年龄"
            chief = props.get("chief_complaint") or "未记录主诉"

            lines.append(f"相似病例{idx}：")
            lines.append(f"- 患者：{gender}，{age}岁")
            lines.append(f"- 主诉：{chief}")
            lines.append(f"- 诊断：{_join(case.get('diseases'))}")
            lines.append(f"- 证候：{_join(case.get('syndromes'))}")
            lines.append(f"- 症状：{_join(case.get('symptoms'))}")
            lines.append(f"- 相似度评分：{case.get('score')}")
            lines.append(f"- 用药：{_get_case_herbs(case, max_herbs=max_herbs_per_case)}")
    else:
        lines.append("- 未检索到相似病例。")

    lines.append("")
    lines.append("【使用要求】")
    lines.append("- 以上知识图谱内容仅作为参考。")
    lines.append("- 若患者当前症状与知识图谱不一致，应优先依据患者当前症状进行辨证。")
    lines.append("- 可结合相似病例中的核心用药，但不得机械照搬。")

    return "\n".join(lines)

def format_disease_properties(props: dict) -> str:
    field_labels = {
        "name": "诊断名称",
        "tcm_name": "中医病名",
        "affected_population": "好发人群",
        "western_summary": "西医概述",
        "pathogenesis_western": "西医病因病机",
        "pathogenesis_tcm": "中医病机",
        "prognosis": "预后",
        "updated_at": "更新时间",
    }

    order = [
        "name",
        "tcm_name",
        "affected_population",
        "western_summary",
        "pathogenesis_western",
        "pathogenesis_tcm",
        "prognosis",
        "updated_at",
    ]

    lines = []
    for key in order:
        value = props.get(key)
        if value:
            lines.append(f"{field_labels[key]}：{value}")

    return "\n".join(lines)


if __name__ == "__main__":
    from medical.config import config
    retriever = TCMKnowledgeRetriever(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=config.NEO4J_PASSWORD,
        database="neo4j",
    )

    # context = retriever.get_syndrome_diagnosis_context(
    #     disease="",
    #     syndrome="",
    #     symptoms=["舌有裂痕"],
    #     herbs=[],
    # )
    # kg_text = format_kg_context_for_llm(context)
    # print(kg_text)

    # 查证候治则治法
    syndrome_treatments = retriever.get_syndrome_treatments(
        syndrome="湿毒蕴肤证",
    )
    print(f"治则治法：{syndrome_treatments}")

    syndrome_relations = retriever.get_syndrome_relations(
        syndrome="湿毒蕴肤证",
    )
    print(f"证候相关知识,常用药材、相似病历、相关疾病：{syndrome_relations}")

    context_relations = retriever.get_disease_syndrome_relations(
        disease="玫瑰痤疮",
        syndrome="湿毒蕴肤证", # 名字写错，也找不到。。。
    )
    print(f"诊断-证候关系：{format_disease_properties(context_relations[0].get("source", {}).get("properties"))}")

    # 诊断关系，包含疾病可能的并发症、疾病鉴别诊断关系、疾病关联生活建议、疾病关联的病例、疾病分期、疾病表现的证候
    disease_relations = retriever.get_disease_relations(
        disease="玫瑰痤疮",
    )
    print(f"诊断关系：{disease_relations}")

    retriever.close()
