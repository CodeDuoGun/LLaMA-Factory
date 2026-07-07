from typing import Any, Optional


class TCMKnowledgeRetriever:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        doctor_id: Optional[str] = None,
    ):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.doctor_id = doctor_id

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: Optional[dict[str, Any]] = None):
        with self.driver.session(database=self.database) as session:
            rows = [r.data() for r in session.run(cypher, params or {})]
        return json_safe(rows)

    def _resolve_doctor_id(self, doctor_id: Optional[str] = None) -> Optional[str]:
        return doctor_id or self.doctor_id

    def get_syndrome_diagnosis_context(
        self,
        disease: str,
        syndrome: str,
        symptoms: Optional[list[str]] = None,
        herbs: Optional[list[str]] = None,
        limit: int = 2,
        doctor_id: Optional[str] = None,
    ) -> dict[str, Any]:
        symptoms = symptoms or []
        herbs = herbs or []
        doctor_id = self._resolve_doctor_id(doctor_id)

        return {
            "诊断": disease,
            "证候": syndrome,
            "doctor_id": doctor_id,
            "病机信息": self.get_pathogenesis(disease, syndrome, doctor_id=doctor_id),
            "疾病证候相关关系": self.get_disease_syndrome_relations(disease, syndrome, doctor_id=doctor_id),
            "证候相关知识": self.get_syndrome_relations(syndrome, doctor_id=doctor_id),
            "疾病相关知识": self.get_disease_relations(disease, doctor_id=doctor_id),
            "临床症状": self.get_clinical_symptoms(disease, syndrome, symptoms, doctor_id=doctor_id),
            "相关药物知识": self.get_herb_knowledge(herbs),
            "相似病例": self.get_similar_cases(disease, syndrome, symptoms, limit, doctor_id=doctor_id),
            # "证候相似病例": self.get_syndrome_similar_cases(syndrome, symptoms, limit),
        }

    def get_disease_syndrome_relations(
        self,
        disease: str,
        syndrome: str,
        limit: int = 50,
        doctor_id: Optional[str] = None,
    ):
        if not disease or not syndrome:
            return []
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (d:Disease {name:$disease})
        WHERE $doctor_id IS NULL OR d.doctor_id = $doctor_id
        MATCH (s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR s.doctor_id = $doctor_id
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
            self.query(cypher, {"disease": disease, "syndrome": syndrome, "limit": limit, "doctor_id": doctor_id})
        )

    def get_disease_properties(self, disease: str, doctor_id: Optional[str] = None):
        if not disease:
            return {}
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (d:Disease {name:$disease})
        WHERE $doctor_id IS NULL OR d.doctor_id = $doctor_id
        RETURN properties(d) AS properties
        LIMIT 1
        """
        rows = self.query(cypher, {"disease": disease, "doctor_id": doctor_id})
        if not rows:
            return {}
        return rows[0].get("properties") or {}

    def get_syndrome_properties(self, syndrome: str, doctor_id: Optional[str] = None):
        if not syndrome:
            return {}
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR s.doctor_id = $doctor_id
        RETURN properties(s) AS properties
        LIMIT 1
        """
        rows = self.query(cypher, {"syndrome": syndrome, "doctor_id": doctor_id})
        if not rows:
            return {}
        return rows[0].get("properties") or {}

    def get_disease_syndrome_properties(
        self,
        disease: str,
        syndrome: str,
        doctor_id: Optional[str] = None,
    ):
        if not disease and not syndrome:
            return {"诊断属性": {}, "证候属性": {}}
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        OPTIONAL MATCH (d:Disease {name:$disease})
        WHERE $doctor_id IS NULL OR d.doctor_id = $doctor_id
        OPTIONAL MATCH (s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR s.doctor_id = $doctor_id
        RETURN
            properties(d) AS disease_properties,
            properties(s) AS syndrome_properties
        LIMIT 1
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome, "doctor_id": doctor_id})
        if not rows:
            return {"诊断属性": {}, "证候属性": {}}
        return {
            "诊断属性": rows[0].get("disease_properties") or {},
            "证候属性": rows[0].get("syndrome_properties") or {},
        }


    def get_syndrome_relations(
        self,
        syndrome: str,
        limit: Optional[int] = None,
        doctor_id: Optional[str] = None,
    ):
        """Query all direct relations connected to a syndrome node.

        默认返回该证候所有一跳入边和出边关系；如果担心结果过多，可传入 limit 截断返回数量。
        COMMONLY_USES 关系通常较多，会在查询后最多保留前 5 条。
        HAS_SYNDROME 关系通常对应相似病历，会在查询后最多保留前 2 条。
        MANIFESTS_AS 关系通常对应相关疾病，会在查询后最多保留前 5 条。
        返回结果会保留关系方向、关系类型、关系属性、相邻节点标签与属性。
        """
        if not syndrome:
            return []
        doctor_id = self._resolve_doctor_id(doctor_id)

        limit_clause = "\n        LIMIT $limit" if limit is not None else ""
        cypher = f"""
        MATCH (s:Syndrome {{name:$syndrome}})
        WHERE $doctor_id IS NULL OR s.doctor_id = $doctor_id
        CALL (s) {{
            MATCH (source)-[r]->(s)
            WHERE $doctor_id IS NULL OR source.doctor_id IS NULL OR source.doctor_id = $doctor_id
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
            WHERE $doctor_id IS NULL OR target.doctor_id IS NULL OR target.doctor_id = $doctor_id
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
        params = {"syndrome": syndrome, "doctor_id": doctor_id}
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

    def get_disease_relations(
        self,
        disease: str,
        limit: Optional[int] = None,
        doctor_id: Optional[str] = None,
    ):
        """Query all direct relations connected to a disease node.

        默认返回该疾病所有一跳入边和出边关系；如果担心结果过多，可传入 limit 截断返回数量。
        查询后会按关系类型限制数量：默认每种关系最多 5 条，HAS_DISEASE 和 MANIFESTS_AS 最多 2 条。
        返回结果会保留关系方向、关系类型、关系属性、相邻节点标签与属性。
        """
        if not disease:
            return []
        doctor_id = self._resolve_doctor_id(doctor_id)

        limit_clause = "\n        LIMIT $limit" if limit is not None else ""
        cypher = f"""
        MATCH (d:Disease {{name:$disease}})
        WHERE $doctor_id IS NULL OR d.doctor_id = $doctor_id
        CALL (d) {{
            MATCH (source)-[r]->(d)
            WHERE $doctor_id IS NULL OR source.doctor_id IS NULL OR source.doctor_id = $doctor_id
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
            WHERE $doctor_id IS NULL OR target.doctor_id IS NULL OR target.doctor_id = $doctor_id
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
        params = {"disease": disease, "doctor_id": doctor_id}
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
        relation_counts = dict.fromkeys(max_counts, 0)
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

    def get_treatment_principles(self, disease: str, syndrome: str, doctor_id: Optional[str] = None):
        """Query treatment principles on Disease-MANIFESTS_AS-Syndrome relations."""
        doctor_id = self._resolve_doctor_id(doctor_id)
        cypher = """
        MATCH (d:Disease {name:$disease})-[r:MANIFESTS_AS]->(s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR (d.doctor_id = $doctor_id AND s.doctor_id = $doctor_id)
        RETURN
            s.name AS syndrome,
            properties(s) AS syndrome_properties,
            properties(r) AS relation_properties
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome, "doctor_id": doctor_id})

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

    def get_pathogenesis(self, disease: str, syndrome: str, doctor_id: Optional[str] = None):
        """Query pathogenesis for a disease-syndrome pair."""
        doctor_id = self._resolve_doctor_id(doctor_id)
        cypher = """
        MATCH (d:Disease {name:$disease})-[r:MANIFESTS_AS]->(s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR (d.doctor_id = $doctor_id AND s.doctor_id = $doctor_id)
        RETURN
            s.name AS syndrome,
            properties(r) AS relation_properties,
            properties(s) AS syndrome_properties
        """
        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome, "doctor_id": doctor_id})

        result = []
        for row in rows:
            rel_props = row.get("relation_properties") or {}
            syn_props = row.get("syndrome_properties") or {}

            pathogenesis = (
                rel_props.get("pathogenesis")
                or rel_props.get("病机")
                or syn_props.get("pathogenesis")
                or syn_props.get("病机")
                or syn_props.get("description")
            )

            if pathogenesis:
                result.append({
                    "name": pathogenesis,
                    "source": "Syndrome.properties / MANIFESTS_AS.properties",
                    "properties": {
                        "relation": rel_props,
                        "syndrome": syn_props,
                    },
                })
        return result

    def get_syndrome_pathogenesis(self, syndrome: str, doctor_id: Optional[str] = None):
        """Query pathogenesis related to a syndrome."""
        if not syndrome:
            return []
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (s:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR s.doctor_id = $doctor_id
        OPTIONAL MATCH (d:Disease)-[r:MANIFESTS_AS]->(s)
        WHERE $doctor_id IS NULL OR d.doctor_id IS NULL OR d.doctor_id = $doctor_id
        RETURN
            d.name AS disease,
            s.name AS syndrome,
            properties(s) AS syndrome_properties,
            CASE WHEN r IS NULL THEN {} ELSE properties(r) END AS relation_properties
        ORDER BY disease ASC
        """
        rows = self.query(cypher, {"syndrome": syndrome, "doctor_id": doctor_id})

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
                or syn_props.get("description")
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

    def get_clinical_symptoms(
        self,
        disease: str,
        syndrome: str,
        input_symptoms: Optional[list[str]] = None,
        doctor_id: Optional[str] = None,
    ):
        input_symptoms = input_symptoms or []
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (c:Case)-[:HAS_DISEASE]->(d:Disease {name:$disease})
        MATCH (c)-[:HAS_SYNDROME]->(s:Syndrome {name:$syndrome})
        MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)
        WHERE $doctor_id IS NULL OR (
            c.doctor_id = $doctor_id
            AND d.doctor_id = $doctor_id
            AND s.doctor_id = $doctor_id
            AND (sym.doctor_id IS NULL OR sym.doctor_id = $doctor_id)
        )
        RETURN
            sym.name AS symptom,
            count(DISTINCT c) AS cases
        ORDER BY cases DESC, symptom ASC
        LIMIT 50
        """

        rows = self.query(cypher, {"disease": disease, "syndrome": syndrome, "doctor_id": doctor_id})

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
        doctor_id: Optional[str] = None,
    ):
        """Query similar cases."""
        symptoms = symptoms or []
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (c:Case)
        WHERE $doctor_id IS NULL OR c.doctor_id = $doctor_id

        OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease)
        WHERE $doctor_id IS NULL OR d.doctor_id IS NULL OR d.doctor_id = $doctor_id
        OPTIONAL MATCH (c)-[:HAS_SYNDROME]->(syn:Syndrome)
        WHERE $doctor_id IS NULL OR syn.doctor_id IS NULL OR syn.doctor_id = $doctor_id
        OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)
        WHERE $doctor_id IS NULL OR sym.doctor_id IS NULL OR sym.doctor_id = $doctor_id

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
                "doctor_id": doctor_id,
            },
        )

    def get_syndrome_similar_cases(
        self,
        syndrome: str,
        symptoms: Optional[list[str]] = None,
        limit: int = 2,
        doctor_id: Optional[str] = None,
    ):
        """Query similar cases under the same syndrome."""
        if not syndrome:
            return []

        symptoms = symptoms or []
        doctor_id = self._resolve_doctor_id(doctor_id)

        cypher = """
        MATCH (c:Case)-[:HAS_SYNDROME]->(syn:Syndrome {name:$syndrome})
        WHERE $doctor_id IS NULL OR (c.doctor_id = $doctor_id AND syn.doctor_id = $doctor_id)

        OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease)
        WHERE $doctor_id IS NULL OR d.doctor_id IS NULL OR d.doctor_id = $doctor_id
        OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom)
        WHERE $doctor_id IS NULL OR sym.doctor_id IS NULL OR sym.doctor_id = $doctor_id

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
                "doctor_id": doctor_id,
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
    "western_summary": "西医概述",
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


_DEFAULT_RELATION_GROUPS = [
    ("常用药物", ("COMMONLY_USES",)),
    ("并发症", ("COMPLICATES_WITH",)),
    ("鉴别诊断", ("DIFFERENTIATE_FROM",)),
    ("生活建议", ("HAS_ADVICE",)),
    ("病例关联", ("HAS_DISEASE", "HAS_SYNDROME")),
    ("疾病分期", ("HAS_STAGE",)),
    ("相关疾病/证候表现", ("MANIFESTS_AS",)),
]


_DISEASE_RELATION_GROUPS = [
    ("并发症", ("COMPLICATES_WITH",)),
    ("鉴别诊断", ("DIFFERENTIATE_FROM",)),
    ("生活建议", ("HAS_ADVICE",)),
    ("病例关联", ("HAS_DISEASE",)),
    ("疾病分期", ("HAS_STAGE",)),
    ("证候表现", ("MANIFESTS_AS",)),
]


def format_relations_as_text(
    relations: list[dict[str, Any]],
    title: str = "【关系知识】",
    max_relations: int = 20,
    empty_text: str = "- 暂无相关关系。",
    relation_groups: Optional[list[tuple[str, tuple[str, ...]]]] = None,
) -> str:
    """Format KG relation objects into grouped readable text."""
    lines = [title]
    if not relations:
        lines.append(empty_text)
        return "\n".join(lines)

    relation_groups = relation_groups or _DEFAULT_RELATION_GROUPS

    total_count = 0
    for group_title, relation_types in relation_groups:
        relation_texts = []
        seen = set()
        for relation in relations:
            if relation.get("relation") not in relation_types:
                continue
            relation_text = _format_relation_for_group(relation)
            if not relation_text or relation_text in seen:
                continue
            relation_texts.append(relation_text)
            seen.add(relation_text)
            if len(relation_texts) >= max_relations:
                break

        if not relation_texts:
            continue
        lines.append(f"{group_title}：")
        for relation_text in relation_texts:
            lines.append(f"- {relation_text}")
        total_count += len(relation_texts)

    if total_count == 0:
        lines.append(empty_text)
    return "\n".join(lines)


def _get_node_id_or_name(node: dict[str, Any]) -> str:
    name = node.get("name")
    properties = node.get("properties") or {}
    return name or properties.get("id") or "未知节点"


def _node_has_label(node: dict[str, Any], label: str) -> bool:
    return label in (node.get("labels") or [])


def _format_relation_for_group(item: dict[str, Any]) -> str | None:
    source = item.get("source") or {}
    target = item.get("target") or {}
    relation = item.get("relation")
    properties = item.get("properties") or {}
    source_name = _get_node_id_or_name(source)
    target_name = _get_node_id_or_name(target)

    if relation == "COMMONLY_USES":
        target_props = target.get("properties") or {}
        details = []
        category = target_props.get("category")
        nature = target_props.get("nature")
        flavors = target_props.get("flavors")
        avg_dose = properties.get("avg_dose")
        count = properties.get("count")
        if category:
            details.append(f"类别：{category}")
        if nature or flavors:
            nature_flavor = "，".join(part for part in [nature, _join(flavors) if flavors else ""] if part)
            details.append(f"性味：{nature_flavor}")
        if avg_dose is not None:
            details.append(f"平均剂量：{avg_dose:g}g")
        if count is not None:
            details.append(f"使用次数：{count}")
        return f"{target_name}（{'；'.join(details)}）" if details else target_name

    if relation == "COMPLICATES_WITH":
        return f"{source_name}可能并发或累及{target_name}。"

    if relation == "DIFFERENTIATE_FROM":
        key_points = _get_relation_property(properties, ["key_points", "鉴别要点", "points"])
        if key_points:
            return f"{source_name}与{target_name}需鉴别；鉴别要点：{key_points}"
        return f"{source_name}与{target_name}需鉴别。"

    if relation == "HAS_ADVICE":
        advice = _get_relation_property(properties, ["advice", "建议", "content", "text", "name"])
        advice = advice or _get_relation_property(target.get("properties") or {}, ["advice", "建议", "content", "text", "name"])
        category = (target.get("properties") or {}).get("category")
        if advice and category:
            return f"{category}：{advice}"
        if advice:
            return str(advice)
        return f"关联生活建议：{target_name}。"

    if relation in ("HAS_DISEASE", "HAS_SYNDROME"):
        case_node = source if _node_has_label(source, "Case") else target
        case_props = case_node.get("properties") or {}
        case_id = case_props.get("id") or _get_node_id_or_name(case_node)
        chief_complaint = case_props.get("chief_complaint")
        patient = _format_case_patient(case_props, include_case_meta=False)
        parts = [f"病例 {case_id}"]
        if patient:
            parts.append(patient)
        if chief_complaint:
            parts.append(f"主诉：{chief_complaint}")
        return "；".join(parts)

    if relation == "HAS_STAGE":
        target_props = target.get("properties") or {}
        manifestation = target_props.get("manifestation")
        treatment = target_props.get("treatment_principle")
        details = []
        if manifestation:
            details.append(f"表现：{manifestation}")
        if treatment:
            details.append(f"治则：{treatment}")
        return f"{target_name}（{'；'.join(details)}）" if details else target_name

    if relation == "MANIFESTS_AS":
        count = properties.get("count")
        if _node_has_label(source, "Disease") and _node_has_label(target, "Syndrome"):
            if count is not None:
                return f"{source_name}可表现为{target_name}（{count}例）"
            return f"{source_name}可表现为{target_name}"
        if count is not None:
            return f"{target_name}（{count}例）"
        return target_name

    return None


def _append_relation_section(
    lines: list[str], title: str, relations: list[dict[str, Any]], max_relations: int
) -> None:
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
        lines.append(f"{title}：")
        for relation_text in relation_texts:
            lines.append(f"- {relation_text}")
        lines.append("")


def format_kg_context_for_llm(
    kg: dict[str, Any],
    max_symptoms: int = 20,
    max_cases: int = 2,
    max_herbs_per_case: int = 20,
    max_relations: int = 20,
) -> str:
    lines = []
    lines.append("【知识图谱参考】")

    # 病机
    pathogenesis = kg.get("病机信息") or []
    lines.append("【病机参考】")
    if pathogenesis:
        for item in pathogenesis:
            lines.append(f"- {item.get('name')}")
    lines.append("")

    _append_relation_section(lines, "【疾病-证候关系参考】", kg.get("疾病证候相关关系") or [], max_relations)
    lines.append(
        format_relations_as_text(
            kg.get("证候相关知识") or [],
            title="【证候关系参考】",
            max_relations=max_relations,
            empty_text="- 知识图谱中暂无相关证候关系。",
        )
    )
    lines.append("")
    lines.append(
        format_relations_as_text(
            kg.get("疾病相关知识") or [],
            title="【疾病知识参考】",
            max_relations=max_relations,
            empty_text="- 知识图谱中暂无相关疾病知识。",
            relation_groups=_DISEASE_RELATION_GROUPS,
        )
    )
    lines.append("")

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
            chief = props.get("chief_complaint") or "未记录主诉"
            patient = _format_case_patient(props) or "未记录性别年龄"

            lines.append(f"相似病例{idx}：")
            lines.append(f"- 患者：{patient}")
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


def _format_case_herbs(case, max_herbs=20):
    """Format case prescription herbs."""
    herbs = []
    for item in case.get("herbs", []) or []:
        name = item.get("name")
        dose = (item.get("prescription_relation") or {}).get("dose")
        if name and dose:
            try:
                herbs.append(f"{name}{float(dose):g}g")
            except (TypeError, ValueError):
                herbs.append(f"{name}{dose}")
        elif name:
            herbs.append(name)
    return "、".join(herbs[:max_herbs]) if herbs else "无"


def _first_non_empty(props, keys):
    """Return first non-empty value from properties."""
    for key in keys:
        value = props.get(key)
        if value not in (None, "", []):
            return value
    return ""


def _format_case_patient(props, include_case_meta=True):
    """Format patient fields while keeping compatibility with old and new KG dumps."""
    gender = _first_non_empty(props, ("gender", "sex", "patient_sex", "患者性别", "性别"))
    age = _first_non_empty(props, ("age", "patient_age", "挂号年龄", "年龄"))
    patient_id = _first_non_empty(props, ("patient_id", "patientId", "患者ID"))
    start_time = _first_non_empty(props, ("start_time", "visit_time", "created_at", "就诊时间"))

    parts = []
    if gender:
        parts.append(str(gender))
    if age:
        age_text = str(age)
        parts.append(age_text if age_text.endswith("岁") else f"{age_text}岁")
    if include_case_meta and patient_id:
        parts.append(f"患者ID：{patient_id}")
    if include_case_meta and start_time:
        parts.append(f"就诊时间：{start_time}")
    return "，".join(parts)


def _format_similar_cases(title, cases, max_cases=3):
    """Format similar cases retrieved from KG."""
    lines = [title]
    if not cases:
        lines.append("- 未检索到相似病例。")
        return "\n".join(lines)

    for idx, case in enumerate(cases[:max_cases], start=1):
        props = case.get("case_properties") or {}
        patient = _format_case_patient(props) or "未记录性别年龄"
        chief = props.get("chief_complaint") or "未记录主诉"
        diseases = "、".join(case.get("diseases") or []) or "无"
        syndromes = "、".join(case.get("syndromes") or []) or "无"
        case_symptoms = "、".join(case.get("symptoms") or []) or "无"
        lines.append(f"相似病例{idx}：")
        lines.append(f"- 患者：{patient}")
        lines.append(f"- 主诉：{chief}")
        lines.append(f"- 诊断：{diseases}")
        lines.append(f"- 证候：{syndromes}")
        lines.append(f"- 症状：{case_symptoms}")
        lines.append(f"- 相似度评分：{case.get('score')}")
        lines.append(f"- 处方用药：{_format_case_herbs(case)}")
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

    syndrome_relations = retriever.get_syndrome_relations(
        syndrome="湿毒蕴肤证",
        doctor_id="wuweiping",
    )
    print("证候相关知识,常用药材、相似病历、相关疾病：")
    print(format_relations_as_text(syndrome_relations, "【证候相关知识】", 10))

    # context_relations = retriever.get_disease_syndrome_relations(
    #     disease="玫瑰痤疮",
    #     syndrome="湿毒蕴肤证", # 名字写错，也找不到。。。
    # )
    # print(f"诊断-证候关系：{format_disease_properties(context_relations[0].get('source', {}).get('properties'))}")

    # similar_cases = retriever.get_similar_cases(
    #     disease="玫瑰痤疮",
    #     syndrome="湿毒蕴肤证",
    #     symptoms=["面部红斑", "丘疹脓疱", "瘙痒"],
    #     limit=3,
    # )
    # print(_format_similar_cases("【相似病例】", similar_cases, 3))
    # 诊断关系，包含疾病可能的并发症、疾病鉴别诊断关系、疾病关联生活建议、疾病关联的病例、疾病分期、疾病表现的证候
    # disease_relations = retriever.get_disease_relations(
    #     disease="玫瑰痤疮",
    # )
    # print(f"疾病知识参考：{format_relations_as_text(disease_relations, "【疾病知识参考】", 10)}")

    retriever.close()
