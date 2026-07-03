# -*- coding: utf-8 -*-
"""中医知识图谱增删改查管理脚本（吴卫平 Neo4j）。

设计目标：
- 与 `app/tcm_agent/custom_knowledge.py` 中 `_GraphDB.VALID_LABELS / VALID_RELS` 保持一致，
  写入/查询都走白名单校验，避免误建节点、误连关系。
- 双形态：可作为 `python -m app.tcm_agent.rag.knowledge_graph_manager ...` CLI 使用，
  也可以 `from app.tcm_agent.rag.knowledge_graph_manager import KnowledgeGraphManager`
  在 Python 里直接调用。
- 通过【证候名 / 诊断结果】等自然输入，自动匹配对应节点类型并返回结构化结果。

核心使用示例：
    from app.tcm_agent.rag.knowledge_graph_manager import KnowledgeGraphManager

    kg = KnowledgeGraphManager()
    # 查询
    kg.get_syndrome("湿热蕴肤证")
    kg.get_treatments_for_disease("痤疮")
    kg.check_incompatibilities(["甘草", "海藻"])
    kg.search_similar_cases(disease="痤疮", syndrome="湿热蕴肤证", limit=5)
    # 写入
    kg.upsert_herb("生地黄", nature="寒", flavors=["甘"], category="清热药")
    kg.link_syndrome_herb("湿热蕴肤证", "黄芩", count=12)
    kg.link_incompatibility("甘草", "海藻", rule="十八反", description="甘草反海藻")
    kg.close()

CLI 示例：
    python -m app.tcm_agent.rag.knowledge_graph_manager syndrome "湿热蕴肤证"
    python -m app.tcm_agent.rag.knowledge_graph_manager disease "痤疮"
    python -m app.tcm_agent.rag.knowledge_graph_manager incompat "甘草" "海藻"
    python -m app.tcm_agent.rag.knowledge_graph_manager herb-upsert 生地黄 --nature 寒 \\
        --flavor 甘 --category 清热药
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 允许直接以 `python app/tcm_agent/rag/knowledge_graph_manager.py` 跑 CLI
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from neo4j import GraphDatabase  # noqa: E402

from app.tcm_agent.config import config  # noqa: E402
from app.tcm_agent.utils.log import logger  # noqa: E402


# ── 与 custom_knowledge._GraphDB 对齐的标签 / 关系白名单 ────────────────────────

VALID_LABELS = {
    "Doctor", "Disease", "Syndrome", "Herb", "Symptom", "Case", "LifestyleAdvice",
    "Function", "Meridian", "Indication", "Contraindication",
    "Preparation", "AdverseEffect", "Complication",
    "HerbNature", "HerbCategory", "HerbIndication",
}

VALID_RELS = {
    "DIAGNOSED", "TREATS", "FREQUENTLY_USES", "HAS_SYNDROME", "HAS_DISEASE",
    "PRESENTS_SYMPTOM", "PRESCRIBED", "COMMONLY_USES", "MANIFESTS_AS",
    "INCOMPATIBLE_WITH",   # 十八反
    "ANTAGONIZES",        # 十九畏
    "HAS_ADVICE", "COMPLICATES_WITH",
    "HAS_FUNCTION", "ENTERS_MERIDIAN", "INDICATED_FOR", "CONTRAINDICATED_FOR",
    "HAS_PREPARATION", "HAS_ADVERSE_EFFECT", "PAIRS_WITH",
    "PROPERTY_OF", "PART_OF", "SUITABLE_FOR",
}

WRITE_FORBIDDEN = ["DETACH DELETE"]  # 危险操作需要单独走 drop_node/drop_rel

# 哪些节点标签在自定义知识图谱里有 identity 字段 `name`（与 custom_knowledge 一致）
NAME_INDEXED_LABELS = {
    "Doctor", "Disease", "Syndrome", "Herb", "Symptom", "Case",
    "Function", "Meridian", "Indication", "Contraindication",
    "Preparation", "AdverseEffect", "Complication",
    "HerbNature", "HerbCategory", "HerbIndication",
    "LifestyleAdvice",
}


# ── 返回结构 ─────────────────────────────────────────────────────────────────

@dataclass
class KGResult:
    """统一的查询结果外壳，便于 CLI 渲染与 Python 消费。"""

    ok: bool = True
    data: Any = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "message": self.message}

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


# ── 通用 Neo4j 连接 ───────────────────────────────────────────────────────────

class KnowledgeGraphManager:
    """知识图谱 CRUD 管理器。

    连接从 `config.NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD` 读取（与现有
    custom_knowledge.py 保持一致），不存在则抛 RuntimeError。
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None):
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        if not self.password:
            raise RuntimeError("NEO4J_PASSWORD 未配置，无法连接图谱")
        self.database = database or getattr(config, "NEO4J_DATABASE", None) or "neo4j"
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    # ── 底层执行（带白名单校验） ──────────────────────────────────────────────

    def _exec(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """任意写读都走这里：标签/关系必须命中白名单，写关键字不被拦截（CRUD
        必须允许 CREATE/MERGE/SET/DELETE，但禁用 DETACH DELETE 默认）。"""
        used_labels = [l for l in _extract_labels(cypher) if l not in VALID_LABELS]
        if used_labels:
            raise ValueError(f"节点标签非法：{used_labels}")
        used_rels = [r for r in _extract_rels(cypher) if r not in VALID_RELS]
        if used_rels:
            raise ValueError(f"关系类型非法：{used_rels}")

        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def _exec_write(self, cypher: str, **params) -> Dict[str, Any]:
        """写操作的轻量包装：返回统计信息；无 stats 返回兜底 dict。"""
        with self._driver.session(database=self.database) as session:
            res = session.run(cypher, **params)
            summary = res.consume()
            counters = summary.counters
            return {
                "nodes_created": counters.nodes_created,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_created": counters.relationships_created,
                "relationships_deleted": counters.relationships_deleted,
                "properties_set": counters.properties_set,
            }

    # ── 查询：按节点类型 ───────────────────────────────────────────────────────

    def get_node(self, label: str, name: str, doctor_id: Optional[str] = None) -> KGResult:
        """按 label+name（或 doctor_id）取一个节点的所有属性。"""
        if label not in VALID_LABELS:
            return KGResult(ok=False, message=f"非法 label: {label}")

        clauses = ["n.name = $name"]
        params: Dict[str, Any] = {"name": name}
        if doctor_id and label in {"Disease", "Syndrome", "Case"}:
            clauses.append("n.doctor_id = $doc")
            params["doc"] = doctor_id
        cypher = f"MATCH (n:{label}) WHERE {' AND '.join(clauses)} RETURN n LIMIT 1"
        rows = self._exec(cypher, **params)
        if not rows:
            return KGResult(ok=False, message=f"未找到 {label}: {name}")
        return KGResult(ok=True, data=_unwrap_node(rows[0]["n"]))

    # ── 查询：证候 (Syndrome) ──────────────────────────────────────────────────

    def get_syndrome(self, syndrome: str, doctor_id: Optional[str] = None) -> KGResult:
        """按证候名取一手信息：常用药表、症见分布、对应疾病、生活调理、相似病例。

        复用 `custom_knowledge._triplets_for_syndrome` 的语义，但返回结构化 dict
        方便脚本化消费。
        """
        if doctor_id:
            syndrome_rows = self._exec(
                "MATCH (s:Syndrome {name:$s, doctor_id:$doc}) RETURN s LIMIT 1",
                s=syndrome, doc=doctor_id,
            )
        else:
            syndrome_rows = self._exec(
                "MATCH (s:Syndrome {name:$s}) RETURN s LIMIT 1",
                s=syndrome,
            )
        if not syndrome_rows:
            return KGResult(ok=False, message=f"证候不存在: {syndrome}")

        # 常用药 top12
        herb_rows = self._exec(
            "MATCH (s:Syndrome {name:$s})-[r:COMMONLY_USES]->(h:Herb) "
            "RETURN h.name AS herb, r.count AS count, r.avg_dose AS dose "
            "ORDER BY r.count DESC LIMIT 12",
            s=syndrome,
        )

        # 症见分布
        symptom_rows = self._exec(
            "MATCH (c:Case)-[:HAS_SYNDROME]->(s:Syndrome {name:$s}) "
            "MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom) "
            "RETURN sym.name AS symptom, count(c) AS cases "
            "ORDER BY cases DESC LIMIT 20",
            s=syndrome,
        )

        # 哪些疾病表现为该证型
        disease_rows = self._exec(
            "MATCH (d:Disease)-[r:MANIFESTS_AS]->(s:Syndrome {name:$s}) "
            "RETURN d.name AS disease, r.count AS cases "
            "ORDER BY r.count DESC LIMIT 10",
            s=syndrome,
        )

        # 相似病历（用该证型的病例，按时间倒序取若干）
        case_rows = self._exec(
            "MATCH (c:Case)-[:HAS_SYNDROME]->(s:Syndrome {name:$s}) "
            "RETURN c.id AS id, c.disease AS disease, c.gender AS gender, "
            "       c.age AS age, c.created_at AS created_at "
            "ORDER BY c.created_at DESC LIMIT 10",
            s=syndrome,
        )

        data = {
            "syndrome": syndrome,
            "common_herbs": [
                {"herb": r["herb"], "count": r.get("count"), "avg_dose": r.get("dose")}
                for r in herb_rows
            ],
            "symptoms": [
                {"symptom": r["symptom"], "cases": r["cases"]} for r in symptom_rows
            ],
            "diseases": [
                {"disease": r["disease"], "cases": r["cases"]} for r in disease_rows
            ],
            "similar_cases": [
                {k: r.get(k) for k in ("id", "disease", "gender", "age", "created_at")}
                for r in case_rows
            ],
        }
        return KGResult(ok=True, data=data)

    # ── 查询：诊断 / 疾病 (Disease) ───────────────────────────────────────────

    def get_treatments_for_disease(self, disease: str, doctor_id: Optional[str] = None) -> KGResult:
        """按疾病取『治则治法』路径：常见证型 + 各证型常用药 + 并发症 + 调理建议。"""
        if doctor_id:
            d_rows = self._exec(
                "MATCH (d:Disease {name:$d, doctor_id:$doc}) RETURN d LIMIT 1",
                d=disease, doc=doctor_id,
            )
        else:
            d_rows = self._exec(
                "MATCH (d:Disease {name:$d}) RETURN d LIMIT 1", d=disease,
            )
        if not d_rows:
            return KGResult(ok=False, message=f"疾病不存在: {disease}")

        syn_rows = self._exec(
            "MATCH (d:Disease {name:$d})-[r:MANIFESTS_AS]->(s:Syndrome) "
            "RETURN s.name AS syndrome, r.count AS cases "
            "ORDER BY r.count DESC LIMIT 5",
            d=disease,
        )

        # 每个证型展开常用药 top10
        syndromes = []
        for s in syn_rows:
            herbs = self._exec(
                "MATCH (s:Syndrome {name:$s})-[r:COMMONLY_USES]->(h:Herb) "
                "RETURN h.name AS herb, r.count AS count, r.avg_dose AS dose "
                "ORDER BY r.count DESC LIMIT 10",
                s=s["syndrome"],
            )
            syndromes.append({
                "syndrome": s["syndrome"],
                "cases": s["cases"],
                "common_herbs": [
                    {"herb": h["herb"], "count": h.get("count"), "avg_dose": h.get("dose")}
                    for h in herbs
                ],
            })

        advice_rows = self._exec(
            "MATCH (d:Disease {name:$d})-[:HAS_ADVICE]->(a:LifestyleAdvice) "
            "RETURN a.category AS category, a.content AS content",
            d=disease,
        )
        complication_rows = self._exec(
            "MATCH (d:Disease {name:$d})-[:COMPLICATES_WITH]->(c:Complication) "
            "RETURN c.name AS complication",
            d=disease,
        )
        return KGResult(ok=True, data={
            "disease": disease,
            "syndromes": syndromes,
            "lifestyle_advice": [
                {"category": r["category"], "content": r["content"]} for r in advice_rows
            ],
            "complications": [r["complication"] for r in complication_rows],
        })

    # ── 查询：相似病历 ────────────────────────────────────────────────────────

    def search_similar_cases(self, disease: str = "", syndrome: str = "",
                             limit: int = 5) -> KGResult:
        """按 disease / syndrome 组合定位相似病历。基本子图：
        (Case)-[:HAS_DISEASE]->(Disease), (Case)-[:HAS_SYNDROME]->(Syndrome),
        (Case)-[:PRESENTS_SYMPTOM]->(Symptom), (Case)-[:DIAGNOSED]->(Doctor)。
        """
        conditions, params = [], {"limit": int(limit)}
        if disease:
            conditions.append("EXISTS { (c)-[:HAS_DISEASE]->(:Disease {name:$d}) }")
            params["d"] = disease
        if syndrome:
            conditions.append("EXISTS { (c)-[:HAS_SYNDROME]->(:Syndrome {name:$s}) }")
            params["s"] = syndrome
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cypher = (
            "MATCH (c:Case) "
            f"{where} "
            "OPTIONAL MATCH (c)-[:HAS_DISEASE]->(d:Disease) "
            "OPTIONAL MATCH (c)-[:HAS_SYNDROME]->(s:Syndrome) "
            "OPTIONAL MATCH (c)-[:PRESENTS_SYMPTOM]->(sym:Symptom) "
            "RETURN c.id AS id, c.gender AS gender, c.age AS age, "
            "       c.created_at AS created_at, "
            "       collect(DISTINCT d.name) AS diseases, "
            "       collect(DISTINCT s.name) AS syndromes, "
            "       collect(DISTINCT sym.name) AS symptoms "
            "ORDER BY c.created_at DESC LIMIT $limit"
        )
        rows = self._exec(cypher, **params)
        return KGResult(ok=True, data={"cases": rows})

    # ── 查询：证候知识（百科式聚合） ───────────────────────────────────────────

    def get_syndrome_knowledge(self, syndrome: str, doctor_id: Optional[str] = None) -> KGResult:
        """科普向的证候聚合：定义 + 病因病机（若有 LifestyleAdvice/doctor notes）
        + 症见分布 + 涵盖疾病 + 医师诊断统计。
        """
        node = self.get_syndrome(syndrome, doctor_id=doctor_id)
        if not node.ok:
            return node

        # 把『核心摘要』+『症见分布』+『涵盖疾病』合并
        base = node.data
        doctor_rows = self._exec(
            "MATCH (d:Doctor)-[:DIAGNOSED]->(c:Case)-[:HAS_SYNDROME]->(s:Syndrome {name:$s}) "
            "RETURN d.name AS doctor, count(c) AS cases "
            "ORDER BY cases DESC LIMIT 5",
            s=syndrome,
        )
        base["doctors"] = [
            {"doctor": r["doctor"], "cases": r["cases"]} for r in doctor_rows
        ]
        return KGResult(ok=True, data=base)

    # ── 查询：十八反 / 十九畏配伍禁忌 ─────────────────────────────────────────

    def check_incompatibilities(self, herbs: Iterable[str]) -> KGResult:
        """对一组药两两校验十八反/十九畏。命中即返回三元组描述。"""
        names = sorted({h.strip() for h in herbs if h and h.strip()})
        if len(names) < 2:
            return KGResult(ok=True, data={"warnings": [], "herbs": names})

        rows = self._exec(
            "MATCH (a:Herb)-[r:INCOMPATIBLE_WITH|ANTAGONIZES]-(b:Herb) "
            "WHERE a.name IN $names AND b.name IN $names AND a.name < b.name "
            "RETURN a.name AS a, b.name AS b, type(r) AS rel, "
            "r.rule AS rule, r.description AS description",
            names=names,
        )
        warnings = []
        for r in rows:
            kind = "十八反" if r["rel"] == "INCOMPATIBLE_WITH" else "十九畏"
            warnings.append({
                "herb_a": r["a"],
                "herb_b": r["b"],
                "kind": kind,
                "rule": r.get("rule"),
                "description": r.get("description"),
            })
        return KGResult(ok=True, data={"herbs": names, "warnings": warnings})

    def list_incompatibilities(self, kind: str = "all", limit: int = 100) -> KGResult:
        """枚举全图配伍禁忌。kind: 'all' | 'INCOMPATIBLE_WITH' | 'ANTAGONIZES'。"""
        rel_types = {
            "all": ["INCOMPATIBLE_WITH", "ANTAGONIZES"],
            "INCOMPATIBLE_WITH": ["INCOMPATIBLE_WITH"],
            "ANTAGONIZES": ["ANTAGONIZES"],
        }.get(kind.upper(), ["INCOMPATIBLE_WITH", "ANTAGONIZES"])

        rows = self._exec(
            "MATCH (a:Herb)-[r]->(b:Herb) WHERE type(r) IN $rels "
            "RETURN a.name AS a, b.name AS b, type(r) AS rel, "
            "r.rule AS rule, r.description AS description LIMIT $limit",
            rels=rel_types, limit=int(limit),
        )
        items = []
        for r in rows:
            label = "十八反" if r["rel"] == "INCOMPATIBLE_WITH" else "十九畏"
            items.append({
                "herb_a": r["a"],
                "herb_b": r["b"],
                "kind": label,
                "rule": r.get("rule"),
                "description": r.get("description"),
            })
        return KGResult(ok=True, data={"items": items, "count": len(items)})

    # ── 写入：节点 ────────────────────────────────────────────────────────────

    def upsert_node(self, label: str, name: str,
                    doctor_id: Optional[str] = None,
                    extra: Optional[Dict[str, Any]] = None) -> KGResult:
        """通用 upsert 节点：name 唯一键（doctor_id 可选）。"""
        if label not in VALID_LABELS or label not in NAME_INDEXED_LABELS:
            return KGResult(ok=False, message=f"该 label 不支持通过 name upsert: {label}")

        props = dict(extra or {})
        props["name"] = name
        if doctor_id and label in {"Disease", "Syndrome", "Case"}:
            props["doctor_id"] = doctor_id
        props.setdefault("updated_at", _now_date())

        set_clauses = ", ".join(f"n.{k} = ${k}" for k in props if k != "name")
        on_create = ", ".join(f"n.{k} = ${k}" for k in props)
        cypher = (
            f"MERGE (n:{label} {{name:$name}}) "
            f"ON CREATE SET {on_create} "
            + (f"ON MATCH SET {set_clauses}" if set_clauses else "")
        )
        stats = self._exec_write(cypher, **props)
        return KGResult(ok=True, data={"label": label, "name": name, "stats": stats})

    def upsert_herb(self, name: str, *, pinyin: Optional[str] = None,
                    nature: Optional[str] = None,
                    flavors: Optional[List[str]] = None,
                    category: Optional[str] = None,
                    toxicity: Optional[str] = None,
                    latin_name: Optional[str] = None,
                    source: Optional[str] = None,
                    reference: Optional[str] = None,
                    textbook_dose: Optional[str] = None,
                    extra: Optional[Dict[str, Any]] = None) -> KGResult:
        """草药节点的便捷 upsert：性味 / 归经 / 药类 / 毒性等。"""
        props: Dict[str, Any] = dict(extra or {})
        if pinyin is not None:
            props["pinyin"] = pinyin
        if nature is not None:
            props["nature"] = nature
        if flavors is not None:
            props["flavors"] = flavors
        if category is not None:
            props["category"] = category
        if toxicity is not None:
            props["toxicity"] = toxicity
        if latin_name is not None:
            props["latin_name"] = latin_name
        if source is not None:
            props["source"] = source
        if reference is not None:
            props["reference"] = reference
        if textbook_dose is not None:
            props["textbook_dose"] = textbook_dose
        return self.upsert_node("Herb", name, extra=props)

    def upsert_syndrome(self, name: str, *, doctor_id: Optional[str] = None,
                        description: Optional[str] = None,
                        extra: Optional[Dict[str, Any]] = None) -> KGResult:
        props: Dict[str, Any] = dict(extra or {})
        if description is not None:
            props["description"] = description
        return self.upsert_node("Syndrome", name, doctor_id=doctor_id, extra=props)

    def upsert_disease(self, name: str, *, doctor_id: Optional[str] = None,
                       description: Optional[str] = None,
                       extra: Optional[Dict[str, Any]] = None) -> KGResult:
        props: Dict[str, Any] = dict(extra or {})
        if description is not None:
            props["description"] = description
        return self.upsert_node("Disease", name, doctor_id=doctor_id, extra=props)

    # ── 写入：关系 ────────────────────────────────────────────────────────────

    def link(self, rel_type: str, from_label: str, from_name: str,
             to_label: str, to_name: str,
             rel_props: Optional[Dict[str, Any]] = None,
             doctor_id: Optional[str] = None) -> KGResult:
        """通用关系 upsert。from_name/to_name 为节点 name（或 (name, doctor_id) 复合）。"""
        if rel_type not in VALID_RELS:
            return KGResult(ok=False, message=f"非法关系类型: {rel_type}")
        if from_label not in VALID_LABELS or to_label not in VALID_LABELS:
            return KGResult(ok=False, message="非法节点标签")

        # 部分节点需要 doctor_id 才能唯一定位
        from_match = "MATCH (a:{L} {{name:$a{cond}}})".format(
            L=from_label, cond=", doctor_id:$doc" if doctor_id and from_label in {"Disease", "Syndrome", "Case"} else "",
        )
        to_match = "MATCH (b:{L} {{name:$b{cond}}})".format(
            L=to_label, cond=", doctor_id:$doc" if doctor_id and to_label in {"Disease", "Syndrome", "Case"} else "",
        )

        rel_props = dict(rel_props or {})
        set_clause = ""
        if rel_props:
            set_clause = "SET " + ", ".join(f"r.{k} = ${k}" for k in rel_props)
            params = {"a": from_name, "b": to_name, **rel_props}
        else:
            params = {"a": from_name, "b": to_name}
        if doctor_id and (from_label in {"Disease", "Syndrome", "Case"} or to_label in {"Disease", "Syndrome", "Case"}):
            params["doc"] = doctor_id

        cypher = (
            f"{from_match} {to_match} "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            + (set_clause if set_clause else "")
        )
        stats = self._exec_write(cypher, **params)
        return KGResult(ok=True, data={
            "from": f"{from_label}:{from_name}",
            "to": f"{to_label}:{to_name}",
            "rel": rel_type,
            "props": rel_props,
            "stats": stats,
        })

    def link_syndrome_herb(self, syndrome: str, herb: str, *,
                           count: Optional[int] = None,
                           avg_dose: Optional[float] = None,
                           doctor_id: Optional[str] = None) -> KGResult:
        """证型 ↔ 草药的常用药关系。"""
        props: Dict[str, Any] = {}
        if count is not None:
            props["count"] = int(count)
        if avg_dose is not None:
            props["avg_dose"] = float(avg_dose)
        return self.link(
            "COMMONLY_USES", "Syndrome", syndrome, "Herb", herb,
            rel_props=props, doctor_id=doctor_id,
        )

    def link_disease_syndrome(self, disease: str, syndrome: str, *,
                              count: Optional[int] = None,
                              doctor_id: Optional[str] = None) -> KGResult:
        props: Dict[str, Any] = {}
        if count is not None:
            props["count"] = int(count)
        return self.link(
            "MANIFESTS_AS", "Disease", disease, "Syndrome", syndrome,
            rel_props=props, doctor_id=doctor_id,
        )

    def link_incompatibility(self, herb_a: str, herb_b: str, *,
                             kind: str = "INCOMPATIBLE_WITH",
                             rule: Optional[str] = None,
                             description: Optional[str] = None) -> KGResult:
        """写入十八反 / 十九畏。kind ∈ {INCOMPATIBLE_WITH, ANTAGONIZES}。"""
        if kind not in {"INCOMPATIBLE_WITH", "ANTAGONIZES"}:
            return KGResult(ok=False, message="kind 必须为 INCOMPATIBLE_WITH 或 ANTAGONIZES")
        props: Dict[str, Any] = {}
        if rule:
            props["rule"] = rule
        if description:
            props["description"] = description
        # 双向链接保证 (a,b) 和 (b,a) 查询都能命中，与现有 fix_herb_incompatibility.cypher 一致
        for direction in [(herb_a, herb_b), (herb_b, herb_a)]:
            self._exec_write(
                f"MATCH (a:Herb {{name:$a}}), (b:Herb {{name:$b}}) "
                f"MERGE (a)-[r:{kind}]->(b) "
                + ("SET " + ", ".join(f"r.{k}=${k}" for k in props) if props else ""),
                a=direction[0], b=direction[1], **props,
            )
        return KGResult(ok=True, data={
            "herb_a": herb_a, "herb_b": herb_b, "kind": kind,
            "rule": rule, "description": description,
        })

    def link_symptom(self, syndrome: str, symptom: str,
                     doctor_id: Optional[str] = None) -> KGResult:
        """为证型登记症见节点，并把 (Case)-[:PRESENTS_SYMPTOM]->(Symptom) 接住。

        这里创建 Symptom 节点（如不存在），然后连 Syndrome-HAS_SYNDROME-Case-
        PRESENTS_SYMPTOM-Symptom 比较重；本函数仅做『自动建 Symptom + 计数关联』，
        若需要病历级统计，请用 bulk_register_symptom_from_cases。
        """
        self.upsert_node("Symptom", symptom)
        cypher = (
            "MATCH (s:Syndrome {name:$s}), (sym:Symptom {name:$sym}) "
            "MERGE (s)-[r:ASSOCIATED_WITH]->(sym) "
            "ON CREATE SET r.created_at = date()"
        )
        stats = self._exec_write(cypher, s=syndrome, sym=symptom)
        return KGResult(ok=True, data={"stats": stats})

    def link_lifestyle_advice(self, disease: str, category: str, content: str) -> KGResult:
        """疾病下的生活方式建议：MATCH LifestyleAdvice(category+content) 再 MERGE。"""
        cypher = (
            "MATCH (d:Disease {name:$d}) "
            "MERGE (a:LifestyleAdvice {category:$cat, content:$content}) "
            "MERGE (d)-[:HAS_ADVICE]->(a)"
        )
        stats = self._exec_write(cypher, d=disease, cat=category, content=content)
        return KGResult(ok=True, data={
            "disease": disease, "category": category, "content": content, "stats": stats,
        })

    # ── 删除：节点 ────────────────────────────────────────────────────────────

    def drop_node(self, label: str, name: str, *,
                  detach: bool = False,
                  doctor_id: Optional[str] = None) -> KGResult:
        """按 label+name 删节点。默认只删无关系的孤立节点；detach=True 时
        连同它发起的关系一起删。"""
        if label not in VALID_LABELS:
            return KGResult(ok=False, message=f"非法 label: {label}")

        cond, params = ["n.name = $name"], {"name": name}
        if doctor_id and label in {"Disease", "Syndrome", "Case"}:
            cond.append("n.doctor_id = $doc")
            params["doc"] = doctor_id
        prefix = "DETACH DELETE n" if detach else "DELETE n"
        cypher = f"MATCH (n:{label}) WHERE {' AND '.join(cond)} {prefix}"
        try:
            stats = self._exec_write(cypher, **params)
        except Exception as e:
            return KGResult(ok=False, message=f"删除失败（节点仍有关系？可加 detach=True）: {e}")
        return KGResult(ok=True, data={"label": label, "name": name, "stats": stats})

    # ── 删除：关系 ────────────────────────────────────────────────────────────

    def drop_relation(self, rel_type: str, from_label: str, from_name: str,
                      to_label: str, to_name: str,
                      doctor_id: Optional[str] = None) -> KGResult:
        """删除两节点间的指定关系（不影响节点本身）。"""
        if rel_type not in VALID_RELS:
            return KGResult(ok=False, message=f"非法关系类型: {rel_type}")
        from_cond = "{name:$a"
        to_cond = "{name:$b"
        if doctor_id and from_label in {"Disease", "Syndrome", "Case"}:
            from_cond += ", doctor_id:$doc"
        if doctor_id and to_label in {"Disease", "Syndrome", "Case"}:
            to_cond += ", doctor_id:$doc"
        from_cond += "}"
        to_cond += "}"

        params: Dict[str, Any] = {"a": from_name, "b": to_name}
        if doctor_id and (from_label in {"Disease", "Syndrome", "Case"} or to_label in {"Disease", "Syndrome", "Case"}):
            params["doc"] = doctor_id

        cypher = (
            f"MATCH (a:{from_label} {from_cond})-[r:{rel_type}]->(b:{to_label} {to_cond}) "
            "DELETE r"
        )
        try:
            stats = self._exec_write(cypher, **params)
        except Exception as e:
            return KGResult(ok=False, message=f"删除失败: {e}")
        return KGResult(ok=True, data={
            "from": f"{from_label}:{from_name}",
            "to": f"{to_label}:{to_name}",
            "rel": rel_type, "stats": stats,
        })

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def stats(self) -> KGResult:
        """全局节点 / 关系数量统计。"""
        cypher = (
            "MATCH (n) UNWIND labels(n) AS lbl WITH lbl, count(*) AS c "
            "ORDER BY c DESC RETURN lbl AS label, c AS count"
        )
        labels = self._exec(cypher)
        cypher2 = (
            "MATCH ()-[r]->() UNWIND type(r) AS rt WITH rt, count(*) AS c "
            "ORDER BY c DESC RETURN rt AS rel, c AS count"
        )
        rels = self._exec(cypher2)
        return KGResult(ok=True, data={"labels": labels, "relationships": rels})

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _extract_labels(cypher: str) -> List[str]:
    """提取 Cypher 里出现的所有节点标签（最简实现，足够白名单校验）。"""
    import re
    return re.findall(r"\((?:\w+\s*:\s*|\s*:)(\w+)", cypher)


def _extract_rels(cypher: str) -> List[str]:
    import re
    return re.findall(r"\[(?:\w+\s*:\s*|\s*:)(\w+)\s*\]", cypher)


def _unwrap_node(node) -> Dict[str, Any]:
    """neo4j Node → dict。处理 list/dict 类型字段。"""
    if node is None:
        return {}
    if isinstance(node, dict):
        return node
    props = dict(node)
    out: Dict[str, Any] = {}
    for k, v in props.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
            try:
                out[k] = [_coerce(x) for x in v]
            except TypeError:
                out[k] = v
        else:
            out[k] = v
    return out


def _coerce(v: Any) -> Any:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _now_date() -> str:
    from datetime import date
    return date.today().isoformat()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_result(res: KGResult) -> None:
    if not res.ok:
        print(f"[FAIL] {res.message}", file=sys.stderr)
        sys.exit(1)
    print(res.to_json())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="knowledge_graph_manager",
        description="中医知识图谱 Neo4j 增删改查 CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 查询
    sub.add_parser("stats", help="节点 / 关系统计")

    sp = sub.add_parser("syndrome", help="按证候聚合查询")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)

    sp = sub.add_parser("disease", help="按疾病聚合查询（治则治法入口）")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)

    sp = sub.add_parser("herb", help="按草药名取节点")
    sp.add_argument("name")

    sp = sub.add_parser("node", help="按 label+name 取一个节点")
    sp.add_argument("label")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)

    sp = sub.add_parser("cases", help="按 disease/syndrome 取相似病历")
    sp.add_argument("--disease", default="")
    sp.add_argument("--syndrome", default="")
    sp.add_argument("--limit", type=int, default=5)

    sp = sub.add_parser("incompat", help="查一组药的十八反/十九畏")
    sp.add_argument("herbs", nargs="+")

    sp = sub.add_parser("incompat-list", help="枚举全图配伍禁忌")
    sp.add_argument("--kind", default="all", choices=["all", "INCOMPATIBLE_WITH", "ANTAGONIZES"])
    sp.add_argument("--limit", type=int, default=50)

    # 写入 / 更新
    sp = sub.add_parser("herb-upsert", help="upsert 草药节点")
    sp.add_argument("name")
    sp.add_argument("--pinyin", default=None)
    sp.add_argument("--nature", default=None)
    sp.add_argument("--flavor", action="append", help="可重复，加口味 list")
    sp.add_argument("--category", default=None)
    sp.add_argument("--toxicity", default=None)
    sp.add_argument("--latin-name", default=None)
    sp.add_argument("--source", default=None)
    sp.add_argument("--reference", default=None)
    sp.add_argument("--textbook-dose", default=None)

    sp = sub.add_parser("syndrome-upsert", help="upsert 证候节点")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)
    sp.add_argument("--description", default=None)

    sp = sub.add_parser("disease-upsert", help="upsert 疾病节点")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)
    sp.add_argument("--description", default=None)

    sp = sub.add_parser("link", help="通用关系 upsert")
    sp.add_argument("--rel", required=True)
    sp.add_argument("--from-label", required=True)
    sp.add_argument("--from-name", required=True)
    sp.add_argument("--to-label", required=True)
    sp.add_argument("--to-name", required=True)
    sp.add_argument("--doctor-id", default=None)
    sp.add_argument("--count", type=int, default=None)
    sp.add_argument("--dose", type=float, default=None)
    sp.add_argument("--rule", default=None)
    sp.add_argument("--description", default=None)

    sp = sub.add_parser("link-symptom", help="把症见挂在证候下")
    sp.add_argument("--syndrome", required=True)
    sp.add_argument("--symptom", required=True)

    sp = sub.add_parser("link-advice", help="把生活方式建议挂在疾病下")
    sp.add_argument("--disease", required=True)
    sp.add_argument("--category", required=True)
    sp.add_argument("--content", required=True)

    sp = sub.add_parser("link-incompat", help="写入一对十八反/十九畏")
    sp.add_argument("herb_a")
    sp.add_argument("herb_b")
    sp.add_argument("--kind", default="INCOMPATIBLE_WITH",
                    choices=["INCOMPATIBLE_WITH", "ANTAGONIZES"])
    sp.add_argument("--rule", default=None)
    sp.add_argument("--description", default=None)

    # 删除
    sp = sub.add_parser("drop-node", help="删一个节点")
    sp.add_argument("label")
    sp.add_argument("name")
    sp.add_argument("--doctor-id", default=None)
    sp.add_argument("--detach", action="store_true", help="连同关系一起删除")

    sp = sub.add_parser("drop-rel", help="删一对关系")
    sp.add_argument("--rel", required=True)
    sp.add_argument("--from-label", required=True)
    sp.add_argument("--from-name", required=True)
    sp.add_argument("--to-label", required=True)
    sp.add_argument("--to-name", required=True)
    sp.add_argument("--doctor-id", default=None)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    kg = KnowledgeGraphManager()
    try:
        if args.cmd == "stats":
            return _print_result(kg.stats())

        if args.cmd == "syndrome":
            return _print_result(kg.get_syndrome(args.name, doctor_id=args.doctor_id))

        if args.cmd == "disease":
            return _print_result(kg.get_treatments_for_disease(args.name, doctor_id=args.doctor_id))

        if args.cmd == "herb":
            return _print_result(kg.get_node("Herb", args.name))

        if args.cmd == "node":
            return _print_result(kg.get_node(args.label, args.name, doctor_id=args.doctor_id))

        if args.cmd == "cases":
            return _print_result(kg.search_similar_cases(args.disease, args.syndrome, args.limit))

        if args.cmd == "incompat":
            return _print_result(kg.check_incompatibilities(args.herbs))

        if args.cmd == "incompat-list":
            return _print_result(kg.list_incompatibilities(args.kind, args.limit))

        if args.cmd == "herb-upsert":
            return _print_result(kg.upsert_herb(
                args.name, pinyin=args.pinyin, nature=args.nature,
                flavors=args.flavor, category=args.category,
                toxicity=args.toxicity, latin_name=args.latin_name,
                source=args.source, reference=args.reference,
                textbook_dose=args.textbook_dose,
            ))

        if args.cmd == "syndrome-upsert":
            return _print_result(kg.upsert_syndrome(
                args.name, doctor_id=args.doctor_id, description=args.description,
            ))

        if args.cmd == "disease-upsert":
            return _print_result(kg.upsert_disease(
                args.name, doctor_id=args.doctor_id, description=args.description,
            ))

        if args.cmd == "link":
            rel_props: Dict[str, Any] = {}
            if args.count is not None:
                rel_props["count"] = args.count
            if args.dose is not None:
                rel_props["avg_dose"] = args.dose
            if args.rule:
                rel_props["rule"] = args.rule
            if args.description:
                rel_props["description"] = args.description
            return _print_result(kg.link(
                args.rel, args.from_label, args.from_name,
                args.to_label, args.to_name,
                rel_props=rel_props or None, doctor_id=args.doctor_id,
            ))

        if args.cmd == "link-symptom":
            return _print_result(kg.link_symptom(args.syndrome, args.symptom))

        if args.cmd == "link-advice":
            return _print_result(kg.link_lifestyle_advice(args.disease, args.category, args.content))

        if args.cmd == "link-incompat":
            return _print_result(kg.link_incompatibility(
                args.herb_a, args.herb_b,
                kind=args.kind, rule=args.rule, description=args.description,
            ))

        if args.cmd == "drop-node":
            return _print_result(kg.drop_node(args.label, args.name, detach=args.detach, doctor_id=args.doctor_id))

        if args.cmd == "drop-rel":
            return _print_result(kg.drop_relation(
                args.rel, args.from_label, args.from_name,
                args.to_label, args.to_name, doctor_id=args.doctor_id,
            ))

        parser.print_help()
        return 1
    finally:
        kg.close()


if __name__ == "__main__":
    sys.exit(main())
