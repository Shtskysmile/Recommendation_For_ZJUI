from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from app.models import (
    CapabilitySpec,
    KnowledgeNode,
    KnowledgeWriteback,
    RepoExtractionResult,
    RetrievedKnowledge,
    ValidationReport,
)


def _tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9@+\-]+", value.lower()))


class KnowledgeGraphStore:
    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        self.seed_path = knowledge_dir / "seed_catalog.json"
        self.export_dir = knowledge_dir / "exports"
        self.graph = nx.DiGraph()
        self._load_seed()
        self.export_graph()

    def _load_seed(self) -> None:
        payload = json.loads(self.seed_path.read_text(encoding="utf-8"))
        for node in payload["nodes"]:
            keywords = node.get("keywords", "")
            normalized_keywords = keywords.split() if isinstance(keywords, str) else list(keywords)
            self.graph.add_node(
                node["id"],
                type=node["type"],
                name=node["name"],
                description=node["description"],
                keywords=normalized_keywords,
            )
        for edge in payload["edges"]:
            self.graph.add_edge(edge["source"], edge["target"], type=edge["type"])

    def _node_score(self, node_id: str, query_tokens: set[str]) -> int:
        node = self.graph.nodes[node_id]
        haystack = " ".join(
            [
                node.get("name", ""),
                node.get("description", ""),
                " ".join(node.get("keywords", [])),
                node_id,
            ]
        ).lower()
        return sum(1 for token in query_tokens if token in haystack)

    def _to_knowledge_node(self, node_id: str) -> KnowledgeNode:
        attrs = self.graph.nodes[node_id]
        return KnowledgeNode(
            node_id=node_id,
            node_type=str(attrs.get("type", "")),
            name=str(attrs.get("name", "")),
            description=str(attrs.get("description", "")),
            keywords=list(attrs.get("keywords", [])),
        )

    def _sorted_nodes(self, node_type: str, query_tokens: set[str]) -> list[KnowledgeNode]:
        node_ids = [node_id for node_id, attrs in self.graph.nodes(data=True) if attrs.get("type") == node_type]
        ranked_ids = sorted(node_ids, key=lambda node_id: (-self._node_score(node_id, query_tokens), node_id))
        return [self._to_knowledge_node(node_id) for node_id in ranked_ids]

    def retrieve(self, spec: CapabilitySpec) -> RetrievedKnowledge:
        query_text = " ".join(
            [
                spec.task_name,
                spec.objective,
                spec.task_type,
                spec.data_profile,
                *spec.constraints,
                *spec.success_metrics,
            ]
        )
        query_tokens = _tokenize(query_text)
        algorithms = self._sorted_nodes("Algorithm", query_tokens)
        capabilities = self._sorted_nodes("Capability", query_tokens)
        metrics = self._sorted_nodes("Metric", query_tokens)
        constraints = self._sorted_nodes("Constraint", query_tokens)

        selected_algorithm_ids = {node.node_id for node in algorithms[: max(2, len(spec.preferred_algorithms) or 2)]}
        failure_patterns = [
            self._to_knowledge_node(source)
            for source, target, attrs in self.graph.edges(data=True)
            if attrs.get("type") == "suggests_fix" and target in selected_algorithm_ids
        ]

        return RetrievedKnowledge(
            capabilities=capabilities[:1],
            algorithms=algorithms[:3],
            metrics=metrics[:3],
            constraints=constraints[:2],
            failure_patterns=failure_patterns,
        )

    def export_payload(self) -> dict[str, list[dict[str, Any]]]:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            payload = {"id": node_id}
            payload.update(attrs)
            nodes.append(payload)

        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            payload = {"source": source, "target": target}
            payload.update(attrs)
            edges.append(payload)
        return {"nodes": nodes, "edges": edges}

    def layout_payload(self) -> dict[str, list[dict[str, Any]]]:
        if self.graph.number_of_nodes() == 0:
            return {"nodes": [], "edges": []}
        layout = nx.spring_layout(self.graph, seed=42)
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            x, y = layout[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("name", node_id),
                    "type": attrs.get("type", ""),
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                }
            )
        edges = [
            {
                "source": source,
                "target": target,
                "type": attrs.get("type", ""),
            }
            for source, target, attrs in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def export_graph(self) -> KnowledgeWriteback:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        payload = self.export_payload()
        json_path = self.export_dir / "knowledge_graph.json"
        graphml_path = self.export_dir / "knowledge_graph.graphml"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        graphml_graph = nx.DiGraph()
        for node in payload["nodes"]:
            node_id = str(node["id"])
            graphml_graph.add_node(
                node_id,
                **{
                    key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else str(value)
                    for key, value in node.items()
                    if key != "id"
                },
            )
        for edge in payload["edges"]:
            graphml_graph.add_edge(
                str(edge["source"]),
                str(edge["target"]),
                **{
                    key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else str(value)
                    for key, value in edge.items()
                    if key not in {"source", "target"}
                },
            )
        nx.write_graphml(graphml_graph, graphml_path)
        return KnowledgeWriteback(
            validation_node_id="seed::graph",
            export_json_path=str(json_path),
            export_graphml_path=str(graphml_path),
        )

    def write_validation_result(
        self,
        task_id: str,
        spec: CapabilitySpec,
        reports: list[ValidationReport],
        best_candidate_id: str,
        capability_version: str | None = None,
    ) -> KnowledgeWriteback:
        validation_node_id = f"validation::{task_id}"
        report_payload = [
            {
                "candidate_id": report.candidate_id,
                "algorithm_key": report.algorithm_key,
                "success": report.success,
                "score": report.score,
                "metrics": report.metrics.model_dump(),
            }
            for report in reports
        ]
        self.graph.add_node(
            validation_node_id,
            type="ValidationResult",
            name=f"Validation {task_id}",
            description=f"Best candidate: {best_candidate_id}",
            best_candidate_id=best_candidate_id,
            task_name=spec.task_name,
            capability_version=capability_version or "unversioned",
            reports=report_payload,
        )
        self.graph.add_edge(validation_node_id, "capability::topk_recommendation", type="verifies")
        for report in reports:
            algorithm_node_id = f"algorithm::{report.algorithm_key}"
            if self.graph.has_node(algorithm_node_id):
                self.graph.add_edge(
                    validation_node_id,
                    algorithm_node_id,
                    type="evaluated",
                    success=report.success,
                    score=report.score,
                )
            if not report.success and report.errors:
                learned_failure_id = f"learned_failure::{task_id}::{report.candidate_id}"
                self.graph.add_node(
                    learned_failure_id,
                    type="FailurePattern",
                    name=f"Learned failure for {report.candidate_id}",
                    description="; ".join(report.errors[:3]),
                    keywords=report.errors[:3],
                )
                self.graph.add_edge(learned_failure_id, algorithm_node_id, type="suggests_fix")
        export = self.export_graph()
        return KnowledgeWriteback(
            validation_node_id=validation_node_id,
            export_json_path=export.export_json_path,
            export_graphml_path=export.export_graphml_path,
        )

    def ingest_repo_extraction(self, result: RepoExtractionResult) -> None:
        repo_node_id = f"repo::{result.repo_name}"
        self.graph.add_node(
            repo_node_id,
            type="Repository",
            name=result.repo_name,
            description=result.readme_excerpt or f"Extracted repository {result.repo_name}",
            keywords=result.dependencies,
            repo_path=result.repo_path,
            output_path=result.output_path or "",
        )

        for dependency in result.dependencies:
            dependency_node_id = f"dependency::{dependency}"
            if not self.graph.has_node(dependency_node_id):
                self.graph.add_node(
                    dependency_node_id,
                    type="Dependency",
                    name=dependency,
                    description=f"Dependency extracted from {result.repo_name}",
                    keywords=[dependency],
                )
            self.graph.add_edge(repo_node_id, dependency_node_id, type="depends_on")

        for file_record in result.files:
            file_node_id = f"repo_file::{result.repo_name}::{file_record.path}"
            self.graph.add_node(
                file_node_id,
                type="SourceFile",
                name=file_record.path,
                description=f"Source file from {result.repo_name}",
                keywords=file_record.imports,
            )
            self.graph.add_edge(repo_node_id, file_node_id, type="contains")
            for symbol in file_record.symbols:
                symbol_node_id = f"repo_symbol::{result.repo_name}::{file_record.path}::{symbol.name}"
                self.graph.add_node(
                    symbol_node_id,
                    type="CodeSymbol",
                    name=symbol.name,
                    description=symbol.docstring or f"{symbol.symbol_type} in {file_record.path}",
                    keywords=[symbol.symbol_type],
                    line_no=symbol.line_no,
                )
                self.graph.add_edge(file_node_id, symbol_node_id, type="defines")
        self.export_graph()
