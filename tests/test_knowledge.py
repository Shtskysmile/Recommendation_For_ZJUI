import json

from app.knowledge import KnowledgeGraphStore
from app.models import CapabilitySpec, ValidationMetrics, ValidationReport


def test_retrieval_returns_recommender_algorithms(test_settings):
    store = KnowledgeGraphStore(test_settings.knowledge_dir)
    spec = CapabilitySpec(
        task_name="movie_homepage",
        objective="Create a Top-K movie recommendation recall workflow.",
        task_type="top_k_recommendation",
        top_k=5,
        constraints=["implicit feedback", "local cpu"],
        success_metrics=["recall@5", "ndcg@5", "coverage"],
        preferred_algorithms=[],
        data_profile="movielens implicit feedback",
    )

    retrieved = store.retrieve(spec)

    algorithm_ids = {node.node_id for node in retrieved.algorithms}
    assert "algorithm::popularity" in algorithm_ids
    assert "algorithm::itemcf" in algorithm_ids


def test_writeback_persists_validation_result(test_settings):
    store = KnowledgeGraphStore(test_settings.knowledge_dir)
    spec = CapabilitySpec(
        task_name="movie_homepage",
        objective="Create a Top-K movie recommendation recall workflow.",
        task_type="top_k_recommendation",
        top_k=5,
        constraints=["implicit feedback", "local cpu"],
        success_metrics=["recall@5", "ndcg@5", "coverage"],
        preferred_algorithms=[],
        data_profile="movielens implicit feedback",
    )
    report = ValidationReport(
        candidate_id="candidate-1",
        algorithm_key="itemcf",
        success=True,
        interface_ok=True,
        stability_ok=True,
        code_path="generated/candidate_1.py",
        score=0.72,
        metrics=ValidationMetrics(
            recall_at_k=0.75,
            ndcg_at_k=0.69,
            coverage=0.5,
            average_latency_ms=1.2,
        ),
        errors=[],
        warnings=[],
    )

    writeback = store.write_validation_result(
        task_id="task-001",
        spec=spec,
        reports=[report],
        best_candidate_id="candidate-1",
    )

    assert writeback.validation_node_id.startswith("validation::task-001")
    exported_graph = test_settings.knowledge_dir / "exports" / "knowledge_graph.json"
    assert exported_graph.exists()
    payload = json.loads(exported_graph.read_text())
    assert any(node["id"] == writeback.validation_node_id for node in payload["nodes"])
