from __future__ import annotations

from app.knowledge import KnowledgeGraphStore
from app.models import CapabilitySpec
from app.search import BeamSearchPlanner


def test_beam_search_ranks_itemcf_first_for_recall_heavy_request(test_settings):
    store = KnowledgeGraphStore(test_settings.knowledge_dir)
    planner = BeamSearchPlanner(beam_width=3)
    spec = CapabilitySpec(
        task_name="movie_homepage",
        objective=(
            "Create a Top-K movie recommendation workflow for a homepage feed. "
            "Optimize recall and ndcg with implicit feedback."
        ),
        task_type="top_k_recommendation",
        dataset_name="movielens_sample",
        top_k=5,
        constraints=["implicit feedback", "local cpu"],
        success_metrics=["recall@5", "ndcg@5", "coverage"],
        preferred_algorithms=[],
        data_profile="movielens implicit feedback",
    )

    retrieved = store.retrieve(spec)
    result = planner.search(spec, retrieved, candidate_count=3)

    assert result.selected_algorithms[0] == "itemcf"
    assert "popularity" in result.selected_algorithms
    assert result.trace


def test_beam_search_ranks_risk_score_first_for_churn_request(test_settings):
    store = KnowledgeGraphStore(test_settings.knowledge_dir)
    planner = BeamSearchPlanner(beam_width=3)
    spec = CapabilitySpec(
        task_name="customer_churn",
        objective="Build a customer churn classification workflow with strong f1 and accuracy.",
        task_type="tabular_classification",
        dataset_name="churn_sample",
        top_k=5,
        constraints=["local cpu"],
        success_metrics=["accuracy", "precision", "recall", "f1"],
        preferred_algorithms=[],
        data_profile="customer churn tabular classification",
    )

    retrieved = store.retrieve(spec)
    result = planner.search(spec, retrieved, candidate_count=2)

    assert result.selected_algorithms[0] == "risk_score"
    assert "majority_class" in result.selected_algorithms
