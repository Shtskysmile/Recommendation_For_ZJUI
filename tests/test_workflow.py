from pathlib import Path

from app.models import GenerationRequest
from app.workflow import CapabilityFactoryService


def test_workflow_runs_end_to_end_and_selects_best_candidate(test_settings):
    service = CapabilityFactoryService.build(test_settings)
    result = service.run_task(
        GenerationRequest(
            task_name="movie_homepage",
            capability_description=(
                "Build a Top-K movie recommendation workflow for a homepage feed. "
                "Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu."
            ),
            top_k=5,
            constraints=["implicit feedback", "local cpu"],
            candidate_count=2,
        )
    )

    assert result.status == "completed"
    assert len(result.candidates) == 2
    assert result.best_candidate is not None
    assert result.best_candidate.algorithm_key == "itemcf"
    assert result.best_candidate.validation is not None
    assert result.best_candidate.validation.success is True
    assert Path(result.report_paths["json"]).exists()
    assert Path(result.report_paths["markdown"]).exists()


def test_workflow_runs_end_to_end_for_classification(test_settings):
    service = CapabilityFactoryService.build(test_settings)
    result = service.run_task(
        GenerationRequest(
            task_name="customer_churn",
            capability_description=(
                "Build a customer churn classification workflow. "
                "Use local cpu, compare baseline and score-based methods, and optimize f1."
            ),
            task_type="tabular_classification",
            dataset_name="churn_sample",
            top_k=5,
            constraints=["local cpu"],
            candidate_count=2,
        )
    )

    assert result.status == "completed"
    assert result.parsed_intent is not None
    assert result.parsed_intent.task_type == "tabular_classification"
    assert len(result.candidates) == 2
    assert result.best_candidate is not None
    assert result.best_candidate.validation is not None
    assert result.best_candidate.validation.profile_name == "tabular_classification"
    assert result.best_candidate.validation.execution_mode == "subprocess_sandbox"
