from pathlib import Path

from app.codegen import CandidateGenerator
from app.data import MovieLensDatasetLoader
from app.models import CapabilitySpec
from app.repair import RepairAgent
from app.validation import CandidateValidator


def test_validator_accepts_popularity_template(test_settings):
    dataset = MovieLensDatasetLoader(test_settings.data_dir / "movielens_sample").load()
    generator = CandidateGenerator(test_settings)
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

    candidate = generator.generate_candidate(
        task_id="task-validation",
        candidate_id="candidate-pop",
        algorithm_key="popularity",
        spec=spec,
    )
    validator = CandidateValidator()
    report = validator.validate(candidate, dataset, top_k=5)

    assert report.interface_ok is True
    assert report.success is True
    assert report.metrics.coverage > 0


def test_repair_agent_recovers_missing_interface(test_settings):
    dataset = MovieLensDatasetLoader(test_settings.data_dir / "movielens_sample").load()
    broken_path = test_settings.generated_dir / "broken_candidate.py"
    broken_path.write_text("class GeneratedRecommender:\n    pass\n")

    validator = CandidateValidator()
    report = validator.validate_path(
        candidate_id="candidate-broken",
        algorithm_key="itemcf",
        code_path=broken_path,
        dataset=dataset,
        top_k=5,
    )

    repair_agent = RepairAgent(test_settings)
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
    repaired_candidate = repair_agent.repair(
        task_id="task-repair",
        candidate_id="candidate-broken",
        algorithm_key="itemcf",
        spec=spec,
        last_report=report,
        attempt=1,
    )
    repaired_report = validator.validate_path(
        candidate_id="candidate-broken",
        algorithm_key="itemcf",
        code_path=Path(repaired_candidate.code_path),
        dataset=dataset,
        top_k=5,
    )

    assert report.success is False
    assert repaired_report.success is True
