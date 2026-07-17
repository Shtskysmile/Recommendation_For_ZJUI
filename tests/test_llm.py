from app.llm import _normalize_capability_payload
from app.models import GenerationRequest


def test_model_dataset_description_falls_back_to_bundled_movielens_data():
    request = GenerationRequest(
        task_name="movie_homepage",
        capability_description="Build a movie recommendation workflow.",
    )

    spec = _normalize_capability_payload(
        {
            "task_type": "top_k_recommendation",
            "dataset_name": "MovieLens (or similar implicit feedback dataset)",
        },
        request,
    )

    assert spec.dataset_name == "movielens_latest_small"


def test_explicit_safe_dataset_name_is_preserved():
    request = GenerationRequest(
        task_name="movie_homepage",
        capability_description="Build a movie recommendation workflow.",
        dataset_name="my_movie_data",
    )

    spec = _normalize_capability_payload(
        {"task_type": "top_k_recommendation", "dataset_name": "made_up_by_model"},
        request,
    )

    assert spec.dataset_name == "my_movie_data"
