from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgorithmDescriptor:
    key: str
    display_name: str
    summary: str
    rationale: str
    family: str
    strengths: tuple[str, ...]
    search_hints: tuple[str, ...]


ALGORITHM_REGISTRY: dict[str, AlgorithmDescriptor] = {
    "popularity": AlgorithmDescriptor(
        key="popularity",
        display_name="Popularity Baseline",
        summary="Use global item frequency as a strong local baseline with clear behavior.",
        rationale="Good for cold-start fallback, coverage checks, and interface validation.",
        family="baseline",
        strengths=("cold_start", "coverage", "simplicity", "local_cpu"),
        search_hints=("baseline", "cold start", "local cpu", "coverage"),
    ),
    "itemcf": AlgorithmDescriptor(
        key="itemcf",
        display_name="Item-Based Collaborative Filtering",
        summary="Score candidate items by similarity to each user's interacted items.",
        rationale="Better aligned with Top-K recommendation quality metrics such as Recall@K and NDCG@K.",
        family="collaborative_filtering",
        strengths=("recall", "ndcg", "implicit_feedback", "ranking"),
        search_hints=("recall", "ndcg", "implicit feedback", "ranking", "top-k"),
    ),
    "genre_affinity": AlgorithmDescriptor(
        key="genre_affinity",
        display_name="Genre Affinity Recommender",
        summary="Estimate user preference from interaction genres and rank unseen items by genre affinity.",
        rationale="Adds a content-oriented candidate that is easy to explain and transfers better to sparse histories.",
        family="content_based",
        strengths=("explainability", "content", "local_cpu", "cold_start"),
        search_hints=("content", "genre", "explain", "cold start", "sparse"),
    ),
}
