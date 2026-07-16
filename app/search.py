from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models import CapabilitySpec, RetrievedKnowledge, SearchTraceStep
from app.plugin_loader import PluginLoader


@dataclass(slots=True)
class SearchResult:
    selected_algorithms: list[str]
    trace: list[SearchTraceStep]
    scores: dict[str, float]


class BeamSearchPlanner:
    def __init__(self, plugin_loader: PluginLoader | None = None, beam_width: int = 3):
        self.plugin_loader = plugin_loader or PluginLoader(Path(__file__).resolve().parents[1])
        self.beam_width = max(beam_width, 1)

    def search(self, spec: CapabilitySpec, knowledge: RetrievedKnowledge, candidate_count: int) -> SearchResult:
        available_plugins = self.plugin_loader.list_algorithm_plugins(spec.task_type)
        requested_count = min(candidate_count, len(available_plugins))
        ranked_algorithms = self._rank_algorithms(spec, knowledge)
        beam: list[tuple[list[str], float]] = [([], 0.0)]
        trace: list[SearchTraceStep] = []

        for depth in range(requested_count):
            expansions: list[tuple[list[str], float]] = []
            for sequence, sequence_score in beam:
                for algorithm_key, algorithm_score in ranked_algorithms:
                    if algorithm_key in sequence:
                        continue
                    marginal_bonus = self._marginal_bonus(sequence, algorithm_key)
                    updated_sequence = [*sequence, algorithm_key]
                    updated_score = sequence_score + algorithm_score + marginal_bonus
                    expansions.append((updated_sequence, updated_score))
            expansions.sort(key=lambda item: (-item[1], item[0]))
            beam = expansions[: self.beam_width]
            if beam:
                best_sequence, best_score = beam[0]
                trace.append(
                    SearchTraceStep(
                        agent="BeamSearchPlanner",
                        summary=f"Expanded beam at depth {depth + 1}",
                        score=round(best_score, 6),
                        state={
                            "depth": depth + 1,
                            "beam_width": self.beam_width,
                            "best_sequence": best_sequence,
                        },
                    )
                )

        score_map = {algorithm_key: score for algorithm_key, score in ranked_algorithms}
        if beam:
            selected_algorithms = sorted(beam[0][0], key=lambda algorithm_key: (-score_map[algorithm_key], algorithm_key))
        else:
            selected_algorithms = [algorithm_key for algorithm_key, _ in ranked_algorithms[:requested_count]]
        return SearchResult(selected_algorithms=selected_algorithms, trace=trace, scores=score_map)

    def _rank_algorithms(self, spec: CapabilitySpec, knowledge: RetrievedKnowledge) -> list[tuple[str, float]]:
        query = " ".join([spec.objective, spec.task_type, spec.data_profile, *spec.constraints, *spec.success_metrics]).lower()
        retrieved_positions = {
            node.node_id.replace("algorithm::", ""): index
            for index, node in enumerate(knowledge.algorithms)
        }
        scores: list[tuple[str, float]] = []
        for plugin in self.plugin_loader.list_algorithm_plugins(spec.task_type):
            algorithm_key = plugin.key
            score = 0.0
            for hint in plugin.search_hints:
                if hint in query:
                    score += 2.0
            for strength in plugin.strengths:
                normalized = strength.replace("_", " ")
                if normalized in query:
                    score += 1.5
            if algorithm_key in spec.preferred_algorithms:
                score += 3.5
            if algorithm_key in retrieved_positions:
                score += max(0.0, 3.0 - retrieved_positions[algorithm_key])
            if algorithm_key == "itemcf":
                if any(metric.startswith("recall") or metric.startswith("ndcg") for metric in spec.success_metrics):
                    score += 4.0
                if "implicit feedback" in query:
                    score += 1.5
            elif algorithm_key == "risk_score":
                if any(metric in {"accuracy", "f1", "precision", "recall"} for metric in spec.success_metrics):
                    score += 4.0
                if "classification" in query or "churn" in query:
                    score += 2.0
            elif algorithm_key == "genre_affinity":
                if any(token in query for token in ("genre", "content", "explain")):
                    score += 2.5
                score -= 0.5
            elif algorithm_key == "popularity":
                if "coverage" in query or "cold start" in query:
                    score += 1.5
            elif algorithm_key == "majority_class":
                if "baseline" in query:
                    score += 2.0
            scores.append((algorithm_key, round(score, 6)))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores

    def _marginal_bonus(self, sequence: list[str], algorithm_key: str) -> float:
        if not sequence:
            return 0.0
        existing_families = {self.plugin_loader.get_algorithm_plugin(key).family for key in sequence}
        candidate_family = self.plugin_loader.get_algorithm_plugin(algorithm_key).family
        return 0.7 if candidate_family not in existing_families else 0.15
