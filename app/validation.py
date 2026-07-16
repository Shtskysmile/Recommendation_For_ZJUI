from __future__ import annotations

import ast
import importlib.util
import math
import time
import tracemalloc
import uuid
from pathlib import Path

from app.data import RecommendationDataset
from app.models import AlgorithmCandidate, ValidationMetrics, ValidationReport


SAFE_IMPORTS = {"__future__", "collections", "math", "pandas"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "input", "__import__"}
FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "httpx", "requests"}


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".", 1)[0]
            if module not in SAFE_IMPORTS:
                self.errors.append(f"Unsafe import blocked: {module}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".", 1)[0]
        if module and module not in SAFE_IMPORTS:
            self.errors.append(f"Unsafe import blocked: {module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden call blocked: {node.func.id}")
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in FORBIDDEN_MODULES:
                self.errors.append(f"Forbidden module call blocked: {node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)


class CandidateValidator:
    def validate(self, candidate: AlgorithmCandidate, dataset: RecommendationDataset, top_k: int) -> ValidationReport:
        return self.validate_path(
            candidate_id=candidate.candidate_id,
            algorithm_key=candidate.algorithm_key,
            code_path=Path(candidate.code_path),
            dataset=dataset,
            top_k=top_k,
        )

    def validate_path(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        dataset: RecommendationDataset,
        top_k: int,
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        metrics = ValidationMetrics()
        interface_ok = False
        stability_ok = False

        try:
            source_code = code_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Could not read candidate code: {exc}")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        safety_errors = self._scan_safety(source_code)
        if safety_errors:
            errors.extend(safety_errors)
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        try:
            module = self._load_module(code_path)
        except Exception as exc:  # pragma: no cover - importlib messages vary
            errors.append(f"Import failed: {exc}")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        recommender_class = getattr(module, "GeneratedRecommender", None)
        if recommender_class is None:
            errors.append("GeneratedRecommender class is missing.")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        try:
            recommender = recommender_class()
        except Exception as exc:
            errors.append(f"Instantiation failed: {exc}")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        missing_methods = [name for name in ("fit", "recommend", "batch_recommend") if not hasattr(recommender, name)]
        if missing_methods:
            errors.append(f"Missing interface methods: {', '.join(missing_methods)}")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        try:
            recommender.fit(dataset.train_df.copy())
            interface_ok = True
            sample_user = next(iter(dataset.test_truth))
            single_recommendations = recommender.recommend(sample_user, k=top_k)
            tracemalloc.start()
            batch_start = time.perf_counter()
            batch_recommendations = recommender.batch_recommend(sorted(dataset.test_truth), k=top_k)
            elapsed = time.perf_counter() - batch_start
            _current_allocated, peak_allocated = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            second_pass = recommender.batch_recommend(sorted(dataset.test_truth), k=top_k)
        except Exception as exc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            errors.append(f"Execution failed: {exc}")
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        if not isinstance(single_recommendations, list):
            errors.append("recommend() must return a list of item ids.")
        if not isinstance(batch_recommendations, dict):
            errors.append("batch_recommend() must return a dict keyed by user id.")

        if errors:
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        normalized_batch: dict[int, list[int]] = {}
        for user_id in sorted(dataset.test_truth):
            raw_items = batch_recommendations.get(user_id)
            if raw_items is None:
                errors.append(f"batch_recommend() missing user {user_id}.")
                continue
            if not isinstance(raw_items, list):
                errors.append(f"Recommendations for user {user_id} must be a list.")
                continue
            normalized_items = [int(item_id) for item_id in raw_items]
            seen_items = set(dataset.user_histories.get(int(user_id), []))
            if any(item_id in seen_items for item_id in normalized_items):
                errors.append(f"Seen item leakage for user {user_id}.")
            if len(set(normalized_items)) != len(normalized_items):
                errors.append(f"Duplicate recommendations for user {user_id}.")
            if len(normalized_items) == 0:
                errors.append(f"Empty recommendation list for user {user_id}.")
            if len(normalized_items) > top_k:
                errors.append(f"Recommendation list exceeds top_k for user {user_id}.")
            if len(normalized_items) < top_k:
                warnings.append(f"Recommendation list shorter than top_k for user {user_id}.")
            normalized_batch[int(user_id)] = normalized_items

        stability_ok = batch_recommendations == second_pass
        if not stability_ok:
            warnings.append("Recommendations are not deterministic across repeated calls.")

        if errors:
            return self._build_report(
                candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
            )

        metrics = self._compute_metrics(normalized_batch, dataset, elapsed, top_k)
        metrics.peak_allocated_kb = round(peak_allocated / 1024, 6)
        return self._build_report(
            candidate_id, algorithm_key, code_path, interface_ok, stability_ok, metrics, errors, warnings
        )

    def _build_report(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        interface_ok: bool,
        stability_ok: bool,
        metrics: ValidationMetrics,
        errors: list[str],
        warnings: list[str],
    ) -> ValidationReport:
        success = interface_ok and not errors
        score = 0.0
        if success:
            score = (metrics.recall_at_k * 0.55) + (metrics.ndcg_at_k * 0.35) + (metrics.coverage * 0.10)
        return ValidationReport(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            success=success,
            interface_ok=interface_ok,
            stability_ok=stability_ok,
            code_path=str(code_path),
            score=round(score, 6),
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

    def _scan_safety(self, source_code: str) -> list[str]:
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return [f"Syntax error: {exc.msg} at line {exc.lineno}"]
        visitor = SafetyVisitor()
        visitor.visit(tree)
        return visitor.errors

    def _load_module(self, code_path: Path):
        module_name = f"generated_candidate_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _compute_metrics(
        self,
        predictions: dict[int, list[int]],
        dataset: RecommendationDataset,
        elapsed_seconds: float,
        top_k: int,
    ) -> ValidationMetrics:
        recall_scores: list[float] = []
        ndcg_scores: list[float] = []
        recommended_items: set[int] = set()

        for user_id, truth_items in dataset.test_truth.items():
            predicted_items = predictions.get(int(user_id), [])[:top_k]
            recommended_items.update(predicted_items)
            hits = [item_id for item_id in predicted_items if item_id in truth_items]
            recall_scores.append(len(hits) / len(truth_items) if truth_items else 0.0)

            dcg = 0.0
            for rank, item_id in enumerate(predicted_items, start=1):
                if item_id in truth_items:
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_hits = min(len(truth_items), top_k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_scores.append(dcg / idcg if idcg else 0.0)

        coverage = len(recommended_items) / max(dataset.item_count, 1)
        average_latency_ms = (elapsed_seconds / max(len(predictions), 1)) * 1000
        return ValidationMetrics(
            recall_at_k=round(sum(recall_scores) / len(recall_scores), 6),
            ndcg_at_k=round(sum(ndcg_scores) / len(ndcg_scores), 6),
            coverage=round(coverage, 6),
            average_latency_ms=round(average_latency_ms, 6),
        )
