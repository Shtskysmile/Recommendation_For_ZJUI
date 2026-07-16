from __future__ import annotations

import importlib.util
import json
import math
import time
import tracemalloc
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from app.data import ClassificationDataset, CustomerChurnDatasetLoader, MovieLensDatasetLoader, RecommendationDataset
from app.models import CapabilitySpec, ValidationMetrics, ValidationReport
from app.settings import AppSettings


class BaseValidationProfile(ABC):
    key: str
    display_name: str
    candidate_class_name: str
    execution_mode = "subprocess_sandbox"

    @abstractmethod
    def load_dataset(self, settings: AppSettings, spec: CapabilitySpec):
        raise NotImplementedError

    @abstractmethod
    def serialize_dataset(self, dataset, target_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_dataset_snapshot(self, source_dir: Path):
        raise NotImplementedError

    @abstractmethod
    def validate_candidate_path(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        dataset_dir: Path,
        top_k: int,
    ) -> ValidationReport:
        raise NotImplementedError

    def _load_module(self, code_path: Path):
        module_name = f"sandbox_candidate_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {code_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _failure_report(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        errors: list[str],
        warnings: list[str] | None = None,
        interface_ok: bool = False,
        stability_ok: bool = False,
        metrics: ValidationMetrics | None = None,
    ) -> ValidationReport:
        return ValidationReport(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            profile_name=self.key,
            success=False,
            interface_ok=interface_ok,
            stability_ok=stability_ok,
            code_path=str(code_path),
            execution_mode=self.execution_mode,
            score=0.0,
            metrics=metrics or ValidationMetrics(),
            errors=errors,
            warnings=warnings or [],
        )


class TopKRecommendationProfile(BaseValidationProfile):
    key = "top_k_recommendation"
    display_name = "Top-K Recommendation Validation"
    candidate_class_name = "GeneratedRecommender"

    def load_dataset(self, settings: AppSettings, spec: CapabilitySpec) -> RecommendationDataset:
        data_dir = settings.data_dir / spec.dataset_name
        return MovieLensDatasetLoader(data_dir).load()

    def serialize_dataset(self, dataset: RecommendationDataset, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset.train_df.to_csv(target_dir / "train.csv", index=False)
        (target_dir / "test_truth.json").write_text(json.dumps(dataset.test_truth, indent=2), encoding="utf-8")
        (target_dir / "catalog.json").write_text(json.dumps(dataset.catalog, indent=2), encoding="utf-8")
        (target_dir / "user_histories.json").write_text(json.dumps(dataset.user_histories, indent=2), encoding="utf-8")
        (target_dir / "metadata.json").write_text(
            json.dumps({"user_count": dataset.user_count, "item_count": dataset.item_count}, indent=2),
            encoding="utf-8",
        )

    def load_dataset_snapshot(self, source_dir: Path) -> RecommendationDataset:
        train_df = pd.read_csv(source_dir / "train.csv")
        test_truth = {
            int(key): [int(item) for item in value]
            for key, value in json.loads((source_dir / "test_truth.json").read_text()).items()
        }
        catalog_payload = json.loads((source_dir / "catalog.json").read_text())
        catalog = {int(key): value for key, value in catalog_payload.items()}
        histories_payload = json.loads((source_dir / "user_histories.json").read_text())
        user_histories = {int(key): [int(item) for item in value] for key, value in histories_payload.items()}
        metadata = json.loads((source_dir / "metadata.json").read_text())
        return RecommendationDataset(
            interactions_df=train_df.copy(),
            train_df=train_df,
            test_truth=test_truth,
            catalog=catalog,
            user_histories=user_histories,
            user_count=int(metadata["user_count"]),
            item_count=int(metadata["item_count"]),
        )

    def validate_candidate_path(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        dataset_dir: Path,
        top_k: int,
    ) -> ValidationReport:
        dataset = self.load_dataset_snapshot(dataset_dir)
        metrics = ValidationMetrics()
        warnings: list[str] = []
        interface_ok = False
        stability_ok = False

        try:
            module = self._load_module(code_path)
        except Exception as exc:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Import failed: {exc}"])

        recommender_class = getattr(module, self.candidate_class_name, None)
        if recommender_class is None:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"{self.candidate_class_name} class is missing."])

        recommender = recommender_class()
        missing_methods = [name for name in ("fit", "recommend", "batch_recommend") if not hasattr(recommender, name)]
        if missing_methods:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Missing interface methods: {', '.join(missing_methods)}"])

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
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Execution failed: {exc}"], interface_ok=interface_ok)

        errors: list[str] = []
        if not isinstance(single_recommendations, list):
            errors.append("recommend() must return a list of item ids.")
        if not isinstance(batch_recommendations, dict):
            errors.append("batch_recommend() must return a dict keyed by user id.")
        if errors:
            return self._failure_report(candidate_id, algorithm_key, code_path, errors, interface_ok=interface_ok)

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
            return self._failure_report(candidate_id, algorithm_key, code_path, errors, warnings, interface_ok, stability_ok)

        metrics = self._compute_metrics(normalized_batch, dataset, elapsed, top_k)
        metrics.peak_allocated_kb = round(peak_allocated / 1024, 6)
        score = (metrics.recall_at_k * 0.55) + (metrics.ndcg_at_k * 0.35) + (metrics.coverage * 0.10)
        return ValidationReport(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            profile_name=self.key,
            success=True,
            interface_ok=interface_ok,
            stability_ok=stability_ok,
            code_path=str(code_path),
            execution_mode=self.execution_mode,
            score=round(score, 6),
            metrics=metrics,
            errors=[],
            warnings=warnings,
        )

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


class TabularClassificationProfile(BaseValidationProfile):
    key = "tabular_classification"
    display_name = "Tabular Classification Validation"
    candidate_class_name = "GeneratedClassifier"

    def load_dataset(self, settings: AppSettings, spec: CapabilitySpec) -> ClassificationDataset:
        data_dir = settings.data_dir / spec.dataset_name
        return CustomerChurnDatasetLoader(data_dir).load()

    def serialize_dataset(self, dataset: ClassificationDataset, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset.train_df.to_csv(target_dir / "train.csv", index=False)
        dataset.test_df.to_csv(target_dir / "test.csv", index=False)
        (target_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_columns": dataset.feature_columns,
                    "label_column": dataset.label_column,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_dataset_snapshot(self, source_dir: Path) -> ClassificationDataset:
        train_df = pd.read_csv(source_dir / "train.csv")
        test_df = pd.read_csv(source_dir / "test.csv")
        metadata = json.loads((source_dir / "metadata.json").read_text())
        return ClassificationDataset(
            train_df=train_df,
            test_df=test_df,
            feature_columns=list(metadata["feature_columns"]),
            label_column=str(metadata["label_column"]),
        )

    def validate_candidate_path(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        dataset_dir: Path,
        top_k: int,
    ) -> ValidationReport:
        dataset = self.load_dataset_snapshot(dataset_dir)
        try:
            module = self._load_module(code_path)
        except Exception as exc:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Import failed: {exc}"])

        classifier_class = getattr(module, self.candidate_class_name, None)
        if classifier_class is None:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"{self.candidate_class_name} class is missing."])

        classifier = classifier_class()
        missing_methods = [name for name in ("fit", "predict", "batch_predict") if not hasattr(classifier, name)]
        if missing_methods:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Missing interface methods: {', '.join(missing_methods)}"])

        try:
            classifier.fit(dataset.train_df.copy())
            tracemalloc.start()
            single_predictions = classifier.predict(dataset.test_features.copy())
            batch_start = time.perf_counter()
            batch_predictions = classifier.batch_predict(dataset.test_features.copy())
            elapsed = time.perf_counter() - batch_start
            _current_allocated, peak_allocated = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            second_pass = classifier.batch_predict(dataset.test_features.copy())
        except Exception as exc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Execution failed: {exc}"], interface_ok=True)

        errors: list[str] = []
        warnings: list[str] = []
        if not isinstance(single_predictions, list):
            errors.append("predict() must return a list of labels.")
        if not isinstance(batch_predictions, list):
            errors.append("batch_predict() must return a list of labels.")
        if errors:
            return self._failure_report(candidate_id, algorithm_key, code_path, errors, warnings, interface_ok=True)
        if len(batch_predictions) != len(dataset.test_df):
            errors.append("Prediction length does not match the test set size.")
            return self._failure_report(candidate_id, algorithm_key, code_path, errors, warnings, interface_ok=True)

        normalized_predictions = [int(label) for label in batch_predictions]
        stability_ok = batch_predictions == second_pass
        if not stability_ok:
            warnings.append("Predictions are not deterministic across repeated calls.")

        metrics = self._compute_metrics(normalized_predictions, dataset, elapsed)
        metrics.peak_allocated_kb = round(peak_allocated / 1024, 6)
        score = (metrics.f1 * 0.45) + (metrics.accuracy * 0.35) + (metrics.precision * 0.20)
        return ValidationReport(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            profile_name=self.key,
            success=True,
            interface_ok=True,
            stability_ok=stability_ok,
            code_path=str(code_path),
            execution_mode=self.execution_mode,
            score=round(score, 6),
            metrics=metrics,
            errors=[],
            warnings=warnings,
        )

    def _compute_metrics(
        self,
        predictions: list[int],
        dataset: ClassificationDataset,
        elapsed_seconds: float,
    ) -> ValidationMetrics:
        truth = [int(value) for value in dataset.test_labels.tolist()]
        tp = sum(1 for pred, actual in zip(predictions, truth) if pred == 1 and actual == 1)
        tn = sum(1 for pred, actual in zip(predictions, truth) if pred == 0 and actual == 0)
        fp = sum(1 for pred, actual in zip(predictions, truth) if pred == 1 and actual == 0)
        fn = sum(1 for pred, actual in zip(predictions, truth) if pred == 0 and actual == 1)
        accuracy = (tp + tn) / max(len(truth), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        average_latency_ms = (elapsed_seconds / max(len(truth), 1)) * 1000
        return ValidationMetrics(
            accuracy=round(accuracy, 6),
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
            average_latency_ms=round(average_latency_ms, 6),
        )
