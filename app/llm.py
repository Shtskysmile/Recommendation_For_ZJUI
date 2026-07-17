from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import httpx

from app.data import DEFAULT_RECOMMENDATION_DATASET
from app.models import CapabilitySpec, CandidatePlan, GenerationRequest, RetrievedKnowledge
from app.plugin_loader import PluginLoader
from app.settings import AppSettings


def _stringify_data_profile(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, nested_value in value.items():
            if isinstance(nested_value, dict):
                nested_summary = ", ".join(f"{nested_key}={nested_item}" for nested_key, nested_item in nested_value.items())
                parts.append(f"{key}: {nested_summary}")
            elif isinstance(nested_value, list):
                parts.append(f"{key}: {', '.join(str(item) for item in nested_value)}")
            else:
                parts.append(f"{key}: {nested_value}")
        return "; ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "movielens implicit feedback")


def _normalize_algorithm_key(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.startswith("algorithm::"):
        return normalized.split("::", 1)[1]
    return normalized


def _normalize_list(value, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _normalize_task_type(value: str | None, request: GenerationRequest) -> str:
    requested = (value or request.task_type or "").strip().lower()
    if requested in {"top_k_recommendation", "tabular_classification"}:
        return requested
    if any(token in requested for token in ("classification", "churn", "分类", "流失")):
        return "tabular_classification"
    return "top_k_recommendation"


def _default_dataset_name(task_type: str) -> str:
    return "churn_sample" if task_type == "tabular_classification" else DEFAULT_RECOMMENDATION_DATASET


def _normalize_dataset_name(request: GenerationRequest, task_type: str) -> str:
    requested_name = str(request.dataset_name or "").strip()
    if requested_name and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", requested_name):
        return requested_name
    return _default_dataset_name(task_type)


def _normalize_capability_payload(payload: dict, request: GenerationRequest) -> CapabilitySpec:
    task_type = _normalize_task_type(payload.get("task_type"), request)
    default_metrics = (
        ["accuracy", "precision", "recall", "f1"]
        if task_type == "tabular_classification"
        else ["recall@k", "ndcg@k", "coverage"]
    )
    default_algorithms = (
        ["risk_score", "majority_class"]
        if task_type == "tabular_classification"
        else ["itemcf", "popularity"]
    )
    constraints = _normalize_list(payload.get("constraints"), request.constraints)
    success_metrics = _normalize_list(payload.get("success_metrics"), default_metrics)
    preferred_algorithms = _normalize_list(payload.get("preferred_algorithms"), default_algorithms)
    return CapabilitySpec(
        task_name=str(payload.get("task_name") or request.task_name),
        objective=str(payload.get("objective") or request.capability_description.strip()),
        task_type=task_type,
        # Dataset selection is a local resource decision. Never turn an LLM's
        # descriptive or hallucinated dataset value into a filesystem path.
        dataset_name=_normalize_dataset_name(request, task_type),
        top_k=int(payload.get("top_k") or request.top_k),
        constraints=constraints,
        success_metrics=success_metrics,
        preferred_algorithms=[_normalize_algorithm_key(item) for item in preferred_algorithms],
        data_profile=_stringify_data_profile(
            payload.get(
                "data_profile",
                "customer churn tabular classification" if task_type == "tabular_classification" else "movielens implicit feedback",
            )
        ),
    )


def _normalize_candidate_plan(plugin_loader: PluginLoader, payload: dict, algorithm_key: str, search_score: float) -> CandidatePlan:
    normalized_key = _normalize_algorithm_key(payload.get("algorithm_key") or algorithm_key)
    descriptor = plugin_loader.get_algorithm_plugin(normalized_key)
    return CandidatePlan(
        algorithm_key=normalized_key,
        display_name=str(payload.get("display_name") or descriptor.display_name),
        rationale=str(payload.get("rationale") or descriptor.rationale),
        plan_summary=str(payload.get("plan_summary") or descriptor.summary),
        search_score=search_score,
        design_explanation=str(
            payload.get("design_explanation")
            or f"Selected by beam search and explained with {descriptor.display_name}."
        ),
    )


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def parse_capability(self, request: GenerationRequest) -> CapabilitySpec:
        raise NotImplementedError

    @abstractmethod
    def explain_algorithm(
        self,
        spec: CapabilitySpec,
        knowledge: RetrievedKnowledge,
        algorithm_key: str,
        search_score: float,
    ) -> CandidatePlan:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    name = "mock"

    def __init__(self, settings: AppSettings):
        self.plugin_loader = PluginLoader(settings.project_root)

    def parse_capability(self, request: GenerationRequest) -> CapabilitySpec:
        description = request.capability_description.lower()
        task_type = _normalize_task_type(request.task_type or description, request)
        payload = {
            "task_name": request.task_name,
            "objective": request.capability_description.strip(),
            "task_type": task_type,
            "dataset_name": request.dataset_name or _default_dataset_name(task_type),
            "top_k": request.top_k,
            "constraints": request.constraints,
        }
        if task_type == "tabular_classification":
            payload["success_metrics"] = ["accuracy", "precision", "recall", "f1"]
            payload["preferred_algorithms"] = ["risk_score", "majority_class"]
            payload["data_profile"] = "customer churn tabular classification"
        else:
            preferred_algorithms = ["itemcf", "popularity"]
            if any(token in description for token in ("genre", "content", "explain")):
                preferred_algorithms.insert(1, "genre_affinity")
            payload["success_metrics"] = ["recall@k", "ndcg@k", "coverage"]
            payload["preferred_algorithms"] = preferred_algorithms
            payload["data_profile"] = "movielens implicit feedback"
        return _normalize_capability_payload(payload, request)

    def explain_algorithm(
        self,
        spec: CapabilitySpec,
        knowledge: RetrievedKnowledge,
        algorithm_key: str,
        search_score: float,
    ) -> CandidatePlan:
        descriptor = self.plugin_loader.get_algorithm_plugin(algorithm_key)
        return CandidatePlan(
            algorithm_key=algorithm_key,
            display_name=descriptor.display_name,
            rationale=descriptor.rationale,
            plan_summary=descriptor.summary,
            search_score=search_score,
            design_explanation=(
                f"Beam search selected {algorithm_key} with score {search_score:.2f} "
                f"for constraints {spec.constraints} and metrics {spec.success_metrics}."
            ),
        )


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(self, settings: AppSettings):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when using a real LLM provider.")
        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.timeout = settings.request_timeout_seconds
        self.plugin_loader = PluginLoader(settings.project_root)
        self.fallback_provider = MockLLMProvider(settings)
        self.name = (
            "siliconflow-openai-compatible"
            if "siliconflow.cn" in self.base_url
            else "openai-compatible"
        )

    def _json_completion(self, system_prompt: str, user_prompt: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content)

    def parse_capability(self, request: GenerationRequest) -> CapabilitySpec:
        system_prompt = (
            "You are extracting a structured algorithm capability specification. "
            "The task_type must be either top_k_recommendation or tabular_classification. "
            "Return JSON with keys: task_name, objective, task_type, dataset_name, top_k, constraints, "
            "success_metrics, preferred_algorithms, data_profile."
        )
        user_prompt = json.dumps(request.model_dump(), ensure_ascii=False)
        try:
            payload = self._json_completion(system_prompt, user_prompt)
            return _normalize_capability_payload(payload, request)
        except Exception:
            return self.fallback_provider.parse_capability(request)

    def explain_algorithm(
        self,
        spec: CapabilitySpec,
        knowledge: RetrievedKnowledge,
        algorithm_key: str,
        search_score: float,
    ) -> CandidatePlan:
        descriptor = self.plugin_loader.get_algorithm_plugin(algorithm_key)
        system_prompt = (
            "You are explaining an algorithm choice in an AI capability factory. "
            "Return JSON with keys: algorithm_key, display_name, rationale, plan_summary, design_explanation."
        )
        user_prompt = json.dumps(
            {
                "spec": spec.model_dump(),
                "knowledge": knowledge.model_dump(),
                "algorithm": {
                    "key": descriptor.key,
                    "display_name": descriptor.display_name,
                    "summary": descriptor.summary,
                    "rationale": descriptor.rationale,
                    "strengths": descriptor.strengths,
                },
                "search_score": search_score,
            },
            ensure_ascii=False,
        )
        try:
            payload = self._json_completion(system_prompt, user_prompt)
            return _normalize_candidate_plan(self.plugin_loader, payload, algorithm_key, search_score)
        except Exception:
            plan = self.fallback_provider.explain_algorithm(spec, knowledge, algorithm_key, search_score)
            return plan.model_copy(
                update={
                    "design_explanation": (
                        f"{plan.design_explanation} Remote explanation timed out or failed, "
                        "so the system fell back to the local template."
                    )
                }
            )


def create_provider(settings: AppSettings) -> LLMProvider:
    if settings.llm_provider in {"openai", "siliconflow"}:
        return OpenAICompatibleLLMProvider(settings)
    return MockLLMProvider(settings)
