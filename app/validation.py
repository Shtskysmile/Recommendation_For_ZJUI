from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.data import ClassificationDataset, RecommendationDataset
from app.models import AlgorithmCandidate, ValidationMetrics, ValidationReport
from app.plugin_loader import PluginLoader
from app.settings import AppSettings


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
    def __init__(
        self,
        settings: AppSettings | None = None,
        plugin_loader: PluginLoader | None = None,
    ) -> None:
        self.runtime_root = Path(__file__).resolve().parents[1]
        self.settings = settings or AppSettings.from_env(self.runtime_root)
        self.plugin_loader = plugin_loader or PluginLoader(self.settings.project_root)

    def validate(
        self,
        candidate: AlgorithmCandidate,
        dataset: RecommendationDataset | ClassificationDataset,
        top_k: int,
        profile_key: str | None = None,
    ) -> ValidationReport:
        return self.validate_path(
            candidate_id=candidate.candidate_id,
            algorithm_key=candidate.algorithm_key,
            code_path=Path(candidate.code_path),
            dataset=dataset,
            top_k=top_k,
            profile_key=profile_key,
        )

    def validate_path(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        dataset: RecommendationDataset | ClassificationDataset,
        top_k: int,
        profile_key: str | None = None,
    ) -> ValidationReport:
        try:
            source_code = code_path.read_text(encoding="utf-8")
        except OSError as exc:
            return self._failure_report(candidate_id, algorithm_key, code_path, [f"Could not read candidate code: {exc}"])

        safety_errors = self._scan_safety(source_code)
        if safety_errors:
            return self._failure_report(candidate_id, algorithm_key, code_path, safety_errors)

        resolved_profile_key = profile_key or self._infer_profile_key(dataset, algorithm_key)
        profile_manifest = self.plugin_loader.get_validation_profile(resolved_profile_key)
        profile = profile_manifest.profile_class()

        with tempfile.TemporaryDirectory(prefix="capability-validator-") as temp_root_raw:
            temp_root = Path(temp_root_raw)
            dataset_dir = temp_root / "dataset"
            manifest_path = temp_root / "manifest.json"
            profile.serialize_dataset(dataset, dataset_dir)
            manifest_path.write_text(
                json.dumps(
                    {
                        "project_root": str(self.settings.project_root),
                        "profile_key": resolved_profile_key,
                        "candidate_id": candidate_id,
                        "algorithm_key": algorithm_key,
                        "code_path": str(code_path.resolve()),
                        "dataset_dir": str(dataset_dir.resolve()),
                        "top_k": top_k,
                        "sandbox_timeout_seconds": self.settings.sandbox_timeout_seconds,
                        "sandbox_memory_limit_mb": self.settings.sandbox_memory_limit_mb,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "app.sandbox_worker", str(manifest_path)],
                    cwd=self.runtime_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.sandbox_timeout_seconds + 5,
                )
            except subprocess.TimeoutExpired:
                return self._failure_report(
                    candidate_id,
                    algorithm_key,
                    code_path,
                    [f"Sandbox execution timed out after {self.settings.sandbox_timeout_seconds} seconds."],
                    profile_name=resolved_profile_key,
                )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            message = stderr or stdout or "Sandbox execution failed."
            return self._failure_report(
                candidate_id,
                algorithm_key,
                code_path,
                [message],
                profile_name=resolved_profile_key,
            )
        if not stdout:
            return self._failure_report(
                candidate_id,
                algorithm_key,
                code_path,
                ["Sandbox execution produced no validation report."],
                profile_name=resolved_profile_key,
            )

        try:
            return ValidationReport.model_validate_json(stdout)
        except Exception as exc:
            message = stderr or f"Could not parse sandbox validation report: {exc}"
            return self._failure_report(
                candidate_id,
                algorithm_key,
                code_path,
                [message],
                profile_name=resolved_profile_key,
            )

    def _scan_safety(self, source_code: str) -> list[str]:
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return [f"Syntax error: {exc.msg} at line {exc.lineno}"]
        visitor = SafetyVisitor()
        visitor.visit(tree)
        return visitor.errors

    def _infer_profile_key(
        self,
        dataset: RecommendationDataset | ClassificationDataset,
        algorithm_key: str,
    ) -> str:
        if isinstance(dataset, RecommendationDataset):
            return "top_k_recommendation"
        if isinstance(dataset, ClassificationDataset):
            return "tabular_classification"

        try:
            plugin = self.plugin_loader.get_algorithm_plugin(algorithm_key)
        except KeyError:
            return "top_k_recommendation"
        if "tabular_classification" in plugin.task_types:
            return "tabular_classification"
        return "top_k_recommendation"

    def _failure_report(
        self,
        candidate_id: str,
        algorithm_key: str,
        code_path: Path,
        errors: list[str],
        warnings: list[str] | None = None,
        profile_name: str = "",
    ) -> ValidationReport:
        return ValidationReport(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            profile_name=profile_name,
            success=False,
            interface_ok=False,
            stability_ok=False,
            code_path=str(code_path),
            execution_mode="subprocess_sandbox",
            score=0.0,
            metrics=ValidationMetrics(),
            errors=errors,
            warnings=warnings or [],
        )
