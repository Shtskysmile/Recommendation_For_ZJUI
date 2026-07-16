from __future__ import annotations

import json
import sys
from pathlib import Path

from app.models import ValidationMetrics, ValidationReport
from app.plugin_loader import PluginLoader


def _apply_resource_limits(cpu_seconds: float, memory_limit_mb: int) -> None:
    try:
        import resource
    except Exception:  # pragma: no cover - platform specific
        return

    hard_cpu = max(int(cpu_seconds) + 1, 2)
    hard_memory = max(int(memory_limit_mb) * 1024 * 1024, 128 * 1024 * 1024)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))
        resource.setrlimit(resource.RLIMIT_AS, (hard_memory, hard_memory))
    except Exception:  # pragma: no cover - platform specific
        return


def main() -> int:
    manifest_path = Path(sys.argv[1]).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _apply_resource_limits(
        cpu_seconds=float(payload.get("sandbox_timeout_seconds", 20.0)),
        memory_limit_mb=int(payload.get("sandbox_memory_limit_mb", 512)),
    )

    project_root = Path(payload["project_root"]).resolve()
    plugin_loader = PluginLoader(project_root)
    profile_manifest = plugin_loader.get_validation_profile(payload["profile_key"])
    profile = profile_manifest.profile_class()

    try:
        report = profile.validate_candidate_path(
            candidate_id=payload["candidate_id"],
            algorithm_key=payload["algorithm_key"],
            code_path=Path(payload["code_path"]).resolve(),
            dataset_dir=Path(payload["dataset_dir"]).resolve(),
            top_k=int(payload.get("top_k", 5)),
        )
    except Exception as exc:
        report = ValidationReport(
            candidate_id=payload["candidate_id"],
            algorithm_key=payload["algorithm_key"],
            profile_name=payload["profile_key"],
            success=False,
            interface_ok=False,
            stability_ok=False,
            code_path=payload["code_path"],
            execution_mode="subprocess_sandbox",
            score=0.0,
            metrics=ValidationMetrics(),
            errors=[f"Sandbox worker failed: {exc}"],
            warnings=[],
        )

    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
