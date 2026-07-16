from __future__ import annotations

from app.models import AlgorithmCandidate, CapabilitySpec, CandidatePlan
from app.plugin_loader import PluginLoader
from app.settings import AppSettings


class CandidateGenerator:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.plugin_loader = PluginLoader(settings.project_root)

    def generate_candidate(
        self,
        task_id: str,
        candidate_id: str,
        algorithm_key: str,
        spec: CapabilitySpec,
        plan: CandidatePlan | None = None,
        version_suffix: str = "",
    ) -> AlgorithmCandidate:
        source_code = self.render_template(algorithm_key=algorithm_key, spec=spec, task_id=task_id)
        target_dir = self.settings.generated_dir / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{candidate_id}{version_suffix}.py" if version_suffix else f"{candidate_id}.py"
        code_path = target_dir / filename
        code_path.write_text(source_code, encoding="utf-8")

        metadata = self.plugin_loader.get_algorithm_plugin(algorithm_key)
        return AlgorithmCandidate(
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            display_name=plan.display_name if plan else metadata.display_name,
            rationale=plan.rationale if plan else metadata.rationale,
            plan_summary=plan.plan_summary if plan else metadata.summary,
            code_path=str(code_path),
            source_code=source_code,
        )

    def render_template(self, algorithm_key: str, spec: CapabilitySpec, task_id: str) -> str:
        plugin = self.plugin_loader.get_algorithm_plugin(algorithm_key)
        return plugin.render(spec, task_id)
