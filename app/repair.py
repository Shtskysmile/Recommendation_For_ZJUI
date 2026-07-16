from __future__ import annotations

from app.codegen import CandidateGenerator
from app.models import AlgorithmCandidate, CapabilitySpec, ValidationReport
from app.settings import AppSettings


class RepairAgent:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.generator = CandidateGenerator(settings)

    def repair(
        self,
        task_id: str,
        candidate_id: str,
        algorithm_key: str,
        spec: CapabilitySpec,
        last_report: ValidationReport,
        attempt: int,
    ) -> AlgorithmCandidate:
        repaired_candidate = self.generator.generate_candidate(
            task_id=task_id,
            candidate_id=candidate_id,
            algorithm_key=algorithm_key,
            spec=spec,
            version_suffix=f"_repair_{attempt}",
        )
        return repaired_candidate.model_copy(
            update={
                "rationale": f"Template repair after validation failure: {'; '.join(last_report.errors[:2])}",
                "plan_summary": "Reset candidate to a known-safe template and re-run validation.",
            }
        )
