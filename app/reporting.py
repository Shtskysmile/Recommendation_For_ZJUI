from __future__ import annotations

import json
from pathlib import Path

from app.models import TaskRecord
from app.settings import AppSettings


class ReportWriter:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def write(self, task: TaskRecord) -> dict[str, str]:
        task_dir = self.settings.reports_dir / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        json_path = task_dir / f"{task.task_id}.json"
        markdown_path = task_dir / f"{task.task_id}.md"

        json_path.write_text(json.dumps(task.model_dump(mode="json"), indent=2), encoding="utf-8")
        markdown_path.write_text(self._to_markdown(task), encoding="utf-8")

        return {"json": str(json_path), "markdown": str(markdown_path)}

    def _to_markdown(self, task: TaskRecord) -> str:
        lines = [
            f"# Validation Report: {task.task_id}",
            "",
            f"- Status: {task.status}",
            f"- Provider: {task.provider_name or 'unknown'}",
            f"- Capability version: {task.capability_version or 'unversioned'}",
            f"- Task: {task.request.task_name}",
            f"- Top-K: {task.request.top_k}",
            "",
            "## Request",
            "",
            task.request.capability_description,
            "",
            "## Parsed Intent",
            "",
            "```json",
            json.dumps(task.parsed_intent.model_dump() if task.parsed_intent else {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Agent Trace",
            "",
            "```json",
            json.dumps([step.model_dump() for step in task.agent_trace], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Candidates",
            "",
        ]

        for candidate in task.candidates:
            report = candidate.validation
            lines.extend(
                [
                    f"### {candidate.display_name} ({candidate.algorithm_key})",
                    "",
                    f"- Candidate ID: {candidate.candidate_id}",
                    f"- Repairs: {candidate.repair_attempts}",
                    f"- Summary: {candidate.plan_summary}",
                    f"- Rationale: {candidate.rationale}",
                    "",
                    "```json",
                    json.dumps(report.model_dump() if report else {}, indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

        if task.best_candidate:
            lines.extend(
                [
                    "## Best Candidate",
                    "",
                    f"- {task.best_candidate.display_name} ({task.best_candidate.algorithm_key})",
                    "",
                ]
            )

        if task.git_context:
            lines.extend(
                [
                    "## Git Context",
                    "",
                    "```json",
                    json.dumps(task.git_context.model_dump(), indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

        if task.repair_history:
            lines.extend(["## Repair History", ""])
            for record in task.repair_history:
                lines.append(
                    f"- {record.candidate_id} attempt {record.attempt}: {record.action} -> {record.outcome}"
                )
            lines.append("")

        return "\n".join(lines)
