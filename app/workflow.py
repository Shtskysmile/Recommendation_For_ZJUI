from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.codegen import CandidateGenerator
from app.gitops import GitRepositoryManager
from app.knowledge import KnowledgeGraphStore
from app.llm import LLMProvider, create_provider
from app.models import (
    AlgorithmCandidate,
    GenerationRequest,
    RepairRecord,
    SearchTraceStep,
    TaskRecord,
    ValidationReport,
    ValidationSummary,
    utc_now_iso,
)
from app.plugin_loader import PluginLoader
from app.repair import RepairAgent
from app.reporting import ReportWriter
from app.search import BeamSearchPlanner
from app.settings import AppSettings
from app.validation import CandidateValidator


class CapabilityFactoryService:
    def __init__(
        self,
        settings: AppSettings,
        knowledge_store: KnowledgeGraphStore,
        llm_provider: LLMProvider,
        generator: CandidateGenerator,
        validator: CandidateValidator,
        repair_agent: RepairAgent,
        report_writer: ReportWriter,
        search_planner: BeamSearchPlanner,
        git_manager: GitRepositoryManager,
        plugin_loader: PluginLoader,
    ):
        self.settings = settings
        self.knowledge_store = knowledge_store
        self.llm_provider = llm_provider
        self.generator = generator
        self.validator = validator
        self.repair_agent = repair_agent
        self.report_writer = report_writer
        self.search_planner = search_planner
        self.git_manager = git_manager
        self.plugin_loader = plugin_loader

    @property
    def provider_name(self) -> str:
        return self.llm_provider.name

    @classmethod
    def build(cls, settings: AppSettings) -> "CapabilityFactoryService":
        settings.ensure_directories()
        plugin_loader = PluginLoader(settings.project_root)
        return cls(
            settings=settings,
            knowledge_store=KnowledgeGraphStore(settings.knowledge_dir),
            llm_provider=create_provider(settings),
            generator=CandidateGenerator(settings),
            validator=CandidateValidator(settings, plugin_loader),
            repair_agent=RepairAgent(settings),
            report_writer=ReportWriter(settings),
            search_planner=BeamSearchPlanner(plugin_loader, beam_width=settings.beam_width),
            git_manager=GitRepositoryManager(settings.project_root),
            plugin_loader=plugin_loader,
        )

    def make_task_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"task-{timestamp}-{uuid4().hex[:8]}"

    def run_task(self, request: GenerationRequest, task_id: str | None = None) -> TaskRecord:
        resolved_task_id = task_id or self.make_task_id()
        git_context = self.git_manager.status()
        capability_version = git_context.head_commit[:8] if git_context.head_commit else f"draft-{resolved_task_id}"
        parsed_intent = self.llm_provider.parse_capability(request)
        profile_manifest = self.plugin_loader.get_profile_for_task_type(parsed_intent.task_type)
        profile = profile_manifest.profile_class()
        dataset = profile.load_dataset(self.settings, parsed_intent)
        retrieved_knowledge = self.knowledge_store.retrieve(parsed_intent)
        search_result = self.search_planner.search(
            spec=parsed_intent,
            knowledge=retrieved_knowledge,
            candidate_count=request.candidate_count,
        )
        agent_trace = [
            SearchTraceStep(
                agent="IntentParser",
                summary="Parsed natural-language capability into a structured task specification.",
                score=0.0,
                state={
                    "task_name": parsed_intent.task_name,
                    "task_type": parsed_intent.task_type,
                    "dataset_name": parsed_intent.dataset_name,
                },
            ),
            SearchTraceStep(
                agent="KnowledgeRetriever",
                summary="Retrieved graph-backed capability knowledge and failure patterns.",
                score=float(len(retrieved_knowledge.algorithms)),
                state={
                    "algorithms": [node.node_id for node in retrieved_knowledge.algorithms],
                    "metrics": [node.node_id for node in retrieved_knowledge.metrics],
                    "profile_key": profile_manifest.key,
                },
            ),
            *search_result.trace,
        ]

        candidates: list[AlgorithmCandidate] = []
        repair_history: list[RepairRecord] = []

        for index, algorithm_key in enumerate(search_result.selected_algorithms, start=1):
            plan = self.llm_provider.explain_algorithm(
                spec=parsed_intent,
                knowledge=retrieved_knowledge,
                algorithm_key=algorithm_key,
                search_score=search_result.scores.get(algorithm_key, 0.0),
            )
            candidate_id = f"{plan.algorithm_key}-{index}"
            candidate = self.generator.generate_candidate(
                task_id=resolved_task_id,
                candidate_id=candidate_id,
                algorithm_key=plan.algorithm_key,
                spec=parsed_intent,
                plan=plan,
            )
            report = self.validator.validate(
                candidate,
                dataset,
                top_k=request.top_k,
                profile_key=profile_manifest.key,
            )
            attempts = 0

            while not report.success and attempts < self.settings.max_repair_attempts:
                attempts += 1
                repaired_candidate = self.repair_agent.repair(
                    task_id=resolved_task_id,
                    candidate_id=candidate_id,
                    algorithm_key=plan.algorithm_key,
                    spec=parsed_intent,
                    last_report=report,
                    attempt=attempts,
                )
                repair_history.append(
                    RepairRecord(
                        candidate_id=candidate_id,
                        algorithm_key=plan.algorithm_key,
                        attempt=attempts,
                        trigger="; ".join(report.errors[:2]) or "validation failure",
                        action="regenerate from safe template",
                        outcome="re-validated",
                    )
                )
                candidate = repaired_candidate.model_copy(
                    update={
                        "display_name": plan.display_name,
                        "rationale": plan.rationale,
                        "plan_summary": plan.plan_summary,
                    }
                )
                report = self.validator.validate(
                    candidate,
                    dataset,
                    top_k=request.top_k,
                    profile_key=profile_manifest.key,
                )

            candidates.append(
                candidate.model_copy(
                    update={
                        "validation": report,
                        "repair_attempts": attempts,
                    }
                )
            )

        best_candidate = self._select_best_candidate(candidates)
        validation_summary = self._build_summary(candidates)
        reports = [candidate.validation for candidate in candidates if candidate.validation]
        writeback = self.knowledge_store.write_validation_result(
            task_id=resolved_task_id,
            spec=parsed_intent,
            reports=[report for report in reports if report is not None],
            best_candidate_id=best_candidate.candidate_id if best_candidate else "none",
            capability_version=capability_version,
        )

        task = TaskRecord(
            task_id=resolved_task_id,
            status="completed" if best_candidate else "failed",
            request=request,
            parsed_intent=parsed_intent,
            retrieved_knowledge=retrieved_knowledge,
            candidates=candidates,
            best_candidate=best_candidate,
            validation_summary=validation_summary,
            repair_history=repair_history,
            agent_trace=agent_trace,
            knowledge_writeback_id=writeback.validation_node_id,
            provider_name=self.provider_name,
            git_context=git_context,
            capability_version=capability_version,
            updated_at=utc_now_iso(),
        )
        initial_paths = self.report_writer.write(task)
        task = task.model_copy(update={"report_paths": initial_paths, "updated_at": utc_now_iso()})
        self.report_writer.write(task)
        return task

    def _select_best_candidate(self, candidates: list[AlgorithmCandidate]) -> AlgorithmCandidate | None:
        successful_candidates = [
            candidate for candidate in candidates if candidate.validation is not None and candidate.validation.success
        ]
        if not successful_candidates:
            return None
        return sorted(
            successful_candidates,
            key=lambda candidate: (-candidate.validation.score, candidate.algorithm_key),
        )[0]

    def _build_summary(self, candidates: list[AlgorithmCandidate]) -> ValidationSummary:
        reports: list[ValidationReport] = [candidate.validation for candidate in candidates if candidate.validation]
        ranking = [
            report.candidate_id
            for report in sorted(reports, key=lambda report: (-report.score, report.candidate_id))
        ]
        best_score = reports and max(report.score for report in reports) or 0.0
        successful_candidates = sum(1 for report in reports if report.success)
        return ValidationSummary(
            successful_candidates=successful_candidates,
            failed_candidates=max(len(candidates) - successful_candidates, 0),
            best_score=best_score,
            ranking=ranking,
        )
