from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class GenerationRequest(BaseModel):
    task_name: str
    capability_description: str
    task_type: str | None = None
    dataset_name: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    constraints: list[str] = Field(default_factory=list)
    candidate_count: int = Field(default=2, ge=1, le=4)


class RepoIngestionRequest(BaseModel):
    repo_path: str


class GitSnapshotRequest(BaseModel):
    message: str = Field(min_length=1)


class CapabilitySpec(BaseModel):
    task_name: str
    objective: str
    task_type: str
    dataset_name: str
    top_k: int
    constraints: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    preferred_algorithms: list[str] = Field(default_factory=list)
    data_profile: str


class KnowledgeNode(BaseModel):
    node_id: str
    node_type: str
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)


class RetrievedKnowledge(BaseModel):
    capabilities: list[KnowledgeNode] = Field(default_factory=list)
    algorithms: list[KnowledgeNode] = Field(default_factory=list)
    metrics: list[KnowledgeNode] = Field(default_factory=list)
    constraints: list[KnowledgeNode] = Field(default_factory=list)
    failure_patterns: list[KnowledgeNode] = Field(default_factory=list)


class CandidatePlan(BaseModel):
    algorithm_key: str
    display_name: str
    rationale: str
    plan_summary: str
    search_score: float = 0.0
    design_explanation: str = ""


class SearchTraceStep(BaseModel):
    agent: str
    summary: str
    score: float = 0.0
    state: dict[str, str | int | float | list[str]] = Field(default_factory=dict)


class ValidationMetrics(BaseModel):
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    coverage: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    average_latency_ms: float = 0.0
    peak_allocated_kb: float = 0.0


class ValidationReport(BaseModel):
    candidate_id: str
    algorithm_key: str
    profile_name: str = ""
    success: bool
    interface_ok: bool
    stability_ok: bool
    code_path: str
    execution_mode: str = "in_process"
    score: float
    metrics: ValidationMetrics
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AlgorithmCandidate(BaseModel):
    candidate_id: str
    algorithm_key: str
    display_name: str
    rationale: str
    plan_summary: str
    code_path: str
    source_code: str
    validation: ValidationReport | None = None
    repair_attempts: int = 0


class RepairRecord(BaseModel):
    candidate_id: str
    algorithm_key: str
    attempt: int
    trigger: str
    action: str
    outcome: str


class ValidationSummary(BaseModel):
    successful_candidates: int
    failed_candidates: int
    best_score: float = 0.0
    ranking: list[str] = Field(default_factory=list)


class GitStatus(BaseModel):
    is_git_repo: bool
    branch: str | None = None
    head_commit: str | None = None
    is_dirty: bool = False
    staged_files: list[str] = Field(default_factory=list)
    unstaged_files: list[str] = Field(default_factory=list)
    untracked_files: list[str] = Field(default_factory=list)


class RepoSymbol(BaseModel):
    name: str
    symbol_type: str
    line_no: int
    docstring: str = ""


class RepoFileRecord(BaseModel):
    path: str
    imports: list[str] = Field(default_factory=list)
    symbols: list[RepoSymbol] = Field(default_factory=list)


class RepoExtractionResult(BaseModel):
    repo_name: str
    repo_path: str
    readme_excerpt: str = ""
    dependencies: list[str] = Field(default_factory=list)
    python_file_count: int = 0
    files: list[RepoFileRecord] = Field(default_factory=list)
    output_path: str | None = None


class KnowledgeWriteback(BaseModel):
    validation_node_id: str
    export_json_path: str
    export_graphml_path: str


class TaskRecord(BaseModel):
    task_id: str
    status: str
    request: GenerationRequest
    parsed_intent: CapabilitySpec | None = None
    retrieved_knowledge: RetrievedKnowledge | None = None
    candidates: list[AlgorithmCandidate] = Field(default_factory=list)
    best_candidate: AlgorithmCandidate | None = None
    validation_summary: ValidationSummary | None = None
    repair_history: list[RepairRecord] = Field(default_factory=list)
    agent_trace: list[SearchTraceStep] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(default_factory=dict)
    knowledge_writeback_id: str | None = None
    provider_name: str | None = None
    git_context: GitStatus | None = None
    capability_version: str | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
