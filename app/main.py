from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import GenerationRequest, GitSnapshotRequest, RepoIngestionRequest
from app.repo_mining import RepoKnowledgeExtractor
from app.settings import AppSettings
from app.tasks import TaskManager
from app.workflow import CapabilityFactoryService


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or AppSettings.from_env()
    service = CapabilityFactoryService.build(resolved_settings)
    manager = TaskManager(service)

    package_root = Path(__file__).resolve().parent
    static_dir = package_root / "ui" / "static"
    template_dir = package_root / "ui" / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    app = FastAPI(title="AI Capability Factory", version="0.1.0")
    app.state.settings = resolved_settings
    app.state.service = service
    app.state.task_manager = manager
    app.state.repo_extractor = RepoKnowledgeExtractor(resolved_settings.knowledge_dir)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "provider_name": service.provider_name,
                "default_top_k": resolved_settings.top_k_default,
                "default_description": (
                    "Build a Top-K movie recommendation workflow for a homepage feed. "
                    "Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu."
                ),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "provider": service.provider_name}

    @app.post("/tasks/generate", status_code=202)
    async def generate_task(payload: GenerationRequest) -> JSONResponse:
        task = manager.submit(payload)
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task.task_id,
                "status": task.status,
                "provider_name": service.provider_name,
            },
        )

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> JSONResponse:
        task = manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return JSONResponse(task.model_dump(mode="json"))

    @app.get("/knowledge/graph")
    async def get_knowledge_graph() -> JSONResponse:
        return JSONResponse(service.knowledge_store.export_payload())

    @app.get("/knowledge/graph/layout")
    async def get_knowledge_graph_layout() -> JSONResponse:
        return JSONResponse(service.knowledge_store.layout_payload())

    @app.post("/knowledge/ingest-repo")
    async def ingest_repo(payload: RepoIngestionRequest) -> JSONResponse:
        repo_path = Path(payload.repo_path).expanduser().resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            raise HTTPException(status_code=400, detail="repo_path must point to an existing directory")
        result = app.state.repo_extractor.extract(repo_path)
        stored = app.state.repo_extractor.write_and_ingest(result, service.knowledge_store)
        return JSONResponse(stored.model_dump(mode="json"))

    @app.get("/git/status")
    async def get_git_status() -> JSONResponse:
        return JSONResponse(service.git_manager.status().model_dump(mode="json"))

    @app.post("/git/snapshot")
    async def create_git_snapshot(payload: GitSnapshotRequest) -> JSONResponse:
        commit_sha = service.git_manager.snapshot(payload.message)
        return JSONResponse({"commit_sha": commit_sha})

    return app


app = create_app()
