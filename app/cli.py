from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from app.models import GenerationRequest
from app.repo_mining import RepoKnowledgeExtractor
from app.settings import AppSettings
from app.workflow import CapabilityFactoryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI capability factory prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI application")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    task_parser = subparsers.add_parser("run-task", help="Run a generation task once")
    task_parser.add_argument("--task-name", default="movie_homepage")
    task_parser.add_argument("--description", required=True)
    task_parser.add_argument("--top-k", type=int, default=5)
    task_parser.add_argument("--constraints", nargs="*", default=["implicit feedback", "local cpu"])
    task_parser.add_argument("--candidate-count", type=int, default=2)

    subparsers.add_parser("export-graph", help="Export the current knowledge graph")

    ingest_parser = subparsers.add_parser("ingest-repo", help="Extract and ingest knowledge from a local repository")
    ingest_parser.add_argument("--repo-path", required=True)

    subparsers.add_parser("git-status", help="Print repository git status")

    snapshot_parser = subparsers.add_parser("git-snapshot", help="Create a git commit snapshot")
    snapshot_parser.add_argument("--message", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = AppSettings.from_env()

    if args.command == "serve":
        uvicorn.run("app.main:create_app", factory=True, host=args.host, port=args.port)
        return 0

    service = CapabilityFactoryService.build(settings)

    if args.command == "run-task":
        result = service.run_task(
            GenerationRequest(
                task_name=args.task_name,
                capability_description=args.description,
                top_k=args.top_k,
                constraints=args.constraints,
                candidate_count=args.candidate_count,
            )
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    if args.command == "export-graph":
        payload = service.knowledge_store.export_payload()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "ingest-repo":
        extractor = RepoKnowledgeExtractor(settings.knowledge_dir)
        result = extractor.extract(Path(args.repo_path))
        stored = extractor.write_and_ingest(result, service.knowledge_store)
        print(json.dumps(stored.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    if args.command == "git-status":
        print(json.dumps(service.git_manager.status().model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    if args.command == "git-snapshot":
        commit_sha = service.git_manager.snapshot(args.message)
        print(json.dumps({"commit_sha": commit_sha}, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
