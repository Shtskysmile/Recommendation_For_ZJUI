from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.settings import AppSettings


@pytest.fixture()
def test_settings(tmp_path: Path) -> AppSettings:
    project_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "sandbox"
    shutil.copytree(project_root / "data", sandbox_root / "data")
    shutil.copytree(project_root / "knowledge", sandbox_root / "knowledge")
    settings = AppSettings(
        project_root=sandbox_root,
        data_dir=sandbox_root / "data",
        knowledge_dir=sandbox_root / "knowledge",
        generated_dir=sandbox_root / "generated",
        reports_dir=sandbox_root / "reports",
        llm_provider="mock",
        openai_model="mock-recommender",
        max_repair_attempts=2,
        top_k_default=5,
    )
    settings.ensure_directories()
    return settings
