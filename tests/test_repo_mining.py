from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.knowledge import KnowledgeGraphStore
from app.repo_mining import RepoKnowledgeExtractor


def test_repo_extractor_collects_symbols_and_dependencies(test_settings, tmp_path: Path):
    repo_root = tmp_path / "sample_repo"
    repo_root.mkdir()
    (repo_root / "requirements.txt").write_text("fastapi==0.139.0\npandas==3.0.3\n", encoding="utf-8")
    (repo_root / "README.md").write_text("# Sample Repo\n\nRecommendation helper functions.\n", encoding="utf-8")
    (repo_root / "service.py").write_text(
        "\n".join(
            [
                "import math",
                "",
                "def score(items: list[int]) -> float:",
                '    """Compute a test score."""',
                "    return float(len(items)) / math.sqrt(4)",
                "",
                "class Recommender:",
                "    def fit(self):",
                "        return True",
            ]
        ),
        encoding="utf-8",
    )

    store = KnowledgeGraphStore(test_settings.knowledge_dir)
    extractor = RepoKnowledgeExtractor(test_settings.knowledge_dir)
    result = extractor.extract(repo_root)
    extractor.write_and_ingest(result, store)

    assert result.repo_name == "sample_repo"
    assert "fastapi" in result.dependencies
    assert any(symbol.name == "score" for file_record in result.files for symbol in file_record.symbols)

    exported = json.loads((test_settings.knowledge_dir / "exports" / "knowledge_graph.json").read_text())
    assert any(node["id"].startswith("repo::sample_repo") for node in exported["nodes"])


def test_git_status_handles_initialized_repo(tmp_path: Path):
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / "README.md").write_text("demo\n", encoding="utf-8")

    from app.gitops import GitRepositoryManager

    manager = GitRepositoryManager(repo_root)
    status = manager.status()

    assert status.is_git_repo is True
    assert "README.md" in status.untracked_files
