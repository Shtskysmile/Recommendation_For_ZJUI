from __future__ import annotations

import subprocess
from pathlib import Path

from app.models import GitStatus


class GitRepositoryManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def status(self) -> GitStatus:
        if not self._is_git_repo():
            return GitStatus(is_git_repo=False)

        branch = self._run_git("symbolic-ref", "--short", "HEAD", allow_failure=True).strip() or None
        head_commit = self._run_git("rev-parse", "HEAD", allow_failure=True).strip() or None
        porcelain = self._run_git("status", "--porcelain").splitlines()
        staged_files: list[str] = []
        unstaged_files: list[str] = []
        untracked_files: list[str] = []

        for line in porcelain:
            if not line:
                continue
            if line.startswith("?? "):
                untracked_files.append(line[3:])
                continue
            staged_flag, unstaged_flag = line[0], line[1]
            filename = line[3:]
            if staged_flag != " ":
                staged_files.append(filename)
            if unstaged_flag != " ":
                unstaged_files.append(filename)

        return GitStatus(
            is_git_repo=True,
            branch=branch,
            head_commit=head_commit,
            is_dirty=bool(staged_files or unstaged_files or untracked_files),
            staged_files=staged_files,
            unstaged_files=unstaged_files,
            untracked_files=untracked_files,
        )

    def snapshot(self, message: str) -> str:
        self._run_git("add", ".")
        self._run_git("commit", "-m", message)
        return self._run_git("rev-parse", "--short", "HEAD").strip()

    def _is_git_repo(self) -> bool:
        return self._run_git("rev-parse", "--is-inside-work-tree", allow_failure=True).strip() == "true"

    def _run_git(self, *args: str, allow_failure: bool = False) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            if allow_failure:
                return ""
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git command failed")
        return completed.stdout
