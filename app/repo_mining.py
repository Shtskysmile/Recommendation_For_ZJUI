from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from app.models import RepoExtractionResult, RepoFileRecord, RepoSymbol


IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".idea"}


class RepoKnowledgeExtractor:
    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        self.output_dir = knowledge_dir / "extracted"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, repo_path: Path) -> RepoExtractionResult:
        resolved_root = repo_path.resolve()
        dependencies = self._extract_dependencies(resolved_root)
        readme_excerpt = self._extract_readme_excerpt(resolved_root)
        files: list[RepoFileRecord] = []

        for python_path in sorted(self._iter_python_files(resolved_root)):
            relative_path = python_path.relative_to(resolved_root).as_posix()
            source_code = python_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                continue
            imports = self._collect_imports(tree)
            symbols = self._collect_symbols(tree)
            files.append(RepoFileRecord(path=relative_path, imports=imports, symbols=symbols))

        return RepoExtractionResult(
            repo_name=resolved_root.name,
            repo_path=str(resolved_root),
            readme_excerpt=readme_excerpt,
            dependencies=dependencies,
            python_file_count=len(files),
            files=files,
        )

    def write_and_ingest(self, result: RepoExtractionResult, store) -> RepoExtractionResult:
        output_path = self.output_dir / f"{result.repo_name}.json"
        serialized = result.model_copy(update={"output_path": str(output_path)})
        output_path.write_text(json.dumps(serialized.model_dump(mode="json"), indent=2), encoding="utf-8")
        store.ingest_repo_extraction(serialized)
        return serialized

    def _iter_python_files(self, repo_root: Path):
        for path in repo_root.rglob("*.py"):
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            yield path

    def _extract_dependencies(self, repo_root: Path) -> list[str]:
        dependencies: list[str] = []
        requirements_path = repo_root / "requirements.txt"
        if requirements_path.exists():
            for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                dependency = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
                if dependency:
                    dependencies.append(dependency)
        return sorted(dict.fromkeys(dependencies))

    def _extract_readme_excerpt(self, repo_root: Path) -> str:
        for filename in ("README.md", "README.rst", "README.txt"):
            readme_path = repo_root / filename
            if readme_path.exists():
                return readme_path.read_text(encoding="utf-8")[:500]
        return ""

    def _collect_imports(self, tree: ast.AST) -> list[str]:
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return sorted(dict.fromkeys(imports))

    def _collect_symbols(self, tree: ast.Module) -> list[RepoSymbol]:
        symbols: list[RepoSymbol] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                symbols.append(
                    RepoSymbol(
                        name=node.name,
                        symbol_type="function",
                        line_no=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    RepoSymbol(
                        name=node.name,
                        symbol_type="class",
                        line_no=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
        return symbols
