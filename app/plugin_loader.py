from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path


def _load_object(import_path: str):
    module_name, object_name = import_path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, object_name)


@dataclass(frozen=True, slots=True)
class AlgorithmPlugin:
    key: str
    display_name: str
    summary: str
    rationale: str
    family: str
    strengths: tuple[str, ...]
    search_hints: tuple[str, ...]
    task_types: tuple[str, ...]
    renderer_path: str

    @property
    def render(self):
        return _load_object(self.renderer_path)


@dataclass(frozen=True, slots=True)
class ValidationProfilePlugin:
    key: str
    display_name: str
    description: str
    task_types: tuple[str, ...]
    class_path: str

    @property
    def profile_class(self):
        return _load_object(self.class_path)


class PluginLoader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.plugin_root = self._resolve_plugin_root(project_root)

    def _resolve_plugin_root(self, project_root: Path) -> Path:
        candidate = project_root / "app" / "plugins"
        if candidate.exists():
            return candidate
        return Path(__file__).resolve().parent / "plugins"

    def list_algorithm_plugins(self, task_type: str | None = None) -> list[AlgorithmPlugin]:
        plugins = list(_load_algorithm_plugins(self.plugin_root / "algorithms").values())
        if task_type is None:
            return plugins
        return [plugin for plugin in plugins if task_type in plugin.task_types]

    def get_algorithm_plugin(self, key: str) -> AlgorithmPlugin:
        return _load_algorithm_plugins(self.plugin_root / "algorithms")[key]

    def list_validation_profiles(self) -> list[ValidationProfilePlugin]:
        return list(_load_validation_profiles(self.plugin_root / "profiles").values())

    def get_validation_profile(self, key: str) -> ValidationProfilePlugin:
        return _load_validation_profiles(self.plugin_root / "profiles")[key]

    def get_profile_for_task_type(self, task_type: str) -> ValidationProfilePlugin:
        for plugin in self.list_validation_profiles():
            if task_type in plugin.task_types:
                return plugin
        raise KeyError(f"No validation profile registered for task_type={task_type}")


@lru_cache(maxsize=1)
def _load_algorithm_plugins(directory: Path) -> dict[str, AlgorithmPlugin]:
    plugins: dict[str, AlgorithmPlugin] = {}
    for manifest_path in sorted(directory.glob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        plugin = AlgorithmPlugin(
            key=payload["key"],
            display_name=payload["display_name"],
            summary=payload["summary"],
            rationale=payload["rationale"],
            family=payload["family"],
            strengths=tuple(payload["strengths"]),
            search_hints=tuple(payload["search_hints"]),
            task_types=tuple(payload["task_types"]),
            renderer_path=payload["renderer_path"],
        )
        plugins[plugin.key] = plugin
    return plugins


@lru_cache(maxsize=1)
def _load_validation_profiles(directory: Path) -> dict[str, ValidationProfilePlugin]:
    profiles: dict[str, ValidationProfilePlugin] = {}
    for manifest_path in sorted(directory.glob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        plugin = ValidationProfilePlugin(
            key=payload["key"],
            display_name=payload["display_name"],
            description=payload["description"],
            task_types=tuple(payload["task_types"]),
            class_path=payload["class_path"],
        )
        profiles[plugin.key] = plugin
    return profiles
