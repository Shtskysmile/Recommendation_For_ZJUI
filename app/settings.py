from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.env import load_env_file


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    data_dir: Path
    knowledge_dir: Path
    generated_dir: Path
    reports_dir: Path
    llm_provider: str = "mock"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    max_repair_attempts: int = 2
    top_k_default: int = 10
    request_timeout_seconds: float = 30.0
    beam_width: int = 3
    sandbox_timeout_seconds: float = 20.0
    sandbox_memory_limit_mb: int = 512

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.knowledge_dir,
            self.generated_dir,
            self.reports_dir,
            self.reports_dir / "tasks",
            self.knowledge_dir / "exports",
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AppSettings":
        root = project_root or Path(__file__).resolve().parents[1]
        load_env_file(root)
        api_key = os.getenv("OPENAI_API_KEY", "")
        default_provider = "siliconflow" if api_key else "mock"
        provider = os.getenv("LLM_PROVIDER", default_provider).strip().lower() or default_provider
        default_base_url = "https://api.siliconflow.cn/v1" if provider == "siliconflow" else "https://api.openai.com/v1"
        settings = cls(
            project_root=root,
            data_dir=root / "data",
            knowledge_dir=root / "knowledge",
            generated_dir=root / "generated",
            reports_dir=root / "reports",
            llm_provider=provider,
            openai_base_url=os.getenv("OPENAI_BASE_URL", default_base_url).strip(),
            openai_api_key=api_key.strip(),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3").strip(),
            max_repair_attempts=int(os.getenv("MAX_REPAIR_ATTEMPTS", "2")),
            top_k_default=int(os.getenv("TOP_K_DEFAULT", "10")),
            request_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            beam_width=int(os.getenv("BEAM_WIDTH", "3")),
            sandbox_timeout_seconds=float(os.getenv("SANDBOX_TIMEOUT_SECONDS", "20")),
            sandbox_memory_limit_mb=int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", "512")),
        )
        settings.ensure_directories()
        return settings
