from __future__ import annotations

from pathlib import Path

from app.settings import AppSettings


def test_settings_load_dotenv_file(monkeypatch, tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://example.com/v1",
                "OPENAI_MODEL=test-model",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)

    settings = AppSettings.from_env(root)

    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == "https://example.com/v1"
    assert settings.openai_model == "test-model"
