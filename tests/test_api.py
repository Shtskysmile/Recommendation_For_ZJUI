import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_index_page_renders(test_settings):
    client = TestClient(create_app(test_settings))
    response = client.get("/")

    assert response.status_code == 200
    assert "Capability Factory" in response.text
    assert "Top-K Recommendation" in response.text


def test_task_api_completes_and_exposes_graph(test_settings):
    client = TestClient(create_app(test_settings))
    response = client.post(
        "/tasks/generate",
        json={
            "task_name": "movie_homepage",
            "capability_description": (
                "Build a Top-K movie recommendation workflow for a homepage feed. "
                "Use implicit feedback, prioritize recall and ndcg, and keep it runnable on local cpu."
            ),
            "top_k": 5,
            "constraints": ["implicit feedback", "local cpu"],
            "candidate_count": 2
        },
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    payload = None
    for _ in range(40):
        task_response = client.get(f"/tasks/{task_id}")
        payload = task_response.json()
        if payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["best_candidate"]["algorithm_key"] in {"itemcf", "popularity"}

    graph_response = client.get("/knowledge/graph")
    assert graph_response.status_code == 200
    graph_payload = graph_response.json()
    assert graph_payload["nodes"]

    layout_response = client.get("/knowledge/graph/layout")
    assert layout_response.status_code == 200
    layout_payload = layout_response.json()
    assert layout_payload["nodes"]
    assert "x" in layout_payload["nodes"][0]


def test_repo_ingestion_api_extracts_repo_knowledge(test_settings, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "requirements.txt").write_text("networkx==3.6.1\n", encoding="utf-8")
    (repo_root / "worker.py").write_text(
        "\n".join(
            [
                "def helper() -> int:",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(test_settings))
    response = client.post("/knowledge/ingest-repo", json={"repo_path": str(repo_root)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_name"] == "repo"
    assert payload["python_file_count"] == 1
