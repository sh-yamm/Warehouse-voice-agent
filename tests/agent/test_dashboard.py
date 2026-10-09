from fastapi.testclient import TestClient

from voiceagent.agent.dashboard import create_app, nearest_rank
from voiceagent.db.repository import Repository

from conftest import NOW, build_world


def make_db(tmp_path):
    path = tmp_path / "w.db"
    repo = Repository(path)
    repo.init_schema()
    build_world(repo, NOW)
    call_id = repo.start_call(1, NOW)
    for kind, ms in [("greeting", 900.0), ("response", 700.0), ("response", 1100.0), ("barge_in", 150.0)]:
        repo.add_turn_metric(call_id, kind, ms, {"contributions": [["llm", "LLM inference", 300.0]]}, NOW)
    repo.finish_call(call_id, NOW, "scheduled", [{"role": "user", "content": "Yes."}])
    repo.close()
    return path


def test_nearest_rank():
    assert nearest_rank([700.0, 1100.0], 50) == 700.0
    assert nearest_rank([700.0, 1100.0], 95) == 1100.0
    assert nearest_rank([], 50) is None


def test_api_calls_summary(tmp_path):
    client = TestClient(create_app(make_db(tmp_path)))
    [call] = client.get("/api/calls").json()
    assert call["customer"] == "Priya Sharma" and call["outcome"] == "scheduled"
    assert (call["responses"], call["p50_ms"], call["p95_ms"]) == (2, 700.0, 1100.0)
    assert call["greeting_ms"] == 900.0 and call["barge_in_p50_ms"] == 150.0


def test_api_call_detail_and_404(tmp_path):
    client = TestClient(create_app(make_db(tmp_path)))
    detail = client.get("/api/calls/1").json()
    assert len(detail["metrics"]) == 4 and detail["transcript"] == [{"role": "user", "content": "Yes."}]
    assert client.get("/api/calls/99").status_code == 404


def test_index_page(tmp_path):
    response = TestClient(create_app(make_db(tmp_path))).get("/")
    assert response.status_code == 200 and "Live calls" in response.text
