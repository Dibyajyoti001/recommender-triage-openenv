from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_runtime_metadata_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    metadata = client.get("/metadata")
    assert metadata.status_code == 200
    metadata_json = metadata.json()
    assert isinstance(metadata_json["name"], str)
    assert isinstance(metadata_json["description"], str)
    assert isinstance(metadata_json["version"], str)

    schema = client.get("/schema")
    assert schema.status_code == 200
    schema_json = schema.json()
    assert "action" in schema_json
    assert "observation" in schema_json
    assert "state" in schema_json

    mcp = client.post("/mcp", json={})
    assert mcp.status_code == 200
    assert mcp.json()["jsonrpc"] == "2.0"


def test_reset_step_state_smoke():
    reset = client.post("/reset", params={"task_id": "task_1"})
    assert reset.status_code == 200
    obs = reset.json()
    assert "candidate_items" in obs
    assert len(obs["candidate_items"]) == 6
    assert "trust_signal" in obs
    assert "feedback_volatility" in obs
    assert "budget_remaining" in obs
    assert "risk_tolerance" in obs
    assert "latency_budget" in obs

    first_item = obs["candidate_items"][0]["item_id"]
    step = client.post(
        "/step",
        json={
            "recommended_item_id": first_item,
            "exploration_flag": False,
            "confidence_score": 0.8,
        },
    )
    assert step.status_code == 200
    payload = step.json()
    assert "observation" in payload
    assert "reward" in payload
    assert "done" in payload

    state = client.get("/state")
    assert state.status_code == 200


def test_visualize_endpoint_returns_regime_series():
    reset = client.post("/reset", params={"task_id": "task_4", "session_id": "viz"})
    assert reset.status_code == 200
    obs = reset.json()

    for _ in range(3):
        first_item = obs["candidate_items"][0]["item_id"]
        step = client.post(
            "/step",
            params={"session_id": "viz"},
            json={
                "recommended_item_id": first_item,
                "exploration_flag": False,
                "confidence_score": 0.8,
            },
        )
        assert step.status_code == 200
        obs = step.json()["observation"]

    viz = client.get("/visualize", params={"session_id": "viz"})
    assert viz.status_code == 200
    payload = viz.json()
    assert payload["task_id"] == "task_4"
    assert len(payload["turns"]) == len(payload["regimes"]) == len(payload["latent_vol"])
    assert payload["regime_labels"]
