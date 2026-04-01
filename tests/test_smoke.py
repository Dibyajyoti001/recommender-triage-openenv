from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_reset_step_state_smoke():
    reset = client.post("/reset", params={"task_id": "task_1"})
    assert reset.status_code == 200
    obs = reset.json()
    assert "candidate_items" in obs
    assert len(obs["candidate_items"]) == 6

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