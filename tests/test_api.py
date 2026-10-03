from fastapi.testclient import TestClient

from src.api import app, get_params


def test_optimize_endpoint(params):
    app.dependency_overrides[get_params] = lambda: params
    client = TestClient(app)
    r = client.post("/optimize", json={"budget": 300, "weeks": 4, "slots_per_week": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total_spend"] <= 300
    assert {"product", "depth", "week"} <= set(body["plan"][0])
    app.dependency_overrides.clear()


def test_rejects_bad_input():
    assert TestClient(app).post("/optimize", json={"budget": -5}).status_code == 422
