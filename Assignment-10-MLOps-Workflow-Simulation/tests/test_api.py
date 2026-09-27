from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

NORMAL_READING = {
    "product_type": "M",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 20,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert "test_f1" in response.json()["metrics"]


def test_predict_normal_reading():
    response = client.post("/predict", json=NORMAL_READING)
    body = response.json()

    assert response.status_code == 200
    assert body["failure"] is False
    assert 0 <= body["failure_probability"] < 0.5


def test_predict_power_overload():
    response = client.post("/predict", json={**NORMAL_READING, "rotational_speed": 1300, "torque": 80.0})

    assert response.status_code == 200
    assert response.json()["failure"] is True


def test_invalid_input_is_rejected():
    response = client.post("/predict", json={**NORMAL_READING, "product_type": "X", "torque": -1})

    assert response.status_code == 422
