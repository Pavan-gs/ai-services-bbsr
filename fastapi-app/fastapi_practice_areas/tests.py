from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_create_practice_area():
    response = client.post(
        "/practice-areas",
        json={"name": "Data Engineering", "floor_number": 3},
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Data Engineering"
