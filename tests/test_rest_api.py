import pytest
from fastapi.testclient import TestClient
from predict_rest_api import app

client = TestClient(app)


def test_predict_success(valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["code_text"] == "ok"
    assert "math_score" in data["data"]


def test_predict_validation_error(invalid_payload):
    response = client.post("/predict", json=invalid_payload)

    # Assertions
    assert response.status_code == 400  # Assuming 400 is used for validation errors
    data = response.json()["detail"]  # Access the `detail` field

    assert data["code"] == -1
    assert data["code_text"] == "error"
    assert "Validation error occurred." in data["message"]
    assert "errors" in data
    assert len(data["errors"]) > 0
    assert data["errors"][0]["field"] == "reading_score"
    assert "Input should be a valid number" in data["errors"][0]["error"]


def test_predict_missing_payload():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data  # Standard FastAPI validation error response


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "FastAPI Prediction Service is running"


@pytest.fixture
def valid_payload():
    return {
        "payload": {
            "data": {
                "gender": "male",
                "race_ethnicity": "group A",
                "parental_level_of_education": "high school",
                "lunch": "standard",
                "test_preparation_course": "none",
                "reading_score": 72.0,
                "writing_score": 74.0,
            }
        }
    }


@pytest.fixture
def invalid_payload():
    return {
        "payload": {
            "data": {
                "gender": "male",
                "race_ethnicity": "group A",
                "parental_level_of_education": "high school",
                "lunch": "standard",
                "test_preparation_course": "none",
                "reading_score": "",
                # Missing "writing_score"
            }
        }
    }
