from fastapi.testclient import TestClient

from app.main import app


def test_health_check():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_disease_catalog_lists_all_assessments():
    response = TestClient(app).get("/api/v1/diseases")
    assert response.status_code == 200
    assert {item["key"] for item in response.json()} == {"diabetes", "heart", "parkinsons", "breast_cancer"}


def test_trained_breast_cancer_model_returns_an_educational_result():
    response = TestClient(app).post("/api/v1/predict/breast_cancer", json={
        "mean radius": 14, "mean texture": 20, "mean perimeter": 90, "mean area": 600,
        "mean smoothness": .1, "mean compactness": .12, "mean concavity": .08,
        "mean concave points": .04, "mean symmetry": .18, "mean fractal dimension": .06,
    })
    assert response.status_code == 200
    assert "not a diagnosis" in response.json()["disclaimer"]
