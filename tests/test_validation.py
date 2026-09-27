from fastapi.testclient import TestClient

from app.main import app


def test_api_rejects_out_of_range_input_before_model_lookup():
    response = TestClient(app).post("/api/v1/predict/diabetes", json={
        "Pregnancies": 1, "Glucose": 999, "BloodPressure": 80, "SkinThickness": 20,
        "Insulin": 80, "BMI": 28, "DiabetesPedigreeFunction": .3, "Age": 30,
    })
    assert response.status_code == 422
