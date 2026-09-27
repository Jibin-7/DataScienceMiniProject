import json

from fastapi.testclient import TestClient

from app import main
from app.main import app


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    return TestClient(app)


def test_feedback_accepts_valid_rating_and_appends_anonymously(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/feedback", json={"disease": "diabetes", "rating": 5, "comment": "  clear result  "})
    assert response.status_code == 200
    assert response.json()["message"] == "Thank you for your feedback!"
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip())
    assert record["disease"] == "diabetes"
    assert record["rating"] == 5
    assert record["comment"] == "clear result"
    assert "submitted_at_utc" in record
    assert set(record) == {"disease", "rating", "comment", "submitted_at_utc"}


def test_feedback_comment_is_optional(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/feedback", json={"disease": "heart", "rating": 1})
    assert response.status_code == 200
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip())
    assert record["comment"] is None


def test_feedback_rejects_rating_out_of_bounds(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    for rating in (0, 6):
        response = client.post("/api/v1/feedback", json={"disease": "parkinsons", "rating": rating})
        assert response.status_code == 422


def test_feedback_rejects_unknown_disease(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/feedback", json={"disease": "migraine", "rating": 3})
    assert response.status_code == 404


def test_feedback_rejects_unknown_fields(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/feedback", json={"disease": "breast_cancer", "rating": 3, "ip": "1.2.3.4"})
    assert response.status_code == 422
