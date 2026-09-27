from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from medisense.ml.inference import predict
from medisense.specs import SPECS, get_spec

app = FastAPI(title="MediSense", version="1.0.0", description="Educational multiple-disease risk assessment using machine learning.")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")


def _input_model(disease: str) -> type[BaseModel]:
    spec = get_spec(disease)
    fields = {feature.key: (float, Field(..., ge=feature.minimum, le=feature.maximum)) for feature in spec.features}
    return create_model(f"{spec.title.replace(' ', '')}Input", __config__=ConfigDict(extra="forbid"), **fields)


FEEDBACK_LOG = ROOT / "data" / "feedback" / "feedback.jsonl"


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(None, max_length=1000)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"specs": SPECS.values()})


@app.get("/assessment/{disease}", response_class=HTMLResponse)
def assessment(request: Request, disease: str):
    try:
        spec = get_spec(disease)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return templates.TemplateResponse(request, "assessment.html", {"spec": spec, "specs": SPECS.values()})


@app.get("/api/v1/diseases")
def diseases():
    return [{"key": spec.key, "title": spec.title, "dataset": spec.dataset, "features": [feature.__dict__ for feature in spec.features]} for spec in SPECS.values()]


@app.post("/api/v1/predict/{disease}")
def assessment_api(disease: str, values: dict):
    try:
        schema = _input_model(disease)
        valid = schema.model_validate(values)
        return predict(disease, valid.model_dump())
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Unable to assess these values: {exc}") from exc


@app.post("/api/v1/feedback")
def feedback_api(payload: FeedbackInput):
    if payload.disease not in SPECS:
        raise HTTPException(404, f"Unknown disease: {payload.disease}")
    record = {
        "disease": payload.disease,
        "rating": payload.rating,
        "comment": (payload.comment or "").strip() or None,
        "submitted_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise HTTPException(500, "Could not store feedback.") from exc
    return {"status": "received", "message": "Thank you for your feedback!"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "MediSense"}
