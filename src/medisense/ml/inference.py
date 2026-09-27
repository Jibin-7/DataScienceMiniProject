from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from medisense.config import MODELS_DIR
from medisense.specs import get_spec


@lru_cache(maxsize=4)
def load_model(disease: str):
    path = MODELS_DIR / f"{disease}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not trained: {disease}. Run `python scripts/train.py --all` first.")
    return joblib.load(path)


def predict(disease: str, payload: dict[str, float]) -> dict:
    spec = get_spec(disease)
    values = {feature.key: float(payload[feature.key]) for feature in spec.features}
    probability = float(load_model(disease).predict_proba(pd.DataFrame([values]))[0, 1])
    risk_label = "Higher preliminary risk" if probability >= .5 else "Lower preliminary risk"
    metrics_path = MODELS_DIR / f"{disease}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {"disease": disease, "title": spec.title, "positive_probability": round(probability * 100, 1), "risk_label": risk_label, "threshold": 50, "top_global_factors": metrics.get("global_feature_importance", [])[:3], "disclaimer": "This educational machine-learning estimate is not a diagnosis and cannot replace a qualified healthcare professional."}
