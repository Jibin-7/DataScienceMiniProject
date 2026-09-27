from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import matplotlib

# Training only writes report images; avoid Tkinter GUI state in worker processes on Windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from medisense.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from medisense.data.loaders import LOADERS


def build_pipeline(model: object, features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)], remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def clean_data(disease: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().drop_duplicates()
    if disease == "diabetes":
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        frame[zero_as_missing] = frame[zero_as_missing].replace(0, np.nan)
    return frame


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "support_vector_machine": CalibratedClassifierCV(SVC(class_weight="balanced", random_state=RANDOM_STATE), cv=5),
    }


def train_disease(disease: str) -> dict:
    frame, target = LOADERS[disease]()
    frame = clean_data(disease, frame)
    target = target.loc[frame.index].reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    features = list(frame.columns)
    X_train, X_test, y_train, y_test = train_test_split(frame, target, test_size=.2, stratify=target, random_state=RANDOM_STATE)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models().items():
        pipeline = build_pipeline(model, features)
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"})
        comparison[name] = {metric: round(float(np.mean(values)), 4) for metric, values in scores.items() if metric.startswith("test_")}
        fitted[name] = pipeline.fit(X_train, y_train)
    selected_name = max(comparison, key=lambda n: comparison[n]["test_roc_auc"])
    selected = fitted[selected_name]
    probability = selected.predict_proba(X_test)[:, 1]
    prediction = (probability >= .5).astype(int)
    metrics = {
        "selected_model": selected_name,
        "selection_metric": "mean 5-fold CV ROC-AUC on the training split",
        "test": {"accuracy": round(float(accuracy_score(y_test, prediction)), 4), "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4), "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 4), "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 4), "roc_auc": round(float(roc_auc_score(y_test, probability)), 4), "average_precision": round(float(average_precision_score(y_test, probability)), 4), "confusion_matrix": confusion_matrix(y_test, prediction).tolist()},
        "cross_validation": comparison,
        "dataset_rows": len(frame), "features": features,
        "trained_at_utc": datetime.now(UTC).isoformat(), "random_state": RANDOM_STATE,
    }
    # Permutation-free global explanation where supported by the selected estimator.
    estimator = selected.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        importances = np.zeros(len(features))
    metrics["global_feature_importance"] = [{"feature": f, "importance": round(float(v), 5)} for f, v in sorted(zip(features, importances), key=lambda item: item[1], reverse=True)]
    joblib.dump(selected, MODELS_DIR / f"{disease}.joblib")
    (MODELS_DIR / f"{disease}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_importance_plot(disease, metrics["global_feature_importance"])
    return metrics


def _save_importance_plot(disease: str, values: list[dict]) -> None:
    top = values[:10][::-1]
    plt.figure(figsize=(8, 4.5))
    plt.barh([item["feature"] for item in top], [item["importance"] for item in top], color="#137a72")
    plt.xlabel("Relative importance / absolute coefficient")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{disease}_feature_importance.png", dpi=150)
    plt.close()
