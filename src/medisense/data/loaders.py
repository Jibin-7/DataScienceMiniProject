from __future__ import annotations

import io
import zipfile
from contextlib import suppress
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

from medisense.config import RAW_DATA_DIR

HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
PARKINSONS_FORM_FEATURES = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
BREAST_FEATURES = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"]


def _download(url: str, destination: Path, validator=None, attempts: int = 3) -> Path:
    """Download atomically so interrupted transfers cannot poison the cache."""
    if destination.exists() and (validator is None or validator(destination)):
        return destination

    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: OSError | None = None
    for _ in range(attempts):
        with suppress(OSError):
            partial.unlink()
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(64 * 1024):
                    output.write(chunk)
            if validator is not None and not validator(partial):
                raise OSError("downloaded file did not pass integrity validation")
            partial.replace(destination)
            return destination
        except OSError as exc:
            last_error = exc
    with suppress(OSError):
        partial.unlink()
    raise RuntimeError(f"Could not download a complete copy of {url} after {attempts} attempts. Check network access, then retry.") from last_error


def load_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_openml(data_id=37, as_frame=True, parser="auto", data_home=RAW_DATA_DIR / "openml")
    frame = data.data.copy()
    frame.columns = [str(c) for c in frame.columns]
    target = data.target.map({"tested_positive": 1, "tested_negative": 0}).astype(int)
    frame.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    return frame, target


def load_heart() -> tuple[pd.DataFrame, pd.Series]:
    path = _download("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", RAW_DATA_DIR / "processed.cleveland.data")
    frame = pd.read_csv(path, names=HEART_COLUMNS, na_values="?")
    target = (pd.to_numeric(frame.pop("target"), errors="coerce") > 0).astype(int)
    return frame.apply(pd.to_numeric, errors="coerce"), target


def load_parkinsons() -> tuple[pd.DataFrame, pd.Series]:
    archive = _download("https://archive.ics.uci.edu/static/public/174/parkinsons.zip", RAW_DATA_DIR / "parkinsons.zip", validator=zipfile.is_zipfile)
    with zipfile.ZipFile(archive) as zf:
        filename = next(name for name in zf.namelist() if name.endswith("parkinsons.data"))
        frame = pd.read_csv(io.BytesIO(zf.read(filename)))
    target = frame.pop("status").astype(int)
    return frame[PARKINSONS_FORM_FEATURES], target


def load_breast_cancer_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    frame = data.frame[BREAST_FEATURES].copy()
    # sklearn labels 0=malignant, 1=benign. Standardize positive class to malignant.
    target = (data.target == 0).astype(int)
    return frame, target


LOADERS = {"diabetes": load_diabetes, "heart": load_heart, "parkinsons": load_parkinsons, "breast_cancer": load_breast_cancer_data}
