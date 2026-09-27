# MediSense: Multiple Disease Prediction System Using Machine Learning

MediSense is a complete mini-project that implements a unified, web-based **preliminary risk assessment** workflow for **Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer**. It follows the presentation pipeline: data collection -> preprocessing -> feature analysis -> model training -> trained disease models -> web application -> user feedback.

> **Medical safety:** This is an educational project, not a medical device. It does not diagnose, treat, or exclude disease. Dataset-specific predictions may be inaccurate, biased, or unsuitable for an individual. Always seek a qualified clinician for symptoms, screening, diagnosis, or treatment decisions.

## What is included

- Public-source dataset adapters for all four required conditions.
- Data cleaning: missing-value parsing, duplicate removal, diabetes inconsistency handling, median imputation, and standardization inside leakage-safe pipelines.
- Candidate model comparison: Logistic Regression, Random Forest, and Support Vector Machine.
- Reproducible stratified train/test split, five-fold cross-validation, model selection by CV ROC-AUC, and saved held-out metrics.
- Persisted `joblib` pipelines, JSON metrics, and feature-importance charts.
- FastAPI backend with a health check, disease catalog, strict input fields, range validation, and prediction endpoints.
- Responsive browser interface covering all four prediction flows and explaining the safety boundary.
- Tests and concise technical documentation.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts/train.py --all
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The first training run requires internet access to obtain public data. Raw UCI files and the OpenML cache are stored under `data/raw/`; model files and metrics are written to `models/`; feature charts are written to `reports/`.

## Train and evaluate

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/train.py --disease diabetes
python scripts/train.py --all
pytest -q
```

For every condition, `models/<disease>.json` contains selected model name, train CV comparison, held-out accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix, feature list, training timestamp, and global feature importance. Report **those generated values**, rather than a universal accuracy claim, in the mini-project report.

## API

- `GET /health` - service readiness.
- `GET /api/v1/diseases` - disease catalog and expected feature fields.
- `POST /api/v1/predict/{disease}` - risk estimate for `diabetes`, `heart`, `parkinsons`, or `breast_cancer`.
- `POST /api/v1/feedback` - anonymous helpfulness rating (1-5) plus optional comment for an assessed disease; appended to `data/feedback/feedback.jsonl`.

All API input must contain each documented numeric field and no unknown fields. A missing model returns HTTP 503 with the training command needed to create it. Inputs outside the browser's documented ranges are stopped on the client; the API schema rejects absent and unknown values.

## Deploy to Render

This repository includes `render.yaml` for a FastAPI web service deployment. The build installs dependencies, downloads the four public datasets, trains all model artifacts, and starts the application. It can take several minutes because training is deliberate and reproducible.

1. Push this project to a GitHub repository.
2. In Render, select **New +** -> **Blueprint** and connect that GitHub repository.
3. Render detects `render.yaml`; approve the `medisense-risk-assessment` service and deploy.
4. Use the generated `https://...onrender.com` address as the live project link.

The free Render plan may spin down after inactivity. Deploying a health-risk educational tool publicly does not make it clinically validated; retain the in-app disclaimer and do not collect real patient data.

## Evaluation and limitations

The project deliberately uses a separate test set and cross-validation, but performance is still limited by small, historic, disease-specific datasets. It does not establish clinical validity, causation, fairness, calibration in real populations, or suitability for triage. The data may have demographic, collection, and label biases. Parkinson's assessment is especially constrained by its voice-recording dataset; breast-cancer fields require specialist measurements. Probability values are model outputs, not personal diagnoses or a prescribed action.

Read [dataset provenance and preprocessing](docs/DATASETS.md) and [architecture and methodology](docs/ARCHITECTURE.md) before presenting the project.

## Project layout

```text
app/                    FastAPI service, templates, responsive frontend
src/medisense/data/     Dataset acquisition adapters
src/medisense/ml/       Cleaning, training, metrics, inference
scripts/train.py        Reproducible training entry point
models/                 Generated pipelines and metrics (ignored by Git)
reports/                Generated feature-analysis charts (ignored by Git)
tests/                  Unit and API smoke tests
docs/                   Dataset and architecture documentation
```
