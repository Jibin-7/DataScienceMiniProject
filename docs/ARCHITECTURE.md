# Architecture

```text
Public datasets
      |
      v
Dataset loaders -> dataset-specific cleaning -> reproducible sklearn pipelines
      |                       |                         |
      |                       |                         +-> CV comparison, held-out metrics, model artifacts
      |                       v
      |                 median imputation + scaling
      v
models/<disease>.joblib and models/<disease>.json
      |
      v
FastAPI API -> validation -> probability + measured-model context -> browser UI
      ^                                                        |
      +----------------------- user feedback / correction -----+
```

## Components

- `src/medisense/data/loaders.py` obtains each dataset through a named source.
- `src/medisense/ml/training.py` owns cleaning, candidate comparison, stratified splitting, persistence, and feature plots.
- `src/medisense/ml/inference.py` applies saved pipelines to one validated record.
- `app/main.py` exposes HTML routes plus JSON endpoints.
- `app/templates/` and `app/static/` provide the responsive interface.

The preprocessor belongs inside the pipeline, avoiding leakage from the held-out set into imputation or scaling. Candidate models are Logistic Regression, Random Forest, and probability-calibrated SVM. The selected model is the one with highest mean five-fold CV ROC-AUC on the training split; the final test set is used only once for reported performance.
