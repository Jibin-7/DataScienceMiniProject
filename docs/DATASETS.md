# Dataset Sources and Handling

MediSense uses public, anonymized learning datasets. They are not representative clinical populations and must not be used for care decisions.

| Assessment | Dataset | Acquisition in this project | Target treatment |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes Database, OpenML dataset 37 | `sklearn.datasets.fetch_openml(data_id=37)` | `tested_positive` becomes 1 |
| Heart Disease | UCI Heart Disease, processed Cleveland file | UCI repository download | Values 1-4 become disease presence (1) |
| Parkinson's Disease | UCI Parkinsons | UCI ZIP download | UCI `status` is used directly |
| Breast Cancer | Wisconsin Diagnostic Breast Cancer | `sklearn.datasets.load_breast_cancer` | `malignant` becomes 1 |

## Dataset notes

- **Diabetes:** The Pima dataset has 768 records and eight predictor variables. Values of zero in glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing because those values are physiologically implausible; median imputation happens inside each model pipeline.
- **Heart disease:** Uses the standard 13 predictors from the processed Cleveland dataset. Question-mark values are parsed as missing and median-imputed within the pipeline.
- **Parkinson's:** The UCI file contains voice measurements. `name` is deliberately excluded. The web form uses 13 published acoustic/nonlinear measures from the dataset rather than every available column to keep input collection practical.
- **Breast cancer:** The website uses the ten `mean` measurements from the Wisconsin Diagnostic dataset. Its positive class is recoded to `malignant` for a consistent risk-oriented API contract.

Raw UCI downloads and the OpenML cache are kept under `data/raw/` and ignored by Git. Downloads are written to a temporary `.part` file, ZIP-validated where applicable, then atomically cached, so an interrupted transfer is retried rather than reused.

## Primary sources

- UCI Heart Disease: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- UCI Parkinsons: <https://archive.ics.uci.edu/dataset/174/parkinsons>
- OpenML Pima diabetes: <https://www.openml.org/d/37>
- scikit-learn breast cancer loader: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>
