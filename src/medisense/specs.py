from __future__ import annotations

from .schemas import DiseaseSpec, Feature


SPECS = {
    "diabetes": DiseaseSpec("diabetes", "Diabetes", "Pima Indians Diabetes (OpenML 37)", "Positive diabetes test", (
        Feature("Pregnancies", "Pregnancies", "count", 0, 20), Feature("Glucose", "Glucose", "mg/dL", 20, 300),
        Feature("BloodPressure", "Diastolic blood pressure", "mm Hg", 20, 200), Feature("SkinThickness", "Skin thickness", "mm", 1, 100),
        Feature("Insulin", "2-hour insulin", "mu U/mL", 1, 1000), Feature("BMI", "Body mass index", "kg/m2", 10, 80, .1),
        Feature("DiabetesPedigreeFunction", "Diabetes pedigree function", "score", 0, 3, .001), Feature("Age", "Age", "years", 18, 120),
    ), "Zero values for physiologically implausible measurements are treated as missing during training."),
    "heart": DiseaseSpec("heart", "Heart Disease", "UCI Heart Disease - Cleveland", "Heart-disease presence", (
        Feature("age", "Age", "years", 18, 120), Feature("sex", "Sex", "0=female, 1=male", 0, 1), Feature("cp", "Chest pain type", "0-3", 0, 3),
        Feature("trestbps", "Resting blood pressure", "mm Hg", 50, 250), Feature("chol", "Serum cholesterol", "mg/dL", 50, 700),
        Feature("fbs", "Fasting blood sugar >120", "0=no, 1=yes", 0, 1), Feature("restecg", "Resting ECG", "0-2", 0, 2),
        Feature("thalach", "Maximum heart rate", "bpm", 40, 250), Feature("exang", "Exercise angina", "0=no, 1=yes", 0, 1), Feature("oldpeak", "ST depression", "score", 0, 10, .1),
        Feature("slope", "ST slope", "0-2", 0, 2), Feature("ca", "Major vessels", "count", 0, 4), Feature("thal", "Thalassemia code", "0-3", 0, 3),
    ), "This model uses the standard 13 Cleveland predictors; UCI target values 1-4 are grouped as presence."),
    "parkinsons": DiseaseSpec("parkinsons", "Parkinson's Disease", "UCI Parkinsons", "Parkinson's status", (
        Feature("MDVP:Fo(Hz)", "Average vocal frequency", "Hz", 50, 300, .01), Feature("MDVP:Fhi(Hz)", "Maximum vocal frequency", "Hz", 50, 700, .01),
        Feature("MDVP:Flo(Hz)", "Minimum vocal frequency", "Hz", 20, 400, .01), Feature("MDVP:Jitter(%)", "Jitter", "%", 0, 5, .001),
        Feature("MDVP:Shimmer", "Shimmer", "ratio", 0, 1, .0001), Feature("NHR", "Noise-to-harmonics ratio", "ratio", 0, 1, .0001),
        Feature("HNR", "Harmonics-to-noise ratio", "dB", 0, 50, .01), Feature("RPDE", "Recurrence period density entropy", "score", 0, 1, .0001),
        Feature("DFA", "Detrended fluctuation analysis", "score", 0, 2, .0001), Feature("spread1", "Nonlinear spread 1", "score", -10, 2, .0001),
        Feature("spread2", "Nonlinear spread 2", "score", 0, 1, .0001), Feature("D2", "Correlation dimension", "score", 0, 5, .0001),
        Feature("PPE", "Pitch period entropy", "score", 0, 1, .0001),
    ), "Voice-measurement inputs only. The application intentionally excludes a person's name from the UCI dataset."),
    "breast_cancer": DiseaseSpec("breast_cancer", "Breast Cancer", "Wisconsin Diagnostic Breast Cancer (scikit-learn/UCI)", "Malignant tumor", (
        Feature("mean radius", "Mean radius", "measurement", 0, 50, .01), Feature("mean texture", "Mean texture", "measurement", 0, 50, .01),
        Feature("mean perimeter", "Mean perimeter", "measurement", 0, 300, .01), Feature("mean area", "Mean area", "measurement", 0, 3000, .01),
        Feature("mean smoothness", "Mean smoothness", "score", 0, 1, .0001), Feature("mean compactness", "Mean compactness", "score", 0, 1, .0001),
        Feature("mean concavity", "Mean concavity", "score", 0, 1, .0001), Feature("mean concave points", "Mean concave points", "score", 0, 1, .0001),
        Feature("mean symmetry", "Mean symmetry", "score", 0, 1, .0001), Feature("mean fractal dimension", "Mean fractal dimension", "score", 0, 1, .0001),
    ), "For a feasible web form, this flow uses the 10 mean measurements from the Wisconsin diagnostic dataset."),
}


def get_spec(disease: str) -> DiseaseSpec:
    try:
        return SPECS[disease]
    except KeyError as exc:
        raise ValueError(f"Unknown disease: {disease}") from exc
