import numpy as np
import pandas as pd

from medisense.ml.training import clean_data


def test_diabetes_replaces_impossible_zeroes_with_missing_values():
    frame = pd.DataFrame({"Pregnancies": [1], "Glucose": [0], "BloodPressure": [0], "SkinThickness": [0], "Insulin": [0], "BMI": [0], "DiabetesPedigreeFunction": [.5], "Age": [30]})
    cleaned = clean_data("diabetes", frame)
    assert cleaned[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isna().all().all()
    assert cleaned["Pregnancies"].iloc[0] == 1


def test_cleaning_removes_duplicate_rows():
    frame = pd.DataFrame({"age": [40, 40], "chol": [200, 200]})
    assert len(clean_data("heart", frame)) == 1
