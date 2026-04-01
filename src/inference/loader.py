import os
import joblib
import json
from typing import List

MODEL_DIR = r"C:\Users\olegs\PycharmProjects\dka_recurrence_project\src\eda\models\catboost"


def load_models() -> List:
    models = []
    for i in range(5):
        path = os.path.join(MODEL_DIR, f"catboost_outer_fold_{i}.pkl")
        models.append(joblib.load(path))
    return models


def load_threshold():
    with open(os.path.join(MODEL_DIR, "thresholds.json"), "r") as f:
        thresholds = json.load(f)
    return float(sorted(thresholds)[len(thresholds)//2])  # медиана