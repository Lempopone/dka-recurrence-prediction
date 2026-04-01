from fastapi import FastAPI
import pandas as pd

from src.inference.loader import load_models, load_threshold
from src.inference.predictor import DKAPredictor
from src.inference.explain import Explainer
from src.inference.clinical import ClinicalInterpreter
import json

app = FastAPI(title="Прогноз рецидива ДКА")

### загрузка данных
df = pd.read_parquet("data/processed/df.parquet")

### загрузка моделей
models = load_models()
threshold = load_threshold()

predictor = DKAPredictor(models, threshold)
explainer = Explainer(models[0])

with open(r"C:\Users\olegs\PycharmProjects\dka_recurrence_project\data\processed\train_ids.json") as f:
    TRAIN_IDS = set(json.load(f))

with open(r"C:\Users\olegs\PycharmProjects\dka_recurrence_project\data\processed\test_ids.json") as f:
    TEST_IDS = set(json.load(f))

def get_dataset_type(patient_id: int) -> str:
    if patient_id in TRAIN_IDS:
        return "train"
    elif patient_id in TEST_IDS:
        return "test"
    else:
        return "unknown"

### список числовых признаков
NUMERIC_FEATURES = [
    'age',
    'diabetes_duration',
    'daily_insulin_dosage',
    'hba1c',
    'creatinine_admission',
    'urea_admission',
    'ph_admission',
    'be_admission',
    'lactate_admission',
    'glucose_admission',
    'cholesterol_total',
    'ldl',
    'hdl',
    'triglycerides'
]

interpreter = ClinicalInterpreter(
    threshold,
    numeric_features=NUMERIC_FEATURES,
    df_reference=df.drop(columns=["target"])
)


@app.get("/predict/{medical_record_id}")
def predict(medical_record_id: int):

    if medical_record_id not in df.index:
        return {"error": "Пациент не найден"}

    X = df.loc[[medical_record_id]].drop(columns=["target"])

    ### прогноз модели
    proba = predictor.predict_proba(X)[0]
    pred = int(proba > threshold)
    confidence = predictor.confidence(proba)

    ### объяснение модели
    shap_vals = explainer.feature_importance_patient(X)
    shap_dict = dict(zip(X.columns, shap_vals.tolist()))

    deviations = explainer.detect_outliers(
        X,
        df.drop(columns=["target"])
    )

    ### клиническая интерпретация

    dataset_type = get_dataset_type(medical_record_id)

    clinical = interpreter.build_response(
        proba,
        shap_dict,
        deviations,
        X
    )

    return {
        "prediction": pred,
        "proba": proba,
        "confidence": confidence,
        "dataset": dataset_type,
        "clinical": clinical,
        "shap": shap_dict
    }