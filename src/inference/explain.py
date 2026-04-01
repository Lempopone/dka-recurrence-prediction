import shap
import numpy as np
import pandas as pd


class Explainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X):
        return self.explainer.shap_values(X)

    def feature_importance_patient(self, X):
        shap_vals = self.shap_values(X)[0]
        return shap_vals

    def detect_outliers(self, X_row, df_reference):
        deviations = {}

        for col in X_row.columns:
            mean = df_reference[col].mean()
            std = df_reference[col].std()

            val = X_row[col].values[0]

            if std == 0:
                continue

            z = (val - mean) / std

            if abs(z) > 2:
                deviations[col] = {
                    "value": float(val),
                    "mean": float(mean),
                    "z_score": float(z)
                }

        return deviations