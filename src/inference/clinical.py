import pandas as pd


class ClinicalInterpreter:
    def __init__(self, threshold, numeric_features: list, df_reference: pd.DataFrame):
        self.threshold = threshold
        self.numeric_features = set(numeric_features)
        self.df_reference = df_reference

    ### уровень риска вместо вероятности
    def risk_level(self, proba: float) -> str:
        if proba < self.threshold - 0.1:
            return "Низкий риск"
        elif proba < self.threshold + 0.1:
            return "Умеренный риск"
        else:
            return "Высокий риск"

    ### ключевые факторы влияния
    def top_drivers(self, shap_dict: dict, top_n: int = 5):
        sorted_feats = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        drivers = []
        for f, v in sorted_feats[:top_n]:
            drivers.append({
                "feature": f,
                "impact": "увеличивает оценку риска рецидива моделью" if v > 0 else "снижает оценку риска моделью"
            })

        return drivers

    ### анализ отклонений для числовых признаков
    def numeric_alerts(self, deviations: dict):
        alerts = []

        for col, v in deviations.items():
            if col not in self.numeric_features:
                continue

            if v["z_score"] > 0:
                direction = "выше нормы"
            else:
                direction = "ниже нормы"

            alerts.append(f"{col} = {v["value"]}, значение {direction}")

        return alerts

    ### анализ редких категорий
    def categorical_alerts(self, X_row: pd.DataFrame):
        alerts = []

        for col in X_row.columns:
            if col in self.numeric_features:
                continue

            value = X_row[col].values[0]

            ### частота категории в обучающей выборке
            freq = (
                self.df_reference[col]
                .value_counts(normalize=True, dropna=False)
                .get(value, 0)
            )

            ### если категория редкая
            if freq < 0.05:
                alerts.append(f"{col}: редкое значение ({value})")

        return alerts

    ### объединение всех алертов
    def build_alerts(self, deviations: dict, X_row: pd.DataFrame):
        alerts = []

        alerts.extend(self.numeric_alerts(deviations))
        alerts.extend(self.categorical_alerts(X_row))

        return alerts

    ### краткое клиническое резюме
    def summary(self, risk_level: str) -> str:
        if risk_level == "Высокий риск":
            return "Требуется повышенное внимание."
        elif risk_level == "Умеренный риск":
            return "Рекомендуется наблюдение."
        else:
            return "Существенных отклонений не выявлено."

    ### финальный ответ для врача
    def build_response(self, proba, shap_dict, deviations, X_row):
        risk = self.risk_level(proba)
        drivers = self.top_drivers(shap_dict)
        alerts = self.build_alerts(deviations, X_row)
        summary = self.summary(risk)

        return {
            "risk_level": risk,
            "drivers": drivers,
            "alerts": alerts,
            "summary": summary
        }