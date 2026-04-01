import numpy as np


class DKAPredictor:
    def __init__(self, models, threshold):
        self.models = models
        self.threshold = threshold

    def predict_proba(self, X):
        preds = []

        for model in self.models:
            p = model.predict_proba(X)[:, 1]
            preds.append(p)

        return np.mean(preds, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def confidence(self, proba):
        # расстояние от threshold
        distance = abs(proba - self.threshold)

        if distance < 0.05:
            return "LOW"
        elif distance < 0.15:
            return "MEDIUM"
        else:
            return "HIGH"