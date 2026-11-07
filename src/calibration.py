"""Platt scaling: turns raw model scores into probabilities you can trust."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, raw_scores, y_true):
        self.model_ = LogisticRegression(C=1e6, max_iter=1000)
        self.model_.fit(_logit(raw_scores), y_true)
        return self

    def predict(self, raw_scores):
        return self.model_.predict_proba(_logit(raw_scores))[:, 1]