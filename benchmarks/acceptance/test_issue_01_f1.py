
import numpy as np
import pytest

from metrics import evaluate_model


class FixedModel:
    """Модель с заранее известными предсказаниями."""

    def predict(self, X):
        return np.array([1, 1, 0, 1, 1, 0])

    def predict_proba(self, X):
        probabilities = np.array( [0.8, 0.7, 0.2, 0.6, 0.55, 0.1])
        return np.column_stack([1 - probabilities, probabilities])


def test_issue_01_f1():
    """F1 должен вычисляться по TP, FP и FN."""

    X = np.zeros((6, 1))
    y_true = np.array([1, 1, 1, 0, 0, 0])

    results = evaluate_model(FixedModel(), X, y_true)

    assert results["f1"] == pytest.approx(4 / 7)
