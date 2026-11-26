
import numpy as np

from dataset import make_dataset
from pipeline import predict_churn_probability


class FixedProbabilityModel:
    """Модель с заранее заданными вероятностями."""

    def predict_proba(self, X):
        return np.array([
            [0.90, 0.10],
            [0.30, 0.70],
            [0.20, 0.80],
        ])


def test_issue_02_probability():
    """Проверяет вероятность положительного класса churn=1."""

    df = make_dataset().head(3)

    probabilities = predict_churn_probability(FixedProbabilityModel(), df, )
    expected = np.array([0.10, 0.70, 0.80])

    np.testing.assert_allclose(probabilities, expected)
