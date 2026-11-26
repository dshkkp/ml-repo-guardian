import numpy as np

from dataset import make_dataset
from pipeline import predict_churn, train_model


def test_issue_03_unknown_category():
    """Модель должна поддерживать неизвестные категории при прогнозировании."""

    dataset = make_dataset(n_samples=300, random_state=42)
    model, X_test, _ = train_model(dataset, random_state=42)

    new_customers = X_test.head(5).copy()
    new_customers["contract_type"] = "enterprise"

    predictions = predict_churn(model, new_customers)

    assert len(predictions) == len(new_customers)
    assert set(np.unique(predictions)).issubset({0, 1})
