import pandas as pd

from core.evaluation import (
    critical_high_recall,
    evaluate_predictions,
    regression_metrics,
    risk_band,
    risk_band_accuracy,
)


def test_risk_band_boundaries():
    assert risk_band(0.10) == "Critical"
    assert risk_band(0.25) == "High"
    assert risk_band(0.50) == "Medium"
    assert risk_band(0.75) == "Low"
    assert risk_band(0.99) == "OK"


def test_regression_metrics_are_computed():
    metrics = regression_metrics(
        pd.Series([1.0, 0.5, 0.0]),
        pd.Series([0.8, 0.4, 0.2]),
    )

    assert round(metrics["mae"], 3) == 0.167
    assert round(metrics["rmse"], 3) == 0.173
    assert "r2" in metrics


def test_risk_band_accuracy():
    accuracy = risk_band_accuracy(
        pd.Series([0.10, 0.40, 0.90]),
        pd.Series([0.20, 0.55, 0.95]),
    )

    assert round(accuracy, 3) == 0.667


def test_critical_high_recall():
    recall = critical_high_recall(
        pd.Series([0.10, 0.40, 0.90]),
        pd.Series([0.20, 0.55, 0.95]),
    )

    assert recall == 0.5


def test_evaluate_predictions_combines_metrics():
    metrics = evaluate_predictions(
        pd.Series([1.0, 0.5, 0.0]),
        pd.Series([0.8, 0.4, 0.2]),
    )

    assert set(metrics) == {
        "mae",
        "rmse",
        "r2",
        "risk_band_accuracy",
        "critical_high_recall",
    }
