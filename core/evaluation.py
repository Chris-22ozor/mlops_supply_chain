"""Model evaluation logic for fulfillment-rate predictions."""

from __future__ import annotations

import math

import pandas as pd


CRITICAL_HIGH_BANDS = {"Critical", "High"}


def risk_band(rate: float) -> str:
    """Map a fulfillment rate to an operational risk band."""

    clipped = _clip_rate(rate)
    if clipped < 0.25:
        return "Critical"
    if clipped < 0.50:
        return "High"
    if clipped < 0.75:
        return "Medium"
    if clipped < 0.99:
        return "Low"
    return "OK"


def risk_band_series(rates: pd.Series) -> pd.Series:
    """Map a series of fulfillment rates to operational risk bands."""

    return rates.apply(risk_band)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute core regression metrics without requiring scikit-learn."""

    actual = _as_float_series(y_true)
    predicted = _as_float_series(y_pred).clip(lower=0.0, upper=1.0)
    error = actual - predicted

    mae = error.abs().mean()
    rmse = math.sqrt((error**2).mean())

    total_sum_squares = ((actual - actual.mean()) ** 2).sum()
    residual_sum_squares = (error**2).sum()
    r2 = 0.0 if total_sum_squares == 0 else 1 - (residual_sum_squares / total_sum_squares)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def risk_band_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Return the share of predictions assigned to the correct risk band."""

    actual_bands = risk_band_series(_as_float_series(y_true))
    predicted_bands = risk_band_series(_as_float_series(y_pred))
    return float((actual_bands == predicted_bands).mean())


def critical_high_recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Return recall for records whose actual band is Critical or High."""

    actual_bands = risk_band_series(_as_float_series(y_true))
    predicted_bands = risk_band_series(_as_float_series(y_pred))

    actual_critical_high = actual_bands.isin(CRITICAL_HIGH_BANDS)
    predicted_critical_high = predicted_bands.isin(CRITICAL_HIGH_BANDS)

    denominator = int(actual_critical_high.sum())
    if denominator == 0:
        return 0.0

    true_positive_count = int((actual_critical_high & predicted_critical_high).sum())
    return true_positive_count / denominator


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression and operational metrics for model evaluation."""

    metrics = regression_metrics(y_true, y_pred)
    metrics["risk_band_accuracy"] = risk_band_accuracy(y_true, y_pred)
    metrics["critical_high_recall"] = critical_high_recall(y_true, y_pred)
    return metrics


def _clip_rate(rate: float) -> float:
    if pd.isna(rate):
        return 0.0
    return min(max(float(rate), 0.0), 1.0)


def _as_float_series(values: pd.Series) -> pd.Series:
    return pd.Series(values, copy=False).astype(float)
