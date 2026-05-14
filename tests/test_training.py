import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from core.features import TARGET_COLUMN
from core.training import (
    build_candidate_model,
    build_mean_baseline,
    split_features_target,
    train_baseline,
    train_candidate_model,
)


def training_dataset():
    return pd.DataFrame(
        {
            "SO_Number": [f"SO_{idx}" for idx in range(12)],
            "Pharmacy_Name": ["Pharmacy_1", "Pharmacy_2"] * 6,
            "Product": ["Product_1", "Product_2", "Product_3"] * 4,
            "Quantity_Ordered": [10, 20, 5, 12, 30, 8, 14, 24, 6, 16, 18, 9],
            "Unit_Price": [100.0, 80.0, 200.0, 95.0, 75.0, 210.0, 90.0, 70.0, 220.0, 88.0, 82.0, 205.0],
            "order_value": [1000.0, 1600.0, 1000.0, 1140.0, 2250.0, 1680.0, 1260.0, 1680.0, 1320.0, 1408.0, 1476.0, 1845.0],
            "inv_qty_on_hand_avg": [20, 0, 10, 22, 0, 11, 18, 1, 9, 24, 0, 12],
            "inv_zero_stock_rate": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.8, 0.0, 0.0, 1.0, 0.1],
            "inv_product_category": ["OTC", "Prescription", "OTC"] * 4,
            "po_receipt_rate_avg": [0.9, 0.4, 0.8, 0.95, 0.35, 0.85, 0.88, 0.45, 0.82, 0.92, 0.3, 0.8],
            "po_short_receipt_rate": [0.1, 0.9, 0.2, 0.1, 0.95, 0.2, 0.2, 0.8, 0.3, 0.1, 1.0, 0.25],
            TARGET_COLUMN: [0.9, 0.3, 0.8, 0.95, 0.25, 0.85, 0.88, 0.4, 0.75, 0.92, 0.2, 0.8],
        }
    )


def test_split_features_target_excludes_identifier_and_target():
    split = split_features_target(training_dataset(), test_size=0.25, random_state=7)

    assert "SO_Number" not in split.X_train.columns
    assert TARGET_COLUMN not in split.X_train.columns
    assert len(split.X_train) == 9
    assert len(split.X_test) == 3


def test_split_features_target_requires_target():
    dataset = training_dataset().drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError):
        split_features_target(dataset)


def test_build_models_return_pipelines():
    split = split_features_target(training_dataset(), test_size=0.25, random_state=7)

    assert isinstance(build_mean_baseline(split.X_train), Pipeline)
    assert isinstance(build_candidate_model(split.X_train), Pipeline)


def test_train_baseline_returns_metrics():
    result = train_baseline(training_dataset(), test_size=0.25, random_state=7)

    assert isinstance(result.model, Pipeline)
    assert "mae" in result.metrics
    assert "critical_high_recall" in result.metrics


def test_train_candidate_model_returns_metrics():
    result = train_candidate_model(training_dataset(), test_size=0.25, random_state=7)

    assert isinstance(result.model, Pipeline)
    assert "mae" in result.metrics
    assert "risk_band_accuracy" in result.metrics
