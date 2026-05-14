"""Run local batch inference for fulfillment-rate predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from core.features import build_inference_dataset
from core.review_rules import ReviewThresholds, add_prediction_context
from core.validation import validate_all_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "batch_predictions.csv"


def main() -> None:
    settings = _load_settings()
    model = _load_model()
    inventory, purchase_orders, sales = _load_source_data(settings)

    validation = validate_all_tables(inventory, purchase_orders, sales)
    if validation.warnings:
        print("Validation warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    if not validation.passed:
        print("Validation failed:")
        for error in validation.errors:
            print(f"  - {error}")
        raise SystemExit(1)

    inference_dataset = build_inference_dataset(inventory, purchase_orders, sales)
    prediction_frame = inference_dataset.loc[
        :,
        ["SO_Number", "Product", "Pharmacy_Name", "Quantity_Ordered", "Unit_Price"],
    ].copy()
    prediction_frame["predicted_fulfillment_rate"] = model.predict(inference_dataset)

    output = add_prediction_context(
        prediction_frame.merge(
            inference_dataset.loc[:, ["SO_Number", "inv_zero_stock_rate", "po_short_receipt_rate"]],
            on="SO_Number",
            how="left",
        ),
        thresholds=_review_thresholds(settings),
    )

    output = output.loc[
        :,
        [
            "SO_Number",
            "Product",
            "Pharmacy_Name",
            "Quantity_Ordered",
            "Unit_Price",
            "predicted_fulfillment_rate",
            "predicted_risk_band",
            "estimated_shortfall_value",
            "review_required",
            "review_reason",
        ],
    ]

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Batch predictions written to {PREDICTIONS_PATH}")
    print(f"Rows scored: {len(output)}")
    print(f"Rows requiring review: {int(output['review_required'].sum())}")


def _load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run scripts/train.py first.")
    return joblib.load(MODEL_PATH)


def _load_source_data(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_settings = settings["data"]
    inventory = pd.read_csv(PROJECT_ROOT / data_settings["product_inventory"])
    purchase_orders = pd.read_csv(PROJECT_ROOT / data_settings["purchase_orders"])
    sales = pd.read_csv(PROJECT_ROOT / data_settings["sales_invoiced"])
    return inventory, purchase_orders, sales


def _review_thresholds(settings: dict) -> ReviewThresholds:
    review_settings = settings.get("human_review", {})
    return ReviewThresholds(
        fulfillment_rate_lt=review_settings.get("fulfillment_rate_lt", 0.50),
        high_value_shortfall=review_settings.get("high_value_shortfall"),
    )


if __name__ == "__main__":
    main()
