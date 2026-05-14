"""Generate diagnostics for model signal, errors, and review load."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from core.features import TARGET_COLUMN, build_training_dataset
from core.training import evaluate_model, split_features_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_diagnostics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "batch_predictions.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run scripts/train.py first.")

    inventory = pd.read_csv(PROJECT_ROOT / "data" / "Product_Inventory.csv")
    purchase_orders = pd.read_csv(PROJECT_ROOT / "data" / "Purchase_Orders.csv")
    sales = pd.read_csv(PROJECT_ROOT / "data" / "Sales_Invoiced.csv")

    dataset = build_training_dataset(inventory, purchase_orders, sales)
    split = split_features_target(dataset)
    model = joblib.load(MODEL_PATH)
    metrics = evaluate_model(model, split.X_test, split.y_test)

    scored_test = split.X_test.copy()
    scored_test[TARGET_COLUMN] = split.y_test
    scored_test["predicted_fulfillment_rate"] = model.predict(split.X_test).clip(0.0, 1.0)
    scored_test["absolute_error"] = (
        scored_test[TARGET_COLUMN] - scored_test["predicted_fulfillment_rate"]
    ).abs()

    diagnostics = {
        "metrics": metrics,
        "target_distribution": _describe_series(dataset[TARGET_COLUMN]),
        "numeric_feature_correlations": _numeric_correlations(dataset),
        "highest_error_products": _group_error(scored_test, "Product"),
        "highest_error_pharmacies": _group_error(scored_test, "Pharmacy_Name"),
        "highest_error_categories": _group_error(scored_test, "inv_product_category"),
        "worst_average_fulfillment_products": _group_target(dataset, "Product"),
        "worst_average_fulfillment_pharmacies": _group_target(dataset, "Pharmacy_Name"),
        "review_queue": _review_summary(),
        "data_limitations": [
            "Sales data has no order or invoice date, so time-based validation is not possible.",
            "Inventory appears as repeated product records, not a clean point-in-time snapshot.",
            "Purchase-order outcomes may be future information unless sales dates are added.",
            "The current candidate only slightly beats the mean baseline, indicating weak predictive signal.",
        ],
        "recommended_next_data_fields": [
            "sales_order_date or invoice_date",
            "available_stock_at_order_time",
            "promised_delivery_date",
            "customer priority or service-level tier",
            "backorder/cancellation reason",
            "warehouse/location serving the order",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Diagnostics written to {REPORT_PATH}")
    print(json.dumps(diagnostics["metrics"], indent=2))


def _describe_series(series: pd.Series) -> dict[str, float]:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def _numeric_correlations(dataset: pd.DataFrame) -> list[dict[str, float | str]]:
    numeric = dataset.select_dtypes(include=["number", "bool"]).copy()
    if TARGET_COLUMN not in numeric:
        return []

    correlations = numeric.corr(numeric_only=True)[TARGET_COLUMN].drop(labels=[TARGET_COLUMN])
    correlations = correlations.dropna().sort_values(key=lambda values: values.abs(), ascending=False)
    return [
        {"feature": feature, "correlation": float(value)}
        for feature, value in correlations.head(15).items()
    ]


def _group_error(df: pd.DataFrame, column: str) -> list[dict[str, float | str | int]]:
    if column not in df:
        return []
    grouped = (
        df.groupby(column)
        .agg(
            rows=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            actual_fulfillment_rate=(TARGET_COLUMN, "mean"),
            predicted_fulfillment_rate=("predicted_fulfillment_rate", "mean"),
        )
        .query("rows >= 2")
        .sort_values("mean_absolute_error", ascending=False)
        .head(10)
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def _group_target(df: pd.DataFrame, column: str) -> list[dict[str, float | str | int]]:
    if column not in df:
        return []
    grouped = (
        df.groupby(column)
        .agg(
            rows=(TARGET_COLUMN, "size"),
            mean_fulfillment_rate=(TARGET_COLUMN, "mean"),
        )
        .query("rows >= 5")
        .sort_values("mean_fulfillment_rate", ascending=True)
        .head(10)
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def _review_summary() -> dict:
    if not PREDICTIONS_PATH.exists():
        return {"available": False}

    predictions = pd.read_csv(PREDICTIONS_PATH)
    summary = {
        "available": True,
        "rows": int(len(predictions)),
        "review_required": int(predictions["review_required"].sum()),
        "review_rate": float(predictions["review_required"].mean()),
        "risk_band_counts": predictions["predicted_risk_band"].value_counts().to_dict(),
    }

    reasons = {}
    for reason in [
        "predicted fulfillment rate below threshold",
        "estimated shortfall value above threshold",
        "product has severe zero-stock history",
        "supplier has severe short-receipt history",
    ]:
        reasons[reason] = int(
            predictions["review_reason"].fillna("").str.contains(reason, regex=False).sum()
        )
    summary["reason_counts"] = reasons
    return summary


if __name__ == "__main__":
    main()
