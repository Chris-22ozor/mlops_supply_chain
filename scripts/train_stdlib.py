"""Dependency-free training fallback for the fulfillment-rate system.

This script is intentionally simple and transparent. It uses only the Python
standard library so the project can run even when scientific Python wheels are
blocked by the environment.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = PROJECT_ROOT / "data" / "Sales_Invoiced.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "training_metrics_stdlib.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model_stdlib.json"


def main() -> None:
    rows = _read_csv(SALES_PATH)
    _validate_sales(rows)

    train_rows, test_rows = _split(rows, test_fraction=0.2)
    global_mean, product_means = _fit_product_mean_model(train_rows)

    baseline_predictions = [global_mean for _ in test_rows]
    candidate_predictions = [_predict_product_mean(row, global_mean, product_means) for row in test_rows]
    actuals = [_fulfillment_rate(row) for row in test_rows]

    report = {
        "row_counts": {
            "sales_invoiced": len(rows),
            "train": len(train_rows),
            "test": len(test_rows),
        },
        "baseline": _metrics(actuals, baseline_predictions),
        "candidate": _metrics(actuals, candidate_predictions),
    }
    report["promotion_check"] = {
        "beats_baseline_mae_by_10pct": _beats_baseline_by_10pct(
            report["baseline"]["mae"],
            report["candidate"]["mae"],
        )
    }

    model = {
        "model_type": "product_mean_fulfillment_rate",
        "global_mean": global_mean,
        "product_means": product_means,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"Training complete. Metrics written to {REPORT_PATH}")
    print(f"Model written to {MODEL_PATH}")
    print(json.dumps(report, indent=2))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _validate_sales(rows: list[dict[str, str]]) -> None:
    required = {
        "SO_Number",
        "Pharmacy_Name",
        "Product",
        "Quantity_Ordered",
        "Quantity_Invoiced",
        "Unit_Price",
    }
    if not rows:
        raise SystemExit("Sales file is empty.")

    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Sales file is missing required columns: {', '.join(sorted(missing))}")

    seen_orders: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if row["SO_Number"] in seen_orders:
            raise SystemExit(f"Duplicate SO_Number at CSV row {index}: {row['SO_Number']}")
        seen_orders.add(row["SO_Number"])

        ordered = _float(row["Quantity_Ordered"], "Quantity_Ordered", index)
        invoiced = _float(row["Quantity_Invoiced"], "Quantity_Invoiced", index)
        unit_price = _float(row["Unit_Price"], "Unit_Price", index)

        if ordered <= 0:
            raise SystemExit(f"Quantity_Ordered must be positive at CSV row {index}.")
        if invoiced < 0 or unit_price < 0:
            raise SystemExit(f"Quantity_Invoiced and Unit_Price must be non-negative at CSV row {index}.")
        if invoiced > ordered:
            raise SystemExit(f"Quantity_Invoiced exceeds Quantity_Ordered at CSV row {index}.")


def _split(rows: list[dict[str, str]], test_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cutoff = int(len(rows) * (1 - test_fraction))
    return rows[:cutoff], rows[cutoff:]


def _fit_product_mean_model(rows: list[dict[str, str]]) -> tuple[float, dict[str, float]]:
    rates_by_product: dict[str, list[float]] = defaultdict(list)
    all_rates: list[float] = []

    for row in rows:
        rate = _fulfillment_rate(row)
        rates_by_product[row["Product"]].append(rate)
        all_rates.append(rate)

    global_mean = sum(all_rates) / len(all_rates)
    product_means = {
        product: sum(rates) / len(rates)
        for product, rates in rates_by_product.items()
    }
    return global_mean, product_means


def _predict_product_mean(row: dict[str, str], global_mean: float, product_means: dict[str, float]) -> float:
    return product_means.get(row["Product"], global_mean)


def _fulfillment_rate(row: dict[str, str]) -> float:
    return _to_float(row["Quantity_Invoiced"]) / _to_float(row["Quantity_Ordered"])


def _metrics(actuals: list[float], predictions: list[float]) -> dict[str, float]:
    errors = [actual - predicted for actual, predicted in zip(actuals, predictions)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    risk_accuracy = sum(
        _risk_band(actual) == _risk_band(predicted)
        for actual, predicted in zip(actuals, predictions)
    ) / len(actuals)
    return {
        "mae": mae,
        "rmse": rmse,
        "risk_band_accuracy": risk_accuracy,
        "critical_high_recall": _critical_high_recall(actuals, predictions),
    }


def _critical_high_recall(actuals: list[float], predictions: list[float]) -> float:
    actual_positive = [_risk_band(value) in {"Critical", "High"} for value in actuals]
    predicted_positive = [_risk_band(value) in {"Critical", "High"} for value in predictions]
    denominator = sum(actual_positive)
    if denominator == 0:
        return 0.0
    true_positives = sum(a and p for a, p in zip(actual_positive, predicted_positive))
    return true_positives / denominator


def _risk_band(rate: float) -> str:
    clipped = max(0.0, min(1.0, rate))
    if clipped < 0.25:
        return "Critical"
    if clipped < 0.50:
        return "High"
    if clipped < 0.75:
        return "Medium"
    if clipped < 0.99:
        return "Low"
    return "OK"


def _beats_baseline_by_10pct(baseline_mae: float, candidate_mae: float) -> bool:
    return candidate_mae <= baseline_mae * 0.90


def _float(raw: str, column: str, row_number: int) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{column} must be numeric at CSV row {row_number}.") from exc


def _to_float(raw: str) -> float:
    return float(raw)


if __name__ == "__main__":
    main()
