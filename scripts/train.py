"""Run local training for the fulfillment-rate model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from core.features import build_training_dataset
from core.training import train_baseline, train_candidate_model
from core.validation import validate_all_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "training_metrics.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def main() -> None:
    settings = _load_settings()
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

    dataset = build_training_dataset(inventory, purchase_orders, sales)
    baseline = train_baseline(dataset)
    candidate = train_candidate_model(dataset)

    report = {
        "row_counts": validation.row_counts,
        "baseline": baseline.metrics,
        "candidate": candidate.metrics,
        "promotion_check": {
            "beats_baseline_mae_by_10pct": _beats_baseline_by_10pct(
                baseline.metrics["mae"],
                candidate.metrics["mae"],
            )
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(candidate.model, MODEL_PATH)

    print(f"Training complete. Metrics written to {REPORT_PATH}")
    print(f"Candidate model written to {MODEL_PATH}")
    print(json.dumps(report, indent=2))


def _load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_source_data(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_settings = settings["data"]
    inventory = pd.read_csv(PROJECT_ROOT / data_settings["product_inventory"])
    purchase_orders = pd.read_csv(PROJECT_ROOT / data_settings["purchase_orders"])
    sales = pd.read_csv(PROJECT_ROOT / data_settings["sales_invoiced"])
    return inventory, purchase_orders, sales


def _beats_baseline_by_10pct(baseline_mae: float, candidate_mae: float) -> bool:
    if baseline_mae <= 0:
        return candidate_mae < baseline_mae
    return candidate_mae <= baseline_mae * 0.90


if __name__ == "__main__":
    main()
