"""Dependency-free batch inference fallback for fulfillment-rate predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = PROJECT_ROOT / "data" / "Sales_Invoiced.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model_stdlib.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "batch_predictions_stdlib.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run scripts/train_stdlib.py first.")

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rows = _read_csv(SALES_PATH)
    output = []

    for row in rows:
        predicted_rate = _predict(row, model)
        shortfall_value = (1 - predicted_rate) * float(row["Quantity_Ordered"]) * float(row["Unit_Price"])
        review_required, review_reason = _review_decision(predicted_rate, shortfall_value)
        output.append(
            {
                "SO_Number": row["SO_Number"],
                "Product": row["Product"],
                "Pharmacy_Name": row["Pharmacy_Name"],
                "Quantity_Ordered": row["Quantity_Ordered"],
                "Unit_Price": row["Unit_Price"],
                "predicted_fulfillment_rate": f"{predicted_rate:.6f}",
                "predicted_risk_band": _risk_band(predicted_rate),
                "estimated_shortfall_value": f"{shortfall_value:.2f}",
                "review_required": str(review_required).lower(),
                "review_reason": review_reason,
            }
        )

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(f"Batch predictions written to {PREDICTIONS_PATH}")
    print(f"Rows scored: {len(output)}")
    print(f"Rows requiring review: {sum(row['review_required'] == 'true' for row in output)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _predict(row: dict[str, str], model: dict) -> float:
    product_means = model["product_means"]
    return float(product_means.get(row["Product"], model["global_mean"]))


def _review_decision(predicted_rate: float, shortfall_value: float) -> tuple[bool, str]:
    reasons = []
    if predicted_rate < 0.50:
        reasons.append("predicted fulfillment rate below threshold")
    if shortfall_value >= 100000:
        reasons.append("estimated shortfall value above threshold")
    return bool(reasons), "; ".join(reasons)


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


if __name__ == "__main__":
    main()
