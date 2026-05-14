"""Create a human review queue from batch predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "batch_predictions.csv"
REVIEW_QUEUE_PATH = PROJECT_ROOT / "reports" / "human_review_queue.csv"


REVIEW_COLUMNS = [
    "SO_Number",
    "Product",
    "Pharmacy_Name",
    "Quantity_Ordered",
    "Unit_Price",
    "predicted_fulfillment_rate",
    "predicted_risk_band",
    "estimated_shortfall_value",
    "review_reason",
    "human_decision",
    "adjusted_quantity",
    "reviewer",
    "review_notes",
    "review_timestamp",
    "final_outcome",
]


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(f"Batch predictions not found at {PREDICTIONS_PATH}. Run scripts/predict_batch.py first.")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    review_queue = predictions[predictions["review_required"]].copy()

    for column in [
        "human_decision",
        "adjusted_quantity",
        "reviewer",
        "review_notes",
        "review_timestamp",
        "final_outcome",
    ]:
        review_queue[column] = ""

    review_queue = review_queue.loc[:, REVIEW_COLUMNS]
    review_queue = review_queue.sort_values(
        by=["estimated_shortfall_value", "predicted_fulfillment_rate"],
        ascending=[False, True],
    )

    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    review_queue.to_csv(REVIEW_QUEUE_PATH, index=False)

    print(f"Human review queue written to {REVIEW_QUEUE_PATH}")
    print(f"Rows requiring review: {len(review_queue)}")


if __name__ == "__main__":
    main()
