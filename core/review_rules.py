"""Human review routing rules for batch predictions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.evaluation import risk_band


@dataclass(frozen=True)
class ReviewThresholds:
    """Business thresholds that control human-review routing."""

    fulfillment_rate_lt: float = 0.40
    high_value_shortfall: float | None = 500_000.0
    zero_stock_rate_gt: float = 0.70
    po_short_receipt_rate_gte: float = 0.90


def add_prediction_context(
    predictions: pd.DataFrame,
    thresholds: ReviewThresholds | None = None,
) -> pd.DataFrame:
    """Add risk bands, shortfall value, and review decisions to predictions."""

    active_thresholds = thresholds or ReviewThresholds()
    df = predictions.copy()
    df["predicted_fulfillment_rate"] = df["predicted_fulfillment_rate"].clip(lower=0.0, upper=1.0)
    df["predicted_risk_band"] = df["predicted_fulfillment_rate"].apply(risk_band)
    df["estimated_shortfall_value"] = (
        (1 - df["predicted_fulfillment_rate"]) * df["Quantity_Ordered"] * df["Unit_Price"]
    )

    decisions = df.apply(lambda row: review_decision(row, active_thresholds), axis=1)
    df["review_required"] = decisions.apply(lambda item: item[0])
    df["review_reason"] = decisions.apply(lambda item: item[1])
    return df


def review_decision(row: pd.Series, thresholds: ReviewThresholds | None = None) -> tuple[bool, str]:
    """Return whether a prediction needs review and a readable reason string."""

    active_thresholds = thresholds or ReviewThresholds()
    trigger_reasons: list[str] = []
    context_reasons: list[str] = []

    predicted_rate = float(row.get("predicted_fulfillment_rate", 1.0))
    if predicted_rate < active_thresholds.fulfillment_rate_lt:
        trigger_reasons.append("predicted fulfillment rate below threshold")

    shortfall_value = row.get("estimated_shortfall_value")
    if (
        active_thresholds.high_value_shortfall is not None
        and shortfall_value is not None
        and not pd.isna(shortfall_value)
        and float(shortfall_value) >= active_thresholds.high_value_shortfall
    ):
        trigger_reasons.append("estimated shortfall value above threshold")

    zero_stock_rate = row.get("inv_zero_stock_rate")
    if (
        zero_stock_rate is not None
        and not pd.isna(zero_stock_rate)
        and float(zero_stock_rate) > active_thresholds.zero_stock_rate_gt
    ):
        context_reasons.append("product has severe zero-stock history")

    po_short_receipt_rate = row.get("po_short_receipt_rate")
    if (
        po_short_receipt_rate is not None
        and not pd.isna(po_short_receipt_rate)
        and float(po_short_receipt_rate) >= active_thresholds.po_short_receipt_rate_gte
    ):
        context_reasons.append("supplier has severe short-receipt history")

    if not trigger_reasons:
        return False, ""

    reasons = trigger_reasons + context_reasons
    return True, "; ".join(reasons)
