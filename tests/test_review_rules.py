import pandas as pd

from core.review_rules import ReviewThresholds, add_prediction_context, review_decision


def test_add_prediction_context_adds_review_fields():
    predictions = pd.DataFrame(
        {
            "SO_Number": ["SO_1"],
            "Quantity_Ordered": [10],
            "Unit_Price": [100.0],
            "predicted_fulfillment_rate": [0.30],
            "inv_zero_stock_rate": [0.0],
            "po_short_receipt_rate": [0.0],
        }
    )

    result = add_prediction_context(predictions)

    assert result["predicted_risk_band"].iloc[0] == "High"
    assert result["estimated_shortfall_value"].iloc[0] == 700.0
    assert result["review_required"].iloc[0]
    assert "predicted fulfillment rate below threshold" in result["review_reason"].iloc[0]


def test_high_value_shortfall_triggers_review():
    row = pd.Series(
        {
            "predicted_fulfillment_rate": 0.80,
            "estimated_shortfall_value": 5_000.0,
            "inv_zero_stock_rate": 0.0,
            "po_short_receipt_rate": 0.0,
        }
    )

    required, reason = review_decision(
        row,
        ReviewThresholds(high_value_shortfall=1_000.0),
    )

    assert required
    assert "estimated shortfall value above threshold" in reason


def test_zero_stock_history_is_context_not_standalone_trigger():
    row = pd.Series(
        {
            "predicted_fulfillment_rate": 0.80,
            "estimated_shortfall_value": 0.0,
            "inv_zero_stock_rate": 0.95,
            "po_short_receipt_rate": 0.0,
        }
    )

    required, reason = review_decision(row)

    assert not required
    assert reason == ""


def test_zero_stock_history_is_added_when_row_is_already_reviewable():
    row = pd.Series(
        {
            "predicted_fulfillment_rate": 0.30,
            "estimated_shortfall_value": 0.0,
            "inv_zero_stock_rate": 0.95,
            "po_short_receipt_rate": 0.0,
        }
    )

    required, reason = review_decision(row)

    assert required
    assert "predicted fulfillment rate below threshold" in reason
    assert "product has severe zero-stock history" in reason


def test_no_trigger_returns_no_review():
    row = pd.Series(
        {
            "predicted_fulfillment_rate": 0.90,
            "estimated_shortfall_value": 100.0,
            "inv_zero_stock_rate": 0.0,
            "po_short_receipt_rate": 0.0,
        }
    )

    required, reason = review_decision(row)

    assert not required
    assert reason == ""
