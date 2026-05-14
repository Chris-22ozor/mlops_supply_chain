"""Staging API for fulfillment-rate prediction.

This endpoint is for shadow/staging use. It returns model recommendations and
human-review routing decisions, but it must not be used for automatic business
actions while production promotion is blocked.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.features import build_inference_dataset
from core.review_rules import ReviewThresholds, add_prediction_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging"
MODEL_PATH = STAGING_DIR / "model.joblib"
MANIFEST_PATH = STAGING_DIR / "deployment_manifest.json"
INVENTORY_PATH = PROJECT_ROOT / "data" / "Product_Inventory.csv"
PURCHASE_ORDERS_PATH = PROJECT_ROOT / "data" / "Purchase_Orders.csv"


app = FastAPI(
    title="Supply-Chain Fulfillment Risk API",
    version="0.1.0",
    description="Staging API for fulfillment-rate prediction and human-review routing.",
)


class PredictionRequest(BaseModel):
    """Single sales-order line to score."""

    SO_Number: str = Field(..., min_length=1)
    Product: str = Field(..., min_length=1)
    Pharmacy_Name: str = Field(..., min_length=1)
    Quantity_Ordered: float = Field(..., gt=0)
    Unit_Price: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    """Prediction response with human-review decision."""

    SO_Number: str
    Product: str
    Pharmacy_Name: str
    Quantity_Ordered: float
    Unit_Price: float
    predicted_fulfillment_rate: float
    predicted_risk_band: str
    estimated_shortfall_value: float
    review_required: bool
    review_reason: str
    deployment_stage: str
    production_promotion_allowed: bool


@app.get("/health")
def health() -> dict:
    """Return API and staging artifact health."""

    manifest = _load_manifest()
    return {
        "status": "ok",
        "deployment_stage": manifest["deployment_stage"],
        "production_promotion_allowed": manifest["production_promotion_allowed"],
        "model_available": MODEL_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one sales-order line and return review routing."""

    manifest = _load_manifest()
    model = _load_model()
    inventory, purchase_orders = _load_context_tables()

    sales = pd.DataFrame([request.model_dump()])
    inference_dataset = build_inference_dataset(inventory, purchase_orders, sales)
    predicted_rate = float(model.predict(inference_dataset)[0])

    prediction_frame = pd.DataFrame(
        [
            {
                "SO_Number": request.SO_Number,
                "Product": request.Product,
                "Pharmacy_Name": request.Pharmacy_Name,
                "Quantity_Ordered": request.Quantity_Ordered,
                "Unit_Price": request.Unit_Price,
                "predicted_fulfillment_rate": predicted_rate,
                "inv_zero_stock_rate": inference_dataset["inv_zero_stock_rate"].iloc[0],
                "po_short_receipt_rate": inference_dataset["po_short_receipt_rate"].iloc[0],
            }
        ]
    )

    output = add_prediction_context(prediction_frame, thresholds=ReviewThresholds()).iloc[0]
    return PredictionResponse(
        SO_Number=str(output["SO_Number"]),
        Product=str(output["Product"]),
        Pharmacy_Name=str(output["Pharmacy_Name"]),
        Quantity_Ordered=float(output["Quantity_Ordered"]),
        Unit_Price=float(output["Unit_Price"]),
        predicted_fulfillment_rate=float(output["predicted_fulfillment_rate"]),
        predicted_risk_band=str(output["predicted_risk_band"]),
        estimated_shortfall_value=float(output["estimated_shortfall_value"]),
        review_required=bool(output["review_required"]),
        review_reason=str(output["review_reason"]),
        deployment_stage=str(manifest["deployment_stage"]),
        production_promotion_allowed=bool(manifest["production_promotion_allowed"]),
    )


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Staging deployment manifest is missing. Run scripts/deploy_staging.py first.",
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Staging model is missing. Run scripts/deploy_staging.py first.",
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_context_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INVENTORY_PATH.exists() or not PURCHASE_ORDERS_PATH.exists():
        raise HTTPException(status_code=503, detail="Inventory or purchase-order source data is missing.")
    inventory = pd.read_csv(INVENTORY_PATH)
    purchase_orders = pd.read_csv(PURCHASE_ORDERS_PATH)
    return inventory, purchase_orders
