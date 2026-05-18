import pandas as pd
from fastapi.testclient import TestClient

from api.app import app


class ConstantModel:
    def predict(self, dataset):
        return [0.6 for _ in range(len(dataset))]


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.app._load_manifest",
        lambda: {
            "deployment_stage": "staging",
            "production_promotion_allowed": False,
        },
    )
    monkeypatch.setattr("api.app._load_model", lambda: ConstantModel())
    monkeypatch.setattr(
        "api.app._load_context_tables",
        lambda: (
            pd.DataFrame(
                [
                    {
                        "Product_Name": "Product_1",
                        "Product_Category": "OTC",
                        "Sales_Price": 100.0,
                        "Cost": 70.0,
                        "Qty_On_Hand": 20,
                        "Lagos": 10,
                        "Ibadan": 10,
                        "Vendors": "Vendor_1",
                        "Class": "Alpha",
                        "Responsible": "Ops",
                        "Date_Added": "2025-01-01",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "PO_Number": "PO_1",
                        "Order_Date": "01/01/2026",
                        "Delivery_Date": "05/01/2026",
                        "Vendor": "Vendor_1",
                        "Product_Name": "Product_1",
                        "Qty_Ordered": 10,
                        "Qty_Received": 8,
                        "Unit_Price": 50.0,
                        "Total": 400.0,
                    }
                ]
            ),
        ),
    )

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["deployment_stage"] == "staging"


def test_predict_endpoint_returns_review_decision(monkeypatch):
    monkeypatch.setattr(
        "api.app._load_manifest",
        lambda: {
            "deployment_stage": "staging",
            "production_promotion_allowed": False,
        },
    )
    monkeypatch.setattr("api.app._load_model", lambda: ConstantModel())
    monkeypatch.setattr(
        "api.app._load_context_tables",
        lambda: (
            pd.DataFrame(
                [
                    {
                        "Product_Name": "Product_1",
                        "Product_Category": "OTC",
                        "Sales_Price": 100.0,
                        "Cost": 70.0,
                        "Qty_On_Hand": 20,
                        "Lagos": 10,
                        "Ibadan": 10,
                        "Vendors": "Vendor_1",
                        "Class": "Alpha",
                        "Responsible": "Ops",
                        "Date_Added": "2025-01-01",
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "PO_Number": "PO_1",
                        "Order_Date": "01/01/2026",
                        "Delivery_Date": "05/01/2026",
                        "Vendor": "Vendor_1",
                        "Product_Name": "Product_1",
                        "Qty_Ordered": 10,
                        "Qty_Received": 8,
                        "Unit_Price": 50.0,
                        "Total": 400.0,
                    }
                ]
            ),
        ),
    )

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "SO_Number": "SO_1",
            "Product": "Product_1",
            "Pharmacy_Name": "Pharmacy_1",
            "Quantity_Ordered": 10,
            "Unit_Price": 100.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deployment_stage"] == "staging"
    assert payload["production_promotion_allowed"] is False
    assert payload["predicted_fulfillment_rate"] == 0.6
    assert payload["review_required"] is False
