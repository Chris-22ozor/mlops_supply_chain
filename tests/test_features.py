import pandas as pd

from core.features import (
    TARGET_COLUMN,
    build_inference_dataset,
    build_inventory_features,
    build_purchase_order_features,
    build_training_dataset,
    feature_columns,
)


def product_inventory_df():
    return pd.DataFrame(
        {
            "Product_Name": ["Product_1", "Product_1", "Product_2"],
            "Product_Category": ["OTC", "OTC", "Prescription"],
            "Sales_Price": [100.0, 120.0, 200.0],
            "Cost": [70.0, 80.0, 150.0],
            "Qty_On_Hand": [20, 0, 10],
            "Lagos": [10, 0, 4],
            "Ibadan": [10, 0, 6],
            "Vendors": ["Vendor_1", "Vendor_2", "Vendor_1"],
            "Class": ["Alpha", "Alpha", "Beta"],
            "Responsible": ["Ops", "Ops", "Procurement"],
            "Date_Added": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )


def purchase_orders_df():
    return pd.DataFrame(
        {
            "PO_Number": ["PO_1", "PO_2", "PO_3"],
            "Order_Date": ["01/01/2026", "03/01/2026", "01/01/2026"],
            "Delivery_Date": ["05/01/2026", "08/01/2026", "03/01/2026"],
            "Vendor": ["Vendor_1", "Vendor_2", "Vendor_1"],
            "Product_Name": ["Product_1", "Product_1", "Product_2"],
            "Qty_Ordered": [10, 20, 5],
            "Qty_Received": [8, 10, 5],
            "Unit_Price": [50.0, 60.0, 80.0],
            "Total": [400.0, 600.0, 400.0],
        }
    )


def sales_invoiced_df():
    return pd.DataFrame(
        {
            "SO_Number": ["SO_1", "SO_2"],
            "Pharmacy_Name": ["Pharmacy_1", "Pharmacy_2"],
            "Product": ["Product_1", "Product_2"],
            "Quantity_Ordered": [10, 4],
            "Quantity_Invoiced": [7, 4],
            "Unit_Price": [100.0, 200.0],
        }
    )


def test_inventory_features_aggregate_to_one_row_per_product():
    features = build_inventory_features(product_inventory_df())

    assert len(features) == 2
    product_1 = features.loc[features["Product_Name"] == "Product_1"].iloc[0]
    assert product_1["inv_qty_on_hand_avg"] == 10
    assert product_1["inv_zero_stock_rate"] == 0.5


def test_purchase_order_features_include_receipt_rate_and_lead_time():
    features = build_purchase_order_features(purchase_orders_df())

    product_1 = features.loc[features["Product_Name"] == "Product_1"].iloc[0]
    assert round(product_1["po_receipt_rate_avg"], 2) == 0.65
    assert product_1["po_lead_time_days_avg"] == 4.5
    assert product_1["po_short_receipt_rate"] == 1.0


def test_training_dataset_contains_target_and_safe_features():
    dataset = build_training_dataset(
        product_inventory_df(),
        purchase_orders_df(),
        sales_invoiced_df(),
    )

    assert len(dataset) == 2
    assert TARGET_COLUMN in dataset.columns
    assert "order_value" in dataset.columns
    assert "po_receipt_rate_avg" in dataset.columns
    assert "inv_zero_stock_rate" in dataset.columns
    assert dataset.loc[dataset["SO_Number"] == "SO_1", TARGET_COLUMN].iloc[0] == 0.7


def test_inference_dataset_excludes_target_and_invoiced_quantity():
    dataset = build_inference_dataset(
        product_inventory_df(),
        purchase_orders_df(),
        sales_invoiced_df(),
    )

    assert TARGET_COLUMN not in dataset.columns
    assert "Quantity_Invoiced" not in dataset.columns


def test_feature_columns_exclude_identifiers_and_target():
    dataset = build_training_dataset(
        product_inventory_df(),
        purchase_orders_df(),
        sales_invoiced_df(),
    )

    columns = feature_columns(dataset)

    assert "SO_Number" not in columns
    assert TARGET_COLUMN not in columns
    assert "Product" in columns
