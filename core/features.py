"""Leakage-safe feature engineering for training and batch inference."""

from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "fulfillment_rate"


def build_training_dataset(
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Build a model-ready training dataset with target and safe features.

    The returned frame keeps sales identifiers for auditability, includes the target,
    and excludes columns that directly reveal the target from the feature set.
    """

    sales_features = _build_sales_base(sales, include_target=True)
    inventory_features = build_inventory_features(inventory)
    purchase_order_features = build_purchase_order_features(purchase_orders)

    dataset = (
        sales_features.merge(
            inventory_features,
            left_on="Product",
            right_on="Product_Name",
            how="left",
        )
        .drop(columns=["Product_Name"])
        .merge(
            purchase_order_features,
            left_on="Product",
            right_on="Product_Name",
            how="left",
        )
        .drop(columns=["Product_Name"])
    )

    return dataset


def build_inference_dataset(
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Build model-ready features for scoring records without target leakage."""

    sales_features = _build_sales_base(sales, include_target=False)
    inventory_features = build_inventory_features(inventory)
    purchase_order_features = build_purchase_order_features(purchase_orders)

    dataset = (
        sales_features.merge(
            inventory_features,
            left_on="Product",
            right_on="Product_Name",
            how="left",
        )
        .drop(columns=["Product_Name"])
        .merge(
            purchase_order_features,
            left_on="Product",
            right_on="Product_Name",
            how="left",
        )
        .drop(columns=["Product_Name"])
    )

    return dataset


def build_inventory_features(inventory: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inventory records to one feature row per product."""

    df = inventory.copy()
    df["margin"] = df["Sales_Price"] - df["Cost"]
    df["margin_pct"] = _safe_divide(df["margin"], df["Sales_Price"])
    df["total_location_stock"] = df["Lagos"] + df["Ibadan"]
    df["zero_stock_flag"] = (df["Qty_On_Hand"] <= 0).astype(int)

    numeric = (
        df.groupby("Product_Name", as_index=False)
        .agg(
            inv_sales_price_avg=("Sales_Price", "mean"),
            inv_cost_avg=("Cost", "mean"),
            inv_qty_on_hand_avg=("Qty_On_Hand", "mean"),
            inv_lagos_avg=("Lagos", "mean"),
            inv_ibadan_avg=("Ibadan", "mean"),
            inv_margin_avg=("margin", "mean"),
            inv_margin_pct_avg=("margin_pct", "mean"),
            inv_total_location_stock_avg=("total_location_stock", "mean"),
            inv_zero_stock_rate=("zero_stock_flag", "mean"),
            inv_record_count=("Product_Name", "size"),
        )
    )

    categorical = (
        df.groupby("Product_Name", as_index=False)
        .agg(
            inv_product_category=("Product_Category", _mode_or_unknown),
            inv_vendor=("Vendors", _mode_or_unknown),
            inv_class=("Class", _mode_or_unknown),
            inv_responsible=("Responsible", _mode_or_unknown),
        )
    )

    return numeric.merge(categorical, on="Product_Name", how="left")


def build_purchase_order_features(purchase_orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate purchase-order history to one feature row per product."""

    df = purchase_orders.copy()
    order_dates = pd.to_datetime(df["Order_Date"], format="%d/%m/%Y", errors="coerce")
    delivery_dates = pd.to_datetime(df["Delivery_Date"], format="%d/%m/%Y", errors="coerce")

    df["po_receipt_rate"] = _safe_divide(df["Qty_Received"], df["Qty_Ordered"])
    df["po_lead_time_days"] = (delivery_dates - order_dates).dt.days
    df["po_short_receipt_flag"] = (df["Qty_Received"] < df["Qty_Ordered"]).astype(int)

    product_features = (
        df.groupby("Product_Name", as_index=False)
        .agg(
            po_ordered_qty_avg=("Qty_Ordered", "mean"),
            po_unit_price_avg=("Unit_Price", "mean"),
            po_receipt_rate_avg=("po_receipt_rate", "mean"),
            po_lead_time_days_avg=("po_lead_time_days", "mean"),
            po_short_receipt_rate=("po_short_receipt_flag", "mean"),
            po_record_count=("Product_Name", "size"),
            po_primary_vendor=("Vendor", _mode_or_unknown),
        )
    )

    return product_features


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return model input columns, excluding identifiers and target columns."""

    excluded = {
        "SO_Number",
        TARGET_COLUMN,
        "Quantity_Invoiced",
    }
    return [column for column in dataset.columns if column not in excluded]


def _build_sales_base(sales: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    columns = ["SO_Number", "Pharmacy_Name", "Product", "Quantity_Ordered", "Unit_Price"]
    df = sales.loc[:, columns].copy()
    df["order_value"] = df["Quantity_Ordered"] * df["Unit_Price"]

    if include_target:
        df[TARGET_COLUMN] = _safe_divide(sales["Quantity_Invoiced"], sales["Quantity_Ordered"])

    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    return result.replace([float("inf"), float("-inf")], pd.NA)


def _mode_or_unknown(series: pd.Series) -> str:
    non_empty = series.dropna().astype(str).str.strip()
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        return "UNKNOWN"
    return str(non_empty.mode().sort_values().iloc[0])
