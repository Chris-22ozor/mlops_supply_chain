"""Data validation logic for source supply-chain tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


PRODUCT_INVENTORY_COLUMNS = {
    "Product_Name",
    "Product_Category",
    "Sales_Price",
    "Cost",
    "Qty_On_Hand",
    "Lagos",
    "Ibadan",
    "Vendors",
    "Class",
    "Responsible",
    "Date_Added",
}

PURCHASE_ORDER_COLUMNS = {
    "PO_Number",
    "Order_Date",
    "Delivery_Date",
    "Vendor",
    "Product_Name",
    "Qty_Ordered",
    "Qty_Received",
    "Unit_Price",
    "Total",
}

SALES_INVOICED_COLUMNS = {
    "SO_Number",
    "Pharmacy_Name",
    "Product",
    "Quantity_Ordered",
    "Quantity_Invoiced",
    "Unit_Price",
}


@dataclass
class ValidationResult:
    """Container for validation failures, warnings, and table sizes."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.row_counts.update(other.row_counts)
        self.passed = self.passed and other.passed
        return self


def validate_product_inventory(df: pd.DataFrame) -> ValidationResult:
    """Validate the product inventory table."""

    result = ValidationResult(row_counts={"product_inventory": len(df)})
    if not _has_required_columns(df, PRODUCT_INVENTORY_COLUMNS, "Product_Inventory", result):
        return result

    _require_non_empty(df, ["Product_Name", "Vendors"], "Product_Inventory", result)
    _require_non_negative(
        df,
        ["Sales_Price", "Cost", "Qty_On_Hand", "Lagos", "Ibadan"],
        "Product_Inventory",
        result,
    )
    _require_dates(df, "Date_Added", "%Y-%m-%d", "Product_Inventory", result)

    location_total = _numeric(df["Lagos"]) + _numeric(df["Ibadan"])
    qty_on_hand = _numeric(df["Qty_On_Hand"])
    mismatch_count = int((location_total > qty_on_hand).sum())
    if mismatch_count:
        result.add_warning(
            "Product_Inventory has "
            f"{mismatch_count} rows where Lagos + Ibadan is greater than Qty_On_Hand."
        )

    return result


def validate_purchase_orders(df: pd.DataFrame) -> ValidationResult:
    """Validate the purchase orders table."""

    result = ValidationResult(row_counts={"purchase_orders": len(df)})
    if not _has_required_columns(df, PURCHASE_ORDER_COLUMNS, "Purchase_Orders", result):
        return result

    _require_non_empty(df, ["PO_Number", "Vendor", "Product_Name"], "Purchase_Orders", result)
    _require_unique(df, "PO_Number", "Purchase_Orders", result)
    _require_non_negative(
        df,
        ["Qty_Ordered", "Qty_Received", "Unit_Price", "Total"],
        "Purchase_Orders",
        result,
    )
    _require_positive(df, ["Qty_Ordered"], "Purchase_Orders", result)

    order_dates = _require_dates(df, "Order_Date", "%d/%m/%Y", "Purchase_Orders", result)
    delivery_dates = _require_dates(df, "Delivery_Date", "%d/%m/%Y", "Purchase_Orders", result)

    if order_dates is not None and delivery_dates is not None:
        bad_delivery_count = int((delivery_dates < order_dates).sum())
        if bad_delivery_count:
            result.add_error(
                "Purchase_Orders has "
                f"{bad_delivery_count} rows where Delivery_Date is before Order_Date."
            )

    qty_ordered = _numeric(df["Qty_Ordered"])
    qty_received = _numeric(df["Qty_Received"])
    over_received_count = int((qty_received > qty_ordered).sum())
    if over_received_count:
        result.add_error(
            "Purchase_Orders has "
            f"{over_received_count} rows where Qty_Received is greater than Qty_Ordered."
        )

    expected_total = (qty_received * _numeric(df["Unit_Price"])).round(2)
    actual_total = _numeric(df["Total"]).round(2)
    bad_total_count = int(((expected_total - actual_total).abs() > 0.01).sum())
    if bad_total_count:
        result.add_error(
            "Purchase_Orders has "
            f"{bad_total_count} rows where Total does not equal Qty_Received * Unit_Price."
        )

    return result


def validate_sales_invoiced(df: pd.DataFrame) -> ValidationResult:
    """Validate the sales invoiced table."""

    result = ValidationResult(row_counts={"sales_invoiced": len(df)})
    if not _has_required_columns(df, SALES_INVOICED_COLUMNS, "Sales_Invoiced", result):
        return result

    _require_non_empty(df, ["SO_Number", "Pharmacy_Name", "Product"], "Sales_Invoiced", result)
    _require_unique(df, "SO_Number", "Sales_Invoiced", result)
    _require_non_negative(
        df,
        ["Quantity_Ordered", "Quantity_Invoiced", "Unit_Price"],
        "Sales_Invoiced",
        result,
    )
    _require_positive(df, ["Quantity_Ordered"], "Sales_Invoiced", result)

    quantity_ordered = _numeric(df["Quantity_Ordered"])
    quantity_invoiced = _numeric(df["Quantity_Invoiced"])
    over_invoiced_count = int((quantity_invoiced > quantity_ordered).sum())
    if over_invoiced_count:
        result.add_error(
            "Sales_Invoiced has "
            f"{over_invoiced_count} rows where Quantity_Invoiced is greater than Quantity_Ordered."
        )

    return result


def validate_cross_table_joins(
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    sales: pd.DataFrame,
) -> ValidationResult:
    """Validate product and vendor coverage across all source tables."""

    result = ValidationResult()

    required = PRODUCT_INVENTORY_COLUMNS | PURCHASE_ORDER_COLUMNS | SALES_INVOICED_COLUMNS
    all_columns = set(inventory.columns) | set(purchase_orders.columns) | set(sales.columns)
    if not required.issubset(all_columns):
        result.add_error("Cross-table validation requires all source tables to have valid schemas.")
        return result

    inventory_products = _clean_set(inventory["Product_Name"])
    po_products = _clean_set(purchase_orders["Product_Name"])
    sales_products = _clean_set(sales["Product"])
    inventory_vendors = _clean_set(inventory["Vendors"])
    po_vendors = _clean_set(purchase_orders["Vendor"])

    _require_subset(sales_products, inventory_products, "sales products", "inventory products", result)
    _require_subset(sales_products, po_products, "sales products", "purchase-order products", result)
    _require_subset(po_products, inventory_products, "purchase-order products", "inventory products", result)
    _require_subset(po_vendors, inventory_vendors, "purchase-order vendors", "inventory vendors", result)

    return result


def validate_all_tables(
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    sales: pd.DataFrame,
) -> ValidationResult:
    """Run every table-level and cross-table validation check."""

    result = ValidationResult()
    result.merge(validate_product_inventory(inventory))
    result.merge(validate_purchase_orders(purchase_orders))
    result.merge(validate_sales_invoiced(sales))

    if result.passed:
        result.merge(validate_cross_table_joins(inventory, purchase_orders, sales))

    return result


def _has_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
    result: ValidationResult,
) -> bool:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        result.add_error(f"{table_name} is missing required columns: {', '.join(missing)}.")
        return False
    return True


def _require_non_empty(
    df: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
    result: ValidationResult,
) -> None:
    for column in columns:
        empty_count = int(df[column].astype("string").str.strip().eq("").fillna(True).sum())
        if empty_count:
            result.add_error(f"{table_name}.{column} has {empty_count} empty values.")


def _require_unique(
    df: pd.DataFrame,
    column: str,
    table_name: str,
    result: ValidationResult,
) -> None:
    duplicate_count = int(df[column].duplicated().sum())
    if duplicate_count:
        result.add_error(f"{table_name}.{column} has {duplicate_count} duplicate values.")


def _require_non_negative(
    df: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
    result: ValidationResult,
) -> None:
    for column in columns:
        values = _numeric(df[column])
        invalid_count = int(values.isna().sum())
        negative_count = int((values < 0).sum())
        if invalid_count:
            result.add_error(f"{table_name}.{column} has {invalid_count} non-numeric values.")
        if negative_count:
            result.add_error(f"{table_name}.{column} has {negative_count} negative values.")


def _require_positive(
    df: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
    result: ValidationResult,
) -> None:
    for column in columns:
        values = _numeric(df[column])
        non_positive_count = int((values <= 0).sum())
        if non_positive_count:
            result.add_error(f"{table_name}.{column} has {non_positive_count} non-positive values.")


def _require_dates(
    df: pd.DataFrame,
    column: str,
    date_format: str,
    table_name: str,
    result: ValidationResult,
) -> pd.Series | None:
    parsed = pd.to_datetime(df[column], format=date_format, errors="coerce")
    bad_count = int(parsed.isna().sum())
    if bad_count:
        result.add_error(f"{table_name}.{column} has {bad_count} invalid dates.")
        return None
    return parsed


def _require_subset(
    actual: set[str],
    expected: set[str],
    actual_name: str,
    expected_name: str,
    result: ValidationResult,
) -> None:
    missing = sorted(actual - expected)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
        result.add_error(
            f"{len(missing)} {actual_name} are missing from {expected_name}: {preview}{suffix}."
        )


def _clean_set(series: pd.Series) -> set[str]:
    return set(series.dropna().astype(str).str.strip()) - {""}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
