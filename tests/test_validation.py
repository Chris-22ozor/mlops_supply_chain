import pandas as pd

from core.validation import (
    validate_all_tables,
    validate_product_inventory,
    validate_purchase_orders,
    validate_sales_invoiced,
)


def product_inventory_df(**overrides):
    data = {
        "Product_Name": ["Product_1"],
        "Product_Category": ["OTC"],
        "Sales_Price": [100.0],
        "Cost": [70.0],
        "Qty_On_Hand": [20],
        "Lagos": [10],
        "Ibadan": [10],
        "Vendors": ["Vendor_1"],
        "Class": ["Alpha"],
        "Responsible": ["Ops"],
        "Date_Added": ["2025-01-01"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def purchase_orders_df(**overrides):
    data = {
        "PO_Number": ["PO_1"],
        "Order_Date": ["01/01/2026"],
        "Delivery_Date": ["05/01/2026"],
        "Vendor": ["Vendor_1"],
        "Product_Name": ["Product_1"],
        "Qty_Ordered": [10],
        "Qty_Received": [8],
        "Unit_Price": [50.0],
        "Total": [400.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def sales_invoiced_df(**overrides):
    data = {
        "SO_Number": ["SO_1"],
        "Pharmacy_Name": ["Pharmacy_1"],
        "Product": ["Product_1"],
        "Quantity_Ordered": [10],
        "Quantity_Invoiced": [7],
        "Unit_Price": [100.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_data_passes():
    result = validate_all_tables(
        product_inventory_df(),
        purchase_orders_df(),
        sales_invoiced_df(),
    )

    assert result.passed
    assert result.errors == []


def test_missing_column_fails():
    df = product_inventory_df().drop(columns=["Product_Name"])

    result = validate_product_inventory(df)

    assert not result.passed
    assert any("missing required columns" in error for error in result.errors)


def test_negative_quantity_fails():
    df = product_inventory_df(Qty_On_Hand=[-1])

    result = validate_product_inventory(df)

    assert not result.passed
    assert any("negative values" in error for error in result.errors)


def test_bad_date_fails():
    df = purchase_orders_df(Order_Date=["2026-01-01"])

    result = validate_purchase_orders(df)

    assert not result.passed
    assert any("invalid dates" in error for error in result.errors)


def test_delivery_before_order_fails():
    df = purchase_orders_df(Order_Date=["05/01/2026"], Delivery_Date=["01/01/2026"])

    result = validate_purchase_orders(df)

    assert not result.passed
    assert any("Delivery_Date is before Order_Date" in error for error in result.errors)


def test_sales_product_missing_from_inventory_fails():
    result = validate_all_tables(
        product_inventory_df(Product_Name=["Product_2"]),
        purchase_orders_df(),
        sales_invoiced_df(),
    )

    assert not result.passed
    assert any("sales products are missing from inventory products" in error for error in result.errors)


def test_inventory_location_mismatch_warns_without_failure():
    df = product_inventory_df(Qty_On_Hand=[5], Lagos=[4], Ibadan=[4])

    result = validate_product_inventory(df)

    assert result.passed
    assert result.warnings


def test_sales_over_invoiced_fails():
    df = sales_invoiced_df(Quantity_Ordered=[5], Quantity_Invoiced=[6])

    result = validate_sales_invoiced(df)

    assert not result.passed
    assert any("Quantity_Invoiced is greater than Quantity_Ordered" in error for error in result.errors)
