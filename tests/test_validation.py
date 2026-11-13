import pandas as pd 
import pytest
from pandera.errors import SchemaErrors
from src.validation.schemas import raw_transaction_schema, clean_transaction_schema

def test_raw_transaction_schema_passes_valid_data():
    """valid raw data should be passed pandera validation without errors"""
    valid_df = pd.DataFrame({
        "invoice_no": ['536365'],
        "stock_code":["85123A"],
        "description":["WHITE HANGING HEART T-LIGHT HOLDER"],
        "quantity": [6],
        "invoice_date":pd.to_datetime(["2010-12-01 08:26:00"]),
        "unit_price": [2.55],
        "customer_id":[17850.0],
        "country":["United Kingdom"],
    })
    validated = raw_transaction_schema.validate(valid_df, lazy=True)
    assert len(validated) == 1

def test_raw_transaction_schema_rejects_negative_price():
    """Negative unit price like bad debt adjustments must fall validation  - customer_id is none and unit price is negative"""
    invalid_df = pd.DataFrame({
            "invoice_no": ["A563185"],
            "stock_code": ["B"],
            "description": ["Adjust bad debt"],
            "quantity": [1],
            "invoice_date": pd.to_datetime(["2011-08-12 14:50:00"]),
            "unit_price": [-53594.36],
            "customer_id": [None],
            "country": ["United Kingdom"],
        }
    )
    with pytest.raises(SchemaErrors):
        raw_transaction_schema.validate(invalid_df, lazy=True)

def test_clean_transaction_schema_passes_valid_cleaned_row():
    """cleaned data with is_cancelled and line_total should pass strict schema"""
    clean_df = pd.DataFrame(
    {
        "invoice_no": ["536365"],
        "stock_code": ["85123A"],
        "description": ["WHITE HANGING HEART T-LIGHT HOLDER"],
        "quantity": [6],
        "invoice_date": pd.to_datetime(["2010-12-01 08:26:00"]),
        "unit_price": [2.55],
        "customer_id": [17850],
        "country": ["United Kingdom"],
        "is_cancelled": [False],
        "line_total": [15.30],
    })
    validated = clean_transaction_schema.validate(clean_df, lazy=True)
    assert len(validated) ==1
                                        