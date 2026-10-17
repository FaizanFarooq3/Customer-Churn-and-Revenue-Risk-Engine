# helper module used to fetch clean data from PostgreSQL

import pandas as pd
from sqlalchemy import text
from src.config import engine


def fetch_clean_transactions() -> pd.DataFrame:
    query = text("""
        SELECT 
            invoice_no, 
            stock_code, 
            quantity, 
            invoice_date, 
            unit_price, 
            customer_id, 
            country,
            is_cancelled, 
            line_total
        FROM clean_transactions
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, con=conn)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df