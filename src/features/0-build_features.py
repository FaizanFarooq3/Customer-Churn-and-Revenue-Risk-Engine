import os
import numpy as np
import pandas as pd
from src.config import engine
from src.features.queries import fetch_clean_transactions

SNAPSHOT_CONFIGS = {
    "train":"2021-03-01",
    "val":"2021-06-01",
    "test":"2021-09-01"
}
HORIZON_DAYS = 90

def build_customer_snapshot(
        df:pd.DataFrame, cutoff_date_str:str, split_name:str, horizon_days:int = 90
) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date_str)
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)


    #1. splitting into past (observation) and future (target windows)
    hist_df = df[df['invoice_date']<cutoff_date].copy()
    future_df = df[
        (df["invoice_date"]>=cutoff_date) & (df['invoice_date']<horizon_end) & (~df['is_cancelled'])
        ].copy()

    #valid historical purchases ( excluding cancellations)
    valid_hist = hist_df[~hist_df['is_cancelled']].copy()

    #2. core rfm + behavioral aggregations based on valid purchases 
    rfm = valid_hist.groupby('customer_id').agg(
        first_purchase = ("invoice_date", "min"),
        last_purchase = ("invoice_date", "max"),
        frequency = ("invoice_no", "nunique"),
        monetary = ("line_total","sum"),
        total_quantity = ("quantity","sum"),
        unique_products = ("stock_code","nunique"),
        country = ("country","last"),
        ).reset_index()

    #recency and tenure in days
    rfm["recency_days"] = (cutoff_date - rfm['last_purchase']).dt.days
    rfm["tenure_days"] = (cutoff_date - rfm['first_purchase']).dt.days

    #Behavioral ratios
    rfm['avg_order_value'] = (rfm["monetary"]/rfm["frequency"]).round(2)
    rfm['avg_basket_size'] = (rfm["total_quantity"]/rfm["frequency"]).round(2)
    rfm['purchase_freq_per_month'] = (rfm["frequency"]/np.maximum(rfm["tenure_days"]/30.0,1.0)).round(4)

    #avg days between consecutive orders (telescoping )
