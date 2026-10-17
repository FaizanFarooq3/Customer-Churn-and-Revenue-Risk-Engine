import os
import numpy as np
import pandas as pd
from src.config import engine
from src.features.queries import fetch_clean_transactions

SNAPSHOT_CONFIGS = {
    "train": "2011-03-01",
    "val": "2011-06-01",
    "test": "2011-09-01",
}
HORIZON_DAYS = 90


def build_customer_snapshot(
    df: pd.DataFrame, cutoff_date_str: str, split_name: str, horizon_days: int = 90
) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(cutoff_date_str)
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)

    # 1. Split strictly into Past (Observation) and Future (Target Window)
    hist_df = df[df["invoice_date"] < cutoff_date].copy()
    future_df = df[
        (df["invoice_date"] >= cutoff_date)
        & (df["invoice_date"] < horizon_end)
        & (~df["is_cancelled"])
    ].copy()

    # Valid historical purchases (excluding cancellations)
    valid_hist = hist_df[~hist_df["is_cancelled"]].copy()

    # 2. Core RFM + Behavioral aggregations on valid purchases
    rfm = valid_hist.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("line_total", "sum"),
        total_quantity=("quantity", "sum"),
        unique_products=("stock_code", "nunique"),
        country=("country", "last"),
    ).reset_index()

    # Recency and Tenure in days
    rfm["recency_days"] = (cutoff_date - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (cutoff_date - rfm["first_purchase"]).dt.days

    # Behavioral ratios
    rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)
    rfm["avg_basket_size"] = (rfm["total_quantity"] / rfm["frequency"]).round(2)
    rfm["purchase_frequency_per_month"] = (
        rfm["frequency"] / np.maximum(rfm["tenure_days"] / 30.0, 1.0)
    ).round(4)

    # Average days between consecutive orders (telescoping sum: (last - first) / (freq - 1))
    rfm["avg_days_between_orders"] = np.where(
        rfm["frequency"] > 1,
        ((rfm["last_purchase"] - rfm["first_purchase"]).dt.days / (rfm["frequency"] - 1)).round(2),
        0.0,
    )

    # Geographic encoding (UK accounts for ~90% of transactions)
    rfm["is_uk"] = (rfm["country"] == "United Kingdom").astype(int)

    # 3. Cancellation rate across all historical invoices
    total_inv = hist_df.groupby("customer_id")["invoice_no"].nunique().rename("total_invoices")
    canc_inv = (
        hist_df[hist_df["is_cancelled"]]
        .groupby("customer_id")["invoice_no"]
        .nunique()
        .rename("cancelled_invoices")
    )
    rfm = rfm.merge(total_inv, on="customer_id", how="left")
    rfm = rfm.merge(canc_inv, on="customer_id", how="left")
    rfm["cancelled_invoices"] = rfm["cancelled_invoices"].fillna(0)
    rfm["cancellation_rate"] = (rfm["cancelled_invoices"] / rfm["total_invoices"]).round(4)

    # 4. Temporal Velocity: Spend in last 90d vs prior 90d (91d to 180d ago)
    cutoff_90 = cutoff_date - pd.Timedelta(days=90)
    cutoff_180 = cutoff_date - pd.Timedelta(days=180)

    spend_last_90 = (
        valid_hist[valid_hist["invoice_date"] >= cutoff_90]
        .groupby("customer_id")["line_total"]
        .sum()
        .rename("spend_last_90d")
    )
    spend_prev_90 = (
        valid_hist[
            (valid_hist["invoice_date"] >= cutoff_180)
            & (valid_hist["invoice_date"] < cutoff_90)
        ]
        .groupby("customer_id")["line_total"]
        .sum()
        .rename("spend_prev_90d")
    )

    rfm = rfm.merge(spend_last_90, on="customer_id", how="left")
    rfm = rfm.merge(spend_prev_90, on="customer_id", how="left")
    rfm["spend_last_90d"] = rfm["spend_last_90d"].fillna(0.0).round(2)
    rfm["spend_prev_90d"] = rfm["spend_prev_90d"].fillna(0.0).round(2)

    # Spending trend ratio (>1 means accelerating spend, <1 means declining spend)
    rfm["spending_trend_90d"] = (
        (rfm["spend_last_90d"] + 1.0) / (rfm["spend_prev_90d"] + 1.0)
    ).round(4)

    # 5. Compute Future Targets in [cutoff_date, cutoff_date + 90 days)
    future_rev = (
        future_df.groupby("customer_id")["line_total"]
        .sum()
        .rename("future_90d_revenue")
    )
    rfm = rfm.merge(future_rev, on="customer_id", how="left")
    rfm["future_90d_revenue"] = rfm["future_90d_revenue"].fillna(0.0).round(2)

    # Customer is at risk (1) if they make 0 purchases in the next 90 days
    rfm["is_at_risk"] = (rfm["future_90d_revenue"] <= 0.0).astype(int)

    # Metadata columns
    rfm["snapshot_date"] = cutoff_date_str
    rfm["split"] = split_name

    # Drop raw timestamp and helper columns not fed into ML models
    drop_cols = ["first_purchase", "last_purchase", "country", "total_invoices", "cancelled_invoices"]
    rfm = rfm.drop(columns=drop_cols)

    return rfm


def run_feature_pipeline():
    print("[1/4] Fetching clean_transactions from PostgreSQL...")
    df = fetch_clean_transactions()
    print(f"      Loaded {len(df):,} clean transactions.")

    print("[2/4] Building temporal snapshots (preventing data leakage)...")
    snapshots = []
    for split_name, cutoff_str in SNAPSHOT_CONFIGS.items():
        snap_df = build_customer_snapshot(df, cutoff_str, split_name, HORIZON_DAYS)
        risk_rate = snap_df["is_at_risk"].mean() * 100
        print(
            f"      -> {split_name.upper()} (Cutoff {cutoff_str}): "
            f"{len(snap_df):,} customers | At-Risk Rate: {risk_rate:.1f}%"
        )
        snapshots.append(snap_df)

    full_feature_df = pd.concat(snapshots, ignore_index=True)

    print("[3/4] Saving feature splits to Parquet in data/processed/...")
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    for split_name in SNAPSHOT_CONFIGS.keys():
        split_df = full_feature_df[full_feature_df["split"] == split_name]
        parquet_path = os.path.join("data", "processed", f"{split_name}_features.parquet")
        split_df.to_parquet(parquet_path, index=False)

    print("[4/4] Saving master 'customer_features' table to PostgreSQL...")
    full_feature_df.to_sql(
        name="customer_features",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi",
    )
    print("Done! Phase 3 complete. Features stored in Parquet and PostgreSQL.")


if __name__ == "__main__":
    run_feature_pipeline()