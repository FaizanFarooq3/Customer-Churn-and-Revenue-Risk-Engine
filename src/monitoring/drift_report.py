import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import mlflow.xgboost

MODEL_EXPORT_PATH = os.getenv("MODEL_PATH", "model_export")
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

FEATURE_COLS = [
    "frequency","monetary","total_quantity", "unique_products", "recency_days", "tenure_days",
    "avg_order_value", "avg_basket_size", "purchase_frequency_per_month","avg_days_between_orders",
    "is_uk", "cancellation_rate", "spend_last_90d", "spend_prev_90d", "spending_trend_90d",
]


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current distributions.
    PSI < 0.10: No significant drift
    0.10 <= PSI < 0.25: Moderate drift
    PSI >= 0.25: Significant drift
    """
    ref_clean = reference.dropna().to_numpy()
    cur_clean = current.dropna().to_numpy()

    # Use quantile breakpoints from reference distribution
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(ref_clean, quantiles))
    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_clean, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(ref_clean))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(cur_clean))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(round(psi, 4))


def generate_drift_monitoring_report():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("[1/4] Loading Reference (Train) and Current (Test) feature snapshots...")
    ref_df = pd.read_parquet(os.path.join("data", "processed", "train_features.parquet"))
    cur_df = pd.read_parquet(os.path.join("data", "processed", "test_features.parquet"))

    print(f"[2/4] Scoring both splits with production model from '{MODEL_EXPORT_PATH}'...")
    model = mlflow.xgboost.load_model(MODEL_EXPORT_PATH)
    ref_df["risk_probability"] = model.predict_proba(ref_df[FEATURE_COLS])[:, 1]
    cur_df["risk_probability"] = model.predict_proba(cur_df[FEATURE_COLS])[:, 1]

    print("[3/4] Running KS-Test and Population Stability Index (PSI) for Data & Prediction Drift...")
    monitored_cols = FEATURE_COLS + ["risk_probability"]
    drift_records = []

    for col in monitored_cols:
        ks_stat, p_value = ks_2samp(ref_df[col], cur_df[col])
        psi_val = calculate_psi(ref_df[col], cur_df[col])

        # Industry standard: drift flagged if PSI >= 0.10 or (KS p-value < 0.01 and KS stat >= 0.05)
        if psi_val >= 0.25:
            status = "SIGNIFICANT DRIFT"
        elif psi_val >= 0.10 or (p_value < 0.01 and ks_stat >= 0.05):
            status = "MODERATE DRIFT"
        else:
            status = "STABLE"

        drift_records.append(
            {
                "feature": col,
                "type": "Prediction" if col == "risk_probability" else "Feature",
                "ref_mean": round(float(ref_df[col].mean()), 3),
                "cur_mean": round(float(cur_df[col].mean()), 3),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_value), 6),
                "psi": psi_val,
                "status": status,
            }
        )

    drift_df = pd.DataFrame(drift_records).sort_values(by="psi", ascending=False)

    # 1. Save structured JSON report
    json_path = os.path.join(REPORTS_DIR, "drift_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_records, f, indent=2)

    # 2. Save visual PSI & Prediction Drift chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left subplot: PSI bar chart across all features + prediction
    colors = [
        "#d9534f" if s == "SIGNIFICANT DRIFT" else ("#f0ad4e" if s == "MODERATE DRIFT" else "#2a9d8f")
        for s in drift_df["status"]
    ]
    axes[0].barh(drift_df["feature"][::-1], drift_df["psi"][::-1], color=colors[::-1])
    axes[0].axvline(0.10, color="#f0ad4e", linestyle="--", linewidth=1.5, label="Moderate Threshold (0.10)")
    axes[0].axvline(0.25, color="#d9534f", linestyle="--", linewidth=1.5, label="Significant Threshold (0.25)")
    axes[0].set_title("Population Stability Index (PSI) by Feature")
    axes[0].set_xlabel("PSI Value")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    # Right subplot: Prediction Drift distribution (Train vs Test risk probabilities)
    axes[1].hist(
        ref_df["risk_probability"],
        bins=25,
        alpha=0.55,
        density=True,
        label="Reference (Train: Mar 2011)",
        color="#264653",
    )
    axes[1].hist(
        cur_df["risk_probability"],
        bins=25,
        alpha=0.55,
        density=True,
        label="Current (Test: Sep 2011)",
        color="#e76f51",
    )
    axes[1].set_title("Prediction Drift: Risk Probability Distribution")
    axes[1].set_xlabel("Predicted Inactivity Probability")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "drift_summary.png")
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    # 3. Save standalone HTML Monitoring Report
    html_path = os.path.join(REPORTS_DIR, "drift_report.html")
    html_table = drift_df.to_html(index=False, classes="drift-table", border=0)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Customer Risk Engine - Data & Prediction Drift Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 32px; background: #f8f9fa; color: #212529; }}
    h1 {{ color: #1d3557; margin-bottom: 4px; }}
    p.sub {{ color: #6c757d; margin-top: 0; }}
    .drift-table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }}
    .drift-table th {{ background: #1d3557; color: #fff; text-align: left; padding: 12px; }}
    .drift-table td {{ padding: 10px 12px; border-bottom: 1px solid #dee2e6; }}
    .drift-table tr:hover {{ background: #f1f3f5; }}
  </style>
</head>
<body>
  <h1>Customer Lifecycle & Revenue Risk Engine — Drift Monitoring Report</h1>
  <p class="sub">Reference Window: Train Split (2011-03-01, n={len(ref_df):,}) vs. Current Window: Test Split (2011-09-01, n={len(cur_df):,})</p>
  {html_table}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[4/4] Drift Analysis Complete!\n")
    print("================ DATA & PREDICTION DRIFT SUMMARY ================")
    print(drift_df.to_string(index=False))
    print("=================================================================")
    print(f"\nSaved HTML Report: '{html_path}'")
    print(f"Saved Drift Plot:  '{chart_path}'")
    print(f"Saved JSON Metrics:'{json_path}'")


if __name__ == "__main__":
    generate_drift_monitoring_report()