import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so plots save cleanly without pop-up windows
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from src.models.evaluate import evaluate_model, log_evaluation_to_mlflow

FIGURES_DIR = os.path.join("reports", "figures")
MODEL_EXPORT_PATH = os.getenv("MODEL_PATH", "model_export")


def load_splits():
    """Load the pre-split Parquet feature files."""
    train = pd.read_parquet(os.path.join("data", "processed", "train_features.parquet"))
    val = pd.read_parquet(os.path.join("data", "processed", "val_features.parquet"))
    test = pd.read_parquet(os.path.join("data", "processed", "test_features.parquet"))
    return train, val, test


def prepare_xy(df: pd.DataFrame):
    """Separate feature matrix X, target y, and recent spend."""
    drop_cols = ["customer_id", "snapshot_date", "split", "is_at_risk", "future_90d_revenue"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["is_at_risk"]
    spend = df["spend_last_90d"]
    return X, y, spend


def plot_roc_and_pr_curves(y_test, preds_dict, save_path):
    """Plot side-by-side ROC and Precision-Recall curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. ROC Curve
    for name, probs in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, linewidth=2, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance")
    axes[0].set_title("ROC Curve Comparison (Test Set)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # 2. Precision-Recall Curve
    for name, probs in preds_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(recall, precision, linewidth=2, label=name)
    axes[1].set_title("Precision-Recall Curve Comparison (Test Set)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration(y_test, preds_dict, save_path):
    """Plot reliability/calibration curve comparing predicted vs actual risk."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    for name, probs in preds_dict.items():
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name)

    ax.set_title("Probability Calibration Curve (Test Set)")
    ax.set_xlabel("Mean Predicted Inactivity Probability")
    ax.set_ylabel("Actual Fraction of Inactive Customers")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_xgb_confusion_matrix(y_test, xgb_probs, save_path):
    """Plot confusion matrix for the production Tuned XGBoost model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred = (xgb_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active (0)", "At Risk (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Tuned XGBoost Confusion Matrix (Test Set)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_shap_explainability(xgb_model, X_test, save_path):
    """Compute SHAP values on Test features and save beeswarm summary plot."""
    # Sample up to 1,000 customers for fast, clean plotting
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)

    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance (Drivers of Customer Inactivity Risk)", pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_comparison_and_explainability():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("[1/5] Loading Train, Val, and Test feature splits...")
    train, val, test = load_splits()
    X_train, y_train, _ = prepare_xy(train)
    X_test, y_test, test_spend = prepare_xy(test)

    # Set up MLflow (uses local tracking server if running, or local mlruns/sqlite)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Customer_Risk_Engine")

    preds_dict = {}
    summary_rows = []

    # 1. Baseline 1: Logistic Regression
    print("[2/5] Training & evaluating Baseline 1: Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        lr_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        mlflow.log_params(lr_params)

        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_params)),
        ])
        lr_pipeline.fit(X_train, y_train)
        lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
        lr_metrics = evaluate_model(y_test, lr_probs, test_spend)
        log_evaluation_to_mlflow(lr_metrics, prefix="test_")

        preds_dict["Logistic Regression"] = lr_probs
        summary_rows.append({"Model": "Logistic Regression", **lr_metrics})

    # 2. Baseline 2: Random Forest
    print("[3/5] Training & evaluating Baseline 2: Random Forest...")
    with mlflow.start_run(run_name="random_forest_baseline"):
        rf_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(rf_params)

        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_model(y_test, rf_probs, test_spend)
        log_evaluation_to_mlflow(rf_metrics, prefix="test_")

        preds_dict["Random Forest"] = rf_probs
        summary_rows.append({"Model": "Random Forest", **rf_metrics})

    # 3. Production Model: Tuned XGBoost (loaded from model_export/)
    print(f"[4/5] Evaluating Production Tuned XGBoost from '{MODEL_EXPORT_PATH}'...")
    xgb_model = mlflow.xgboost.load_model(MODEL_EXPORT_PATH)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_probs, test_spend)
    preds_dict["XGBoost (Optuna Tuned)"] = xgb_probs
    summary_rows.append({"Model": "XGBoost (Optuna Tuned)", **xgb_metrics})

    # 4. Generate Diagnostic Plots & SHAP Explainability
    print("[5/5] Generating ROC/PR, Calibration, Confusion Matrix, and SHAP plots...")
    roc_pr_path = os.path.join(FIGURES_DIR, "model_comparison_roc_pr.png")
    cal_path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")

    plot_roc_and_pr_curves(y_test, preds_dict, roc_pr_path)
    plot_calibration(y_test, preds_dict, cal_path)
    plot_xgb_confusion_matrix(y_test, xgb_probs, cm_path)
    plot_shap_explainability(xgb_model, X_test, shap_path)

    # Log all diagnostic plots and comparison table to an MLflow evaluation run
    with mlflow.start_run(run_name="xgb_tuned_diagnostics_and_shap"):
        log_evaluation_to_mlflow(xgb_metrics, prefix="test_")
        mlflow.log_artifact(roc_pr_path, artifact_path="plots")
        mlflow.log_artifact(cal_path, artifact_path="plots")
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(shap_path, artifact_path="plots")

    # Print clean comparison table to terminal
# Print clean comparison table to terminal
    comparison_df = pd.DataFrame(summary_rows)
    print("\n================ MODEL COMPARISON ON TEST SET ================")
    print(comparison_df.to_string(index=False))
    print("==============================================================")
    print(f"\nAll 4 plots saved to '{FIGURES_DIR}/' and logged to MLflow!")


if __name__ == "__main__":
    run_comparison_and_explainability()