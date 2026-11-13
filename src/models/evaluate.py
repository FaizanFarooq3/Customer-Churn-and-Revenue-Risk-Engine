import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, precision_score, average_precision_score, brier_score_loss
import mlflow

def calculate_business_metrics(y_true, y_pred_proba, recent_spend):
    """Calculates the expected revenue at risk based on predicted probabilities"""
    df = pd.DataFrame({
        'true_risk':y_true,
        'risk_prob':y_pred_proba,
        'recent_spend':recent_spend
    })

    #expected value formula: Prob * potential_value
    df['expected_risk_value'] = df['risk_prob'] * df['recent_spend']

    #calculate the precision on the top 10% highest risk customers
    threshold_90th = np.percentile(df['risk_prob'],90)
    top_10_percent = df[df['risk_prob'] >= threshold_90th]
    top_10_precision = precision_score(top_10_percent['true_risk'], 
                                       (top_10_percent['risk_prob'] > 0.5).astype(int))
    return {
        "total_expected_revenue_at_risk" : df['expected_risk_value'].sum(),
        "top_10_percent_precision":top_10_precision
    }

def evaluate_model(y_true, y_pred_proba, recent_spend, threshold=0.5):
    """it generates a comphrensive dictionary of ml and business metrics"""
    y_pred_class = (y_pred_proba > threshold).astype(int)

    metrics = {
        "roc-auc":roc_auc_score(y_true, y_pred_proba),
        "pr_auc":average_precision_score(y_true, y_pred_proba),
        "brier_score":brier_score_loss(y_true, y_pred_proba), #it measures prob calibration
        "precision":precision_score(y_true, y_pred_class),
        "recall":recall_score(y_true,y_pred_class)
        }
    business_metrics = calculate_business_metrics(y_true, y_pred_proba, recent_spend)
    metrics.update(business_metrics)
    return metrics

def log_evaluation_to_mlflow(metrics, prefix="test_"):
    """Logs the dictionary of metrics to the active mlflow run"""
    mlflow_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(mlflow_metrics)
