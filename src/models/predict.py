import os 
import pandas as pd
import mlflow.xgboost
from src.config import engine
from src.features.queries import fetch_clean_transactions
from src.features.build_features import build_customer_snapshot

def run_batch_scoring(cutoff_date="2011-12-01", model_version="2"):
    """it generates predictions for all customers at a specific point in time"""
    print(f'[1/4] extracting live data up to {cutoff_date}')
    raw_df = fetch_clean_transactions()

    print(f"[2/4] generating current feature snapshot")
    #build features representing the current state of the customer
    features_df = build_customer_snapshot(raw_df, cutoff_date, split_name="live", horizon_days=90)

    #prepare for model input
    drop_cols = ['customer_id','snapshot_date','split', 'is_at_risk','future_90d_revenue']
    x_live = features_df.drop(columns=[col for col in drop_cols if col in features_df.columns])

    print(f"[3/4] loading the model version {model_version} from mlflow")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.xgboost.load_model(f'models:/Risk_Xgboost_Model/{model_version}')

    # GPU inference for this batch job(Checking both methods depending on whether MLflow loads a Booster or Scikit-Learn wrapper)
    if hasattr(model, 'set_param'):
        model.set_param({"device": "cuda"})
    else:
        model.set_params(device="cuda")


    print('[4/4] scoring customers')
    probabilities = model.predict_proba(x_live)[:,1]

    #attach predictions back to customer ids
    results_df = pd.DataFrame({
        "customer_id": features_df["customer_id"],
        "prediction_date": cutoff_date,
        "risk_probability": probabilities,
        "recent_90d_spend": features_df["spend_last_90d"]
    })

    results_df['expected_revenue_at_risk'] = results_df['risk_probability'] * results_df['recent_90d_spend']

    #save to db for dashboard consumption
    print('saving batch predictions to PostgreSQL')
    results_df.to_sql("customer_risk_scores" ,con=engine, if_exists="replace", index=False)
    print("done! batch scoring complete")

if __name__ == "__main__":
    run_batch_scoring(cutoff_date="2011-12-01")
    #in real envinorment this date would be dynamic (eg- pd.TimeStamp.today())