import os 
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

def load_data():
    """to load the pre-split parquet feature files """
    train = pd.read_parquet(os.path.join("data", "processed","train_features.parquet"))
    val = pd.read_parquet(os.path.join("data",'processed',"val_features.parquet"))
    test = pd.read_parquet(os.path.join("data",'processed',"test_features.parquet"))
    return train, val, test

def prepare_features(df):
    """to seprate features x and y and metadata"""
    # columns we do not want the model to train on
    drop_colns = ["customer_id","snapshot_date","split","is_at_risk","future_90d_revenue"]

    X = df.drop(columns=[col for col in drop_colns if col in df.columns])
    y = df["is_at_risk"]
    return X, y

def train_and_evaluate():
    print('[1/4] loading feature data')
    train, val, test = load_data()

    x_train, y_train = prepare_features(train)
    x_val, y_val = prepare_features(val)
    x_test, y_test = prepare_features(test)

    #to setup mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Customer_Risk_Enging")

    print("[2/4] Intializing mlflow run and training xgboost")
    with mlflow.start_run(run_name = "xgb_baseline"):

        #1.define model parameters
        params = {
            "objective" : "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "n_estimators": 150,
            "random_state": 42
        }

        # to log parameters to mlflow
        mlflow.log_params(params)

        #2. to train the model
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set =[(x_val, y_val)], verbose=False)

        print('[3/4] evaluating the model on test set')
        #predict probabilities ( likelihood of being at risk)
        y_pred_proba = model.predict_proba(x_test)[:,1]
        y_pred_class = (y_pred_proba > 0.5).astype(int)


        #calculate ml metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        precision  = precision_score(y_test, y_pred_class)
        recall = recall_score(y_test, y_pred_class)

        mlflow.log_metrics({"test_roc_auc":roc_auc, "test_pr_auc":pr_auc,
                            "test_precision":precision, "test_recall":recall
                            })
        print(f" ROC-AUC: {roc_auc:.4f}")
        print(f" PR-AUC: {pr_auc:.4f}")

        print("[4/4] calculating business metrics (expected revenue at risk)")
        #to calculate expected revenue at risk, we use a proxy for their future value
        #a standard business proxy is their spending over the previous 90 days
        #formula: expected_risk = prob of inactivity * recent 90d spend

        test_with_preds = test.copy()
        test_with_preds["risk_probability"] = y_pred_proba
        test_with_preds["expected_revenue_at_risk"] = (
            test_with_preds["risk_probability"] * test_with_preds["spend_last_90d"]
        )
        total_portfolio_risk = test_with_preds['expected_revenue_at_risk'].sum()

        #to log out custom business metrics
        mlflow.log_metric("total_expected_revenue_at_risk", total_portfolio_risk)
        print(f"total expected portfolio revenue at risk: £{total_portfolio_risk:,.2f}")

        #3. save the model to mlflow
        print("Logging model to mlflow registry")
        mlflow.xgboost.log_model(
            xgb_model=model, artifact_path="xgboost-model", registered_model_name="Risk_Xgboost_Model"
        )
        print("done! run complete. check http://127.0.0.1:5000 to see the tracked experiment")

if __name__ == "__main__":
    train_and_evaluate()