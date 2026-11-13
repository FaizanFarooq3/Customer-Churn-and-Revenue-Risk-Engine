import os 
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import optuna
from src.models.evaluate import evaluate_model, log_evaluation_to_mlflow

def load_data():
    train = pd.read_parquet(os.path.join("data","processed","train_features.parquet"))
    val = pd.read_parquet(os.path.join("data","processed","val_features.parquet"))
    test = pd.read_parquet(os.path.join("data","processed","test_features.parquet"))
    return train, val, test

def prepare_features(df):
    drop_colns = ['customer_id','snapshot_date','split','is_at_risk','future_90d_revenue']
    x = df.drop(columns=[col for col in drop_colns if col in df.columns])
    y = df['is_at_risk']
    return x, y, df['spend_last_90d']

def optimize_hyperparameters(x_train, y_train, x_val, y_val, n_trials=15):
    """uses optuna to find the best xgboost parameters optimizing for pr-auc"""
    print(f"running {n_trials} optuna trials")

    def objective(trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric":"aucpr", #to optimize for precision recall auc
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42  ,
            "device": "cuda",    # to use the GPU
            "tree_method": "hist" #uses histogram based appraoch  

        }
        model = xgb.XGBClassifier(**param)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        #evaluation on validation set
        preds = model.predict_proba(x_val)[:,1]
        metrics = evaluate_model(y_val, preds, recent_spend=pd.Series(0,index=y_val.index))
        return metrics['pr_auc']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
    
def train_and_evaluate():
    print("[1/4] loading the feature data")
    train, val, test = load_data()

    x_train, y_train, _ = prepare_features(train)
    x_val, y_val, val_spend = prepare_features(val)
    x_test, y_test, test_spend = prepare_features(test)

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("Customer_Risk_Engine")

    print("[2/4] tuning the hyperparameters")
    best_params = optimize_hyperparameters(x_train, y_train, x_val, y_val, n_trials=20)
    print(f"best params found:{best_params}")

    # to make the final production model also trains on the GPU
    best_params["device"] = "cuda"
    best_params["tree_method"] = "hist"

    print("[3/4] intializing the mllfow run and training final model")
    with mlflow.start_run(run_name="xgb_tuned"):
        mlflow.log_params(best_params)

        model = xgb.XGBClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        print('[4/4] evaluating the model on test set')
        y_pred_proba = model.predict_proba(x_test)[:,1]

        #use our new evaluation model
        metrics = evaluate_model(y_test, y_pred_proba, test_spend)
        log_evaluation_to_mlflow(metrics, prefix="test_")

        print(f" roc-auc: {metrics['roc-auc']:.4f}")
        print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"Brier Score (Calibration): {metrics['brier_score']:.4f}")
        print(f"Total Expected Portfolio Revenue at Risk: £{metrics['total_expected_revenue_at_risk']:,.2f}")

        print("Logging model to MLflow registry...")
        mlflow.xgboost.log_model(
            xgb_model=model,artifact_path='xgboost-model',
            registered_model_name="Risk_Xgboost_Model"
        )
        print('check mlflow ui for the tuned model')

if __name__ == "__main__":
    train_and_evaluate()
