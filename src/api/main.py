import os
from contextlib import asynccontextmanager
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from src.api.schemas import CustomerFeaturesRequest, RiskPredictionResponse

# Global variables
MODEL = None
MODEL_VERSION = "1" 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model from MLflow when the server starts."""
    global MODEL
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri = f"models:/Risk_Xgboost_Model/{MODEL_VERSION}"
    
    print(f"Loading model from MLflow: {model_uri}")
    try:
        MODEL = mlflow.xgboost.load_model(model_uri)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        
    yield # Yield control to the FastAPI application
    
    # Code after yield executes on shutdown
    MODEL = None

# Initialize FastAPI app with the lifespan context manager
app = FastAPI(
    title="Customer Risk Engine API",
    description="Real-time inference service for customer inactivity prediction.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(request: CustomerFeaturesRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # 1. Convert JSON request to DataFrame matching XGBoost training columns
    input_data = pd.DataFrame([request.model_dump()])
    
    # 2. Generate Prediction (Probability of class 1)
    try:
        # predict_proba returns array like [[prob_class_0, prob_class_1]]
        risk_prob = float(MODEL.predict_proba(input_data)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # 3. Apply Business Logic
    if risk_prob >= 0.75:
        band = "high"
    elif risk_prob >= 0.40:
        band = "medium"
    else:
        band = "low"

    revenue_at_risk = round(risk_prob * request.spend_last_90d, 2)

    # 4. Return formatted response
    return RiskPredictionResponse(
        risk_probability=round(risk_prob, 4),
        risk_band=band,
        expected_revenue_at_risk=revenue_at_risk,
        model_version=MODEL_VERSION
    )