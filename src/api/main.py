import os
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.schemas import CustomerFeaturesRequest, RiskPredictionResponse

# 1. Define globals first
MODEL = None
MODEL_VERSION = "2" 
MODEL_PATH = os.getenv("MODEL_PATH", "model_export")

def assign_segment_and_action(risk_prob: float, spend_last_90d: float) -> tuple[str, str]:
    """it maps risk prob and recent spend into a business segment and action"""
    if risk_prob >= 0.50:
        if spend_last_90d >= 500.0:
            return (
                "High-value at risk", "Immediate personal outreach + vip retenetion offer",
            )
        elif spend_last_90d > 0.0:
            return ("At risk", "Trigger automated win-back email + time limited discount",)
        else: 
            return ("Low-value inactive", "Low-cost automated re-engagement newsletter only",)
    else:
        return ("Healthy","Standard lifecycle marketing + cross-sell recommendations",)


# 2. Define the lifespan function second
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model from MLflow when the server starts."""
    global MODEL

    print(f"Loading the Model from: {MODEL_PATH}")
    
    try:
        MODEL = mlflow.xgboost.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise #crashes on startup so broken model is never hidden

    #it reads the model from model_export folder instead of mlflow --which was causing the error in ci      
    yield
    # Cleanup on shutdown
    MODEL = None

# 3. Initialize the app WITH the lifespan function third
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

    input_data = pd.DataFrame([request.model_dump()])
    
    try:
        risk_prob = float(MODEL.predict_proba(input_data)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    if risk_prob >= 0.75:
        band = "high"
    elif risk_prob >= 0.40:
        band = "medium"
    else:
        band = "low"

    revenue_at_risk = round(risk_prob * request.spend_last_90d, 2)
    segment, action = assign_segment_and_action(risk_prob, request.spend_last_90d)

    return RiskPredictionResponse(
        risk_probability=round(risk_prob, 4),
        risk_band=band,
        expected_revenue_at_risk=revenue_at_risk,
        customer_segment=segment,
        recommended_action=action,
        model_version=MODEL_VERSION,
    )