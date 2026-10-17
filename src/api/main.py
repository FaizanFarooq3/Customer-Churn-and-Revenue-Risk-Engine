import os
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.schemas import CustomerFeaturesRequest, RiskPredictionResponse

# 1. Define globals first
MODEL = None
MODEL_VERSION = "2" 
MODEL_PATH = os.getenv("MODEL_PATH", "MODEL_EXPORT")

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

    return RiskPredictionResponse(
        risk_probability=round(risk_prob, 4),
        risk_band=band,
        expected_revenue_at_risk=revenue_at_risk,
        model_version=MODEL_VERSION
    )