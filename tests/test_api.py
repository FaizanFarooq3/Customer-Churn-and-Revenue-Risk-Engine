import json
import os
from fastapi.testclient import TestClient
from src.api.main import app

SAMPLE_REQUUEST_PATH = os.path.join("tests","sample_request.json")

def test_health_check_endpoint():
    """Verify GET / returns 200 and model_version 2"""
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == "healthy"
        assert data['model_version'] == '2'


def test_predict_endpoint_with_sample_json():
    """verify POST /predict loads model_export and scores sample_request.json"""
    with open(SAMPLE_REQUUEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with TestClient(app) as client:
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        data = response.json()

        #check all expected keys exist
        assert "risk_probability" in data
        assert "risk_band" in data
        assert "expected_revenue_at_risk" in data
        assert "customer_segment" in data
        assert "recommended_action" in data
        assert data['model_version'] == '2'

        #prob must be between 0 and 1
        assert 0.0 <= data['risk_probability'] <=1.0
        assert data['risk_band'] in {"low", "medium", "high"}
        assert data["customer_segment"] in {"High-value at risk", "At risk", "Low-value inactive", "Healthy",}