# Customer Lifecycle & Revenue Risk Engine

![CI Status](https://github.com/FaizanFarooq3/Customer-Churn-and-Revenue-Risk-Engine/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.143.3-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

An end-to-end **Machine Learning Engineering & MLOps system** that predicts **customer inactivity risk**, quantifies **Expected 90-Day Revenue at Risk**, and assigns **actionable retention segments** on 1.07M+ retail transactions.

---

## 1. Business Problem & Why It Matters

Standard churn tutorials stop at binary classification (`1 = Churned`, `0 = Active`). In production e-commerce, a raw churn probability is insufficient for decision-making:
- A customer with a **92% risk score** who spends **£20/year** does not justify expensive retention outreach.
- A customer with a **65% risk score** who spends **£1,500/quarter** represents urgent revenue loss.

This engine solves the real business problem by combining **calibrated inactivity probability** with **customer monetary velocity**:

$$\text{Expected Revenue at Risk} = P(\text{Inactivity in Next 90 Days}) \times \text{Spend}_{\text{Last 90 Days}}$$

### Real-Time API Output (`/predict`)
```json
{
  "risk_probability": 0.1766,
  "risk_band": "low",
  "expected_revenue_at_risk": 114.76,
  "customer_segment": "Healthy",
  "recommended_action": "Standard lifecycle marketing + cross-sell recommendations",
  "model_version": "2"
}
```

### Actionable Customer Segmentation Matrix

| Risk Condition | 90-Day Spend | Segment | Recommended Business Action |
| :--- | :--- | :--- | :--- |
| `risk_prob >= 0.50` | `>= £500` | **High-value at risk** | Immediate personal outreach + VIP retention offer |
| `risk_prob >= 0.50` | `£0.01 – £499.99` | **At risk** | Trigger automated win-back email + time-limited discount |
| `risk_prob >= 0.50` | `£0.00` | **Low-value inactive** | Low-cost automated re-engagement newsletter only |
| `risk_prob < 0.50` | Any | **Healthy** | Standard lifecycle marketing + cross-sell recommendations |

---

## 2. System Architecture

```text
  ┌──────────────────────────┐
  │ UCI Online Retail II     │  1,067,371 raw transactions (Dec 2009 – Dec 2011)
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ PostgreSQL + Pandera     │  Ingests raw data -> Validates schema & filters bad-debt/guest
  │ Data Validation Gate     │  anomalies -> Stores 794,433 verified rows in clean_transactions
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ Point-in-Time Feature    │  Strict temporal cutoffs (Train / Val / Test) to prevent
  │ Engineering (15 Features)│  data leakage -> RFM, order velocity, cancellation rate
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ XGBoost + Optuna +       │  20 Optuna trials maximizing PR-AUC -> Tracks experiments,
  │ MLflow Model Registry    │  Brier calibration & Revenue at Risk -> Exports production artifact
  └────────────┬─────────────┘
               ▼
  ┌──────────────────────────┐
  │ FastAPI + Docker         │  Self-contained container loads exported model from /app/model_export
  │ + GitHub Actions CI      │  (Zero runtime dependency on MLflow server) -> Tested via Pytest
  └──────────────────────────┘
```

---

## 3. Preventing Temporal Data Leakage

Instead of a random `train_test_split` (which leaks future customer behavior into historical features), the feature pipeline constructs strict **point-in-time customer snapshots**:

- **Observation Window ($X$):** Only transactions strictly *before* the cutoff date (`invoice_date < cutoff_date`) are used to compute the 15 RFM and behavioral velocity features.
- **Prediction Window ($y$):** Only transactions in the **90 days following** the cutoff date (`[cutoff_date, cutoff_date + 90 days)`) determine whether the customer went inactive (`is_at_risk = 1` if 90-day spend is £0).

| Split | Cutoff Date | Customers | At-Risk Rate | Target Window (90 Days) |
| :--- | :--- | :---: | :---: | :--- |
| **Train** | `2011-03-01` | 4,513 | 64.9% | Mar 01, 2011 – May 30, 2011 |
| **Validation** | `2011-06-01` | 4,909 | 62.6% | Jun 01, 2011 – Aug 30, 2011 |
| **Test** | `2011-09-01` | 5,225 | 55.8% | Sep 01, 2011 – Nov 30, 2011 |

---

## 4. Model Performance (Test Split)

Hyperparameters were tuned across 20 trials using **Optuna** optimizing Validation **PR-AUC** and logged to the **MLflow Model Registry** (`Risk_Xgboost_Model`):

| Model Version | ROC-AUC | PR-AUC | Brier Score (Calibration) | Top-10% Risk Precision | Test Portfolio Revenue at Risk |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v1: XGBoost Baseline** | 0.7744 | 0.7946 | — | — | £445,723.11 |
| **v2: XGBoost (Optuna Tuned)** | **0.7996** | **0.8227** | **0.1656** | **0.9197** | **£350,829.22** |

---

## 5. Repository Structure

```text
customer-risk-engine/
├── .github/workflows/main.yml   # CI pipeline: runs Pytest + builds & smoke-tests Docker image
├── docker/
│   └── Dockerfile               # Production Python 3.13-slim image baking in src/ and model_export/
├── model_export/                # Exported production XGBoost artifact for zero-dependency serving
├── src/
│   ├── config.py                # SQLAlchemy PostgreSQL engine configuration
│   ├── ingestion/
│   │   └── load_to_postgres.py  # Loads raw Excel sheets into PostgreSQL (raw_transactions)
│   ├── validation/
│   │   ├── schemas.py           # Pandera schemas for raw and cleaned transaction tables
│   │   ├── run_validation.py    # Audits raw data quality and surfaces schema violations
│   │   └── clean_data.py        # Cleans records, flags cancellations & writes clean_transactions
│   ├── features/
│   │   ├── queries.py           # Extracts validated transactions from PostgreSQL
│   │   └── build_features.py    # Point-in-time RFM + velocity feature engineering (Parquet + DB)
│   ├── models/
│   │   ├── evaluate.py          # ML metrics, Brier score, top-decile precision & Revenue at Risk
│   │   ├── xgbbaseline_train.py # Baseline XGBoost training run logged to MLflow
│   │   ├── train.py             # Optuna hyperparameter tuning + MLflow registration
│   │   └── predict.py           # Batch scoring pipeline writing to customer_risk_scores table
│   └── api/
│       ├── schemas.py           # Pydantic v2 request/response contracts
│       └── main.py              # FastAPI service (/ and /predict) loading from model_export/
├── tests/
│   ├── sample_request.json      # Sample customer feature payload for API testing
│   ├── test_validation.py       # Unit tests for Pandera data quality schemas
│   ├── test_features.py         # Unit tests verifying zero temporal leakage in feature logic
│   └── test_api.py              # FastAPI TestClient integration tests against model_export/
├── docker-compose.yml           # Local container orchestration
├── requirements.txt             # Pinned production dependencies for Docker API container
├── requirements-dev.txt         # Full test & pipeline dependencies for local dev and CI
└── README.md
```

---

## 6. Quickstart Guide

### A. Run the Containerized API (No Local DB or MLflow Required)
Because the production model artifact is baked into `model_export/`, you can build and run the inference service immediately with Docker:

```bash
docker-compose up --build
```
- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `GET http://localhost:8000/`

### B. Run the Automated Test Suite
```bash
pip install -r requirements-dev.txt
pytest -v
```

### C. Run the Full Offline Training Pipeline (Requires Local PostgreSQL & Dataset)
1. Place `online_retail_II.xlsx` ([UCI Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)) in `data/raw/` and configure `.env` with your PostgreSQL credentials.
2. Execute each pipeline stage sequentially:
```bash
# 1. Ingest raw Excel sheets to PostgreSQL
python -m src.ingestion.load_to_postgres

# 2. Validate and clean transactions with Pandera
python -m src.validation.clean_data

# 3. Build leak-free temporal feature snapshots (Train/Val/Test)
python -m src.features.build_features

# 4. Tune XGBoost with Optuna and log to MLflow
python -m src.models.train

# 5. Run batch scoring across all active customers
python -m src.models.predict
```