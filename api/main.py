"""
FastAPI — Crude Oil Production Intelligence API
================================================
Endpoints:
  POST /predict/production  → forecast next-day oil output (XGBoost)
  POST /predict/anomaly     → flag whether a day's readings are anomalous
  GET  /health              → service health check
  GET  /docs                → auto-generated Swagger UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import os

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "Crude Oil Production Intelligence API",
    description  = (
        "ML-powered API for oil production forecasting and anomaly detection. "
        "Based on Volve oilfield production data (North Sea, Norway). "
        "Models: XGBoost (baseline) · BiLSTM (primary) · Isolation Forest (anomalies)."
    ),
    version      = "1.0.0",
    contact      = {"name": "Fortune", "url": "https://github.com/fortuneiyoha"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "..", "models")

xgb_model      = None
anomaly_model  = None
anomaly_scaler = None
feat_names     = None


def load_models():
    global xgb_model, anomaly_model, anomaly_scaler, feat_names
    try:
        xgb_model      = joblib.load(os.path.join(MODELS_DIR, "xgboost_production.pkl"))
        anomaly_model  = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
        anomaly_scaler = joblib.load(os.path.join(MODELS_DIR, "anomaly_scaler.pkl"))
        feat_names     = joblib.load(os.path.join(MODELS_DIR, "anomaly_features.pkl"))
    except FileNotFoundError as e:
        print(f"Warning: Could not load models – {e}")


@app.on_event("startup")
async def startup_event():
    load_models()
    print("✅ Models loaded. API ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ProductionInput(BaseModel):
    """Features for next-day production forecast (XGBoost)."""
    gas_vol:       float = Field(..., example=850000.0,  description="Gas volume Sm³/day")
    water_vol:     float = Field(..., example=320.0,     description="Water volume Sm³/day")
    avg_gor:       float = Field(..., example=210.5,     description="Gas-oil ratio Sm³/Sm³")
    avg_wct:       float = Field(..., example=0.18,      description="Water cut fraction 0-1")
    avg_pres:      float = Field(..., example=220.0,     description="Downhole pressure bar")
    active_wells:  int   = Field(..., example=5,          description="Active producing wells")
    oil_lag1:      float = Field(..., example=1850.0,    description="Yesterday oil vol Sm³")
    oil_lag7:      float = Field(..., example=1920.0,    description="7-day lag oil vol Sm³")
    oil_lag30:     float = Field(..., example=2100.0,    description="30-day lag oil vol Sm³")
    oil_roll_mean7:  float = Field(..., example=1880.0,  description="7-day rolling mean Sm³")
    oil_roll_mean30: float = Field(..., example=1950.0,  description="30-day rolling mean Sm³")
    day_of_year:   int   = Field(..., example=180,        description="Day of year 1-365")
    month:         int   = Field(..., example=6,           description="Month 1-12")
    year:          int   = Field(..., example=2015,        description="Year")


class AnomalyInput(BaseModel):
    """Features for anomaly detection."""
    oil_vol:            float = Field(..., example=1850.0,  description="Daily oil vol Sm³")
    gas_vol:            float = Field(..., example=850000.0)
    water_vol:          float = Field(..., example=320.0)
    avg_gor:            float = Field(..., example=210.5)
    avg_wct:            float = Field(..., example=0.18)
    avg_pres:           float = Field(..., example=220.0)
    active_wells:       int   = Field(..., example=5)
    oil_vol_roll_mean7: float = Field(..., example=1880.0)
    oil_vol_roll_std7:  float = Field(..., example=85.0)
    oil_vol_roll_mean30:float = Field(..., example=1950.0)
    oil_vol_roll_std30: float = Field(..., example=120.0)
    oil_daily_change:   float = Field(..., example=-40.0)
    oil_pct_change:     float = Field(..., example=-0.021)
    oil_per_well:       float = Field(..., example=370.0)
    cum_oil_norm:       float = Field(..., example=0.72,    description="Normalised cumulative oil 0-1")


class ProductionResponse(BaseModel):
    predicted_oil_vol:     float
    unit:                  str
    model:                 str
    confidence_interval:   dict
    interpretation:        str


class AnomalyResponse(BaseModel):
    is_anomaly:     bool
    anomaly_score:  float
    risk_level:     str
    interpretation: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pad_xgb_features(data: dict, n_expected: int = 45) -> np.ndarray:
    """
    XGBoost was trained on 45 features. We map the API inputs to the
    most important ones and zero-fill the rest.
    """
    vec = np.zeros(n_expected)
    mapping = {
        0:  data.get("gas_vol", 0),
        1:  data.get("water_vol", 0),
        2:  data.get("avg_gor", 0),
        3:  data.get("avg_wct", 0),
        4:  data.get("avg_pres", 0),
        5:  data.get("active_wells", 0),
        6:  data.get("oil_lag1", 0),
        7:  data.get("oil_lag7", 0),
        8:  data.get("oil_lag30", 0),
        9:  data.get("oil_roll_mean7", 0),
        10: data.get("oil_roll_mean30", 0),
        11: data.get("day_of_year", 0),
        12: data.get("month", 0),
        13: data.get("year", 0),
        14: np.sin(2 * np.pi * data.get("day_of_year", 1) / 365.25),
        15: np.cos(2 * np.pi * data.get("day_of_year", 1) / 365.25),
    }
    for idx, val in mapping.items():
        vec[idx] = val
    return vec.reshape(1, -1)


def _interpret_production(val: float) -> str:
    if val > 3000:
        return "High production — field performing above average."
    elif val > 1500:
        return "Moderate production — normal decline-phase output."
    elif val > 500:
        return "Low production — late-life or partial shutdown likely."
    else:
        return "Very low production — potential shutdown or major issue."


def _risk_level(score: float) -> str:
    if score < -0.15:
        return "HIGH"
    elif score < -0.08:
        return "MEDIUM"
    else:
        return "LOW"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": xgb_model is not None,
        "version": "1.0.0",
    }


@app.post("/predict/production", response_model=ProductionResponse)
def predict_production(data: ProductionInput):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _pad_xgb_features(data.dict())
    pred = float(xgb_model.predict(X)[0])
    pred = max(pred, 0)
    margin = pred * 0.08     # ±8% CI based on training MAPE

    return ProductionResponse(
        predicted_oil_vol   = round(pred, 2),
        unit                = "Sm³/day",
        model               = "XGBoost (n_estimators=800)",
        confidence_interval = {
            "lower": round(max(pred - margin, 0), 2),
            "upper": round(pred + margin, 2),
        },
        interpretation = _interpret_production(pred),
    )


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(data: AnomalyInput):
    if anomaly_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([data.dict()])
    # Keep only features the model was trained on
    available = [f for f in feat_names if f in row.columns]
    missing   = [f for f in feat_names if f not in row.columns]
    for col in missing:
        row[col] = 0.0
    row = row[feat_names]

    X      = anomaly_scaler.transform(row.values)
    pred   = anomaly_model.predict(X)[0]        # 1=normal, -1=anomaly
    score  = float(anomaly_model.score_samples(X)[0])
    is_anom = pred == -1

    risk = _risk_level(score)
    if is_anom:
        interp = (f"⚠️  ANOMALY DETECTED (score={score:.3f}). "
                  f"Possible equipment failure or unplanned shutdown. "
                  f"Immediate inspection recommended.")
    else:
        interp = (f"✅ Normal production pattern (score={score:.3f}). "
                  f"No intervention required.")

    return AnomalyResponse(
        is_anomaly     = is_anom,
        anomaly_score  = round(score, 4),
        risk_level     = risk,
        interpretation = interp,
    )


@app.get("/wells/summary")
def wells_summary():
    """Return a mock summary of well production status."""
    wells = [
        {"well": "NO 15/9-F-1 C",  "status": "producing", "last_oil_vol": 186.3,  "anomaly": False},
        {"well": "NO 15/9-F-4",    "status": "producing", "last_oil_vol": 142.5,  "anomaly": False},
        {"well": "NO 15/9-F-5",    "status": "producing", "last_oil_vol": 98.7,   "anomaly": False},
        {"well": "NO 15/9-F-11",   "status": "shutdown",  "last_oil_vol": 0.0,    "anomaly": True},
        {"well": "NO 15/9-F-12",   "status": "producing", "last_oil_vol": 321.8,  "anomaly": False},
        {"well": "NO 15/9-F-14",   "status": "producing", "last_oil_vol": 175.2,  "anomaly": False},
        {"well": "NO 15/9-F-15 D", "status": "producing", "last_oil_vol": 89.4,   "anomaly": False},
    ]
    return {
        "field":        "Volve (NO 15/9)",
        "report_date":  "2016-12-30",
        "total_wells":  7,
        "active_wells": 6,
        "total_oil":    sum(w["last_oil_vol"] for w in wells),
        "wells":        wells,
    }
