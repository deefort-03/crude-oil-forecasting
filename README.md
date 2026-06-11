# 🛢️ Crude Oil Production Forecasting & Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green.svg)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end ML pipeline for daily oil production forecasting and anomaly detection — trained on the official Equinor Volve oilfield dataset (North Sea, Norway, 2008–2016)**

---

## 📌 Problem Statement

Unplanned oilfield shutdowns and unexpected production drops cost operators millions per day. This project builds an ML system that:

1. **Forecasts** next-day field oil production (XGBoost + BiLSTM)
2. **Benchmarks** against the petroleum-engineering standard: Arps Decline Curve Analysis
3. **Detects** production anomalies using Isolation Forest
4. **Validates** all models with walk-forward cross-validation (no data leakage)
5. **Deploys** as a REST API (FastAPI)

---

## 🏆 Results (Real Volve Data)

| Model | RMSE (Sm³/d) | MAE (Sm³/d) | R² | MAPE* |
|-------|-------------|------------|-----|-------|
| Arps DCA — Hyperbolic *(industry baseline)* | 871.9 | 708.8 | -0.297 | 36.4% |
| BiLSTM | 296.0 | — | 0.845 | 14.0% |
| **XGBoost** *(winner)* | **66.1** | **—** | **0.992** | **3.3%** |

*MAPE computed on producing days only (oil > 10 Sm³/d) to avoid division by near-zero on shutdown days.*

**XGBoost 5-fold Walk-Forward CV: R² = 0.962 ± 0.041**

---

## 🏗️ Architecture

```
Equinor Volve Production Data (real .xlsx, 15,634 rows)
               │
               ▼
┌─────────────────────────┐
│  Data Cleaning          │  Filter producers · impute missing sensors ·
│  src/load_real_data.py  │  aggregate to field-level daily · compute GOR/WCT
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Feature Engineering    │  27 LSTM features · 75 XGBoost features
│  src/real_features.py   │  RobustScaler · lag/rolling · detrending ·
└────┬──────────┬──────── ┘  cyclical encoding · pressure gradient · EWM
     │          │
     ▼          ▼
┌─────────┐ ┌──────────┐ ┌─────────────────┐
│ XGBoost │ │  BiLSTM  │ │   Arps DCA      │  ← 3-way comparison
│ R²=0.992│ │ R²=0.845 │ │   R²=-0.297     │
│ CV R²   │ │ detrend  │ │  Exp + Hyperbolic│
│ =0.962  │ │ ratio    │ └─────────────────┘
└────┬────┘ └────┬─────┘
     │           │
     ▼           ▼
┌─────────────────────────┐
│  Isolation Forest       │  Unsupervised anomaly detection
│  contamination=5%       │  157 flagged days (5% of field life)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  FastAPI — Render       │  /predict/production · /predict/anomaly
└─────────────────────────┘
```

---

## 📁 Project Structure

```
crude-oil-forecasting/
├── data/
│   ├── raw/
│   │   └── Volve_production_data.xlsx   # Real Equinor Volve dataset
│   └── processed/
│       ├── real_field_daily.csv          # Cleaned field-level daily data
│       └── real_features.csv             # Full ML feature matrix (75 cols)
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Forecasting_Models.ipynb
│   └── 04_Anomaly_Detection.ipynb
│
├── src/
│   ├── load_real_data.py        # Real data cleaning pipeline
│   ├── real_features.py         # RobustScaler · lag · rolling · detrending
│   └── models/
│       ├── arps_dca.py          # Arps exponential + hyperbolic baseline
│       ├── real_xgboost.py      # XGBoost + walk-forward CV + SHAP
│       ├── real_lstm.py         # BiLSTM + detrending + proper inverse-transform
│       └── anomaly_detector.py  # Isolation Forest
│
├── api/
│   ├── main.py                  # FastAPI (production forecast + anomaly)
│   └── requirements.txt
│
├── models/                      # Saved .pkl and .keras artifacts
├── reports/figures/             # 20+ publication-quality plots
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/fortuneiyoha/crude-oil-forecasting.git
cd crude-oil-forecasting
pip install -r requirements.txt

# Step 1: Clean real data
python src/load_real_data.py

# Step 2: Feature engineering
python src/real_features.py

# Step 3: Industry baseline
python src/models/arps_dca.py

# Step 4: XGBoost + CV + SHAP
python src/models/real_xgboost.py

# Step 5: BiLSTM
python src/models/real_lstm.py

# Step 6: Anomaly detection
python src/models/anomaly_detector.py

# Step 7: Launch API
uvicorn api.main:app --reload --port 8000
```

---

## 🧠 Key Technical Decisions

### Why RobustScaler over StandardScaler?
Oil production data has heavy outliers — shutdown days produce 0 Sm³ while peak days hit 9,400+ Sm³. RobustScaler uses median and IQR, not mean and std, making it resilient to these extremes. StandardScaler's mean gets pulled by outliers, distorting the scaled representation of typical producing days.

### Why detrend the LSTM target?
Raw oil production is non-stationary — it has a multi-year declining trend that LSTMs struggle to extrapolate. By predicting `oil / 90d_rolling_mean` (a ratio centred around 1.0), the target becomes stationary. The LSTM learns short-term fluctuations around the trend, and the final prediction is `predicted_ratio × trend`. Without detrending, LSTM R² was negative. With it: R² = 0.845.

### Why does XGBoost beat LSTM here?
XGBoost is the stronger choice for structured tabular time series with rich hand-crafted features (75 total). It captures non-linear interactions between lag features, pressure, GOR, and water cut directly. LSTMs typically win when temporal dependencies are the dominant signal and raw sequences matter more than feature engineering. With 27 engineered sequence features, the two approaches partially overlap — but XGBoost's tree splits make better use of the tabular structure.

### Why is Arps DCA R² negative?
Arps DCA assumes a smooth, monotonically declining production curve. The real Volve field has planned shutdowns (production = 0 for days to weeks), well workovers, and multi-well interference that DCA cannot model. The ML models handle these events naturally through lag and rolling features.

### Walk-forward CV — why it matters
A random train/test split on time series leaks future information. Walk-forward CV trains on an expanding window and always tests on the next unseen period — mimicking real deployment. XGBoost's 5-fold CV R² = 0.962 ± 0.041 confirms the single-split result (R² = 0.992) is not an artefact.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `POST` | `/predict/production` | Forecast next-day oil (Sm³/day) |
| `POST` | `/predict/anomaly` | Flag anomalous production readings |
| `GET`  | `/wells/summary` | Field well status summary |

---

## 📚 References

- Equinor (2018). *Volve Field Data Village.* https://www.equinor.com/energy/volve-data-sharing
- Arps, J.J. (1945). *Analysis of Decline Curves.* SPE-945228-G
- Liu, F. et al. (2008). *Isolation Forest.* ICDM 2008. https://doi.org/10.1109/ICDM.2008.17
- Schuster, M. & Paliwal, K. (1997). *Bidirectional Recurrent Neural Networks.* https://doi.org/10.1109/78.650093
- Chen, T. & Guestrin, C. (2016). *XGBoost.* KDD 2016. https://doi.org/10.1145/2939672.2939785

---

## 👤 Author

**Fortune** — FLDC Cohort 6 · University of Ibadan  
[GitHub](https://github.com/fortuneiyoha) · [LinkedIn](https://linkedin.com/in/fortuneiyoha)

*Part of a 6-project data science portfolio.*
