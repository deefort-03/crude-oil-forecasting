"""
Feature Engineering on Real Volve Data
========================================
RobustScaler (outlier-resistant) · lag/rolling features ·
cyclical encoding · detrending · anomaly features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib, os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models",         exist_ok=True)

TARGET = "oil_vol"
SEQ_LEN = 30

def load():
    df = pd.read_csv("data/processed/real_field_daily.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)

def add_time_features(df):
    df = df.copy()
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["year"]         = df["date"].dt.year
    df["year_idx"]     = (df["date"] - df["date"].min()).dt.days / 365.25
    df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    return df

def add_lag_features(df, lags=[1,2,3,7,14,30,60,90]):
    df = df.copy()
    for lag in lags:
        df[f"oil_lag{lag}"]   = df[TARGET].shift(lag)
        df[f"gas_lag{lag}"]   = df["gas_vol"].shift(lag)
    return df

def add_rolling_features(df, windows=[7,14,30,60,90]):
    df = df.copy()
    s = df[TARGET].shift(1)
    for w in windows:
        df[f"oil_rmean{w}"]  = s.rolling(w, min_periods=1).mean()
        df[f"oil_rstd{w}"]   = s.rolling(w, min_periods=1).std().fillna(0)
        df[f"oil_rmax{w}"]   = s.rolling(w, min_periods=1).max()
        df[f"oil_rmin{w}"]   = s.rolling(w, min_periods=1).min()
    # EWM (exponentially-weighted mean) – captures recency
    df["oil_ewm7"]  = s.ewm(span=7,  adjust=False).mean()
    df["oil_ewm30"] = s.ewm(span=30, adjust=False).mean()
    return df

def add_domain_features(df):
    df = df.copy()
    df["oil_daily_chg"]  = df[TARGET].diff().fillna(0)
    df["oil_pct_chg"]    = df[TARGET].pct_change().replace([np.inf,-np.inf],0).fillna(0)
    df["cum_oil_norm"]   = df[TARGET].cumsum() / (df[TARGET].sum() + 1e-6)
    df["oil_per_well"]   = df[TARGET] / (df["active_wells"] + 1e-6)
    df["liq_ratio"]      = df["oil_vol"] / (df["oil_vol"] + df["water_vol"] + 1e-6)
    # Pressure gradient (downhole - wellhead)
    df["pres_gradient"]  = df["avg_downhole_pres"] - df["avg_whp"]
    # Productivity Index proxy: oil per unit choke opening
    df["prod_index"]     = df["oil_vol"] / (df["avg_choke_size"] + 1e-6)
    return df

def detrend(df):
    """Add ratio-to-trend feature (key for LSTM stationarity)."""
    df = df.copy()
    trend = df[TARGET].rolling(90, min_periods=1, center=True).mean()
    df["trend90"]         = trend
    df["oil_ratio_trend"] = df[TARGET] / (trend + 1e-6)
    return df

def add_anomaly_features(df):
    df = df.copy()
    for w in [7, 30]:
        rm = df[TARGET].rolling(w, min_periods=1, center=True).mean()
        rs = df[TARGET].rolling(w, min_periods=1).std().fillna(1)
        df[f"oil_zscore{w}"]     = (df[TARGET] - rm) / (rs + 1e-6)
        df[f"oil_dev_mean{w}"]   = df[TARGET] - rm
    roll7 = df[TARGET].rolling(7, min_periods=1).mean().shift(1)
    df["sudden_drop_ratio"] = df[TARGET] / (roll7 + 1e-6)
    df["pres_deviation"]    = df["avg_downhole_pres"] - \
        df["avg_downhole_pres"].rolling(14, min_periods=1).mean()
    return df

def build_lstm_sequences(scaled: np.ndarray):
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i, 0])          # oil_vol ratio-to-trend at col 0
    return np.array(X), np.array(y)

def prepare(test_ratio=0.20):
    df = load()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_domain_features(df)
    df = detrend(df)
    df = add_anomaly_features(df)
    df = df.dropna().reset_index(drop=True)
    df.to_csv("data/processed/real_features.csv", index=False)

    # ── LSTM features (stationary: ratio-to-trend as target) ─────────────
    lstm_cols = [
        "oil_ratio_trend",        # TARGET for LSTM
        "gas_vol","water_vol",
        "avg_downhole_pres","avg_downhole_temp","avg_whp","avg_wht",
        "avg_choke_size","avg_annulus_press","pres_gradient",
        "active_wells","on_stream_hrs",
        "oil_rmean7","oil_rmean30","oil_rstd7",
        "oil_ewm7","oil_ewm30",
        "gor","water_cut","liq_ratio",
        "doy_sin","doy_cos","month_sin","month_cos","year_idx",
        "oil_daily_chg","oil_pct_chg",
    ]
    lstm_cols = [c for c in lstm_cols if c in df.columns]
    lstm_df   = df[lstm_cols].fillna(0)

    n_train = int(len(lstm_df) * (1 - test_ratio))
    # RobustScaler: median/IQR — resistant to production spikes & zeros
    scaler  = RobustScaler()
    scaler.fit(lstm_df.values[:n_train])
    scaled  = scaler.transform(lstm_df.values)
    joblib.dump(scaler, "models/real_lstm_scaler.pkl")
    joblib.dump(df["trend90"].values, "models/real_trend90.pkl")

    X, y_ratio = build_lstm_sequences(scaled)
    split = int(len(X) * (1 - test_ratio))
    trend90_seq = df["trend90"].values[SEQ_LEN:]

    lstm_out = dict(
        X_train=X[:split], X_test=X[split:],
        y_train=y_ratio[:split], y_test=y_ratio[split:],
        trend_train=trend90_seq[:split], trend_test=trend90_seq[split:],
        oil_train=df[TARGET].values[SEQ_LEN:SEQ_LEN+split],
        oil_test =df[TARGET].values[SEQ_LEN+split:],
        scaler=scaler, n_features=len(lstm_cols),
        dates_test=df["date"].values[SEQ_LEN+split:],
    )

    # ── XGBoost features ──────────────────────────────────────────────────
    xgb_exclude = {"date", TARGET, "trend90", "oil_ratio_trend"}
    xgb_cols    = [c for c in df.columns if c not in xgb_exclude]
    X_xgb       = df[xgb_cols].fillna(0).values
    y_xgb       = df[TARGET].values
    sp = int(len(X_xgb) * (1 - test_ratio))

    xgb_out = dict(
        X_train=X_xgb[:sp], X_test=X_xgb[sp:],
        y_train=y_xgb[:sp], y_test=y_xgb[sp:],
        feature_names=xgb_cols,
        dates_test=df["date"].values[sp:],
    )

    # ── Anomaly features ──────────────────────────────────────────────────
    anom_cols = [c for c in df.columns
                 if c not in {"date",TARGET,"trend90","oil_ratio_trend"}
                 and not c.startswith("oil_lag")
                 and not c.startswith("gas_lag")]
    anom_out  = dict(X=df[anom_cols].fillna(0).values,
                     feature_names=anom_cols, df=df)

    print(f"  LSTM   train/test : {lstm_out['X_train'].shape} / {lstm_out['X_test'].shape}")
    print(f"  XGBoost train/test: {xgb_out['X_train'].shape} / {xgb_out['X_test'].shape}")
    print(f"  Features (LSTM)   : {len(lstm_cols)}")
    print(f"  Features (XGBoost): {len(xgb_cols)}")
    return lstm_out, xgb_out, anom_out, df

if __name__ == "__main__":
    print("Building features on real Volve data...")
    l, x, a, df = prepare()
    print("\n✅ Feature engineering complete.")
