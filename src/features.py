"""
Feature Engineering Pipeline
==============================
Transforms raw production data into ML-ready features for:
  1. Production forecasting (LSTM + XGBoost)
  2. Anomaly detection (Isolation Forest)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib, os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_raw():
    df = pd.read_csv("data/raw/volve_production.csv", parse_dates=["DATEPRD"])
    df = df.sort_values(["WELL_BORE_CODE", "DATEPRD"]).reset_index(drop=True)
    return df


def build_field_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all wells into a daily field-level time series.
    This is what the LSTM will forecast (total field oil output).
    """
    field = (
        df.groupby("DATEPRD")
        .agg(
            oil_vol    =("BORE_OIL_VOL",   "sum"),
            gas_vol    =("BORE_GAS_VOL",   "sum"),
            water_vol  =("BORE_WAT_VOL",   "sum"),
            avg_gor    =("GOR",            "mean"),
            avg_wct    =("WATER_CUT",      "mean"),
            avg_pres   =("AVG_DOWNHOLE_PRESSURE", "mean"),
            active_wells=("BORE_OIL_VOL",  lambda x: (x > 10).sum()),
        )
        .reset_index()
        .rename(columns={"DATEPRD": "date"})
    )
    field = field.set_index("date").asfreq("D").reset_index()
    field = field.ffill().fillna(0)
    return field


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features that capture seasonality / maintenance cycles."""
    df = df.copy()
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["date"].dt.year
    df["year_frac"]    = df["year"] + df["day_of_year"] / 365.25
    # Cyclical encoding so day_of_year is continuous at year boundary
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "oil_vol",
                     lags: list = None) -> pd.DataFrame:
    """Lag features for autoregressive modeling."""
    if lags is None:
        lags = [1, 2, 3, 7, 14, 30, 60, 90]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "oil_vol",
                         windows: list = None) -> pd.DataFrame:
    """Rolling statistics to capture trend and variability."""
    if windows is None:
        windows = [7, 14, 30, 90]
    df = df.copy()
    for w in windows:
        df[f"{target}_roll_mean{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std{w}"]  = df[target].shift(1).rolling(w, min_periods=1).std().fillna(0)
        df[f"{target}_roll_max{w}"]  = df[target].shift(1).rolling(w, min_periods=1).max()
        df[f"{target}_roll_min{w}"]  = df[target].shift(1).rolling(w, min_periods=1).min()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-specific petroleum engineering features."""
    df = df.copy()
    # Day-over-day production change
    df["oil_daily_change"]  = df["oil_vol"].diff().fillna(0)
    df["oil_pct_change"]    = df["oil_vol"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    # Liquids ratio
    total_liq = df["oil_vol"] + df["water_vol"] + 1e-6
    df["oil_fraction"]      = df["oil_vol"] / total_liq
    # Cumulative production (depletion indicator)
    df["cum_oil"]           = df["oil_vol"].cumsum()
    df["cum_oil_norm"]      = df["cum_oil"] / df["cum_oil"].max()
    # Production efficiency (oil per active well)
    df["oil_per_well"]      = df["oil_vol"] / (df["active_wells"] + 1e-6)
    return df


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set specifically for anomaly detection.
    Focuses on deviations from expected behavior.
    """
    df = df.copy()
    # Deviation from rolling mean
    for w in [7, 30]:
        roll_mean = df["oil_vol"].rolling(w, min_periods=1, center=True).mean()
        df[f"oil_dev_from_mean{w}"] = df["oil_vol"] - roll_mean
        df[f"oil_zscore{w}"] = (df["oil_vol"] - roll_mean) / (
            df["oil_vol"].rolling(w, min_periods=1).std().fillna(1) + 1e-6
        )
    # Sudden production drop flag (>40% drop vs 7-day avg)
    roll7 = df["oil_vol"].rolling(7, min_periods=1).mean().shift(1)
    df["sudden_drop_ratio"] = df["oil_vol"] / (roll7 + 1e-6)
    # Pressure anomaly
    roll_pres = df["avg_pres"].rolling(14, min_periods=1).mean()
    df["pressure_deviation"] = df["avg_pres"] - roll_pres
    return df


def create_lstm_sequences(data: np.ndarray, seq_len: int = 30):
    """Reshape flat array into (samples, timesteps, features) for LSTM."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])          # 0 = oil_vol (first column)
    return np.array(X), np.array(y)


def prepare_forecasting_data(seq_len: int = 30, test_ratio: float = 0.15):
    """
    Full pipeline → returns train/test splits for LSTM and XGBoost.
    """
    df_raw = load_raw()
    field  = build_field_timeseries(df_raw)

    field = add_time_features(field)
    field = add_lag_features(field)
    field = add_rolling_features(field)
    field = add_derived_features(field)
    field = field.dropna().reset_index(drop=True)

    # ── LSTM data ────────────────────────────────────────────────────────────
    lstm_features = [
        "oil_vol", "gas_vol", "water_vol", "avg_gor", "avg_wct", "avg_pres",
        "active_wells", "doy_sin", "doy_cos", "oil_daily_change",
        "oil_vol_roll_mean7", "oil_vol_roll_mean30",
    ]
    lstm_df = field[lstm_features].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(lstm_df)
    joblib.dump(scaler, "models/lstm_scaler.pkl")

    X_lstm, y_lstm = create_lstm_sequences(scaled, seq_len)

    split = int(len(X_lstm) * (1 - test_ratio))
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

    # ── XGBoost data ─────────────────────────────────────────────────────────
    xgb_exclude = {"date", "oil_vol"}
    xgb_features = [c for c in field.columns if c not in xgb_exclude]
    X_xgb = field[xgb_features].values
    y_xgb = field["oil_vol"].values

    split_xgb = int(len(X_xgb) * (1 - test_ratio))
    X_train_xgb, X_test_xgb = X_xgb[:split_xgb], X_xgb[split_xgb:]
    y_train_xgb, y_test_xgb = y_xgb[:split_xgb], y_xgb[split_xgb:]

    # ── Anomaly data ──────────────────────────────────────────────────────────
    anomaly_df  = build_anomaly_features(field)
    anomaly_cols = [c for c in anomaly_df.columns
                    if c not in {"date","oil_vol"} and "anomaly" not in c.lower()]
    X_anomaly   = anomaly_df[anomaly_cols].fillna(0).values

    # Save processed datasets
    field.to_csv("data/processed/field_features.csv", index=False)
    anomaly_df.to_csv("data/processed/anomaly_features.csv", index=False)
    np.save("data/processed/X_train_lstm.npy", X_train_lstm)
    np.save("data/processed/X_test_lstm.npy",  X_test_lstm)
    np.save("data/processed/y_train_lstm.npy", y_train_lstm)
    np.save("data/processed/y_test_lstm.npy",  y_test_lstm)

    print(f"  LSTM  train/test : {X_train_lstm.shape} / {X_test_lstm.shape}")
    print(f"  XGBoost train/test: {X_train_xgb.shape} / {X_test_xgb.shape}")
    print(f"  Anomaly features  : {X_anomaly.shape}")
    print(f"  Features saved to data/processed/")

    return {
        "field":        field,
        "anomaly_df":   anomaly_df,
        "lstm": {
            "X_train": X_train_lstm, "X_test": X_test_lstm,
            "y_train": y_train_lstm, "y_test": y_test_lstm,
            "scaler":  scaler, "seq_len": seq_len, "n_features": len(lstm_features),
        },
        "xgb": {
            "X_train": X_train_xgb, "X_test": X_test_xgb,
            "y_train": y_train_xgb, "y_test": y_test_xgb,
            "feature_names": xgb_features,
        },
        "anomaly": {
            "X": X_anomaly,
            "feature_names": anomaly_cols,
        },
    }


if __name__ == "__main__":
    print("Building feature sets...")
    data = prepare_forecasting_data()
    print("\n✅ Feature engineering complete.")
