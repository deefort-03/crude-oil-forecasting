"""
LSTM Production Forecasting — Deep Learning Model
===================================================
Bidirectional LSTM with dropout, BatchNorm, and early stopping.
Uses StandardScaler (robust across train/test distribution shift).
Target: daily field oil production (Sm³/day).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense,
                                      Dropout, BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)

plt.rcParams.update({
    "figure.facecolor": "#0f0f23", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa", "ytick.color": "#aaa", "text.color": "#e0e0e0",
    "grid.color": "#2a2a4a", "grid.linestyle": "--", "grid.alpha": 0.5,
})

SEQ_LEN = 30


# ── Data prep ────────────────────────────────────────────────────────────────
def prepare_data():
    field = pd.read_csv("data/processed/field_features.csv", parse_dates=["date"])

    features = [
        "oil_vol", "gas_vol", "water_vol", "avg_gor", "avg_wct", "avg_pres",
        "active_wells", "doy_sin", "doy_cos", "oil_daily_change",
        "oil_vol_roll_mean7", "oil_vol_roll_mean30",
    ]
    feat_df = field[features].copy().fillna(0)

    # StandardScaler — handles distribution shift better than MinMaxScaler
    scaler = StandardScaler()
    split_idx = int(len(feat_df) * 0.85)
    scaler.fit(feat_df.values[:split_idx])
    scaled = scaler.transform(feat_df.values)
    joblib.dump(scaler, "models/lstm_std_scaler.pkl")

    # Build sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i, 0])   # oil_vol index 0

    X, y = np.array(X), np.array(y)

    # Time-based split
    split = int(len(X) * 0.85)
    return (X[:split], X[split:], y[:split], y[split:],
            scaler, field["date"].values[SEQ_LEN:])


def inverse_oil(y_scaled, scaler):
    n = scaler.n_features_in_
    dummy = np.zeros((len(y_scaled), n))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]


# ── Model ────────────────────────────────────────────────────────────────────
def build_model(seq_len, n_feat):
    model = Sequential([
        Input(shape=(seq_len, n_feat)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_training(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")
    a1.plot(history.history["loss"],     color="#00d4ff", label="Train")
    a1.plot(history.history["val_loss"], color="#ff6b6b", label="Val", linestyle="--")
    a1.set_title("MSE Loss"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(history.history["mae"],      color="#00d4ff", label="Train")
    a2.plot(history.history["val_mae"],  color="#ff6b6b", label="Val", linestyle="--")
    a2.set_title("MAE"); a2.set_xlabel("Epoch"); a2.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/10_lstm_training.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 10_lstm_training.png")


def plot_predictions(y_true, y_pred, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LSTM BiLSTM — Oil Production Forecasting", fontsize=16, fontweight="bold")

    # Time series
    ax = axes[0, 0]; ax.set_title("Actual vs Predicted (Test)", fontsize=12)
    ax.plot(y_true, color="#00d4ff", lw=1.5, label="Actual")
    ax.plot(y_pred, color="#ff6b6b", lw=1.5, label="Predicted", linestyle="--")
    ax.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.12, color="#ffd93d")
    ax.set_ylabel("Oil (Sm³/day)"); ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Metrics box
    mt = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    axes[0,1].axis("off")
    axes[0,1].text(0.5, 0.5, mt, ha="center", va="center", fontsize=14,
                   transform=axes[0,1].transAxes,
                   bbox=dict(boxstyle="round,pad=1", facecolor="#1a1a2e",
                             edgecolor="#00d4ff", alpha=0.9))
    axes[0,1].set_title("Test Set Metrics", fontsize=12)

    # Scatter
    ax2 = axes[1, 0]; ax2.set_title("Scatter: Actual vs Predicted", fontsize=12)
    ax2.scatter(y_true, y_pred, alpha=0.4, s=10, color="#c77dff")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5)
    ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")

    # Error distribution
    ax3 = axes[1, 1]; ax3.set_title("Residual Distribution", fontsize=12)
    res = y_true - y_pred
    ax3.hist(res, bins=40, color="#6bcb77", alpha=0.75, edgecolor="none")
    ax3.axvline(0, color="white", lw=1.5, linestyle="--")
    ax3.axvline(res.mean(), color="#ffd93d", lw=1.5, label=f"Mean={res.mean():.1f}")
    ax3.set_xlabel("Residual (Sm³/day)"); ax3.legend()

    plt.tight_layout()
    plt.savefig("reports/figures/11_lstm_predictions.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 11_lstm_predictions.png")


def plot_comparison(xgb_m, lstm_m):
    metrics = ["RMSE", "MAE", "R²", "MAPE"]
    x, w = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("XGBoost vs BiLSTM — Performance Comparison", fontsize=14, fontweight="bold")
    bars1 = ax.bar(x - w/2, [xgb_m[m] for m in metrics], w, label="XGBoost",  color="#00d4ff", alpha=0.85)
    bars2 = ax.bar(x + w/2, [lstm_m[m] for m in metrics], w, label="BiLSTM",   color="#ff6b6b", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score"); ax.legend(fontsize=11)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/12_model_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 12_model_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Preparing data...")
    X_tr, X_te, y_tr, y_te, scaler, dates = prepare_data()
    print(f"  Train {X_tr.shape} | Test {X_te.shape}")

    print("Building model...")
    model = build_model(SEQ_LEN, X_tr.shape[2])
    model.summary(print_fn=lambda x: None)  # silent

    print("Training (max 150 epochs, early stopping patience=20)...")
    history = model.fit(
        X_tr, y_tr,
        epochs=150, batch_size=32, validation_split=0.1,
        callbacks=[
            EarlyStopping("val_loss", patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )
    print(f"  Stopped at epoch {len(history.history['loss'])}")

    # Evaluate
    y_pred_sc  = model.predict(X_te, verbose=0).flatten()
    y_true_raw = inverse_oil(y_te,       scaler)
    y_pred_raw = inverse_oil(y_pred_sc,  scaler)

    lstm_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)),
        "MAE":  mean_absolute_error(y_true_raw, y_pred_raw),
        "R²":   r2_score(y_true_raw, y_pred_raw),
        "MAPE": float(np.mean(np.abs((y_true_raw - y_pred_raw) / (y_true_raw + 1e-6))) * 100),
    }
    print("\n  ── LSTM Metrics ─────────────────────")
    for k, v in lstm_metrics.items():
        print(f"  {k}: {v:.4f}")

    model.save("models/lstm_production.keras")
    np.save("models/lstm_predictions.npy", y_pred_raw)
    np.save("models/lstm_true.npy",        y_true_raw)
    print("  ✓ Model saved")

    # XGBoost metrics (reload)
    xgb_model = joblib.load("models/xgboost_production.pkl")
    field = pd.read_csv("data/processed/field_features.csv", parse_dates=["date"])
    feat_cols = [c for c in field.columns if c not in {"date","oil_vol"}]
    X_xgb, y_xgb = field[feat_cols].values, field["oil_vol"].values
    split = int(len(X_xgb) * 0.85)
    y_xp = xgb_model.predict(X_xgb[split:])
    xgb_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_xgb[split:], y_xp)),
        "MAE":  mean_absolute_error(y_xgb[split:], y_xp),
        "R²":   r2_score(y_xgb[split:], y_xp),
        "MAPE": float(np.mean(np.abs((y_xgb[split:] - y_xp)/(y_xgb[split:]+1e-6)))*100),
    }

    print("\nGenerating plots:")
    plot_training(history)
    plot_predictions(y_true_raw, y_pred_raw, lstm_metrics)
    plot_comparison(xgb_metrics, lstm_metrics)
    print("\n✅ LSTM training complete.")
