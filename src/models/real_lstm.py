"""
BiLSTM — Real Volve Data
Predicts ratio-to-trend (stationary target), properly inverse-transformed.
"""
import numpy as np, pandas as pd, joblib, os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42); np.random.seed(42)
SEQ_LEN = 30

plt.rcParams.update({
    "figure.facecolor":"#0f0f23","axes.facecolor":"#1a1a2e","axes.edgecolor":"#444",
    "axes.labelcolor":"#e0e0e0","xtick.color":"#aaa","ytick.color":"#aaa",
    "text.color":"#e0e0e0","grid.color":"#2a2a4a","grid.linestyle":"--","grid.alpha":0.5,
})

os.chdir("/home/claude/crude-oil-forecasting")
df = pd.read_csv("data/processed/real_features.csv", parse_dates=["date"])

lstm_cols = [c for c in [
    "oil_ratio_trend","gas_vol","water_vol",
    "avg_downhole_pres","avg_downhole_temp","avg_whp","avg_wht",
    "avg_choke_size","avg_annulus_press","pres_gradient",
    "active_wells","on_stream_hrs",
    "oil_rmean7","oil_rmean30","oil_rstd7",
    "oil_ewm7","oil_ewm30","gor","water_cut","liq_ratio",
    "doy_sin","doy_cos","month_sin","month_cos","year_idx",
    "oil_daily_chg","oil_pct_chg",
] if c in df.columns]

data  = df[lstm_cols].fillna(0).values
trend = df["trend90"].values
oil   = df["oil_vol"].values
dates = df["date"].values
n     = len(data)
split = int(n * 0.80)

# ── Scale INSIDE training window ──────────────────────────────────────────
sc = RobustScaler()
sc.fit(data[:split])
scaled = sc.transform(data)
joblib.dump(sc, "models/real_lstm_scaler.pkl")

# ── Sequences: scaled features → scaled ratio ─────────────────────────────
X, y_sc = [], []
for i in range(SEQ_LEN, n):
    X.append(scaled[i-SEQ_LEN:i])
    y_sc.append(scaled[i, 0])    # scaled col-0 = scaled oil_ratio_trend
X, y_sc = np.array(X), np.array(y_sc)

sp2 = int(len(X) * 0.80)
print(f"Train seqs: {sp2} | Test seqs: {len(X)-sp2} | Features: {len(lstm_cols)}")

# ── Inverse-transform helper: scaled col-0 → actual ratio ─────────────────
def inv_col0(y_sc_col0, scaler):
    """Recover the original oil_ratio_trend from its scaled value."""
    n   = len(y_sc_col0)
    nf  = scaler.n_features_in_
    buf = np.zeros((n, nf))
    buf[:, 0] = y_sc_col0
    return scaler.inverse_transform(buf)[:, 0]

# ── Model ─────────────────────────────────────────────────────────────────
m = Sequential([
    Input(shape=(SEQ_LEN, len(lstm_cols))),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation="relu"),
    Dense(1),
])
m.compile(optimizer=Adam(5e-4), loss="huber", metrics=["mae"])

print("Training BiLSTM (max 120 epochs)...")
history = m.fit(
    X[:sp2], y_sc[:sp2],
    epochs=120, batch_size=32, validation_split=0.1,
    callbacks=[
        EarlyStopping("val_loss", patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=0),
    ],
    verbose=0,
)
print(f"  Stopped at epoch {len(history.history['loss'])}")

# ── Reconstruct oil: inv-scale ratio → multiply by trend ─────────────────
y_pred_sc  = m.predict(X[sp2:], verbose=0).flatten()
y_pred_ratio = inv_col0(y_pred_sc, sc)          # back to ratio space
trend_te     = trend[SEQ_LEN + sp2:]
oil_te       = oil[SEQ_LEN + sp2:]
dates_te     = dates[SEQ_LEN + sp2:]
n_out        = len(y_pred_ratio)
y_pred_oil   = np.clip(y_pred_ratio[:n_out] * trend_te[:n_out], 0, None)
y_true_oil   = oil_te[:n_out]
dates_out    = dates_te[:n_out]

print(f"\n  Ratio range  : {y_pred_ratio.min():.3f} – {y_pred_ratio.max():.3f}")
print(f"  Pred oil     : {y_pred_oil.min():.1f} – {y_pred_oil.max():.1f} Sm³/d")
print(f"  True oil     : {y_true_oil.min():.1f} – {y_true_oil.max():.1f} Sm³/d")

mask = y_true_oil > 10
metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_true_oil, y_pred_oil)),
    "MAE":  mean_absolute_error(y_true_oil, y_pred_oil),
    "R²":   r2_score(y_true_oil, y_pred_oil),
    "MAPE": float(np.mean(np.abs((y_true_oil[mask]-y_pred_oil[mask])
                                  /(y_true_oil[mask]+1e-6)))*100),
}
print(f"\n  BiLSTM → RMSE={metrics['RMSE']:,.1f}  R²={metrics['R²']:.4f}  MAPE={metrics['MAPE']:.2f}%")

m.save("models/real_lstm.keras")
np.save("models/real_lstm_pred.npy",    y_pred_oil)
np.save("models/real_lstm_true.npy",    y_true_oil)
np.save("models/real_lstm_dates.npy",   dates_out)
np.save("models/real_lstm_loss.npy",    np.array(history.history["loss"]))
np.save("models/real_lstm_valloss.npy", np.array(history.history["val_loss"]))
joblib.dump(metrics, "models/real_lstm_metrics.pkl")
