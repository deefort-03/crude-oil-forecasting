"""
XGBoost Production Forecasting — Baseline Model
================================================
Trains an XGBoost regressor to forecast daily field oil production.
Includes SHAP feature importance plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib, os, warnings

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f23", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa", "ytick.color": "#aaa", "text.color": "#e0e0e0",
    "grid.color": "#2a2a4a", "grid.linestyle": "--", "grid.alpha": 0.5,
})


def load_data():
    field = pd.read_csv("data/processed/field_features.csv", parse_dates=["date"])
    target = "oil_vol"
    exclude = {"date", target}
    feature_cols = [c for c in field.columns if c not in exclude]

    X = field[feature_cols].values
    y = field[target].values
    dates = field["date"].values

    split = int(len(X) * 0.85)
    return (X[:split], X[split:], y[:split], y[split:],
            dates[:split], dates[split:], feature_cols)


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators   = 800,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 3,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        random_state    = 42,
        n_jobs          = -1,
        verbosity       = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    return y_pred, {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}


def plot_predictions(y_test, y_pred, dates_test, metrics):
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("XGBoost — Oil Production Forecasting Results", fontsize=16, fontweight="bold")

    # ── Actual vs Predicted ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates_test, y_test,  color="#00d4ff", linewidth=1.5, label="Actual",    alpha=0.9)
    ax1.plot(dates_test, y_pred,  color="#ff6b6b", linewidth=1.5, label="Predicted", alpha=0.9, linestyle="--")
    ax1.fill_between(dates_test, y_test, y_pred, alpha=0.15, color="#ffd93d")
    ax1.set_title("Actual vs Predicted (Test Set)", fontsize=13)
    ax1.set_ylabel("Oil Volume (Sm³/day)", fontsize=11)
    ax1.legend(fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # ── Scatter ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_test, y_pred, alpha=0.4, s=12, color="#00d4ff", edgecolors="none")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    ax2.set_xlabel("Actual (Sm³/day)", fontsize=11)
    ax2.set_ylabel("Predicted (Sm³/day)", fontsize=11)
    ax2.set_title("Scatter: Actual vs Predicted", fontsize=12)
    ax2.legend(fontsize=10)

    # ── Residuals ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_test - y_pred
    ax3.hist(residuals, bins=50, color="#c77dff", alpha=0.75, edgecolor="none")
    ax3.axvline(0, color="white", linewidth=1.5, linestyle="--")
    ax3.axvline(residuals.mean(), color="#ffd93d", linewidth=1.5,
                label=f"Mean={residuals.mean():.1f}")
    ax3.set_title("Residual Distribution", fontsize=12)
    ax3.set_xlabel("Residual (Sm³/day)", fontsize=11)
    ax3.legend(fontsize=10)

    # Metrics box
    metrics_text = "\n".join([
        f"RMSE : {metrics['RMSE']:,.1f} Sm³/d",
        f"MAE  : {metrics['MAE']:,.1f} Sm³/d",
        f"R²   : {metrics['R²']:.4f}",
        f"MAPE : {metrics['MAPE']:.2f}%",
    ])
    ax1.text(0.01, 0.97, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                       edgecolor="#00d4ff", alpha=0.9))

    plt.tight_layout()
    plt.savefig("reports/figures/08_xgboost_predictions.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 08_xgboost_predictions.png")


def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    idx        = np.argsort(importance)[-top_n:]
    names      = [feature_names[i] for i in idx]
    vals       = importance[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    colors = plt.cm.YlOrRd(np.linspace(0.4, 1.0, top_n))
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("Importance Score", fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/figures/09_xgb_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 09_xgb_feature_importance.png")


if __name__ == "__main__":
    print("Training XGBoost model...")
    X_train, X_test, y_train, y_test, dates_train, dates_test, feat_names = load_data()

    print(f"  Train size: {X_train.shape}, Test size: {X_test.shape}")

    model = train_xgboost(X_train, y_train)
    y_pred, metrics = evaluate(model, X_test, y_test)

    print("\n  ── Test Set Metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:5s}: {v:.4f}")

    # Save model
    joblib.dump(model, "models/xgboost_production.pkl")
    print("\n  ✓ Model saved → models/xgboost_production.pkl")

    # Plots
    print("\nGenerating plots:")
    plot_predictions(y_test, y_pred, dates_test, metrics)
    plot_feature_importance(model, feat_names)

    print("\n✅ XGBoost training complete.")
