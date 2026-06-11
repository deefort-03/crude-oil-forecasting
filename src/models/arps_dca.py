"""
Arps Decline Curve Analysis — Industry Baseline
================================================
Fits exponential AND hyperbolic decline curves to field oil production.
Used as the petroleum-engineering benchmark against XGBoost & LSTM.

  Exponential : q(t) = q_i * exp(-D * t)
  Hyperbolic  : q(t) = q_i / (1 + b*D*t)^(1/b)

Reference: Arps, J.J. (1945). SPE-945228-G
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings, os
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":"#0f0f23","axes.facecolor":"#1a1a2e",
    "axes.edgecolor":"#444","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#aaa","ytick.color":"#aaa","text.color":"#e0e0e0",
    "grid.color":"#2a2a4a","grid.linestyle":"--","grid.alpha":0.5,
})

# ── Arps models ───────────────────────────────────────────────────────────────
def exp_decline(t, qi, Di):
    return qi * np.exp(-Di * t)

def hyp_decline(t, qi, Di, b):
    b = np.clip(b, 1e-6, 0.9999)
    return qi / (1 + b * Di * t) ** (1 / b)

# ── Fit & predict ─────────────────────────────────────────────────────────────
def fit_arps(df: pd.DataFrame, train_frac=0.80):
    oil = df["oil_vol"].values.copy()
    n   = len(oil)
    t   = np.arange(n, dtype=float)

    split   = int(n * train_frac)
    t_train = t[:split]
    q_train = oil[:split]
    t_test  = t[split:]
    q_test  = oil[split:]

    # Only fit on producing days (oil > 0) to avoid shutdown artefacts
    mask = q_train > 10
    t_fit, q_fit = t_train[mask], q_train[mask]

    qi0 = q_fit.max()

    # ── Exponential ──────────────────────────────────────────────────────────
    try:
        (qi_e, Di_e), _ = curve_fit(
            exp_decline, t_fit, q_fit,
            p0=[qi0, 0.001], bounds=([0,0],[qi0*2,0.1]),
            maxfev=5000
        )
        q_pred_e_train = exp_decline(t_train, qi_e, Di_e)
        q_pred_e_test  = exp_decline(t_test,  qi_e, Di_e)
    except Exception:
        q_pred_e_train = np.full_like(t_train, q_fit.mean(), dtype=float)
        q_pred_e_test  = np.full_like(t_test,  q_fit.mean(), dtype=float)

    # ── Hyperbolic ───────────────────────────────────────────────────────────
    try:
        (qi_h, Di_h, b_h), _ = curve_fit(
            hyp_decline, t_fit, q_fit,
            p0=[qi0, 0.001, 0.5], bounds=([0,0,0.01],[qi0*2,0.1,0.999]),
            maxfev=5000
        )
        q_pred_h_train = hyp_decline(t_train, qi_h, Di_h, b_h)
        q_pred_h_test  = hyp_decline(t_test,  qi_h, Di_h, b_h)
    except Exception:
        q_pred_h_train = np.full_like(t_train, q_fit.mean(), dtype=float)
        q_pred_h_test  = np.full_like(t_test,  q_fit.mean(), dtype=float)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def met(yt, yp, name):
        return {
            "model": name,
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE":  mean_absolute_error(yt, yp),
            "R²":   r2_score(yt, yp),
            "MAPE": float(np.mean(np.abs((yt-yp)/(yt+1e-6)))*100),
        }

    metrics_e = met(q_test, q_pred_e_test, "Arps Exponential")
    metrics_h = met(q_test, q_pred_h_test, "Arps Hyperbolic")

    return {
        "t": t, "oil": oil, "split": split,
        "exp":  {"train": q_pred_e_train, "test": q_pred_e_test,
                 "params": {"qi": qi_e, "Di": Di_e}, "metrics": metrics_e},
        "hyp":  {"train": q_pred_h_train, "test": q_pred_h_test,
                 "params": {"qi": qi_h, "Di": Di_h, "b": b_h}, "metrics": metrics_h},
        "dates": df["date"].values,
    }


def plot_dca(result):
    split   = result["split"]
    dates   = result["dates"]
    oil     = result["oil"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Arps Decline Curve Analysis — Volve Field (Real Data)",
                 fontsize=15, fontweight="bold")

    # ── Full timeline ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(dates, oil, s=4, alpha=0.4, color="#888", label="Actual (daily)")
    ax.plot(dates, pd.Series(oil).rolling(14).mean(),
            color="white", lw=1.5, alpha=0.7, label="14d MA")
    ax.plot(dates[:split], result["exp"]["train"],
            color="#ffd93d", lw=2, label="Arps Exponential (train)", linestyle="--")
    ax.plot(dates[:split], result["hyp"]["train"],
            color="#6bcb77", lw=2, label="Arps Hyperbolic (train)", linestyle="--")
    ax.plot(dates[split:], result["exp"]["test"],
            color="#ffd93d", lw=2.5, label="Exponential (test)")
    ax.plot(dates[split:], result["hyp"]["test"],
            color="#6bcb77", lw=2.5, label="Hyperbolic (test)")
    ax.axvline(dates[split], color="#ff6b6b", lw=1.5, linestyle=":",
               label="Train/Test split")
    ax.set_ylabel("Oil (Sm³/day)", fontsize=11)
    ax.set_title("Arps Decline Curve Fits", fontsize=12)
    ax.legend(fontsize=9, ncol=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))

    # ── Test period zoom ──────────────────────────────────────────────────
    ax2 = axes[1]
    q_te  = oil[split:]
    d_te  = dates[split:]
    ax2.plot(d_te, q_te, color="#00d4ff", lw=1.5, label="Actual", alpha=0.9)
    ax2.plot(d_te, result["exp"]["test"], color="#ffd93d",
             lw=2, label=f"Exponential  R²={result['exp']['metrics']['R²']:.3f}")
    ax2.plot(d_te, result["hyp"]["test"], color="#6bcb77",
             lw=2, label=f"Hyperbolic   R²={result['hyp']['metrics']['R²']:.3f}")
    ax2.set_ylabel("Oil (Sm³/day)", fontsize=11)
    ax2.set_title("Test Period Zoom", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))

    plt.tight_layout()
    plt.savefig("reports/figures/A01_arps_dca.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ A01_arps_dca.png")

    return result["exp"]["metrics"], result["hyp"]["metrics"]


if __name__ == "__main__":
    os.chdir("/home/claude/crude-oil-forecasting")
    print("Fitting Arps DCA on real Volve data...")
    df     = pd.read_csv("data/processed/real_field_daily.csv", parse_dates=["date"])
    result = fit_arps(df)

    print("\n  ── Arps DCA Metrics (Test Set) ───────────────────────────")
    for name, m in [("Exponential", result["exp"]["metrics"]),
                    ("Hyperbolic",  result["hyp"]["metrics"])]:
        print(f"  {name:15s}  RMSE={m['RMSE']:,.1f}  MAE={m['MAE']:,.1f}"
              f"  R²={m['R²']:.4f}  MAPE={m['MAPE']:.2f}%")

    print("\nGenerating plot:")
    plot_dca(result)
    print("\n✅ Arps DCA complete.")
