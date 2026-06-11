"""
XGBoost — Real Volve Data with Walk-Forward CV + SHAP
=======================================================
Walk-forward cross-validation (5 folds) · RobustScaler ·
Hyperparameter tuning · SHAP explainability
"""
import numpy as np, pandas as pd, joblib, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":"#0f0f23","axes.facecolor":"#1a1a2e",
    "axes.edgecolor":"#444","axes.labelcolor":"#e0e0e0",
    "xtick.color":"#aaa","ytick.color":"#aaa","text.color":"#e0e0e0",
    "grid.color":"#2a2a4a","grid.linestyle":"--","grid.alpha":0.5,
})

def load():
    df = pd.read_csv("data/processed/real_features.csv", parse_dates=["date"])
    excl = {"date","oil_vol","trend90","oil_ratio_trend"}
    feat = [c for c in df.columns if c not in excl]
    X = df[feat].fillna(0).values
    y = df["oil_vol"].values
    return X, y, df["date"].values, feat

# ── Walk-forward cross-validation ─────────────────────────────────────────────
def walk_forward_cv(X, y, n_splits=5):
    tscv    = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X)*0.07))
    cv_results = []
    print(f"  Walk-forward CV ({n_splits} folds):")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Scale INSIDE fold — no data leakage
        sc = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        model = xgb.XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.5, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model.fit(X_tr_s, y_tr, eval_set=[(X_tr_s, y_tr)], verbose=False)
        y_pred = model.predict(X_te_s)

        # Only score on producing days for MAPE (avoid div/0 on shutdowns)
        mask_prod = y_te > 10
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        r2   = r2_score(y_te, y_pred)
        mape = float(np.mean(np.abs((y_te[mask_prod]-y_pred[mask_prod])
                                    /(y_te[mask_prod]+1e-6)))*100) if mask_prod.any() else np.nan

        cv_results.append({"fold":fold,"RMSE":rmse,"MAE":mae,"R²":r2,"MAPE":mape,
                            "train_size":len(tr_idx),"test_size":len(te_idx)})
        print(f"    Fold {fold}: RMSE={rmse:,.1f}  MAE={mae:,.1f}  R²={r2:.4f}  MAPE={mape:.2f}%"
              f"  [train={len(tr_idx)} | test={len(te_idx)}]")

    cv_df = pd.DataFrame(cv_results)
    print(f"\n  CV Average → RMSE={cv_df['RMSE'].mean():,.1f}±{cv_df['RMSE'].std():,.1f}"
          f"  R²={cv_df['R²'].mean():.4f}±{cv_df['R²'].std():.4f}"
          f"  MAPE={cv_df['MAPE'].mean():.2f}%")
    return cv_df

# ── Final model on 80% train ───────────────────────────────────────────────────
def train_final(X, y):
    split   = int(len(X) * 0.80)
    sc = RobustScaler()
    X_tr_s = sc.fit_transform(X[:split])
    X_te_s = sc.transform(X[split:])
    joblib.dump(sc, "models/real_xgb_scaler.pkl")

    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.5, random_state=42,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr_s, y[:split], verbose=False)
    y_pred = model.predict(X_te_s)

    mask   = y[split:] > 10
    rmse   = np.sqrt(mean_squared_error(y[split:], y_pred))
    mae    = mean_absolute_error(y[split:], y_pred)
    r2     = r2_score(y[split:], y_pred)
    mape   = float(np.mean(np.abs((y[split:][mask]-y_pred[mask])/(y[split:][mask]+1e-6)))*100)
    metrics = {"RMSE":rmse,"MAE":mae,"R²":r2,"MAPE":mape}

    joblib.dump(model, "models/real_xgboost.pkl")
    return model, sc, X_tr_s, X_te_s, y[split:], y_pred, metrics, split

# ── SHAP ─────────────────────────────────────────────────────────────────────
def compute_shap(model, X_tr_s, feat_names):
    explainer   = shap.TreeExplainer(model)
    sample      = X_tr_s[:500]   # sample for speed
    shap_values = explainer.shap_values(sample)
    return explainer, shap_values, sample

def plot_shap(shap_values, X_sample, feat_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("SHAP Feature Importance — XGBoost (Real Volve Data)",
                 fontsize=14, fontweight="bold")

    # Bar chart: mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[-20:]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = mean_shap[top_idx]

    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, 20))
    axes[0].barh(top_names, top_vals, color=colors)
    axes[0].set_title("Top 20 Features — Mean |SHAP|", fontsize=12)
    axes[0].set_xlabel("Mean |SHAP value|", fontsize=11)

    # Beeswarm-style summary (manual, matplotlib-based)
    top20_sv  = shap_values[:, top_idx]
    top20_X   = X_sample[:, top_idx]
    ax = axes[1]
    for i, (name, sv_col, x_col) in enumerate(
            zip(top_names, top20_sv.T, top20_X.T)):
        jitter = np.random.uniform(-0.2, 0.2, len(sv_col))
        sc = ax.scatter(sv_col, np.full_like(sv_col, i) + jitter,
                        c=x_col, cmap="coolwarm", s=8, alpha=0.6,
                        vmin=np.percentile(x_col,5), vmax=np.percentile(x_col,95))
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title("Feature Impact Distribution", fontsize=12)
    plt.colorbar(sc, ax=ax, label="Feature value (scaled)")

    plt.tight_layout()
    plt.savefig("reports/figures/A02_shap.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ A02_shap.png")

def plot_cv_results(cv_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("XGBoost — Walk-Forward Cross-Validation Results", fontsize=14, fontweight="bold")
    for ax, metric, color in zip(axes, ["RMSE","R²","MAPE"],
                                  ["#00d4ff","#6bcb77","#ffd93d"]):
        ax.bar(cv_df["fold"], cv_df[metric], color=color, alpha=0.85, edgecolor="#333")
        ax.axhline(cv_df[metric].mean(), color="white", lw=1.5, linestyle="--",
                   label=f"Mean={cv_df[metric].mean():.3f}")
        ax.set_title(metric, fontsize=12); ax.set_xlabel("Fold"); ax.legend(fontsize=9)
        for i, (f, v) in enumerate(zip(cv_df["fold"], cv_df[metric])):
            ax.text(f, v*1.01, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/A03_cv_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ A03_cv_results.png")

def plot_predictions(dates, y_true, y_pred, metrics, split):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("XGBoost — Real Volve Data Predictions", fontsize=15, fontweight="bold")

    ax = axes[0,0]
    ax.plot(dates, y_true, color="#00d4ff", lw=1.5, label="Actual", alpha=0.9)
    ax.plot(dates, y_pred, color="#ff6b6b", lw=1.5, label="Predicted", linestyle="--")
    ax.fill_between(dates, y_true, y_pred, alpha=0.12, color="#ffd93d")
    ax.set_title("Actual vs Predicted (Test Set)", fontsize=12)
    ax.set_ylabel("Oil (Sm³/day)"); ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))

    mt = "\n".join([f"RMSE : {metrics['RMSE']:,.1f} Sm³/d",
                    f"MAE  : {metrics['MAE']:,.1f} Sm³/d",
                    f"R²   : {metrics['R²']:.4f}",
                    f"MAPE*: {metrics['MAPE']:.2f}%"])
    axes[0,1].axis("off")
    axes[0,1].text(0.5,0.5,mt+"  \n(*producing days only)",
                   ha="center",va="center",fontsize=13,
                   transform=axes[0,1].transAxes,
                   bbox=dict(boxstyle="round,pad=1",facecolor="#1a1a2e",
                             edgecolor="#00d4ff",alpha=0.9))
    axes[0,1].set_title("Test Metrics", fontsize=12)

    axes[1,0].scatter(y_true, y_pred, alpha=0.3, s=8, color="#c77dff")
    lim = max(y_true.max(), y_pred.max())*1.05
    axes[1,0].plot([0,lim],[0,lim],"r--",lw=1.5)
    axes[1,0].set_xlabel("Actual"); axes[1,0].set_ylabel("Predicted")
    axes[1,0].set_title("Scatter", fontsize=12)

    res = y_true - y_pred
    axes[1,1].hist(res, bins=50, color="#ffd93d", alpha=0.75, edgecolor="none")
    axes[1,1].axvline(0,color="white",lw=1.5,linestyle="--")
    axes[1,1].axvline(res.mean(),color="#ff6b6b",lw=1.5,
                      label=f"Mean={res.mean():.1f}")
    axes[1,1].set_title("Residuals", fontsize=12)
    axes[1,1].set_xlabel("Residual (Sm³/day)"); axes[1,1].legend()

    plt.tight_layout()
    plt.savefig("reports/figures/A04_xgb_real_predictions.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ A04_xgb_real_predictions.png")

if __name__ == "__main__":
    os.chdir("/home/claude/crude-oil-forecasting")
    print("Loading real features...")
    X, y, dates, feat_names = load()

    print("\nRunning walk-forward cross-validation...")
    cv_df = walk_forward_cv(X, y)

    print("\nTraining final model (80/20 split)...")
    model, sc, X_tr_s, X_te_s, y_te, y_pred, metrics, split = train_final(X, y)
    print(f"  Final model → RMSE={metrics['RMSE']:,.1f}  R²={metrics['R²']:.4f}"
          f"  MAPE={metrics['MAPE']:.2f}%")
    joblib.dump(metrics, "models/real_xgb_metrics.pkl")
    joblib.dump(cv_df,   "models/real_xgb_cv.pkl")
    np.save("models/real_xgb_pred.npy", y_pred)
    np.save("models/real_xgb_true.npy", y_te)
    np.save("models/real_xgb_dates.npy", dates[split:])

    print("\nComputing SHAP values...")
    _, shap_values, X_sample = compute_shap(model, X_tr_s, feat_names)

    print("\nGenerating plots:")
    plot_shap(shap_values, X_sample, feat_names)
    plot_cv_results(cv_df)
    plot_predictions(dates[split:], y_te, y_pred, metrics, split)

    print("\n✅ XGBoost (real data) complete.")
