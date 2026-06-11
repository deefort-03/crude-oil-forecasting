"""
Anomaly Detection — Isolation Forest
=====================================
Detects abnormal production events:
  - Equipment failures (sudden production drops)
  - Planned shutdowns
  - Well test spikes
  - Unexpected pressure anomalies

Uses Isolation Forest (unsupervised) trained on normal production windows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib, os, warnings

warnings.filterwarnings("ignore")
os.makedirs("models",           exist_ok=True)
os.makedirs("reports/figures",  exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f0f23", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa", "ytick.color": "#aaa", "text.color": "#e0e0e0",
    "grid.color": "#2a2a4a", "grid.linestyle": "--", "grid.alpha": 0.5,
})


def load_data():
    field = pd.read_csv("data/processed/field_features.csv",   parse_dates=["date"])
    events_raw = pd.read_csv("data/raw/well_events.csv", parse_dates=["date"])

    # Build ground-truth anomaly dates at field level
    anomaly_dates = set(events_raw["date"].dt.date.unique())

    # Expand event windows: mark day-of + 3 following days as anomaly
    expanded_anomaly = set()
    for d in anomaly_dates:
        for offset in range(4):
            expanded_anomaly.add(d + pd.Timedelta(days=offset))

    field["true_anomaly"] = field["date"].dt.date.apply(
        lambda d: 1 if d in expanded_anomaly else 0
    )
    return field


def build_features(field: pd.DataFrame):
    """Construct feature matrix for the Isolation Forest."""
    features = [
        # Production volumes
        "oil_vol", "gas_vol", "water_vol",
        # Domain ratios
        "avg_gor", "avg_wct", "avg_pres", "active_wells",
        # Statistical deviation features
        "oil_vol_roll_mean7", "oil_vol_roll_std7",
        "oil_vol_roll_mean30", "oil_vol_roll_std30",
        # Change metrics
        "oil_daily_change", "oil_pct_change",
        # Depletion indicators
        "oil_per_well", "cum_oil_norm",
    ]
    available = [f for f in features if f in field.columns]
    X = field[available].fillna(0).values
    return X, available


def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Isolation Forest contamination=0.05 (we expect ~5% anomalous days).
    Tuned for imbalanced, real-world production data.
    """
    model = IsolationForest(
        n_estimators   = 200,
        max_samples    = "auto",
        contamination  = 0.05,
        max_features   = 1.0,
        random_state   = 42,
        n_jobs         = -1,
    )
    model.fit(X_train)
    return model


def plot_anomaly_timeline(field, predictions):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.suptitle("Anomaly Detection — Isolation Forest on Field Production",
                 fontsize=15, fontweight="bold")

    anomaly_mask = predictions == -1
    dates = field["date"].values

    # Oil production
    ax = axes[0]
    ax.plot(dates, field["oil_vol"], color="#00d4ff", lw=1.2, label="Oil Vol", zorder=2)
    ax.scatter(dates[anomaly_mask], field["oil_vol"].values[anomaly_mask],
               color="#ff6b6b", s=20, zorder=3, label="Detected Anomaly", alpha=0.8)
    ax.set_ylabel("Oil (Sm³/day)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title("Oil Production with Detected Anomalies", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Downhole pressure
    ax2 = axes[1]
    ax2.plot(dates, field["avg_pres"], color="#ffd93d", lw=1.2, label="Downhole Pressure")
    ax2.scatter(dates[anomaly_mask], field["avg_pres"].values[anomaly_mask],
                color="#ff6b6b", s=20, zorder=3, alpha=0.8)
    ax2.set_ylabel("Pressure (bar)", fontsize=11)
    ax2.set_title("Downhole Pressure Anomalies", fontsize=12)
    ax2.legend(fontsize=10)

    # Water cut
    ax3 = axes[2]
    ax3.plot(dates, field["avg_wct"] * 100, color="#6bcb77", lw=1.2, label="Water Cut (%)")
    ax3.scatter(dates[anomaly_mask], field["avg_wct"].values[anomaly_mask] * 100,
                color="#ff6b6b", s=20, zorder=3, alpha=0.8)
    ax3.set_ylabel("Water Cut (%)", fontsize=11)
    ax3.set_title("Water Cut Anomalies", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig("reports/figures/13_anomaly_timeline.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 13_anomaly_timeline.png")


def plot_anomaly_scores(field, scores, predictions):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Isolation Forest — Anomaly Scores", fontsize=14, fontweight="bold")

    dates = field["date"].values
    colors = ["#ff6b6b" if p == -1 else "#00d4ff" for p in predictions]

    ax1.scatter(dates, scores, c=colors, s=8, alpha=0.6)
    ax1.axhline(np.percentile(scores, 5), color="#ffd93d", linestyle="--",
                linewidth=1.5, label="5th pctile threshold")
    ax1.set_ylabel("Anomaly Score", fontsize=11)
    ax1.set_title("Anomaly Score per Day (Red = Flagged)", fontsize=12)
    ax1.legend(fontsize=10)

    ax2.hist(scores, bins=60, color="#c77dff", alpha=0.75, edgecolor="none")
    ax2.axvline(np.percentile(scores, 5), color="#ffd93d", linestyle="--",
                linewidth=2, label="Decision boundary")
    ax2.set_xlabel("Anomaly Score", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Score Distribution", fontsize=12)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("reports/figures/14_anomaly_scores.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 14_anomaly_scores.png")


def plot_anomaly_confusion(y_true, y_pred):
    # Convert Isolation Forest (-1/1) to binary (1/0)
    y_pred_bin = (y_pred == -1).astype(int)

    cm = confusion_matrix(y_true, y_pred_bin)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Anomaly Detection — Evaluation", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Anomaly"],
                yticklabels=["Normal","Anomaly"], ax=ax1,
                annot_kws={"size": 14})
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("Actual",    fontsize=12)
    ax1.set_title("Confusion Matrix", fontsize=12)

    # Event type breakdown
    events = pd.read_csv("data/raw/well_events.csv")
    event_counts = events["event_type"].value_counts()
    bars = ax2.barh(event_counts.index, event_counts.values,
                    color=["#ff6b6b", "#ffd93d", "#00d4ff"][:len(event_counts)])
    ax2.set_title("Anomaly Events by Type (Ground Truth)", fontsize=12)
    ax2.set_xlabel("Count", fontsize=11)
    for bar, v in zip(bars, event_counts.values):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(v), va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("reports/figures/15_anomaly_evaluation.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    print("  ✓ 15_anomaly_evaluation.png")


if __name__ == "__main__":
    print("Loading data...")
    field = load_data()
    X, feat_names = build_features(field)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Ground-truth anomalies: {field['true_anomaly'].sum()} days "
          f"({field['true_anomaly'].mean()*100:.1f}%)")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nTraining Isolation Forest...")
    model = train_isolation_forest(X_scaled)
    predictions = model.predict(X_scaled)  # 1=normal, -1=anomaly
    scores      = model.score_samples(X_scaled)

    n_anomalies = (predictions == -1).sum()
    print(f"  Flagged anomalies : {n_anomalies} days ({n_anomalies/len(predictions)*100:.1f}%)")

    # Evaluate vs ground truth
    y_true     = field["true_anomaly"].values
    y_pred_bin = (predictions == -1).astype(int)

    print("\n  ── Classification Report ─────────────")
    print(classification_report(y_true, y_pred_bin,
                                target_names=["Normal","Anomaly"]))

    # Save
    joblib.dump(model,       "models/isolation_forest.pkl")
    joblib.dump(scaler,      "models/anomaly_scaler.pkl")
    joblib.dump(feat_names,  "models/anomaly_features.pkl")
    print("  ✓ Models saved")

    print("\nGenerating plots:")
    plot_anomaly_timeline(field, predictions)
    plot_anomaly_scores(field, scores, predictions)
    plot_anomaly_confusion(y_true, predictions)

    print("\n✅ Anomaly detection complete.")
