"""
Exploratory Data Analysis — Volve Production Dataset
=====================================================
Generates publication-quality plots saved to reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f23",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff", "#ff9a3c", "#4cc9f0"]
WELLS   = None   # filled after loading


def load_data():
    df = pd.read_csv("data/raw/volve_production.csv", parse_dates=["DATEPRD"])
    events = pd.read_csv("data/raw/well_events.csv",  parse_dates=["date"])
    global WELLS
    WELLS = df["WELL_BORE_CODE"].unique()
    return df, events


# ── Plot 1: Field-level production overview ──────────────────────────────────
def plot_field_overview(df):
    field = df.groupby("DATEPRD")[["BORE_OIL_VOL","BORE_GAS_VOL","BORE_WAT_VOL"]].sum()
    field = field.resample("W").mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Volve Field — Total Weekly Production Overview", fontsize=16, fontweight="bold", y=1.01)

    labels = ["Oil (Sm³/day)", "Gas (Sm³/day)", "Water (Sm³/day)"]
    cols   = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]
    colors = ["#00d4ff", "#ffd93d", "#ff6b6b"]

    for ax, col, label, color in zip(axes, cols, labels, colors):
        ax.fill_between(field.index, field[col], alpha=0.35, color=color)
        ax.plot(field.index, field[col], color=color, linewidth=1.5, label=label)
        ax.set_ylabel(label, fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(loc="upper right", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig("reports/figures/01_field_overview.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 01_field_overview.png")


# ── Plot 2: Per-well oil production ─────────────────────────────────────────
def plot_well_production(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Individual Well Oil Production (30-Day Moving Average)", fontsize=15, fontweight="bold")

    for i, well in enumerate(WELLS):
        wdf = df[df["WELL_BORE_CODE"] == well].set_index("DATEPRD")["BORE_OIL_VOL"]
        wdf = wdf.resample("D").sum().rolling(30, min_periods=1).mean()
        short = well.split("-")[-1]
        ax.plot(wdf.index, wdf.values, color=PALETTE[i % len(PALETTE)],
                linewidth=1.8, label=f"Well {short}", alpha=0.9)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Oil Volume (Sm³/day)", fontsize=12)
    ax.legend(fontsize=10, ncol=2, framealpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig("reports/figures/02_well_production.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 02_well_production.png")


# ── Plot 3: Missing data heatmap ─────────────────────────────────────────────
def plot_missing_data(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Data Quality Analysis", fontsize=15, fontweight="bold")

    # Missing % bar chart
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    axes[0].barh(missing_pct.index, missing_pct.values,
                 color=["#ff6b6b" if v > 0 else "#6bcb77" for v in missing_pct.values])
    axes[0].set_xlabel("Missing Values (%)", fontsize=11)
    axes[0].set_title("Missing Values by Column", fontsize=12)
    for i, v in enumerate(missing_pct.values):
        axes[0].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)

    # Zero-production days per well
    zero_days = {}
    for well in WELLS:
        wdf = df[df["WELL_BORE_CODE"] == well]
        zero_days[well.split("-")[-1]] = (wdf["BORE_OIL_VOL"] == 0).sum()

    wells_short = list(zero_days.keys())
    counts = list(zero_days.values())
    bars = axes[1].bar(wells_short, counts, color=PALETTE[:len(wells_short)])
    axes[1].set_title("Zero-Production Days per Well", fontsize=12)
    axes[1].set_ylabel("Days", fontsize=11)
    axes[1].set_xlabel("Well", fontsize=11)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(count), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/figures/03_data_quality.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 03_data_quality.png")


# ── Plot 4: Correlation heatmap ──────────────────────────────────────────────
def plot_correlation(df):
    num_cols = ["BORE_OIL_VOL","BORE_GAS_VOL","BORE_WAT_VOL",
                "WELL_BORE_HOURS","AVG_DOWNHOLE_PRESSURE","GOR","WATER_CUT"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, ax=ax,
                annot_kws={"size": 9})

    short_labels = ["Oil Vol","Gas Vol","Water Vol","Hours","Pressure","GOR","WCT"]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/figures/04_correlation.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 04_correlation.png")


# ── Plot 5: Distribution plots ───────────────────────────────────────────────
def plot_distributions(df):
    prod = df[df["BORE_OIL_VOL"] > 10]   # filter shutdowns for distributions
    cols = ["BORE_OIL_VOL","BORE_GAS_VOL","GOR","WATER_CUT","AVG_DOWNHOLE_PRESSURE"]
    titles = ["Oil Volume (Sm³/d)","Gas Volume (Sm³/d)","Gas-Oil Ratio","Water Cut","Downhole Pressure (bar)"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Feature Distributions (Producing Days Only)", fontsize=14, fontweight="bold")

    for ax, col, title, color in zip(axes, cols, titles, PALETTE):
        data = prod[col].dropna()
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.5, label=f"μ={data.mean():.1f}")
        ax.axvline(data.median(), color="#ffd93d", linestyle=":", linewidth=1.5, label=f"m={data.median():.1f}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8, framealpha=0.2)

    plt.tight_layout()
    plt.savefig("reports/figures/05_distributions.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 05_distributions.png")


# ── Plot 6: GOR & Water Cut evolution ────────────────────────────────────────
def plot_reservoir_depletion(df):
    field = df[df["BORE_OIL_VOL"] > 10].copy()
    field = field.set_index("DATEPRD")
    monthly = field.resample("ME")[["GOR","WATER_CUT"]].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Reservoir Depletion Indicators", fontsize=15, fontweight="bold")

    ax1.plot(monthly.index, monthly["GOR"], color="#ffd93d", linewidth=2)
    ax1.fill_between(monthly.index, monthly["GOR"], alpha=0.2, color="#ffd93d")
    ax1.set_ylabel("Gas-Oil Ratio (Sm³/Sm³)", fontsize=11)
    ax1.set_title("Rising GOR → Gas Cap Expansion / Reservoir Depletion", fontsize=10, color="#aaa")

    ax2.plot(monthly.index, monthly["WATER_CUT"] * 100, color="#ff6b6b", linewidth=2)
    ax2.fill_between(monthly.index, monthly["WATER_CUT"] * 100, alpha=0.2, color="#ff6b6b")
    ax2.set_ylabel("Water Cut (%)", fontsize=11)
    ax2.set_title("Rising Water Cut → Aquifer Influx", fontsize=10, color="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("reports/figures/06_reservoir_depletion.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 06_reservoir_depletion.png")


# ── Plot 7: Annual production summary ───────────────────────────────────────
def plot_annual_summary(df):
    df["year"] = df["DATEPRD"].dt.year
    annual = df.groupby("year")[["BORE_OIL_VOL","BORE_GAS_VOL","BORE_WAT_VOL"]].sum() / 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Annual Field Production (Million Sm³)", fontsize=15, fontweight="bold")

    x = np.arange(len(annual))
    w = 0.28
    ax.bar(x - w, annual["BORE_OIL_VOL"], w, label="Oil",   color="#00d4ff", alpha=0.85)
    ax.bar(x,     annual["BORE_GAS_VOL"]/1000, w, label="Gas (÷1000)", color="#ffd93d", alpha=0.85)
    ax.bar(x + w, annual["BORE_WAT_VOL"], w, label="Water", color="#ff6b6b", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.astype(str), rotation=0)
    ax.set_ylabel("Volume (MMSm³)", fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    plt.tight_layout()
    plt.savefig("reports/figures/07_annual_summary.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f23")
    plt.close()
    print("  ✓ 07_annual_summary.png")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(df, events):
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Records        : {len(df):,}")
    print(f"  Wells          : {df['WELL_BORE_CODE'].nunique()}")
    print(f"  Date range     : {df['DATEPRD'].min().date()} → {df['DATEPRD'].max().date()}")
    print(f"  Total Oil (Sm³): {df['BORE_OIL_VOL'].sum():,.0f}")
    print(f"  Total Gas (Sm³): {df['BORE_GAS_VOL'].sum():,.0f}")
    print(f"  Avg daily oil  : {df[df['BORE_OIL_VOL']>0]['BORE_OIL_VOL'].mean():.1f} Sm³/day")
    print(f"  Anomaly events : {len(events)}")
    print("="*55)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running EDA...")
    df, events = load_data()
    print_summary(df, events)

    print("\nGenerating plots:")
    plot_field_overview(df)
    plot_well_production(df)
    plot_missing_data(df)
    plot_correlation(df)
    plot_distributions(df)
    plot_reservoir_depletion(df)
    plot_annual_summary(df)

    print("\n✅ All EDA plots saved to reports/figures/")
