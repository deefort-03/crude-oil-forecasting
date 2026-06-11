"""
Volve-like Synthetic Dataset Generator
---------------------------------------
Generates realistic oilfield production data inspired by the
Equinor Volve dataset (North Sea, Norway).

Wells: 7 producer wells based on actual Volve well names
Period: Jan 2008 – Dec 2016 (daily resolution)
Features: oil, gas, water volumes, operational hours, GOR, WCT
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ── Well definitions ────────────────────────────────────────────────────────
WELLS = {
    "NO 15/9-F-1 C": {"start": "2008-01-01", "peak_oil": 1800, "peak_day": 180,  "decline_rate": 0.0010},
    "NO 15/9-F-4":   {"start": "2008-06-01", "peak_oil": 2200, "peak_day": 150,  "decline_rate": 0.0012},
    "NO 15/9-F-5":   {"start": "2009-01-01", "peak_oil": 1500, "peak_day": 120,  "decline_rate": 0.0009},
    "NO 15/9-F-11":  {"start": "2009-07-01", "peak_oil": 900,  "peak_day": 90,   "decline_rate": 0.0008},
    "NO 15/9-F-12":  {"start": "2010-03-01", "peak_oil": 2600, "peak_day": 200,  "decline_rate": 0.0014},
    "NO 15/9-F-14":  {"start": "2011-01-01", "peak_oil": 1100, "peak_day": 100,  "decline_rate": 0.0007},
    "NO 15/9-F-15 D":{"start": "2012-06-01", "peak_oil": 750,  "peak_day": 80,   "decline_rate": 0.0006},
}

END_DATE = "2016-12-31"

# ── Production curve (ramp-up → plateau → decline) ──────────────────────────
def production_curve(t, peak_oil, peak_day, decline_rate):
    """Triangular ramp-up then exponential decline."""
    ramp = peak_oil * (t / peak_day)
    decline = peak_oil * np.exp(-decline_rate * (t - peak_day))
    oil = np.where(t < peak_day, ramp, decline)
    oil = np.maximum(oil, 0)
    return oil


def inject_anomalies(oil_series, dates):
    """
    Inject realistic anomalies into production data:
    - Planned shutdowns (workovers / maintenance)
    - Unplanned equipment failures (sudden drop → slow recovery)
    - Brief spikes (well testing)
    """
    oil = oil_series.copy()
    n = len(oil)
    event_log = []

    # Planned shutdowns (every ~300 days, last 7-14 days)
    for shutdown_start in range(300, n - 14, 320 + np.random.randint(-30, 30)):
        duration = np.random.randint(7, 15)
        oil[shutdown_start: shutdown_start + duration] = 0
        event_log.append((dates[shutdown_start], "planned_shutdown", duration))

    # Unplanned equipment failure (~2 per well life)
    for _ in range(2):
        fail_day = np.random.randint(60, n - 60)
        drop_pct = np.random.uniform(0.4, 0.75)
        recovery_days = np.random.randint(20, 45)
        for d in range(recovery_days):
            recovery_factor = 1 - drop_pct * np.exp(-0.1 * d)
            if fail_day + d < n:
                oil[fail_day + d] *= recovery_factor
        event_log.append((dates[fail_day], "equipment_failure", recovery_days))

    # Brief production spikes (well test / stimulation)
    for _ in range(3):
        spike_day = np.random.randint(30, n - 5)
        duration = np.random.randint(2, 5)
        oil[spike_day: spike_day + duration] *= np.random.uniform(1.2, 1.6)
        event_log.append((dates[spike_day], "well_test_spike", duration))

    return oil, event_log


# ── Main generator ──────────────────────────────────────────────────────────
def generate_volve_data():
    records = []
    all_events = []

    for well_name, params in WELLS.items():
        start = pd.Timestamp(params["start"])
        end   = pd.Timestamp(END_DATE)
        dates = pd.date_range(start, end, freq="D")
        n     = len(dates)
        t     = np.arange(n)

        # Base oil production curve
        oil_base = production_curve(
            t,
            params["peak_oil"],
            params["peak_day"],
            params["decline_rate"],
        )

        # Add noise (operational variability ±8%)
        noise = np.random.normal(1.0, 0.08, n)
        oil   = oil_base * noise

        # Inject anomalies
        oil, events = inject_anomalies(oil, dates)
        for ev_date, ev_type, ev_dur in events:
            all_events.append({"well": well_name, "date": ev_date,
                                "event_type": ev_type, "duration_days": ev_dur})

        # Operational hours (24h when producing, 0 during shutdowns)
        hours = np.where(oil > 10, np.random.uniform(22, 24, n), 0.0)

        # Gas-Oil Ratio (Sm3/Sm3) – increases as reservoir matures
        gor_base = 180 + t * 0.05
        gor = gor_base * np.random.normal(1.0, 0.05, n)

        # Water Cut (fraction) – logistic growth over well life
        wct = 1 / (1 + np.exp(-0.003 * (t - n * 0.55)))
        wct = np.clip(wct + np.random.normal(0, 0.02, n), 0.0, 0.95)

        # Gas and water volumes derived from oil
        gas   = oil * gor
        water = oil * (wct / (1 - wct + 1e-6))

        # Wellhead pressure (bar) – inversely correlated with production
        pressure = 250 - 0.05 * oil_base + np.random.normal(0, 5, n)
        pressure = np.clip(pressure, 50, 350)

        for i, d in enumerate(dates):
            records.append({
                "DATEPRD":        d,
                "WELL_BORE_CODE": well_name,
                "BORE_OIL_VOL":   round(max(oil[i], 0), 2),    # Sm3/day
                "BORE_GAS_VOL":   round(max(gas[i], 0), 2),    # Sm3/day
                "BORE_WAT_VOL":   round(max(water[i], 0), 2),  # Sm3/day
                "WELL_BORE_HOURS":round(hours[i], 2),           # hrs/day
                "AVG_DOWNHOLE_PRESSURE": round(pressure[i], 2), # bar
                "GOR":            round(gor[i], 2),             # Sm3/Sm3
                "WATER_CUT":      round(wct[i], 4),             # fraction
                "FLOW_KIND":      "production",
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["DATEPRD", "WELL_BORE_CODE"]).reset_index(drop=True)

    events_df = pd.DataFrame(all_events)

    return df, events_df


if __name__ == "__main__":
    print("Generating Volve-like production dataset...")
    df, events_df = generate_volve_data()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(f"{out_dir}/volve_production.csv", index=False)
    events_df.to_csv(f"{out_dir}/well_events.csv", index=False)

    print(f"✓ Production records : {len(df):,}")
    print(f"✓ Wells              : {df['WELL_BORE_CODE'].nunique()}")
    print(f"✓ Date range         : {df['DATEPRD'].min().date()} → {df['DATEPRD'].max().date()}")
    print(f"✓ Anomaly events     : {len(events_df)}")
    print(f"✓ Saved to           : {os.path.abspath(out_dir)}")
    print("\nSample:")
    print(df.head(3).to_string())
