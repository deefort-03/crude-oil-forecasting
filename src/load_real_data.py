"""
Real Volve Data Loader & Cleaner
==================================
Loads the official Equinor Volve production dataset,
filters to producer wells, handles missing values,
and saves a clean field-level daily CSV.
"""
import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

def load_and_clean():
    print("Loading real Volve production data...")
    df = pd.read_excel(
        "data/raw/Volve_production_data.xlsx",
        sheet_name="Daily Production Data",
        parse_dates=["DATEPRD"],
    )
    print(f"  Raw rows       : {len(df):,}")

    # ── 1. Keep producer wells only ──────────────────────────────────────────
    prod = df[
        (df["FLOW_KIND"] == "production") &
        (df["WELL_TYPE"] == "OP")
    ].copy()
    print(f"  Producer rows  : {len(prod):,}")

    # ── 2. Drop metadata columns we don't need ───────────────────────────────
    drop_cols = ["NPD_WELL_BORE_CODE","NPD_WELL_BORE_NAME","NPD_FIELD_CODE",
                 "NPD_FIELD_NAME","NPD_FACILITY_CODE","NPD_FACILITY_NAME",
                 "AVG_CHOKE_UOM","BORE_WI_VOL","FLOW_KIND","WELL_TYPE"]
    prod.drop(columns=[c for c in drop_cols if c in prod.columns], inplace=True)

    # ── 3. Missing value strategy ────────────────────────────────────────────
    # AVG_ANNULUS_PRESS: 1271 missing → median impute per well
    # AVG_CHOKE_SIZE_P : 242 missing  → forward-fill then median
    # AVG_DOWNHOLE_PRESSURE/TEMP: 181 → forward-fill per well
    # AVG_WHP_P/WHT_P  : <15 missing  → linear interpolate
    for well in prod["WELL_BORE_CODE"].unique():
        mask = prod["WELL_BORE_CODE"] == well
        for col in ["AVG_DOWNHOLE_PRESSURE","AVG_DOWNHOLE_TEMPERATURE",
                    "AVG_DP_TUBING","AVG_ANNULUS_PRESS","AVG_CHOKE_SIZE_P"]:
            if col in prod.columns:
                med = prod.loc[mask, col].median()
                prod.loc[mask, col] = (
                    prod.loc[mask, col]
                    .ffill()
                    .bfill()
                    .fillna(med if not pd.isna(med) else 0)
                )
        for col in ["AVG_WHP_P","AVG_WHT_P"]:
            if col in prod.columns:
                prod.loc[mask, col] = (
                    prod.loc[mask, col]
                    .interpolate(method="linear")
                    .ffill().bfill().fillna(0)
                )

    # ── 4. Aggregate to field-level daily ────────────────────────────────────
    field = (
        prod.groupby("DATEPRD")
        .agg(
            oil_vol          =("BORE_OIL_VOL",              "sum"),
            gas_vol          =("BORE_GAS_VOL",              "sum"),
            water_vol        =("BORE_WAT_VOL",              "sum"),
            on_stream_hrs    =("ON_STREAM_HRS",             "sum"),
            avg_downhole_pres=("AVG_DOWNHOLE_PRESSURE",     "mean"),
            avg_downhole_temp=("AVG_DOWNHOLE_TEMPERATURE",  "mean"),
            avg_dp_tubing    =("AVG_DP_TUBING",             "mean"),
            avg_annulus_press=("AVG_ANNULUS_PRESS",         "mean"),
            avg_choke_size   =("AVG_CHOKE_SIZE_P",          "mean"),
            avg_whp          =("AVG_WHP_P",                 "mean"),
            avg_wht          =("AVG_WHT_P",                 "mean"),
            active_wells     =("BORE_OIL_VOL",  lambda x: (x > 0).sum()),
        )
        .reset_index()
        .rename(columns={"DATEPRD":"date"})
    )

    # ── 5. Reindex to continuous daily, forward-fill gaps ────────────────────
    full_range = pd.date_range(field["date"].min(), field["date"].max(), freq="D")
    field = (
        field.set_index("date")
        .reindex(full_range)
        .ffill()
        .fillna(0)
        .reset_index()
        .rename(columns={"index":"date"})
    )

    # ── 6. Derived production ratios ─────────────────────────────────────────
    total_liq = field["oil_vol"] + field["water_vol"] + 1e-6
    field["gor"]         = field["gas_vol"] / (field["oil_vol"] + 1e-6)
    field["water_cut"]   = field["water_vol"] / total_liq
    field["oil_fraction"]= field["oil_vol"]   / total_liq

    field.to_csv("data/processed/real_field_daily.csv", index=False)
    print(f"  Field daily rows: {len(field):,}")
    print(f"  Date range      : {field['date'].min().date()} → {field['date'].max().date()}")
    print(f"  Oil range       : {field['oil_vol'].min():.0f} – {field['oil_vol'].max():.0f} Sm³/day")
    print(f"  Columns         : {list(field.columns)}")
    return field

if __name__ == "__main__":
    field = load_and_clean()
    print("\n✅ Real data loaded and cleaned.")
