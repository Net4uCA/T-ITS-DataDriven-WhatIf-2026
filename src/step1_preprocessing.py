"""
step1_preprocessing.py — Caricamento e pulizia dei dati osservati
Output: output_v2/trips_clean.csv
"""

import os
import pandas as pd
import numpy as np
import config


def parse_coords(s):
    try:
        parts = str(s).strip().split()
        return float(parts[0]), float(parts[1])
    except Exception:
        return np.nan, np.nan


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1 — Preprocessing spostamenti osservati")
    print("=" * 60)

    df = pd.read_csv(config.FILE_TRIPS)
    print(f"  Righe caricate:       {len(df):>7}")

    df.rename(columns={
        "Id spostamento":                  "trip_id",
        "Utente":                          "user_id",
        "Modalita":                        "mode",
        "Modalità":                        "mode",
        "Data ora inizio":                 "datetime_start",
        "Durata secondi":                  "duration_s",
        "Distanza percorsa metri":         "distance_m",
        "Coordinate di partenza":          "coord_origin",
        "Coordinate di arrivo":            "coord_dest",
        "estimated_car_duration(secondi)": "est_car_duration_s",
        "estimated_car_distance(metri)":   "est_car_distance_m",
    }, inplace=True)

    if "utente" in df.columns and "user_id" not in df.columns:
        df.rename(columns={"utente": "user_id"}, inplace=True)

    # Filtra modalità
    all_modes = config.MODES_PV | config.MODES_PT
    df = df[df["mode"].isin(all_modes)].copy()
    print(f"  Dopo filtro modalità: {len(df):>7}")

    # Status OTP
    df = df[df["status"] == "OK"].copy()
    print(f"  Dopo filtro status:   {len(df):>7}")

    # Coordinate
    df[["lat_o", "lon_o"]] = pd.DataFrame(
        df["coord_origin"].apply(parse_coords).tolist(), index=df.index)
    df[["lat_d", "lon_d"]] = pd.DataFrame(
        df["coord_dest"].apply(parse_coords).tolist(), index=df.index)

    # Bounding box
    bb = (
        df["lat_o"].between(config.LAT_MIN, config.LAT_MAX) &
        df["lon_o"].between(config.LON_MIN, config.LON_MAX) &
        df["lat_d"].between(config.LAT_MIN, config.LAT_MAX) &
        df["lon_d"].between(config.LON_MIN, config.LON_MAX)
    )
    df = df[bb].copy()
    print(f"  Dopo bounding box:    {len(df):>7}")

    df = df.dropna(subset=["duration_s", "lat_o", "lon_o", "lat_d", "lon_d"])
    print(f"  Dopo dropna base:     {len(df):>7}")

    df["is_pv"] = df["mode"].isin(config.MODES_PV)

    df["gc_j0"] = config.BETA["ride"] * df["duration_s"] / 60.0

    # Segnala PT senza est_car_duration_s (non potranno avere GC_jk in step2)
    pt_mask = ~df["is_pv"]
    n_missing = (pt_mask & df["est_car_duration_s"].isna()).sum()
    if n_missing > 0:
        print(f"  Attenzione: {n_missing} trip PT senza est_car_duration_s")

    keep = [
        "trip_id", "user_id", "mode", "is_pv",
        "datetime_start", "duration_s", "distance_m",
        "lat_o", "lon_o", "lat_d", "lon_d",
        "est_car_duration_s", "est_car_distance_m",
        "gc_j0",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    pv_count = df["is_pv"].sum()
    pt_count = (~df["is_pv"]).sum()
    print(f"\n  Distribuzione nel dataset pulito:")
    print(f"    PV: {pv_count:>6}  GC_j0 media: {df.loc[df['is_pv'],'gc_j0'].mean():.1f} min")
    print(f"    PT: {pt_count:>6}  GC_j0 media: {df.loc[~df['is_pv'],'gc_j0'].mean():.1f} min")
    print(f"    Totale: {len(df)}")

    out = os.path.join(config.OUTPUT_DIR, "trips_clean.csv")
    df.to_csv(out, index=False)
    print(f"\n  Salvato: {out}")
    return df


if __name__ == "__main__":
    run()
