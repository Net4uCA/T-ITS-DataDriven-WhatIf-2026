"""
step1_preprocessing.py — Caricamento e pulizia dei dati osservati
Output: output_v2/trips_clean.csv

Operazioni:
 - Carica il CSV degli spostamenti osservati
 - Filtra per modalità di interesse (PV e PT)
 - Filtra per status OTP = OK
 - Filtra per bounding box geografica
 - Normalizza colonne e assegna flag is_pv
 - Esporta trips_clean.csv
"""

import os
import pandas as pd
import numpy as np
import config

# ───────────────────────────────────────────────────────────────────────────────────────
def parse_coords(s):
    """Converte 'lat lon' → (lat, lon) come float."""
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

    # ── 1. Carica CSV principale ────────────────────────────────────────────────────────
    df = pd.read_csv(config.FILE_TRIPS)
    print(f"  Righe caricate:       {len(df):>7}")

    # ── 2. Rinomina colonne chiave ──────────────────────────────────────────────────────
    df.rename(columns={
        "Id spostamento":             "trip_id",
        "Utente":                     "user_id",
        "Modalità":                   "mode",
        "Data ora inizio":            "datetime_start",
        "Durata secondi":             "duration_s",
        "Distanza percorsa metri":    "distance_m",
        "Coordinate di partenza":     "coord_origin",
        "Coordinate di arrivo":       "coord_dest",
        "estimated_car_duration(secondi)": "est_car_duration_s",
        "estimated_car_distance(metri)":   "est_car_distance_m",
    }, inplace=True)

    # Colonna utente (duplicata come "utente" minuscolo nel CSV originale)
    if "utente" in df.columns and "user_id" not in df.columns:
        df.rename(columns={"utente": "user_id"}, inplace=True)

    # ── 3. Filtra modalità rilevanti (vedi config.py per dettagli) ──────────────────────
    all_modes = config.MODES_PV | config.MODES_PT
    df = df[df["mode"].isin(all_modes)].copy()
    print(f"  Dopo filtro modalità: {len(df):>7}")

    # ── 4. Filtra per status OTP OK ─────────────────────────────────────────────────────
    df = df[df["status"] == "OK"].copy()
    print(f"  Dopo filtro status:   {len(df):>7}")

    # ── 5. Parsing coordinate ───────────────────────────────────────────────────────────
    df[["lat_o", "lon_o"]] = pd.DataFrame(
        df["coord_origin"].apply(parse_coords).tolist(), index=df.index)
    df[["lat_d", "lon_d"]] = pd.DataFrame(
        df["coord_dest"].apply(parse_coords).tolist(), index=df.index)

    # ── 6. Bounding box ─────────────────────────────────────────────────────────────────
    bb = (
        df["lat_o"].between(config.LAT_MIN, config.LAT_MAX) &
        df["lon_o"].between(config.LON_MIN, config.LON_MAX) &
        df["lat_d"].between(config.LAT_MIN, config.LAT_MAX) &
        df["lon_d"].between(config.LON_MIN, config.LON_MAX)
    )
    df = df[bb].copy()
    print(f"  Dopo bounding box:    {len(df):>7}")

    # ── 7. Rimuovi righe con dati essenziali mancanti ───────────────────────────────────
    df = df.dropna(subset=["duration_s", "lat_o", "lon_o", "lat_d", "lon_d"])
    print(f"  Dopo dropna base:     {len(df):>7}")

    # ── 8. Flag PV / PT ─────────────────────────────────────────────────────────────────
    df["is_pv"] = df["mode"].isin(config.MODES_PV)

    # Per PV: GC_j0 = β_ride × duration_s / 60 (door-to-door car trip)
    # Per PT: GC_j0 = β_ride × est_car_duration_s / 60 (alternative car trip)
    df["gc_j0"] = np.where(
        df["is_pv"],
        config.BETA["ride"] * df["duration_s"] / 60.0,
        config.BETA["ride"] * df["est_car_duration_s"] / 60.0,
    )

    # Rimuovi PT rows dove est_car_duration_s manca (non possiamo calcolare gc_j0)
    before = len(df)
    df = df.dropna(subset=["gc_j0"])
    print(f"  Dopo dropna gc_j0:    {len(df):>7}  (rimossi {before-len(df)} PT senza est_car_duration)")

    # ── 9. Colonne di output ────────────────────────────────────────────────────────────
    keep = [
        "trip_id", "user_id", "mode", "is_pv",
        "datetime_start", "duration_s", "distance_m",
        "lat_o", "lon_o", "lat_d", "lon_d",
        "est_car_duration_s", "est_car_distance_m",
        "gc_j0",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    # ── 10. Statistiche ─────────────────────────────────────────────────────────────────
    pv_count = df["is_pv"].sum()
    pt_count = (~df["is_pv"]).sum()
    print(f"\n  Distribuzione modalità nel dataset pulito:")
    print(f"    Veicolo privato (PV): {pv_count}")
    print(f"    Trasporto pubblico (PT): {pt_count}")
    print(f"    Totale: {len(df)}")

    # ── 11. Salva output ────────────────────────────────────────────────────────────────
    out = os.path.join(config.OUTPUT_DIR, "trips_clean.csv")
    df.to_csv(out, index=False)
    print(f"\n  ✓ Salvato: {out}")
    return df


if __name__ == "__main__":
    run()
