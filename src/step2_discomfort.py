"""
step2_discomfort.py — Calcolo GC_jk e Delta_D per PV e PT trips
Output: output_v2/alternatives_gc.csv
        output_v2/delta_d_per_trip.csv

Delta_D = GC_PT - GC_PV
"""

import os
import numpy as np
import pandas as pd
import config


# ─────────────────────────────────────────────────────────
def parse_time_s(t):
    """'HH:MM:SS' -> secondi dall'inizio della giornata."""
    try:
        h, m, s = str(t).strip().split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return np.nan


def transfer_wait_s(walk_to_time, next_bus_from_time):
    """
    Attesa al trasbordo in secondi (capped a TRANSFER_WAIT_CAP_S).
    """
    t_walk_end = parse_time_s(walk_to_time)
    t_bus_dep  = parse_time_s(next_bus_from_time)
    if np.isnan(t_walk_end) or np.isnan(t_bus_dep):
        return np.nan
    wait = t_bus_dep - t_walk_end
    if wait < 0:
        wait += 86400   # midnight crossing
    return min(wait, config.TRANSFER_WAIT_CAP_S)


def extract_phases_pt_alternative(row, offset_s):
    """
    Estrae T_walk, T_wait, T_ride, T_trans da una riga wide del CSV alternative OTP.
    Ritorna None se l'alternativa e' walk-only (nessun leg ride).
    """
    T_walk  = 0.0
    T_wait  = max(0.0, float(offset_s))
    T_ride  = 0.0
    T_trans = 0.0
    has_ride = False

    iw = row.get("initial_walk_duration")
    if pd.notna(iw):
        T_walk += float(iw)

    fb = row.get("first_bus_duration")
    if pd.notna(fb):
        T_ride += float(fb); has_ride = True

    mw1 = row.get("mid_walk1_duration")
    if pd.notna(mw1):
        T_walk += float(mw1)
        w1 = transfer_wait_s(row.get("mid_walk1_to_time"),
                             row.get("second_bus_from_time"))
        if pd.notna(w1):
            T_trans += w1

    sb = row.get("second_bus_duration")
    if pd.notna(sb):
        T_ride += float(sb); has_ride = True

    mw2 = row.get("mid_walk2_duration")
    if pd.notna(mw2):
        T_walk += float(mw2)
        w2 = transfer_wait_s(row.get("mid_walk2_to_time"),
                             row.get("third_bus_from_time"))
        if pd.notna(w2):
            T_trans += w2

    tb = row.get("third_bus_duration")
    if pd.notna(tb):
        T_ride += float(tb); has_ride = True

    fw = row.get("final_walk_duration")
    if pd.notna(fw):
        T_walk += float(fw)

    if not has_ride:
        return None
    return T_walk, T_wait, T_ride, T_trans


def calc_gc_pt(T_walk_s, T_wait_s, T_ride_s, T_trans_s):
    """GC di un itinerario PT (Eq. 2) in beta-weighted minutes."""
    return (
        config.BETA["walk"]  * T_walk_s  / 60.0 +
        config.BETA["wait"]  * T_wait_s  / 60.0 +
        config.BETA["ride"]  * T_ride_s  / 60.0 +
        config.BETA["trans"] * T_trans_s / 60.0
    )


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 2 — Calcolo GC_jk e Delta_D")
    print("=" * 60)
    print("  Convenzione: Delta_D = GC_PT - GC_PV (sempre)")
    print()

    trips = pd.read_csv(os.path.join(config.OUTPUT_DIR, "trips_clean.csv"))
    alt   = pd.read_csv(config.FILE_ALT)
    off   = pd.read_csv(config.FILE_OFFSETS)

    pv_trips = trips[trips["is_pv"]].copy()
    pt_trips = trips[~trips["is_pv"]].copy()

    print(f"  Trip PV (auto/moto):  {len(pv_trips):>7}")
    print(f"  Trip PT (bus):        {len(pt_trips):>7}")

    # =========================================================
    # BLOCCO A — Trip PV: GC_jk dalle leg OTP
    # =========================================================
    print()
    print("  [A] Calcolo Delta_D per trip PV -> alternativa PT")

    alt_valid = alt[alt["status"].isna() & alt["first_bus_duration"].notna()].copy()
    print(f"      Alternative PT valide nel CSV: {len(alt_valid):>7}")

    off_ok = off[off["status"] == "OK"][
        ["id_spostamento", "utente", "id_alternativa", "offset(secondi)"]
    ].copy()
    alt_valid = alt_valid.merge(off_ok,
                                on=["id_spostamento", "utente", "id_alternativa"],
                                how="left")

    alt_valid["offset_corretto"] = alt_valid["offset(secondi)"].apply(
        lambda x: (x + 86400) if (pd.notna(x) and x < 0) else x
    ).fillna(0.0)

    records_pv = []
    skipped_walk_only = 0

    for _, row in alt_valid.iterrows():
        result = extract_phases_pt_alternative(row, row["offset_corretto"])
        if result is None:
            skipped_walk_only += 1
            continue
        T_walk_s, T_wait_s, T_ride_s, T_trans_s = result
        gc_jk_pt = calc_gc_pt(T_walk_s, T_wait_s, T_ride_s, T_trans_s)

        records_pv.append({
            "trip_id":       row["id_spostamento"],
            "user_id":       row["utente"],
            "alt_id":        row["id_alternativa"],
            "T_walk_min":    T_walk_s  / 60.0,
            "T_wait_min":    T_wait_s  / 60.0,
            "T_ride_min":    T_ride_s  / 60.0,
            "T_trans_min":   T_trans_s / 60.0,
            "GC_jk":         gc_jk_pt,
            "alt_type":      "PT",
        })

    print(f"      Walk-only scartate:        {skipped_walk_only:>7}")
    print(f"      Alternative calcolate:     {len(records_pv):>7}")

    alts_pv = pd.DataFrame(records_pv)

    # Merge con trips PV per GC_j0
    alts_pv = alts_pv.merge(
        pv_trips[["trip_id", "user_id", "mode", "is_pv",
                  "gc_j0", "lat_o", "lon_o", "lat_d", "lon_d",
                  "datetime_start", "duration_s", "est_car_duration_s"]],
        on=["trip_id", "user_id"], how="inner"
    )

    # Delta_D = GC_PT - GC_PV
    alts_pv["delta_D"] = alts_pv["GC_jk"] - alts_pv["gc_j0"]

    # Miglior alternativa per ogni viaggio PV (min Delta_D)
    best_pv_idx = alts_pv.groupby("trip_id")["delta_D"].idxmin()
    best_pv = alts_pv.loc[best_pv_idx].copy().reset_index(drop=True)

    print(f"\n      Trip PV con Delta_D valido: {len(best_pv):>6}")
    print(f"      Delta_D medio PV: {best_pv['delta_D'].mean():.2f} min  "
          f"(mediana: {best_pv['delta_D'].median():.2f})")

    # =========================================================
    # BLOCCO B — Trip PT: GC_jk = beta_ride x est_car_duration / 60
    # =========================================================
    print()
    print("  [B] Calcolo Delta_D per trip PT -> alternativa PV")
    print("      (usa est_car_duration_s dal CSV spostamenti, NON il CSV alternative)")

    pt_valid = pt_trips.dropna(subset=["est_car_duration_s"]).copy()
    n_dropped = len(pt_trips) - len(pt_valid)
    if n_dropped > 0:
        print(f"      Rimossi {n_dropped} trip PT senza est_car_duration_s")

    # GC_jk per l'alternativa PV: solo fase ride, door-to-door
    pt_valid["GC_jk"]     = config.BETA["ride"] * pt_valid["est_car_duration_s"] / 60.0
    pt_valid["alt_type"]  = "PV"

    # Delta_D = GC_PT (osservato) - GC_PV (alternativa auto)
    # = gc_j0 - GC_jk
    # che e' equivalente a GC_PT - GC_PV con il segno corretto
    pt_valid["delta_D"] = pt_valid["gc_j0"] - pt_valid["GC_jk"]

    # Colonne fasi: per i PT trips la leg osservata e' approssimata a ride unica
    # T_walk, T_wait, T_trans = 0 (non disponibili per il bus osservato)
    # T_ride = duration_s / 60
    pt_valid["T_walk_min"]  = 0.0
    pt_valid["T_wait_min"]  = 0.0
    pt_valid["T_ride_min"]  = pt_valid["duration_s"] / 60.0
    pt_valid["T_trans_min"] = 0.0
    pt_valid["alt_id"]      = "car_otp"

    # Per i PT trips non c'e' selezione tra alternative (una sola alternativa auto)
    best_pt = pt_valid[[
        "trip_id", "user_id", "alt_id", "alt_type",
        "T_walk_min", "T_wait_min", "T_ride_min", "T_trans_min",
        "GC_jk", "mode", "is_pv", "gc_j0",
        "lat_o", "lon_o", "lat_d", "lon_d",
        "datetime_start", "duration_s", "est_car_duration_s",
        "delta_D",
    ]].copy().reset_index(drop=True)

    print(f"\n      Trip PT con Delta_D valido: {len(best_pt):>6}")
    print(f"      Delta_D medio PT: {best_pt['delta_D'].mean():.2f} min  "
          f"(mediana: {best_pt['delta_D'].median():.2f})")

    # =========================================================
    # Unione PV + PT
    # =========================================================
    # Allinea colonne
    pv_cols = set(best_pv.columns)
    pt_cols = set(best_pt.columns)

    for c in pv_cols - pt_cols:
        best_pt[c] = np.nan
    for c in pt_cols - pv_cols:
        best_pv[c] = np.nan

    best = pd.concat([best_pv, best_pt], ignore_index=True)

    print()
    print(f"  Totale viaggi con Delta_D: {len(best):>7}")
    pv_d = best.loc[best["is_pv"],  "delta_D"]
    pt_d = best.loc[~best["is_pv"], "delta_D"]
    print(f"    PV: n={len(pv_d)}  media={pv_d.mean():.1f}  mediana={pv_d.median():.1f}  std={pv_d.std():.1f}")
    print(f"    PT: n={len(pt_d)}  media={pt_d.mean():.1f}  mediana={pt_d.median():.1f}  std={pt_d.std():.1f}")
    print()
    print(f"    Delta_D PV > 0: {(pv_d > 0).sum()} ({100*(pv_d>0).mean():.1f}%)")
    print(f"    Delta_D PT > 0: {(pt_d > 0).sum()} ({100*(pt_d>0).mean():.1f}%)")
    print(f"    Delta_D PT < 0: {(pt_d < 0).sum()} ({100*(pt_d<0).mean():.1f}%)")

    # Salva
    out_alts = os.path.join(config.OUTPUT_DIR, "alternatives_gc.csv")
    out_best = os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv")

    alts_pv.to_csv(out_alts, index=False)
    best.to_csv(out_best, index=False)

    print(f"\n  Salvato: {out_alts}")
    print(f"  Salvato: {out_best}")

    return alts_pv, best


if __name__ == "__main__":
    run()
