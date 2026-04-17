"""
step2_discomfort.py — Conversione wide→leg + calcolo GC e ΔD
Inputs:  output_v2/trips_clean.csv
         alternative_otp_*.csv
         time_offsets_*.csv
Outputs: output_v2/alternatives_gc.csv   (una riga per alternativa con fasi e GC)
         output_v2/delta_d_per_trip.csv  (una riga per viaggio: migliore alternativa)

Modello matematico (paper v2, Eq. 2-5):
  GC_jk = β_walk·T_walk + β_wait·T_wait + β_ride·T_ride + β_trans·T_trans
  ΔD_jk = GC_jk − GC_j0
  Per ogni viaggio: seleziona alternativa con ΔD minimo (best_alternative).
"""

import os
import numpy as np
import pandas as pd
import config


# ──────────────────────────────────────────────────────────────────────────────────────
# Helper: parse "HH:MM:SS" → secondi dall'inizio della giornata
# ──────────────────────────────────────────────────────────────────────────────────────
def parse_time_s(t):
    """'HH:MM:SS' → float secondi. Ritorna NaN se non parsabile."""
    try:
        h, m, s = str(t).strip().split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return np.nan


def transfer_wait_s(walk_to_time, next_bus_from_time):
    """
    Calcola l'attesa al trasbordo in secondi.
    wait = next_bus_from_time - walk_to_time
    Applica il cap a TRANSFER_WAIT_CAP_S.
    Se il risultato è negativo (midnight crossing), aggiunge 24h.
    """
    t_walk_end = parse_time_s(walk_to_time)
    t_bus_dep  = parse_time_s(next_bus_from_time)

    if np.isnan(t_walk_end) or np.isnan(t_bus_dep):
        return np.nan

    wait = t_bus_dep - t_walk_end
    if wait < 0:
        wait += 86400  # correzione midnight crossing
    return min(wait, config.TRANSFER_WAIT_CAP_S)


# ──────────────────────────────────────────────────────────────────────────────────────
# Conversione wide → fasi aggregate (T_walk, T_wait, T_ride, T_trans)
# per una singola riga del CSV delle alternative
# ──────────────────────────────────────────────────────────────────────────────────────
def extract_phases(row, offset_s):
    """
    Dato un dict/Series con le colonne wide dell'alternativa OTP
    e l'offset temporale (secondi, già corretto per midnight),
    ritorna:
      T_walk_s, T_wait_s, T_ride_s, T_trans_s

    Logica:
      T_walk  = Σ walk_duration di tutti i leg wk e tr (secondi)
      T_wait  = offset_s  (attesa alla prima fermata = tempo dall'istante
                           di partenza osservato al momento in cui OTP
                           fa partire il viaggio)
      T_ride  = Σ bus_duration di tutti i leg rd (secondi)
      T_trans = Σ wait_time di tutti i leg tr (secondi, capped)
    """
    T_walk  = 0.0
    T_wait  = max(0.0, float(offset_s))   # garantisce non negativo
    T_ride  = 0.0
    T_trans = 0.0

    has_ride = False  # flag: almeno un leg rd presente

    # ── leg 1: initial_walk ────────────────────────────────────────────────────────────
    iw_dur = row.get("initial_walk_duration")
    if pd.notna(iw_dur):
        T_walk += float(iw_dur)

    # ── leg 2: first_bus ───────────────────────────────────────────────────────────────
    fb_dur = row.get("first_bus_duration")
    if pd.notna(fb_dur):
        T_ride += float(fb_dur)
        has_ride = True

    # ── leg 3: mid_walk1 + wait trasbordo 1 ───────────────────────────────────────────
    mw1_dur = row.get("mid_walk1_duration")
    if pd.notna(mw1_dur):
        T_walk += float(mw1_dur)
        wait1 = transfer_wait_s(row.get("mid_walk1_to_time"),
                                row.get("second_bus_from_time"))
        if pd.notna(wait1):
            T_trans += wait1

    # ── leg 4: second_bus ──────────────────────────────────────────────────────────────
    sb_dur = row.get("second_bus_duration")
    if pd.notna(sb_dur):
        T_ride += float(sb_dur)
        has_ride = True

    # ── leg 5: mid_walk2 + wait trasbordo 2 ───────────────────────────────────────────
    mw2_dur = row.get("mid_walk2_duration")
    if pd.notna(mw2_dur):
        T_walk += float(mw2_dur)
        wait2 = transfer_wait_s(row.get("mid_walk2_to_time"),
                                row.get("third_bus_from_time"))
        if pd.notna(wait2):
            T_trans += wait2

    # ── leg 6: third_bus ───────────────────────────────────────────────────────────────
    tb_dur = row.get("third_bus_duration")
    if pd.notna(tb_dur):
        T_ride += float(tb_dur)
        has_ride = True

    # ── leg 7: final_walk ──────────────────────────────────────────────────────────────
    fw_dur = row.get("final_walk_duration")
    if pd.notna(fw_dur):
        T_walk += float(fw_dur)

    if not has_ride:
        return None  # alternativa walk-only → scartata

    return T_walk, T_wait, T_ride, T_trans


# ──────────────────────────────────────────────────────────────────────────────────────
# Calcolo GC_jk (Eq. 2) — in β-weighted minutes
# ──────────────────────────────────────────────────────────────────────────────────────
def calc_gc_jk(T_walk_s, T_wait_s, T_ride_s, T_trans_s):
    """
    GC_jk = β_walk·T_walk_min + β_wait·T_wait_min
           + β_ride·T_ride_min + β_trans·T_trans_min
    """
    return (
        config.BETA["walk"]  * T_walk_s  / 60.0 +
        config.BETA["wait"]  * T_wait_s  / 60.0 +
        config.BETA["ride"]  * T_ride_s  / 60.0 +
        config.BETA["trans"] * T_trans_s / 60.0
    )


# ──────────────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ──────────────────────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 2 — Calcolo ΔD (Differential Generalized Discomfort)")
    print("=" * 60)

    # ── 1. Carica dati ─────────────────────────────────────────────────────────────────
    trips = pd.read_csv(os.path.join(config.OUTPUT_DIR, "trips_clean.csv"))
    alt   = pd.read_csv(config.FILE_ALT)
    off   = pd.read_csv(config.FILE_OFFSETS)

    print(f"  Spostamenti puliti:   {len(trips):>7}")
    print(f"  Alternative raw:      {len(alt):>7}")
    print(f"  Offsets:              {len(off):>7}")

    # ── 2. Filtra alternative NO_VALID_ROUTES ──────────────────────────────────────────
    alt = alt[alt["status"].isna()].copy()   # NaN = valida (no status = ok)
    print(f"  Alternative valide:   {len(alt):>7}  (dopo filtro status)")

    # ── 3. Filtra alternative walk-only ────────────────────────────────────────────────
    alt = alt[alt["first_bus_duration"].notna()].copy()
    print(f"  Alternative con bus:  {len(alt):>7}  (dopo filtro walk-only)")

    # ── 4. Merge con offsets ───────────────────────────────────────────────────────────
    off_ok = off[off["status"] == "OK"][
        ["id_spostamento", "utente", "id_alternativa", "offset(secondi)"]
    ].copy()
    alt = alt.merge(off_ok,
                    on=["id_spostamento", "utente", "id_alternativa"],
                    how="left")

    # ── 5. Correzione midnight crossing sugli offset ───────────────────────────────────
    #   Se offset < 0 → l'alternativa OTP era già partita prima
    #   del viaggio osservato (midnight crossing) → aggiungi 24h
    alt["offset_corretto"] = alt["offset(secondi)"].apply(
        lambda x: (x + 86400) if (pd.notna(x) and x < 0) else x
    )
    # Dove l'offset manca, usa 0 (nessun attesa iniziale stimata)
    alt["offset_corretto"] = alt["offset_corretto"].fillna(0.0)

    # ── 6. Calcola fasi (T_walk, T_wait, T_ride, T_trans) ─
    records = []
    skipped_walk_only = 0

    for _, row in alt.iterrows():
        result = extract_phases(row, row["offset_corretto"])
        if result is None:
            skipped_walk_only += 1
            continue
        T_walk_s, T_wait_s, T_ride_s, T_trans_s = result

        gc_jk = calc_gc_jk(T_walk_s, T_wait_s, T_ride_s, T_trans_s)

        records.append({
            "trip_id":       row["id_spostamento"],
            "user_id":       row["utente"],
            "alt_id":        row["id_alternativa"],
            "T_walk_min":    T_walk_s  / 60.0,
            "T_wait_min":    T_wait_s  / 60.0,
            "T_ride_min":    T_ride_s  / 60.0,
            "T_trans_min":   T_trans_s / 60.0,
            "GC_jk":         gc_jk,
        })

    print(f"\n  Walk-only scartate:   {skipped_walk_only:>7}")
    print(f"  Alternative calcolate:{len(records):>7}")

    alts_gc = pd.DataFrame(records)

    # ── 7. Merge con trips per ottenere GC_j0 e metadati ───────────────────────────────
    alts_gc = alts_gc.merge(
        trips[["trip_id", "user_id", "mode", "is_pv",
               "gc_j0", "lat_o", "lon_o", "lat_d", "lon_d",
               "datetime_start", "duration_s"]],
        on=["trip_id", "user_id"], how="inner"
    )

    # ── 8. ΔD_jk = GC_jk − GC_j0 (Eq. 4) ───────────────────────────────────────────────
    alts_gc["delta_D"] = alts_gc["GC_jk"] - alts_gc["gc_j0"]

    # ── 9. Per ogni viaggio: seleziona la migliore alternativa (min ΔD)
    best_idx = alts_gc.groupby("trip_id")["delta_D"].idxmin()
    best = alts_gc.loc[best_idx].copy().reset_index(drop=True)

    # ── 10. Statistiche ────────────────────────────────────────────────────────────────
    pv_mask = best["is_pv"]
    pt_mask = ~best["is_pv"]
    print(f"\n  Viaggi con ΔD valido: {len(best):>7}")
    print(f"    di cui PV (auto/moto): {pv_mask.sum()}")
    print(f"    di cui PT (bus):       {pt_mask.sum()}")

    if pv_mask.sum() > 0:
        pv_d = best.loc[pv_mask, "delta_D"]
        print(f"\n  ΔD utenti PV — media: {pv_d.mean():.2f}, "
              f"mediana: {pv_d.median():.2f}, std: {pv_d.std():.2f}")
        print(f"    p25={pv_d.quantile(.25):.2f}, "
              f"p75={pv_d.quantile(.75):.2f}, "
              f"p95={pv_d.quantile(.95):.2f}")

    if pt_mask.sum() > 0:
        pt_d = best.loc[pt_mask, "delta_D"]
        print(f"\n  ΔD utenti PT — media: {pt_d.mean():.2f}, "
              f"mediana: {pt_d.median():.2f}, std: {pt_d.std():.2f}")
        print(f"    p25={pt_d.quantile(.25):.2f}, "
              f"p75={pt_d.quantile(.75):.2f}, "
              f"p95={pt_d.quantile(.95):.2f}")

    # ── 11. Salva output ───────────────────────────────────────────────────────────────
    out_gc   = os.path.join(config.OUTPUT_DIR, "alternatives_gc.csv")
    out_best = os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv")

    alts_gc.to_csv(out_gc,   index=False)
    best.to_csv(out_best, index=False)

    print(f"\n  ✓ Salvato: {out_gc}")
    print(f"  ✓ Salvato: {out_best}")

    return alts_gc, best


if __name__ == "__main__":
    run()
