"""
step3_propensity.py — Calibrazione della sigmoide di propensione (Eq. 6)
Input:  output_v2/delta_d_per_trip.csv
Output: output_v2/propensity_per_trip.csv
        output_v2/sigmoid_params.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import config


# ─────────────────────────────────────────────────────────
def sigmoid(delta_d, p_max, g, d0_flex):
    """Eq. 6: sigmoide logistica per la propensione all'uso del PT."""
    return p_max / (1.0 + np.exp(g * (delta_d - d0_flex)))


def r_squared(y_true, y_pred):
    """Coefficiente di determinazione R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ─────────────────────────────────────────────────────────
def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 3 — Calibrazione sigmoide di propensione")
    print("=" * 60)

    # ── 1. Carica dati ────────────────────────────────────
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv"))
    print(f"  Viaggi con ΔD:        {len(df):>7}")

    # ── 2. Costruisci i bin empirici ──────────────────────
    # Usiamo il range [-15, 120] che copre entrambe le distribuzioni
    # ma si può adattare automaticamente
    d_min = max(df["delta_D"].quantile(0.01), -20)
    d_max = min(df["delta_D"].quantile(0.99), 150)

    bins = np.linspace(d_min, d_max, config.N_BINS_SIGMOID + 1)
    df["bin"] = pd.cut(df["delta_D"], bins=bins, include_lowest=True)

    bin_stats = df.groupby("bin", observed=True).agg(
        n_total  = ("is_pv", "count"),
        n_pv     = ("is_pv", "sum"),
    ).reset_index()

    bin_stats["n_pt"] = bin_stats["n_total"] - bin_stats["n_pv"]
    # Quota PT = frazione di utenti bus in quel bin di ΔD
    bin_stats["pt_share"] = bin_stats["n_pt"] / bin_stats["n_total"]
    bin_stats["bin_center"] = bins[:-1] + np.diff(bins) / 2

    # Rimuovi bin con meno di 5 osservazioni (troppo rumorosi)
    bin_stats = bin_stats[bin_stats["n_total"] >= 5].dropna(
        subset=["pt_share", "bin_center"])

    print(f"  Bin validi per fit:   {len(bin_stats):>7}")

    # ── 3. Fit sigmoide ───────────────────────────────────
    x = bin_stats["bin_center"].values
    y = bin_stats["pt_share"].values

    p0     = config.SIGMOID_P0.copy()
    p0[2]  = float(np.mean(df["delta_D"]))   # D0_flex inizializzato alla media

    try:
        popt, pcov = curve_fit(
            sigmoid, x, y,
            p0=p0,
            bounds=config.SIGMOID_BOUNDS,
            maxfev=10000,
        )
        p_max, g, d0_flex = popt
        perr = np.sqrt(np.diag(pcov))

    except RuntimeError as e:
        print(f"  ⚠ curve_fit non ha converguto: {e}")
        print("  Usando valori di default.")
        p_max, g, d0_flex = config.SIGMOID_P0
        perr = [np.nan, np.nan, np.nan]

    y_pred = sigmoid(x, p_max, g, d0_flex)
    r2 = r_squared(y, y_pred)

    print(f"\n  Parametri sigmoide calibrati:")
    print(f"    P_max  = {p_max:.4f}  (±{perr[0]:.4f})")
    print(f"    g      = {g:.4f}  (±{perr[1]:.4f})")
    print(f"    D0_flex= {d0_flex:.4f}  (±{perr[2]:.4f})")
    print(f"    R²     = {r2:.4f}")

    # Punti di interesse sulla sigmoide
    prop_high = sigmoid(d0_flex - 1/g * np.log(2), p_max, g, d0_flex) if g > 0 else np.nan
    print(f"\n  Zona di indifferenza (propensione ~ P_max/2): ΔD ≈ {d0_flex:.1f}")

    # ── 4. Calcola propensione per ogni viaggio ───────────
    df["propensity"] = sigmoid(df["delta_D"], p_max, g, d0_flex)

    # ── 5. Salva parametri ────────────────────────────────
    params = {
        "p_max":   float(p_max),
        "g":       float(g),
        "d0_flex": float(d0_flex),
        "r2":      float(r2),
        "perr_p_max":   float(perr[0]),
        "perr_g":       float(perr[1]),
        "perr_d0_flex": float(perr[2]),
    }
    params_path = os.path.join(config.OUTPUT_DIR, "sigmoid_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # Salva anche i bin empirici per i grafici
    bin_stats_path = os.path.join(config.OUTPUT_DIR, "sigmoid_bins.csv")
    bin_stats.to_csv(bin_stats_path, index=False)

    # ── 6. Salva dataset con propensità ───────────────────
    out = os.path.join(config.OUTPUT_DIR, "propensity_per_trip.csv")
    df.to_csv(out, index=False)

    print(f"\n  ✓ Salvato: {out}")
    print(f"  ✓ Salvato: {params_path}")

    return df, params, bin_stats


if __name__ == "__main__":
    run()
