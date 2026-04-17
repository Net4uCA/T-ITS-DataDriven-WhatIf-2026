"""
step4_corridors.py — Griglia spazio-temporale, H-index, corridoi prioritari
Input:  output_v2/propensity_per_trip.csv
        output_v2/sigmoid_params.json
Output: output_v2/corridors_all.csv
        output_v2/priority_corridors.csv
"""

import os
import json
import numpy as np
import pandas as pd
import config


# ─────────────────────────────────────────────────────────
# Conversione coordinate geografiche → indice cella griglia
# Usa proiezione metrica approssimata (Cagliari lat≈39.2°)
# ─────────────────────────────────────────────────────────
LAT_ORIGIN  = 39.15
LON_ORIGIN  = 8.95
DEG_PER_M_LAT = 1.0 / 111_320          # ~1 grado lat ≈ 111.32 km
DEG_PER_M_LON = 1.0 / (111_320 * np.cos(np.radians(39.2)))


def coord_to_cell(lat, lon, cell_size_m=config.CELL_SIZE_M):
    """Converte (lat, lon) → (row, col) della cella 500×500 m."""
    row = int((lat - LAT_ORIGIN) / (cell_size_m * DEG_PER_M_LAT))
    col = int((lon - LON_ORIGIN) / (cell_size_m * DEG_PER_M_LON))
    return row, col


def cell_centroid(row_idx, col_idx, cell_size_m=config.CELL_SIZE_M):
    """Ritorna (lat, lon) del centroide di una cella."""
    lat = LAT_ORIGIN + (row_idx + 0.5) * cell_size_m * DEG_PER_M_LAT
    lon = LON_ORIGIN + (col_idx + 0.5) * cell_size_m * DEG_PER_M_LON
    return lat, lon


def parse_datetime_to_slot(dt_str):
    """
    Converte 'dd-mm-yyyy HH:MM' o 'HH:MM:SS' → slot di 10 minuti.
    Ritorna intero: minuti dall'inizio della giornata // 10.
    """
    try:
        # Prova formato completo
        import datetime
        for fmt in ("%d-%m-%Y %H:%M", "%H:%M:%S", "%H:%M"):
            try:
                dt = datetime.datetime.strptime(str(dt_str).strip(), fmt)
                return (dt.hour * 60 + dt.minute) // config.TIME_SLOT_MIN
            except ValueError:
                continue
    except Exception:
        pass
    return np.nan


def slot_to_time_range(slot):
    """Converte lo slot intero in stringa 'HH:MM−HH:MM'."""
    start_min = int(slot) * config.TIME_SLOT_MIN
    end_min   = start_min + config.TIME_SLOT_MIN
    return f"{start_min//60:02d}:{start_min%60:02d}−{end_min//60:02d}:{end_min%60:02d}"


# ─────────────────────────────────────────────────────────
def dominant_phase(row):
    """
    Identifica la fase dominante (con peso β) e suggerisce la leva.
    Usa le medie dei tempi β-pesati nel corridoio.
    """
    components = {
        "walk":  config.BETA["walk"]  * row.get("avg_walk_min",  0),
        "wait":  config.BETA["wait"]  * row.get("avg_wait_min",  0),
        "ride":  config.BETA["ride"]  * row.get("avg_ride_min",  0),
        "trans": config.BETA["trans"] * row.get("avg_trans_min", 0),
    }
    dom = max(components, key=components.get)
    lever_map = {
        "walk":  "y^stops  (aggiungere fermate / ridurre accesso a piedi)",
        "wait":  "y^freq   (aumentare frequenza / ridurre attesa)",
        "ride":  "y^speed  (bus lane / transit signal priority)",
        "trans": "y^sync   (sincronizzare orari al trasbordo)",
    }
    return dom, lever_map[dom]


# ─────────────────────────────────────────────────────────
def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 4 — Corridoi prioritari e H-index")
    print("=" * 60)

    # ── 1. Carica dati ────────────────────────────────────
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "propensity_per_trip.csv"))
    print(f"  Viaggi con propensità: {len(df):>7}")

    # ── 2. Assegna cella origine e destinazione ───────────
    orig_cells = df.apply(lambda r: coord_to_cell(r["lat_o"], r["lon_o"]), axis=1)
    dest_cells = df.apply(lambda r: coord_to_cell(r["lat_d"], r["lon_d"]), axis=1)

    df["origin_row"], df["origin_col"] = zip(*orig_cells)
    df["dest_row"],   df["dest_col"]   = zip(*dest_cells)
    df["origin_cell"] = df.apply(lambda r: f"({int(r.origin_row)},{int(r.origin_col)})", axis=1)
    df["dest_cell"]   = df.apply(lambda r: f"({int(r.dest_row)},{int(r.dest_col)})", axis=1)

    # ── 3. Assegna slot temporale ─────────────────────────
    df["time_slot"] = df["datetime_start"].apply(parse_datetime_to_slot)
    df = df.dropna(subset=["time_slot"])
    df["time_slot"] = df["time_slot"].astype(int)
    print(f"  Dopo dropna time_slot: {len(df):>7}")

    # ── 4. Filtra outlier celle (celle negative o troppo lontane) ──
    valid_cells = (
        (df["origin_row"] >= 0) & (df["origin_col"] >= 0) &
        (df["dest_row"]   >= 0) & (df["dest_col"]   >= 0) &
        (df["origin_row"] < 100) & (df["dest_row"] < 100)
    )
    df = df[valid_cells].copy()
    print(f"  Dopo filtro celle:     {len(df):>7}")

    # ── 5. Raggruppa in corridoi ──────────────────────────
    grp = df.groupby(["origin_cell", "dest_cell", "time_slot"])

    corridors = grp.agg(
        demand         = ("trip_id",       "count"),
        avg_propensity = ("propensity",     "mean"),
        avg_delta_d    = ("delta_D",        "mean"),
        avg_walk_min   = ("T_walk_min",     "mean"),
        avg_wait_min   = ("T_wait_min",     "mean"),
        avg_ride_min   = ("T_ride_min",     "mean"),
        avg_trans_min  = ("T_trans_min",    "mean"),
    ).reset_index()

    # ── 6. H-index (Eq. 7) ───────────────────────────────
    corridors["h_index"] = (
        corridors["demand"] * (1.0 - corridors["avg_propensity"])
    )

    print(f"\n  Corridoi totali:       {len(corridors):>7}")
    print(f"  Corridoi con H > 0:   {(corridors['h_index'] > 0).sum():>7}")

    # ── 7. Aggiungi informazioni sui percorsi (route names) ──
    try:
        alt_full = pd.read_csv(config.FILE_ALT)
        alt_full = alt_full[alt_full["first_bus_duration"].notna()][
            ["id_spostamento", "first_bus_short_name",
             "second_bus_short_name", "third_bus_short_name"]
        ].copy()

        def get_routes(row):
            r = []
            for col in ["first_bus_short_name","second_bus_short_name","third_bus_short_name"]:
                if pd.notna(row.get(col)):
                    r.append(str(row[col]))
            return "+".join(r) if r else ""

        alt_full["routes"] = alt_full.apply(get_routes, axis=1)
        alt_full = alt_full[["id_spostamento","routes"]].drop_duplicates("id_spostamento")
        df = df.merge(alt_full, left_on="trip_id", right_on="id_spostamento", how="left")

        corr_routes = df.groupby(
            ["origin_cell", "dest_cell", "time_slot"]
        )["routes"].apply(
            lambda s: ", ".join(sorted(set(filter(None, s))))
        ).reset_index()

        corridors = corridors.merge(corr_routes,
                                    on=["origin_cell","dest_cell","time_slot"],
                                    how="left")
    except Exception:
        corridors["routes"] = ""

    # ── 8. Aggiungi centroidi per mappa ───────────────────
    def extract_row_col(cell_str):
        try:
            parts = cell_str.strip("()").split(",")
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0

    origin_rc = corridors["origin_cell"].apply(extract_row_col)
    dest_rc   = corridors["dest_cell"].apply(extract_row_col)

    origin_lats, origin_lons, dest_lats, dest_lons = [], [], [], []
    for (r, c) in origin_rc:
        lat, lon = cell_centroid(r, c)
        origin_lats.append(lat); origin_lons.append(lon)
    for (r, c) in dest_rc:
        lat, lon = cell_centroid(r, c)
        dest_lats.append(lat); dest_lons.append(lon)

    corridors["origin_lat"] = origin_lats
    corridors["origin_lon"] = origin_lons
    corridors["dest_lat"]   = dest_lats
    corridors["dest_lon"]   = dest_lons
    corridors["time_range"] = corridors["time_slot"].apply(slot_to_time_range)

    # ── 9. Corridoi prioritari ────────────────────────────
    top = (
        corridors[corridors["demand"] >= config.MIN_DEMAND]
        .sort_values("h_index", ascending=False)
        .head(config.N_PRIORITY)
        .reset_index(drop=True)
    )
    top["rank"] = range(1, len(top) + 1)

    # Fase dominante e leva suggerita
    dom_phases, levers = [], []
    for _, row in top.iterrows():
        dom, lev = dominant_phase(row)
        dom_phases.append(dom)
        levers.append(lev)
    top["dominant_phase"] = dom_phases
    top["suggested_lever"] = levers

    # ── 10. ID stringa per ogni corridoio ─────────────────
    top["corridor_id"] = top.apply(
        lambda r: f"{r.origin_cell}→{r.dest_cell} {r.time_range}", axis=1)
    corridors["corridor_id"] = corridors.apply(
        lambda r: f"{r.origin_cell}→{r.dest_cell} {r.time_range}", axis=1)

    # ── 11. Stampa sommario ───────────────────────────────
    print(f"\n  Top-{config.N_PRIORITY} corridoi prioritari:")
    print(f"  {'Rank':>4} {'Corridoio':<45} {'Dem':>5} {'Prop':>6} {'H':>6} {'ΔD':>6} {'Leva'}")
    print("  " + "-" * 95)
    for _, r in top.iterrows():
        print(f"  {int(r['rank']):>4} {r['corridor_id']:<45} "
              f"{int(r['demand']):>5} {r['avg_propensity']:>6.3f} "
              f"{r['h_index']:>6.2f} {r['avg_delta_d']:>6.1f}  "
              f"{r['dominant_phase']}")

    # ── 12. Salva output ──────────────────────────────────
    cols_out = [
        "rank", "corridor_id", "origin_cell", "dest_cell", "time_range",
        "demand", "avg_propensity", "h_index",
        "avg_delta_d",
        "avg_walk_min", "avg_wait_min", "avg_ride_min", "avg_trans_min",
        "origin_lat", "origin_lon", "dest_lat", "dest_lon",
        "dominant_phase", "suggested_lever", "routes",
    ]
    cols_out = [c for c in cols_out if c in top.columns]

    out_all  = os.path.join(config.OUTPUT_DIR, "corridors_all.csv")
    out_prio = os.path.join(config.OUTPUT_DIR, "priority_corridors.csv")

    corridors.to_csv(out_all, index=False)
    top[cols_out].to_csv(out_prio, index=False)

    print(f"\n  ✓ Salvato: {out_all}")
    print(f"  ✓ Salvato: {out_prio}")

    return corridors, top


if __name__ == "__main__":
    run()
