"""
step6_candidates.py — Generazione candidati Qλ per tutte e 4 le leve
Inputs:  output_v2/priority_corridors.csv
         output_v2/propensity_per_trip.csv
         output_v2/sigmoid_params.json
         alternative_otp_*.csv
         gtfs/ctm/  gtfs/arst/
Outputs: output_v2/candidates/
           trips_in_corridors.csv     — mapping trip→corridoio
           candidates_freq.csv        — Qfreq
           candidates_sync.csv        — Qsync
           candidates_stops.csv       — Qstops
           candidates_speed.csv       — Qspeed
           lookup_freq.json           — (route,q) → Δt_wait, cost
           lookup_sync.json           — (transfer_leg,offset) → new_wait
           lookup_stops.json          — (candidate,trip) → access_gain_m
           lookup_speed.json          — (route,intervention) → Δt_ride_min, cost
           ga_input_summary.json      — dimensioni vettore y e meta info

Leve:
  y^freq  — aumento frequenza (riduce T_wait)
  y^sync  — sincronizzazione orari trasbordo (riduce T_trans)
  y^stops — nuove fermate lungo shape (riduce T_walk)
  y^speed — velocità commerciale (riduce T_ride): TSP, Bus Lane, Queue Jump
"""

import os, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import config

os.makedirs(config.CANDIDATES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))

def parse_time_s(t):
    try:
        h, m, s = str(t).strip().split(":")
        return int(h)*3600 + int(m)*60 + int(s)
    except:
        return None

def clean_route(s):
    if pd.isna(s): return None
    s = str(s).strip()
    return s[:-2] if s.endswith(".0") else s

def to_slot(s):
    try:
        for fmt in ('%d-%m-%Y %H:%M', '%H:%M:%S', '%H:%M'):
            try:
                dt = datetime.datetime.strptime(str(s).strip(), fmt)
                return (dt.hour*60 + dt.minute) // config.TIME_SLOT_MIN
            except: pass
    except: pass
    return None

LAT_O = 39.15; LON_O = 8.95
DEG_LAT = 1/111_320
DEG_LON  = 1/(111_320 * math.cos(math.radians(39.2)))
CELL_M   = config.CELL_SIZE_M

def coord_to_cell(lat, lon):
    r = int((lat - LAT_O) / (CELL_M * DEG_LAT))
    c = int((lon - LON_O) / (CELL_M * DEG_LON))
    return f"({r},{c})"

# ─────────────────────────────────────────────────────────
# Carica dati base
# ─────────────────────────────────────────────────────────
print("="*60)
print("STEP 6 — Generazione candidati Qλ")
print("="*60)

top    = pd.read_csv(os.path.join(config.OUTPUT_DIR, "priority_corridors.csv"))
prop   = pd.read_csv(os.path.join(config.OUTPUT_DIR, "propensity_per_trip.csv"))
trips  = pd.read_csv(os.path.join(config.OUTPUT_DIR, "trips_clean.csv"))
alt    = pd.read_csv(config.FILE_ALT)
off    = pd.read_csv(config.FILE_OFFSETS)
with open(os.path.join(config.OUTPUT_DIR, "sigmoid_params.json")) as f:
    sig = json.load(f)

# Pulizia nomi route
for col in ["first_bus_short_name","second_bus_short_name","third_bus_short_name"]:
    alt[col] = alt[col].apply(clean_route)

# ─────────────────────────────────────────────────────────
# GTFS unificato CTM + ARST
# ─────────────────────────────────────────────────────────
def load_gtfs(path):
    d = {}
    for fname in ["routes","trips","stops","stop_times","shapes"]:
        fp = os.path.join(path, f"{fname}.txt")
        if os.path.exists(fp):
            d[fname] = pd.read_csv(fp)
    return d

ctm  = load_gtfs(config.GTFS_CTM)
arst = load_gtfs(config.GTFS_ARST)

# Merge routes+trips per lookup route_short_name→route_id
def build_route_lookup(gtfs):
    tr = gtfs["trips"].merge(gtfs["routes"][["route_id","route_short_name"]], on="route_id")
    return tr[["route_id","route_short_name","trip_id","shape_id"]].drop_duplicates()

ctm_rl  = build_route_lookup(ctm)
arst_rl = build_route_lookup(arst)
all_rl  = pd.concat([ctm_rl, arst_rl], ignore_index=True)

# Stop_times con route_short_name
def build_st_full(gtfs, rl):
    return gtfs["stop_times"].merge(
        rl[["trip_id","route_short_name"]].drop_duplicates(), on="trip_id", how="left")

ctm_st_full  = build_st_full(ctm,  ctm_rl)
arst_st_full = build_st_full(arst, arst_rl)
all_st_full  = pd.concat([ctm_st_full, arst_st_full], ignore_index=True)
all_st_full["dep_s"] = all_st_full["departure_time"].apply(parse_time_s)

# Tutti gli stops
all_stops = pd.concat([
    ctm["stops"][["stop_id","stop_name","stop_lat","stop_lon"]],
    arst["stops"][["stop_id","stop_name","stop_lat","stop_lon"]]
], ignore_index=True).drop_duplicates("stop_id")

# Tutti gli shapes
all_shapes = pd.concat([
    ctm["shapes"], arst["shapes"]
], ignore_index=True)

print(f"  GTFS caricato: {len(all_rl['route_short_name'].unique())} route, "
      f"{len(all_stops)} stops, {len(all_shapes)} shape points")

# ─────────────────────────────────────────────────────────
# MAPPING: trip → corridoio prioritario
# ─────────────────────────────────────────────────────────
print("\n[0] Mapping trip → corridoio")

prop["origin_cell"] = prop.apply(lambda r: coord_to_cell(r.lat_o, r.lon_o), axis=1)
prop["dest_cell"]   = prop.apply(lambda r: coord_to_cell(r.lat_d, r.lon_d), axis=1)
prop["time_slot"]   = prop["datetime_start"].apply(to_slot)

# Costruisci dizionario corridoio → lista trip_id
corridor_trips = {}   # corr_id → [trip_id, ...]
trip_corr_map  = {}   # trip_id → corr_id

for _, row in top.iterrows():
    oc = row["origin_cell"]; dc = row["dest_cell"]
    tr_str = row["time_range"].split("−")[0]
    h, m = tr_str.split(":"); slot = (int(h)*60+int(m))//config.TIME_SLOT_MIN
    cid = row["corridor_id"]

    mask = (prop["origin_cell"]==oc) & (prop["dest_cell"]==dc) & (prop["time_slot"]==slot)
    tids = prop.loc[mask, "trip_id"].tolist()
    corridor_trips[cid] = tids
    for tid in tids:
        trip_corr_map[tid] = cid

# Salva mapping
tic_rows = []
for cid, tids in corridor_trips.items():
    rank_row = top[top["corridor_id"]==cid].iloc[0]
    for tid in tids:
        tic_rows.append({"corridor_id": cid, "rank": int(rank_row["rank"]),
                         "trip_id": tid})
tic_df = pd.DataFrame(tic_rows)
tic_df.to_csv(os.path.join(config.CANDIDATES_DIR, "trips_in_corridors.csv"), index=False)
print(f"  Corridoi: {len(corridor_trips)}, trip mappati: {len(trip_corr_map)}")

# Alternative per i trip PV nei corridoi (servono per benefit)
alt_valid = alt[alt["status"].isna() & alt["first_bus_duration"].notna()].copy()
off_ok = off[off["status"]=="OK"][["id_spostamento","id_alternativa","offset(secondi)"]].copy()
alt_valid = alt_valid.merge(off_ok, on=["id_spostamento","id_alternativa"], how="left")
alt_valid["offset_corretto"] = alt_valid["offset(secondi)"].apply(
    lambda x: (x+86400) if (pd.notna(x) and x<0) else x).fillna(0)

# PV trips con alternativa nei corridoi
alt_in_corr = alt_valid[alt_valid["id_spostamento"].isin(trip_corr_map.keys())].copy()
alt_in_corr["corridor_id"] = alt_in_corr["id_spostamento"].map(trip_corr_map)
print(f"  Alternative PV nei corridoi: {len(alt_in_corr)}")

# ─────────────────────────────────────────────────────────
# Calcolo headway per route dal GTFS
# ─────────────────────────────────────────────────────────
def compute_headway(route_name, st_full):
    rdata = st_full[st_full["route_short_name"]==route_name].copy()
    if len(rdata)==0: return None, None, None
    first = rdata[rdata["stop_sequence"]==rdata.groupby("trip_id")["stop_sequence"].transform("min")]
    deps = first["dep_s"].dropna().sort_values().values
    if len(deps)<2: return None, None, None
    diffs = np.diff(deps)
    diffs = diffs[(diffs>0) & (diffs<7200)]
    hw = float(np.median(diffs))/60 if len(diffs)>0 else None
    # Cycle time
    cycle = rdata.groupby("trip_id").apply(
        lambda g: (g["dep_s"].max()-g["dep_s"].min())/60 if g["dep_s"].notna().sum()>1 else None)
    cycle = cycle.dropna()
    ct = float(cycle.median()) if len(cycle)>0 else None
    n_trips = rdata["trip_id"].nunique()
    return hw, ct, n_trips

# Pre-calcola headway per tutte le route che compaiono nei corridoi
route_set = set()
for cid, tids in corridor_trips.items():
    ta = alt_in_corr[alt_in_corr["corridor_id"]==cid]
    for col in ["first_bus_short_name","second_bus_short_name","third_bus_short_name"]:
        route_set.update(ta[col].dropna().tolist())

route_headways = {}
for rname in sorted(route_set):
    hw, ct, nt = compute_headway(rname, all_st_full)
    if hw:
        route_headways[rname] = {"headway_min": hw, "cycle_min": ct, "n_trips_day": nt}

print(f"\n  Route nei corridoi con headway GTFS: {len(route_headways)}/{len(route_set)}")

# ─────────────────────────────────────────────────────────
# LEVA 1 — y^freq: candidati frequenza
# ─────────────────────────────────────────────────────────
print("\n[1] Generazione Qfreq (frequenza)")

freq_candidates = []
lookup_freq = {}   # (route, q_idx) → {delta_wait_min, cost_eur, new_headway}

for rname, rinfo in route_headways.items():
    hw0 = rinfo["headway_min"]
    ct  = rinfo["cycle_min"] or hw0 * 4
    f0  = 1.0 / hw0   # corse/min

    # Frequenza massima: min(1/H_min, MAX_FREQ_MULT × f0)
    f_max = min(1.0/config.H_MIN_MIN, config.MAX_FREQ_MULT * f0)
    # Numero massimo di veicoli aggiuntivi
    n_veh_base = max(1, round(ct / hw0))
    n_veh_max  = max(1, round(ct * f_max))
    q_max = n_veh_max - n_veh_base

    for q in range(1, min(q_max+1, 6)):   # max +5 veicoli per route
        n_new   = n_veh_base + q
        f_new   = n_new / ct   # corse/min
        hw_new  = 1.0 / f_new  # min
        if hw_new < config.H_MIN_MIN:
            continue

        # Riduzione attesa media: Δt_wait = hw0/2 - hw_new/2
        delta_wait = hw0/2 - hw_new/2   # minuti

        # Costo globale: q veicoli aggiuntivi × costo operativo
        cost = q * config.C_VEHICLE_OPEX_EUR

        cand_id = f"freq_{rname}_+{q}veh"
        freq_candidates.append({
            "candidate_id":  cand_id,
            "lever":         "freq",
            "route":         rname,
            "q_vehicles":    q,
            "headway_0_min": hw0,
            "headway_new_min": hw_new,
            "delta_wait_min":  delta_wait,
            "cost_eur":      cost,
        })
        lookup_freq[cand_id] = {
            "route":          rname,
            "delta_wait_min": delta_wait,
            "cost_eur":       cost,
            "headway_new_min": hw_new,
            "q_vehicles":     q,
        }

cands_freq = pd.DataFrame(freq_candidates)
cands_freq.to_csv(os.path.join(config.CANDIDATES_DIR,"candidates_freq.csv"), index=False)
print(f"  Candidati Qfreq: {len(cands_freq)}  "
      f"(per {cands_freq['route'].nunique()} route)")

# ─────────────────────────────────────────────────────────
# LEVA 2 — y^sync: candidati sincronizzazione
# ─────────────────────────────────────────────────────────
print("\n[2] Generazione Qsync (sincronizzazione trasbordi)")

DELTA_OFFSET = 2.0   # minuti — passo discretizzazione

sync_candidates = []
lookup_sync = {}   # (transfer_leg_id, cand_id) → new_wait_min

transfer_legs = []   # raccoglie tutti i leg di trasbordo nei corridoi

leg_id_counter = 0
for _, arow in alt_in_corr.iterrows():
    cid = arow["corridor_id"]

    # mid_walk1: trasbordo bus1→bus2
    if pd.notna(arow.get("mid_walk1_duration")) and pd.notna(arow.get("second_bus_short_name")):
        t_walk_end  = parse_time_s(arow.get("mid_walk1_to_time"))
        t_bus2_dep  = parse_time_s(arow.get("second_bus_from_time"))
        if t_walk_end and t_bus2_dep:
            baseline_wait = (t_bus2_dep - t_walk_end) % 3600
            baseline_wait = min(baseline_wait/60, config.TRANSFER_WAIT_CAP_S/60)
            r_arr = arow.get("first_bus_short_name")
            r_dep = arow.get("second_bus_short_name")
            if r_arr and r_dep:
                leg_id = f"tleg_{leg_id_counter}"
                leg_id_counter += 1
                transfer_legs.append({
                    "leg_id":        leg_id,
                    "corridor_id":   cid,
                    "trip_id":       arow["id_spostamento"],
                    "alt_id":        arow["id_alternativa"],
                    "route_arr":     r_arr,
                    "route_dep":     r_dep,
                    "baseline_wait_min": baseline_wait,
                    "walk_dur_min":  arow["mid_walk1_duration"]/60,
                })

    # mid_walk2: trasbordo bus2→bus3
    if pd.notna(arow.get("mid_walk2_duration")) and pd.notna(arow.get("third_bus_short_name")):
        t_walk_end  = parse_time_s(arow.get("mid_walk2_to_time"))
        t_bus3_dep  = parse_time_s(arow.get("third_bus_from_time"))
        if t_walk_end and t_bus3_dep:
            baseline_wait = (t_bus3_dep - t_walk_end) % 3600
            baseline_wait = min(baseline_wait/60, config.TRANSFER_WAIT_CAP_S/60)
            r_arr = arow.get("second_bus_short_name")
            r_dep = arow.get("third_bus_short_name")
            if r_arr and r_dep:
                leg_id = f"tleg_{leg_id_counter}"
                leg_id_counter += 1
                transfer_legs.append({
                    "leg_id":        leg_id,
                    "corridor_id":   cid,
                    "trip_id":       arow["id_spostamento"],
                    "alt_id":        arow["id_alternativa"],
                    "route_arr":     r_arr,
                    "route_dep":     r_dep,
                    "baseline_wait_min": baseline_wait,
                    "walk_dur_min":  arow["mid_walk2_duration"]/60,
                })

tlegs_df = pd.DataFrame(transfer_legs) if transfer_legs else pd.DataFrame()

# Per ogni route coinvolta nei trasbordi, genera offset candidates
routes_in_transfers = set()
if len(tlegs_df) > 0:
    routes_in_transfers.update(tlegs_df["route_arr"].tolist())
    routes_in_transfers.update(tlegs_df["route_dep"].tolist())

sync_cand_list = []
route_offsets = {}  # route → lista di offset candidati (minuti)

for rname in routes_in_transfers:
    hw = route_headways.get(rname, {}).get("headway_min")
    if hw is None:
        hw = 10.0  # default
    # Range offset: [-H/2, ..., 0, ..., +H/2]
    n_steps = int(hw / DELTA_OFFSET)
    offsets = [round(-hw/2 + i*DELTA_OFFSET, 1) for i in range(n_steps+1)]
    if 0.0 not in offsets:
        offsets.append(0.0)
    offsets = sorted(set(offsets))
    route_offsets[rname] = offsets

    for off_val in offsets:
        if off_val == 0.0:
            continue   # offset 0 = baseline, non è un candidato attivo
        cand_id = f"sync_{rname}_off{off_val:+.1f}"
        cost    = config.C_COORD_ROUTE_EUR   # costo coordinamento

        sync_cand_list.append({
            "candidate_id": cand_id,
            "lever":        "sync",
            "route":        rname,
            "offset_min":   off_val,
            "cost_eur":     cost,
        })

cands_sync = pd.DataFrame(sync_cand_list) if sync_cand_list else pd.DataFrame()

# Lookup table: (leg_id, cand_id_arr, cand_id_dep) → new_wait_min
# Per semplicità computazionale, pre-calcola per ogni leg e ogni coppia di offsets
# (solo offset del feeder e del connettore)
lookup_sync = {}
if len(tlegs_df) > 0:
    for _, leg in tlegs_df.iterrows():
        lid   = leg["leg_id"]
        r_arr = leg["route_arr"]
        r_dep = leg["route_dep"]
        bw    = leg["baseline_wait_min"]
        wk    = leg["walk_dur_min"]

        hw_dep = route_headways.get(r_dep, {}).get("headway_min", 10.0)
        offsets_arr = route_offsets.get(r_arr, [0.0])
        offsets_dep = route_offsets.get(r_dep, [0.0])

        entry = {"baseline_wait_min": bw, "hw_dep_min": hw_dep,
                 "walk_dur_min": wk, "offsets": {}}
        for oa in offsets_arr:
            for od in offsets_dep:
                # new_wait = (baseline_wait + od - oa) mod hw_dep
                new_wait = (bw + od - oa) % hw_dep
                # Apply safety margin
                if new_wait < config.T_SAFE_MIN:
                    new_wait = new_wait + hw_dep   # miss connection → wait full cycle
                new_wait = min(new_wait, config.TRANSFER_WAIT_CAP_S/60)
                entry["offsets"][f"{oa:+.1f}|{od:+.1f}"] = new_wait
        lookup_sync[lid] = entry

if len(cands_sync) > 0:
    cands_sync.to_csv(os.path.join(config.CANDIDATES_DIR,"candidates_sync.csv"), index=False)
tlegs_df.to_csv(os.path.join(config.CANDIDATES_DIR,"transfer_legs.csv"), index=False)
print(f"  Transfer legs nei corridoi: {len(tlegs_df)}")
print(f"  Candidati Qsync: {len(cands_sync)}  "
      f"(per {len(routes_in_transfers)} route con trasbordo)")

# ─────────────────────────────────────────────────────────
# LEVA 3 — y^stops: candidati fermate
# ─────────────────────────────────────────────────────────
print("\n[3] Generazione Qstops (nuove fermate)")

def get_shape_for_route(rname, rl, shapes):
    """Restituisce la shape geometry (lat,lon) per la route."""
    sids = rl[rl["route_short_name"]==rname]["shape_id"].dropna().unique()
    if len(sids)==0: return pd.DataFrame()
    # Usa la shape più lunga (più rappresentativa)
    best = None; best_n = 0
    for sid in sids:
        sh = shapes[shapes["shape_id"]==sid].sort_values("shape_pt_sequence")
        if len(sh) > best_n:
            best = sh; best_n = len(sh)
    return best[["shape_pt_lat","shape_pt_lon"]].values if best is not None else np.array([])

def interpolate_shape(pts, step_m=100):
    """Genera punti a distanza step_m lungo la polyline."""
    candidates = []
    acc = 0.0
    if len(pts) < 2: return candidates
    for i in range(len(pts)-1):
        la1,lo1 = pts[i]; la2,lo2 = pts[i+1]
        seg_len = haversine_m(la1,lo1,la2,lo2)
        if seg_len == 0: continue
        while acc < seg_len:
            t = acc / seg_len
            candidates.append((la1+t*(la2-la1), lo1+t*(lo2-lo1)))
            acc += step_m
        acc -= seg_len
    return candidates

def min_dist_to_stops(lat, lon, stops_df):
    """Distanza minima in metri da un punto al set di stops esistenti."""
    dists = stops_df.apply(
        lambda r: haversine_m(lat, lon, r.stop_lat, r.stop_lon), axis=1)
    return dists.min()

stops_candidates = []
lookup_stops = {}  # cand_id → {trips: [{trip_id, access_gain_m, dwell_pen_min}]}

# Trips PV nei corridoi con dati walk
pv_trip_ids = set(alt_in_corr["id_spostamento"].unique())

# Dati posizione origine per i trip
trip_positions = trips[trips["trip_id"].isin(trip_corr_map.keys())][
    ["trip_id","lat_o","lon_o"]].set_index("trip_id")

# Walk distances attuali (dalla miglior alternativa per ogni trip)
dd = pd.read_csv(os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv"))
dd_pv = dd[dd["is_pv"]].set_index("trip_id")[["T_walk_min"]]

cand_stop_id = 0
for rname in sorted(route_set):
    sh_pts = get_shape_for_route(rname, all_rl, all_shapes)
    if len(sh_pts) < 2:
        continue

    cand_pts = interpolate_shape(sh_pts, config.STOP_CANDIDATE_STEP_M)
    if not cand_pts:
        continue

    # Stops esistenti su questa route
    st_route = all_st_full[all_st_full["route_short_name"]==rname]["stop_id"].unique()
    ex_stops = all_stops[all_stops["stop_id"].isin(st_route)]

    for lat_c, lon_c in cand_pts:
        # Filtro bounding box
        if not (config.LAT_MIN <= lat_c <= config.LAT_MAX and
                config.LON_MIN <= lon_c <= config.LON_MAX):
            continue

        # Distanza minima da stops esistenti
        if len(ex_stops) > 0:
            d_min = min(haversine_m(lat_c, lon_c, r.stop_lat, r.stop_lon)
                        for _, r in ex_stops.iterrows())
            if d_min < config.STOP_MIN_SPACING_M:
                continue

        cand_id = f"stop_{rname}_{cand_stop_id}"
        cand_stop_id += 1

        # Calcola benefit per ogni trip PV nel corridoio che usa questa route
        trip_benefits = []
        corr_trips_using_route = alt_in_corr[
            (alt_in_corr["first_bus_short_name"]==rname) |
            (alt_in_corr["second_bus_short_name"]==rname) |
            (alt_in_corr["third_bus_short_name"]==rname)
        ]

        for _, ta in corr_trips_using_route.iterrows():
            tid = ta["id_spostamento"]
            if tid not in trip_positions.index:
                continue
            lat_o = trip_positions.loc[tid,"lat_o"]
            lon_o = trip_positions.loc[tid,"lon_o"]

            # Distanza attuale dall'origine al candidato (come proxy della nuova walk)
            d_to_cand = haversine_m(lat_o, lon_o, lat_c, lon_c)
            # Walk attuale (dalla miglior alternativa)
            walk_min_curr = dd_pv.loc[tid,"T_walk_min"] if tid in dd_pv.index else None
            if walk_min_curr is None:
                continue
            walk_m_curr = walk_min_curr * 60 * config.WALK_SPEED_MPS

            # Access gain: max(0, walk_corrente - walk_nuova)
            walk_new_m = min(d_to_cand, walk_m_curr)
            gain_m = max(0.0, walk_m_curr - walk_new_m)

            if gain_m > 0:
                # Dwell penalty: tempo perso dagli utenti già a bordo
                dwell_pen_min = config.DWELL_TIME_S / 60.0
                trip_benefits.append({
                    "trip_id":      tid,
                    "access_gain_m": gain_m,
                    "dwell_pen_min": dwell_pen_min,
                })

        if not trip_benefits:
            continue   # nessun benefit reale → scarta candidato

        stops_candidates.append({
            "candidate_id": cand_id,
            "lever":        "stops",
            "route":        rname,
            "lat":          lat_c,
            "lon":          lon_c,
            "n_trips_benefit": len(trip_benefits),
            "cost_eur":     config.C_STOP_CAPEX_EUR,
        })
        lookup_stops[cand_id] = {
            "route": rname,
            "lat":   lat_c,
            "lon":   lon_c,
            "cost_eur": config.C_STOP_CAPEX_EUR,
            "trips": trip_benefits,
        }

cands_stops = pd.DataFrame(stops_candidates) if stops_candidates else pd.DataFrame()
if len(cands_stops) > 0:
    cands_stops.to_csv(os.path.join(config.CANDIDATES_DIR,"candidates_stops.csv"), index=False)
print(f"  Candidati Qstops: {len(cands_stops)}  "
      f"(per {cands_stops['route'].nunique() if len(cands_stops)>0 else 0} route)")

# ─────────────────────────────────────────────────────────
# LEVA 4 — y^speed: candidati velocità commerciale
# ─────────────────────────────────────────────────────────
print("\n[4] Generazione Qspeed (velocità commerciale)")

def route_total_length_km(rname, rl, shapes):
    sh_pts = get_shape_for_route(rname, rl, shapes)
    if len(sh_pts) < 2: return None
    total = sum(haversine_m(sh_pts[i][0],sh_pts[i][1],
                            sh_pts[i+1][0],sh_pts[i+1][1])
                for i in range(len(sh_pts)-1))
    return total / 1000   # km

speed_candidates = []
lookup_speed = {}

# Velocità commerciale baseline per route (da GTFS: lunghezza shape / tempo trip)
def route_baseline_speed(rname, rl, shapes, st_full):
    L_km = route_total_length_km(rname, rl, shapes)
    if L_km is None: return None, None
    rdata = st_full[st_full["route_short_name"]==rname]
    trip_dur = rdata.groupby("trip_id").apply(
        lambda g: (g["dep_s"].max()-g["dep_s"].min())/3600
        if g["dep_s"].notna().sum()>1 else None).dropna()
    t_h = float(trip_dur.median()) if len(trip_dur)>0 else None
    if t_h is None or t_h <= 0: return None, L_km
    return L_km/t_h, L_km   # km/h, km

for rname in sorted(route_set):
    v0_kmh, L_km = route_baseline_speed(rname, all_rl, all_shapes, all_st_full)
    if v0_kmh is None or L_km is None or v0_kmh <= 0:
        continue

    # Ride time baseline per gli utenti nel corridoio che usano questa route
    trips_using_r = alt_in_corr[
        (alt_in_corr["first_bus_short_name"]==rname) |
        (alt_in_corr["second_bus_short_name"]==rname) |
        (alt_in_corr["third_bus_short_name"]==rname)
    ]

    # Durata ride stimata per ogni trip (proporzionale alla distanza percorsa)
    avg_ride_min = None
    if len(trips_using_r) > 0:
        ride_times = []
        for col_d, col_dur in [
            ("first_bus_short_name",  "first_bus_duration"),
            ("second_bus_short_name", "second_bus_duration"),
            ("third_bus_short_name",  "third_bus_duration"),
        ]:
            mask = trips_using_r[col_d]==rname
            vals = trips_using_r.loc[mask, col_dur].dropna()
            ride_times.extend(vals.tolist())
        avg_ride_min = np.mean(ride_times)/60 if ride_times else None

    if avg_ride_min is None or avg_ride_min <= 0:
        continue

    for itype, iparams in config.SPEED_CATALOG.items():
        dv = iparams["gain_kmh"]

        # Benefit: riduzione ride time
        # t_ride_new = t_ride_0 × v0 / (v0+Δv)
        # Δt_ride = t_ride_0 × (1 - v0/(v0+Δv))
        delta_ride_min = avg_ride_min * (1.0 - v0_kmh/(v0_kmh + dv))

        # Costo intervento
        if iparams["unit"] == "km":
            cost = iparams["cost_eur"] * L_km
        elif iparams["unit"] == "intersection":
            n_int = max(1, round(L_km * 3))   # ~3 incroci/km su arterie urbane
            cost = iparams["cost_eur"] * n_int
        else:   # junction
            n_j = max(1, round(L_km * 2))
            cost = iparams["cost_eur"] * n_j

        # Costo quadratico (Eq. 21): gamma × L × Δv²
        cost_q = config.GAMMA_SPEED * L_km * dv**2
        cost_total = cost + cost_q

        cand_id = f"speed_{rname}_{itype}"
        speed_candidates.append({
            "candidate_id":    cand_id,
            "lever":           "speed",
            "route":           rname,
            "intervention":    itype,
            "v0_kmh":          round(v0_kmh,2),
            "dv_kmh":          dv,
            "L_km":            round(L_km,2),
            "delta_ride_min":  round(delta_ride_min,3),
            "cost_eur":        round(cost_total,0),
        })
        lookup_speed[cand_id] = {
            "route":          rname,
            "intervention":   itype,
            "delta_ride_min": delta_ride_min,
            "cost_eur":       cost_total,
            "trip_ids":       trips_using_r["id_spostamento"].tolist(),
        }

cands_speed = pd.DataFrame(speed_candidates) if speed_candidates else pd.DataFrame()
if len(cands_speed) > 0:
    cands_speed.to_csv(os.path.join(config.CANDIDATES_DIR,"candidates_speed.csv"), index=False)
print(f"  Candidati Qspeed: {len(cands_speed)}  "
      f"(per {cands_speed['route'].nunique() if len(cands_speed)>0 else 0} route, "
      f"3 tipi intervento)")

# ─────────────────────────────────────────────────────────
# Salva lookup tables e riepilogo input GA
# ─────────────────────────────────────────────────────────
print("\n[5] Salvataggio lookup tables")

with open(os.path.join(config.CANDIDATES_DIR,"lookup_freq.json"),  "w") as f:
    json.dump(lookup_freq,  f, indent=2)
with open(os.path.join(config.CANDIDATES_DIR,"lookup_sync.json"),  "w") as f:
    json.dump(lookup_sync,  f, indent=2)
with open(os.path.join(config.CANDIDATES_DIR,"lookup_stops.json"), "w") as f:
    json.dump(lookup_stops, f, indent=2)
with open(os.path.join(config.CANDIDATES_DIR,"lookup_speed.json"), "w") as f:
    json.dump(lookup_speed, f, indent=2)

# Riepilogo per il GA: ordine dei candidati nel vettore y
all_cand_ids = (
    cands_freq["candidate_id"].tolist() +
    (cands_sync["candidate_id"].tolist() if len(cands_sync)>0 else []) +
    (cands_stops["candidate_id"].tolist() if len(cands_stops)>0 else []) +
    cands_speed["candidate_id"].tolist()
)

ga_meta = {
    "n_freq":   len(cands_freq),
    "n_sync":   len(cands_sync) if len(cands_sync)>0 else 0,
    "n_stops":  len(cands_stops) if len(cands_stops)>0 else 0,
    "n_speed":  len(cands_speed) if len(cands_speed)>0 else 0,
    "n_total":  len(all_cand_ids),
    "candidate_order": all_cand_ids,
    "corridor_ids":    top["corridor_id"].tolist(),
    "sigmoid_params":  sig,
}
with open(os.path.join(config.CANDIDATES_DIR,"ga_input_summary.json"),"w") as f:
    json.dump(ga_meta, f, indent=2)

print(f"\n  Riepilogo vettore y:")
print(f"    y^freq  : {ga_meta['n_freq']:>4} candidati")
print(f"    y^sync  : {ga_meta['n_sync']:>4} candidati")
print(f"    y^stops : {ga_meta['n_stops']:>4} candidati")
print(f"    y^speed : {ga_meta['n_speed']:>4} candidati")
print(f"    |y| totale: {ga_meta['n_total']}")
print(f"    Spazio soluzioni: 2^{ga_meta['n_total']} ≈ 10^{ga_meta['n_total']*0.301:.0f}")
print(f"\n  Salvati in: {config.CANDIDATES_DIR}/")
