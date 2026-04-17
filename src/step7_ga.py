"""
step7_ga.py  —  Genetic Algorithm con metrica benefit corretta
=============================================================

Output:
    output_v2/ga_best_solution.json
    output_v2/ga_convergence.csv
    output_v2/ga_results_corridors.csv
    output_v2/fig_FG_ga_results.png
"""

import os, json, math, time, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

random.seed(config.GA_SEED)
np.random.seed(config.GA_SEED)

# ─────────────────────────────────────────────────────────────────
# Caricamento dati
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7  —  Genetic Algorithm (metrica: utenti convertiti)")
print("=" * 60)

CDIR = config.CANDIDATES_DIR

with open(os.path.join(CDIR, "ga_input_summary.json")) as f: meta   = json.load(f)
with open(os.path.join(CDIR, "lookup_freq.json"))       as f: lk_freq  = json.load(f)
with open(os.path.join(CDIR, "lookup_sync.json"))       as f: lk_sync  = json.load(f)
with open(os.path.join(CDIR, "lookup_stops.json"))      as f: lk_stops = json.load(f)
with open(os.path.join(CDIR, "lookup_speed.json"))      as f: lk_speed = json.load(f)

sig     = meta["sigmoid_params"]
P_MAX   = sig["p_max"];  G_SIG = sig["g"];  D0_FLEX = sig["d0_flex"]

top   = pd.read_csv(os.path.join(config.OUTPUT_DIR, "priority_corridors.csv"))
tic   = pd.read_csv(os.path.join(CDIR, "trips_in_corridors.csv"))
dd    = pd.read_csv(os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv"))
tlegs = pd.read_csv(os.path.join(CDIR, "transfer_legs.csv")) \
        if os.path.exists(os.path.join(CDIR, "transfer_legs.csv")) \
        else pd.DataFrame()

cands_freq  = pd.read_csv(os.path.join(CDIR, "candidates_freq.csv"))
cands_sync  = pd.read_csv(os.path.join(CDIR, "candidates_sync.csv"))  \
              if os.path.exists(os.path.join(CDIR, "candidates_sync.csv"))  else pd.DataFrame()
cands_stops = pd.read_csv(os.path.join(CDIR, "candidates_stops.csv")) \
              if os.path.exists(os.path.join(CDIR, "candidates_stops.csv")) else pd.DataFrame()
cands_speed = pd.read_csv(os.path.join(CDIR, "candidates_speed.csv")) \
              if os.path.exists(os.path.join(CDIR, "candidates_speed.csv")) else pd.DataFrame()

N_FREQ  = meta["n_freq"];  N_SYNC = meta["n_sync"]
N_STOPS = meta["n_stops"]; N_SPEED = meta["n_speed"]
N       = meta["n_total"];  CAND_ORDER = meta["candidate_order"]

IDX_FREQ  = list(range(0, N_FREQ))
IDX_SYNC  = list(range(N_FREQ, N_FREQ + N_SYNC))
IDX_STOPS = list(range(N_FREQ + N_SYNC, N_FREQ + N_SYNC + N_STOPS))
IDX_SPEED = list(range(N_FREQ + N_SYNC + N_STOPS, N))

print(f"  Vettore y: |y|={N}  "
      f"[freq:{N_FREQ} | sync:{N_SYNC} | stops:{N_STOPS} | speed:{N_SPEED}]")

# ─────────────────────────────────────────────────────────────────
# Strutture di look-up rapide (pre-calcolo O(1) per fitness eval)
# ─────────────────────────────────────────────────────────────────
def sigmoid(d):
    try:
        return P_MAX / (1.0 + math.exp(G_SIG * (d - D0_FLEX)))
    except OverflowError:
        return 0.0 if G_SIG * (d - D0_FLEX) > 0 else P_MAX

# ΔD baseline per ogni trip PV nei corridoi
dd_pv = dd[dd["is_pv"]].set_index("trip_id")[
    ["delta_D", "T_walk_min", "T_wait_min", "T_ride_min", "T_trans_min"]
].to_dict("index")

# trip_id → corridoio
trip_to_corr = dict(zip(tic["trip_id"], tic["corridor_id"]))

# Domanda per corridoio (da top)
demand_by_corr = dict(zip(top["corridor_id"], top["demand"].astype(int)))

# ΔD baseline per corridoio (media sui trip del corridoio)
prop_base_by_corr = {}
for _, crow in top.iterrows():
    cid  = crow["corridor_id"]
    tids = tic[tic["corridor_id"] == cid]["trip_id"].tolist()
    vals = [sigmoid(dd_pv[t]["delta_D"]) for t in tids if t in dd_pv]
    prop_base_by_corr[cid] = float(np.mean(vals)) if vals else 0.0

# Struttura freq: route → [(idx_y, delta_wait_min, cost_eur, n_veh)]
freq_by_route = {}
for i_rel, cid in enumerate(CAND_ORDER[:N_FREQ]):
    if cid in lk_freq:
        r = lk_freq[cid]["route"]
        freq_by_route.setdefault(r, []).append(
            (i_rel, lk_freq[cid]["delta_wait_min"],
             lk_freq[cid]["cost_eur"], lk_freq[cid]["q_vehicles"]))

# Struttura sync: route → [(idx_y, offset_min, cost_eur, cand_id)]
sync_by_route = {}
if len(cands_sync) > 0:
    for i_rel, cid in enumerate(CAND_ORDER[N_FREQ:N_FREQ + N_SYNC]):
        row = cands_sync[cands_sync["candidate_id"] == cid]
        if len(row) > 0:
            r = row.iloc[0]["route"]
            sync_by_route.setdefault(r, []).append(
                (N_FREQ + i_rel, float(row.iloc[0]["offset_min"]),
                 float(row.iloc[0]["cost_eur"]), cid))

# Struttura stops e speed indicizzate per posizione nel vettore y
stops_by_idx = {N_FREQ + N_SYNC + i: lk_stops[cid]
                for i, cid in enumerate(CAND_ORDER[N_FREQ + N_SYNC:
                                                   N_FREQ + N_SYNC + N_STOPS])
                if cid in lk_stops}
speed_by_idx = {N_FREQ + N_SYNC + N_STOPS + i: lk_speed[cid]
                for i, cid in enumerate(CAND_ORDER[N_FREQ + N_SYNC + N_STOPS:])
                if cid in lk_speed}

tleg_list = tlegs.to_dict("records") if len(tlegs) > 0 else []

# ─────────────────────────────────────────────────────────────────
# Funzione di valutazione fitness
# Metrica: users_converted = Σ ΔProp_odh × Dem_odh
# ─────────────────────────────────────────────────────────────────
def evaluate(y):
    """
    Ritorna (fitness, users_converted, total_cost_eur, violation, per_corridor_dict).
    fitness = users_converted − penalty × violation
    """
    beta = config.BETA

    # ── Leggi attivazioni dal cromosoma ──────────────────
    # Freq: delta_wait per route (prendi il candidato attivato con Δ maggiore)
    route_delta_wait = {}
    freq_cost = 0.0;  n_veh_added = 0
    for route, cands in freq_by_route.items():
        for (idx, dw, cost, nv) in cands:
            if y[idx] == 1:
                if dw > route_delta_wait.get(route, 0.0):
                    route_delta_wait[route] = dw
                    freq_cost   += cost
                    n_veh_added += nv

    # Sync: offset per route
    route_offset = {}
    sync_cost = 0.0;  n_sync_routes = 0
    for route, cands in sync_by_route.items():
        for (idx, off_val, cost, cid) in cands:
            if y[idx] == 1:
                route_offset[route] = off_val
                sync_cost    += cost
                n_sync_routes += 1
                break

    stops_active = [idx for idx in stops_by_idx if y[idx] == 1]
    stops_cost   = sum(stops_by_idx[i]["cost_eur"] for i in stops_active)

    speed_active = [idx for idx in speed_by_idx if y[idx] == 1]
    speed_cost   = sum(speed_by_idx[i]["cost_eur"] for i in speed_active)

    total_cost = freq_cost + sync_cost + stops_cost + speed_cost
    capex_cost = stops_cost + speed_cost
    opex_cost  = freq_cost + sync_cost

    # ── Calcola ΔProp per ogni trip e aggrega per corridoio ─
    users_converted = 0.0
    per_corridor = {}

    for _, crow in top.iterrows():
        cid  = crow["corridor_id"]
        dem  = int(crow["demand"])
        tids = tic[tic["corridor_id"] == cid]["trip_id"].tolist()

        delta_prop_sum = 0.0
        n_valid = 0

        for tid in tids:
            if tid not in dd_pv:
                continue
            base = dd_pv[tid]
            dD   = base["delta_D"]

            # Guadagno freq (riduce T_wait): usa la route migliore tra quelle
            # che servono questo corridoio
            dw_gain = max(
                (route_delta_wait.get(r, 0.0)
                 for r in freq_by_route.keys()),
                default=0.0
            )
            gain_wait = beta["wait"] * dw_gain

            # Guadagno sync (riduce T_trans)
            gain_trans = 0.0
            for leg in tleg_list:
                if leg["trip_id"] != tid:
                    continue
                r_arr = leg["route_arr"];  r_dep = leg["route_dep"]
                oa = route_offset.get(r_arr, 0.0)
                od = route_offset.get(r_dep, 0.0)
                key = f"{oa:+.1f}|{od:+.1f}"
                leg_lu   = lk_sync.get(leg["leg_id"], {})
                bw       = leg["baseline_wait_min"]
                new_wait = leg_lu.get("offsets", {}).get(key, bw)
                gain_trans += beta["trans"] * (bw - new_wait)

            # Guadagno stops (riduce T_walk)
            gain_walk = 0.0
            for idx in stops_active:
                info = stops_by_idx[idx]
                for t_info in info.get("trips", []):
                    if t_info["trip_id"] == tid:
                        gain_m   = t_info["access_gain_m"]
                        gain_min = gain_m / (config.WALK_SPEED_MPS * 60)
                        gain_walk += (beta["walk"] * gain_min
                                      - beta["ride"] * t_info["dwell_pen_min"])

            # Guadagno speed (riduce T_ride)
            gain_ride = 0.0
            for idx in speed_active:
                info = speed_by_idx[idx]
                if tid in info.get("trip_ids", []):
                    gain_ride += beta["ride"] * info["delta_ride_min"]

            total_gain = gain_wait + gain_trans + gain_walk + gain_ride
            dD_new     = max(dD - total_gain, -20.0)

            prop_old = sigmoid(dD)
            prop_new = min(sigmoid(dD_new), P_MAX)
            delta_prop_sum += (prop_new - prop_old)
            n_valid += 1

        # ΔProp media del corridoio × domanda = utenti convertiti in questo corridoio
        if n_valid > 0:
            dp_mean = delta_prop_sum / n_valid
        else:
            dp_mean = 0.0

        users_c = dp_mean * dem       # utenti convertiti in questo corridoio
        users_converted += users_c

        per_corridor[cid] = {
            "prop_base":       prop_base_by_corr.get(cid, 0.0),
            "prop_new":        min(prop_base_by_corr.get(cid, 0.0) + dp_mean, P_MAX),
            "delta_prop":      dp_mean,
            "users_converted": users_c,
            "demand":          dem,
        }

    # ── Vincoli ───────────────────────────────────────────
    viol = 0.0
    if total_cost   > config.BUDGET_TOTAL_EUR: viol += total_cost - config.BUDGET_TOTAL_EUR
    if capex_cost   > config.BUDGET_CAPEX_EUR: viol += capex_cost - config.BUDGET_CAPEX_EUR
    if opex_cost    > config.BUDGET_OPEX_EUR:  viol += opex_cost  - config.BUDGET_OPEX_EUR
    if n_veh_added  > config.K_FLEET_MAX:      viol += (n_veh_added  - config.K_FLEET_MAX)  * 1e5
    if n_sync_routes > config.K_SYNC_MAX:      viol += (n_sync_routes - config.K_SYNC_MAX)  * 1e5

    # Fitness = utenti convertiti − penalità (in unità comparabili a users)
    # Normalizzazione penalità: €1 violazione ≈ 1/SOCIAL_VALUE utenti persi
    penalty_users = viol / config.SOCIAL_VALUE_PER_USER_EUR
    fitness = users_converted - penalty_users

    costs = {"freq": freq_cost, "sync": sync_cost,
             "stops": stops_cost, "speed": speed_cost, "total": total_cost}

    return fitness, users_converted, total_cost, viol, per_corridor, costs

# ─────────────────────────────────────────────────────────────────
# Repair: unicità per route (freq e sync)
# ─────────────────────────────────────────────────────────────────
def repair(y):
    y = y.copy()
    # Freq: tieni solo il candidato con delta_wait massimo per route
    for route, cands in freq_by_route.items():
        active = [(idx, dw) for (idx, dw, *_) in cands if y[idx] == 1]
        if len(active) > 1:
            best = max(active, key=lambda x: x[1])[0]
            for idx, _ in active:
                if idx != best: y[idx] = 0
    # Sync: un solo offset per route
    for route, cands in sync_by_route.items():
        active = [idx for (idx, *_) in cands if y[idx] == 1]
        for idx in active[1:]:
            y[idx] = 0
    return y

# ─────────────────────────────────────────────────────────────────
# Operatori GA
# ─────────────────────────────────────────────────────────────────
def random_individual():
    y = np.zeros(N, dtype=int)
    for i in range(N):
        if random.random() < 2.0 / N:
            y[i] = 1
    return repair(y)

def greedy_seed():
    """Attiva il candidato freq più conveniente (minor costo, maggior gain) per route."""
    y = np.zeros(N, dtype=int)
    for route, cands in freq_by_route.items():
        if cands:
            best = max(cands, key=lambda x: x[1] / max(x[2], 1))   # Δwait/cost
            y[best[0]] = 1
    return repair(y)

def tournament(pop, fits, k=config.GA_TOURNEY_K):
    idxs = random.sample(range(len(pop)), k)
    return pop[max(idxs, key=lambda i: fits[i])].copy()

def crossover(p1, p2):
    if random.random() > config.GA_PC:
        return p1.copy(), p2.copy()
    mask = np.random.randint(0, 2, N).astype(bool)
    return repair(np.where(mask, p1, p2)), repair(np.where(mask, p2, p1))

def mutate(y):
    pm    = 1.0 / N
    child = y.copy()
    for i in range(N):
        if random.random() < pm:
            child[i] ^= 1
    return repair(child)

# ─────────────────────────────────────────────────────────────────
# Esecuzione GA
# ─────────────────────────────────────────────────────────────────
print(f"\n  Budget: €{config.BUDGET_TOTAL_EUR:,.0f}  "
      f"(CAPEX €{config.BUDGET_CAPEX_EUR:,.0f} | OPEX €{config.BUDGET_OPEX_EUR:,.0f})")
print(f"  Valore sociale: €{config.SOCIAL_VALUE_PER_USER_EUR:,.0f}/utente convertito")
print(f"  GA: pop={config.GA_POP_SIZE}, gen_max={config.GA_N_GEN}, "
      f"k_sync_max={config.K_SYNC_MAX}, k_fleet_max={config.K_FLEET_MAX}\n")

population = [random_individual() for _ in range(config.GA_POP_SIZE - 2)]
population += [greedy_seed(), np.zeros(N, dtype=int)]
fitnesses  =  [evaluate(y)[0] for y in population]

best_fit = max(fitnesses)
best_y   = population[int(np.argmax(fitnesses))].copy()
history  = [best_fit]
stag     = 0

print(f"  Gen   0 | best_users={best_fit:.3f}")

t0 = time.time()
gen = 0
for gen in range(1, config.GA_N_GEN + 1):
    new_pop = []
    while len(new_pop) < config.GA_POP_SIZE:
        c1, c2 = crossover(tournament(population, fitnesses),
                           tournament(population, fitnesses))
        new_pop += [mutate(c1), mutate(c2)]
    population = new_pop[:config.GA_POP_SIZE]
    fitnesses  = [evaluate(y)[0] for y in population]

    gen_best = max(fitnesses)
    if gen_best > best_fit + 1e-9:
        best_fit = gen_best
        best_y   = population[int(np.argmax(fitnesses))].copy()
        stag = 0
    else:
        stag += 1

    history.append(best_fit)

    if gen % 50 == 0:
        print(f"  Gen {gen:>3} | best_users={best_fit:.3f} | stag={stag} | "
              f"t={time.time()-t0:.1f}s")
    if stag >= config.GA_STAGNATION:
        print(f"  Convergenza gen {gen}  (stagnazione {stag})")
        break

t_tot = time.time() - t0

# ─────────────────────────────────────────────────────────────────
# Valutazione soluzione ottimale
# ─────────────────────────────────────────────────────────────────
_, users_conv, cost_tot, viol, per_corr, costs_bk = evaluate(best_y)

cpu = cost_tot / users_conv if users_conv > 1e-9 else float("inf")  # €/utente

activated = [(CAND_ORDER[i], i) for i in range(N) if best_y[i] == 1]
lever_of   = lambda i: ("freq"  if i < N_FREQ else
                        "sync"  if i < N_FREQ + N_SYNC else
                        "stops" if i < N_FREQ + N_SYNC + N_STOPS else "speed")

print(f"\n{'='*60}")
print(f"  SOLUZIONE OTTIMALE")
print(f"{'='*60}")
print(f"  Utenti convertiti:   {users_conv:.2f}")
print(f"  Costo totale:        €{cost_tot:,.0f}")
print(f"  Costo/utente:        €{cpu:,.0f}/utente")
print(f"  Valore sociale netto:€{users_conv*config.SOCIAL_VALUE_PER_USER_EUR - cost_tot:,.0f}")
print(f"  Break-down costi:    freq=€{costs_bk['freq']:,.0f}  "
      f"sync=€{costs_bk['sync']:,.0f}  "
      f"stops=€{costs_bk['stops']:,.0f}  "
      f"speed=€{costs_bk['speed']:,.0f}")
print(f"\n  Interventi attivati ({len(activated)}):")
for cid, idx in activated:
    print(f"    [{lever_of(idx)}]  {cid}")

# ─────────────────────────────────────────────────────────────────
# Risultati per corridoio
# ─────────────────────────────────────────────────────────────────
print(f"\n  Tabella IV — Risultati per corridoio:")
hdr = f"  {'C':>3} {'Ora':>12} {'Dem':>4} {'H':>5}  "     \
      f"{'Prop_0':>6} {'Prop_f':>6} {'ΔProp':>6} {'Users_c':>7} {'Leve'}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

rows = []
levers_used_global = set()
for cand_id, idx in activated:
    levers_used_global.add(lever_of(idx))

for _, crow in top.iterrows():
    cid  = crow["corridor_id"]
    rank = int(crow["rank"])
    pc   = per_corr.get(cid, {})
    dem  = int(crow["demand"])

    levers_activated = set()
    for cand_id, idx in activated:
        lv = lever_of(idx)
        cand_row = (cands_freq if lv == "freq" else
                    cands_sync if lv == "sync" else
                    cands_stops if lv == "stops" else cands_speed)
        if len(cand_row) > 0:
            cand_info = cand_row[cand_row["candidate_id"] == cand_id]
            if len(cand_info) > 0:
                r = cand_info.iloc[0].get("route", "")
                # Verifica se la route è usata in questo corridoio
                from_this = tic[tic["corridor_id"] == cid]["trip_id"].tolist()
                if any(t in trip_to_corr and trip_to_corr[t] == cid
                       for t in from_this):
                    levers_activated.add(f"y^{lv}")

    lev_str = "+".join(sorted(levers_activated)) if levers_activated else "—"
    p0  = pc.get("prop_base", 0.0)
    pf  = pc.get("prop_new",  0.0)
    dp  = pc.get("delta_prop", 0.0)
    uc  = pc.get("users_converted", 0.0)

    print(f"  C{rank:>2} {crow['time_range']:>12} {dem:>4} {crow['h_index']:>5.2f}  "
          f"{p0:>6.3f} {pf:>6.3f} {dp:>+6.3f} {uc:>7.2f}")

    rows.append({
        "rank":            rank,
        "corridor_id":     cid,
        "time_range":      crow["time_range"],
        "demand":          dem,
        "h_index":         round(float(crow["h_index"]), 2),
        "avg_delta_d":     round(float(crow["avg_delta_d"]), 2),
        "prop_initial":    round(p0, 4),
        "prop_final":      round(pf, 4),
        "delta_prop":      round(dp, 4),
        "users_converted": round(uc, 3),
        "dominant_phase":  crow["dominant_phase"],
        "levers_activated": lev_str,
        "cost_contribution_eur": round(cost_tot / max(len(activated), 1), 0),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(config.OUTPUT_DIR, "ga_results_corridors.csv"), index=False)

# ─────────────────────────────────────────────────────────────────
# Salva soluzione + convergenza
# ─────────────────────────────────────────────────────────────────
solution = {
    "users_converted":   users_conv,
    "total_cost_eur":    cost_tot,
    "cost_per_user_eur": cpu,
    "social_value_net_eur": users_conv * config.SOCIAL_VALUE_PER_USER_EUR - cost_tot,
    "costs_breakdown":   costs_bk,
    "violations":        viol,
    "n_generations":     gen,
    "t_seconds":         round(t_tot, 1),
    "activated_candidates": [CAND_ORDER[i] for i in range(N) if best_y[i] == 1],
    "y": best_y.tolist(),
    "sigmoid_params": {"p_max": P_MAX, "g": G_SIG, "d0_flex": D0_FLEX},
}
with open(os.path.join(config.OUTPUT_DIR, "ga_best_solution.json"), "w") as f:
    json.dump(solution, f, indent=2)

pd.DataFrame({"generation": range(len(history)), "best_fitness": history}).to_csv(
    os.path.join(config.OUTPUT_DIR, "ga_convergence.csv"), index=False)

# ─────────────────────────────────────────────────────────────────
# Grafici F e G
# ─────────────────────────────────────────────────────────────────
COLORS = {"freq": "#E74C3C", "sync": "#9B59B6", "stops": "#27AE60", "speed": "#2980B9"}
C_BASE = "#BDC3C7";  C_NEW = "#27AE60"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── F: Convergenza GA ────────────────────────────────────
ax = axes[0]
ax.plot(history, color="#2980B9", lw=2, zorder=3)
ax.fill_between(range(len(history)), min(history), history, alpha=0.15, color="#2980B9")
ax.set_xlabel("Generazione", fontsize=11)
ax.set_ylabel("Utenti convertiti (best fitness)", fontsize=11)
ax.set_title("F — Convergenza GA", fontsize=12, fontweight="bold")
ax.axhline(best_fit, color="#E74C3C", ls="--", lw=1, label=f"Ottimo: {best_fit:.2f} utenti")
ax.legend(fontsize=9)

# ── G: Propensità prima/dopo per corridoio ───────────────
ax = axes[1]
corr_labels = [f"C{r['rank']}\n{r['time_range']}" for r in rows]
p_init  = [r["prop_initial"]    for r in rows]
p_final = [r["prop_final"]      for r in rows]
uc_vals = [r["users_converted"] for r in rows]
x = np.arange(len(rows));  w = 0.35

b1 = ax.bar(x - w/2, p_init,  w, color=C_BASE, alpha=0.85, label="Baseline",     zorder=3)
b2 = ax.bar(x + w/2, p_final, w, color=C_NEW,  alpha=0.85, label="Post-intervento", zorder=3)

for bar, val in zip(b1, p_init):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#555")
for bar, uc, val in zip(b2, uc_vals, p_final):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    if uc > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.055,
                f"+{uc:.1f}u", ha="center", va="bottom", fontsize=6, color="#27AE60")

ax.set_xticks(x)
ax.set_xticklabels(corr_labels, fontsize=7.5)
ax.set_ylabel("Propensione al PT  Prop(ΔD)", fontsize=11)
ax.set_ylim(0, min(P_MAX * 1.2, 1.15))
ax.axhline(P_MAX, color="grey", ls=":", lw=1, alpha=0.7, label=f"P_max={P_MAX:.2f}")
ax.set_title("G — Propensità: baseline vs post-intervento\n"
             f"(Costo totale €{cost_tot:,.0f} | "
             f"€{cpu:,.0f}/utente | "
             f"{users_conv:.1f} utenti convertiti)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "fig_FG_ga_results.png"),
            bbox_inches="tight", dpi=150)
plt.close(fig)

print(f"\n  Salvato: {config.OUTPUT_DIR}/ga_results_corridors.csv")
print(f"  Salvato: {config.OUTPUT_DIR}/ga_best_solution.json")
print(f"  Salvato: {config.OUTPUT_DIR}/fig_FG_ga_results.png")
