"""
step8_results.py  —  Analisi costo-efficienza e grafici paper-ready
====================================================================

Output:
  output_v2/cost_efficiency_by_corridor.csv
  output_v2/fig_H_cost_efficiency.png      — scatter bubble paper-ready
  output_v2/fig_I_table4_heatmap.png       — heatmap Tabella IV
  (stampa LaTeX snippet Tabella IV)
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import config

print("=" * 60)
print("STEP 8  —  Analisi costo-efficienza (paper-ready)")
print("=" * 60)

CDIR = config.CANDIDATES_DIR

# ── Carica dati ───────────────────────────────────────────
with open(os.path.join(CDIR, "ga_input_summary.json")) as f: meta = json.load(f)
with open(os.path.join(CDIR, "lookup_freq.json"))       as f: lk_freq  = json.load(f)
with open(os.path.join(CDIR, "lookup_stops.json"))      as f: lk_stops = json.load(f)
with open(os.path.join(CDIR, "lookup_speed.json"))      as f: lk_speed = json.load(f)
with open(os.path.join(CDIR, "lookup_sync.json"))       as f: lk_sync  = json.load(f)
with open(os.path.join(config.OUTPUT_DIR, "ga_best_solution.json")) as f: sol = json.load(f)

sig   = meta["sigmoid_params"]
P_MAX = sig["p_max"];  G_SIG = sig["g"];  D0_FLEX = sig["d0_flex"]

def sigmoid(d):
    try:    return P_MAX / (1.0 + math.exp(G_SIG*(d - D0_FLEX)))
    except: return 0.0 if G_SIG*(d - D0_FLEX) > 0 else P_MAX

top     = pd.read_csv(os.path.join(config.OUTPUT_DIR, "priority_corridors.csv"))
tic     = pd.read_csv(os.path.join(CDIR, "trips_in_corridors.csv"))
dd      = pd.read_csv(os.path.join(config.OUTPUT_DIR, "delta_d_per_trip.csv"))
res_ga  = pd.read_csv(os.path.join(config.OUTPUT_DIR, "ga_results_corridors.csv"))
tlegs   = pd.read_csv(os.path.join(CDIR, "transfer_legs.csv")) \
          if os.path.exists(os.path.join(CDIR, "transfer_legs.csv")) else pd.DataFrame()

cands_freq  = pd.read_csv(os.path.join(CDIR, "candidates_freq.csv"))
cands_sync  = pd.read_csv(os.path.join(CDIR, "candidates_sync.csv"))  \
              if os.path.exists(os.path.join(CDIR, "candidates_sync.csv"))  else pd.DataFrame()
cands_stops = pd.read_csv(os.path.join(CDIR, "candidates_stops.csv")) \
              if os.path.exists(os.path.join(CDIR, "candidates_stops.csv")) else pd.DataFrame()
cands_speed = pd.read_csv(os.path.join(CDIR, "candidates_speed.csv")) \
              if os.path.exists(os.path.join(CDIR, "candidates_speed.csv")) else pd.DataFrame()

dd_pv = dd[dd["is_pv"]].set_index("trip_id")[["delta_D","T_walk_min","T_wait_min","T_ride_min","T_trans_min"]].to_dict("index")

# ─────────────────────────────────────────────────────────
# Funzione: valuta UN singolo candidato su UN singolo corridoio
# Ritorna: (users_converted, cost_eur, delta_prop_mean)
# ─────────────────────────────────────────────────────────
def eval_single_candidate(cid_corr, cand_id, lever):
    """Valuta l'intervento isolato su un corridoio."""
    beta = config.BETA
    tids = tic[tic["corridor_id"] == cid_corr]["trip_id"].tolist()
    dem  = top[top["corridor_id"] == cid_corr]["demand"].values[0]

    delta_prop_sum = 0.0
    n_valid = 0

    for tid in tids:
        if tid not in dd_pv:
            continue
        dD   = dd_pv[tid]["delta_D"]
        gain = 0.0

        if lever == "freq" and cand_id in lk_freq:
            info = lk_freq[cand_id]
            gain = beta["wait"] * info["delta_wait_min"]

        elif lever == "sync":
            for _, leg in tlegs[tlegs["trip_id"] == tid].iterrows() if len(tlegs) > 0 else []:
                leg_lu = lk_sync.get(leg["leg_id"], {})
                # Usa l'offset del candidato sul feeder route
                row = cands_sync[cands_sync["candidate_id"] == cand_id]
                if len(row) == 0: continue
                off_val = float(row.iloc[0]["offset_min"])
                r_dep   = row.iloc[0]["route"]
                key_0   = f"+0.0|{off_val:+.1f}"   # feeder invariato, connector shiftato
                bw      = leg["baseline_wait_min"]
                nw      = leg_lu.get("offsets", {}).get(key_0, bw)
                gain   += beta["trans"] * (bw - nw)

        elif lever == "stops" and cand_id in lk_stops:
            info = lk_stops[cand_id]
            for t_info in info.get("trips", []):
                if t_info["trip_id"] == tid:
                    gm    = t_info["access_gain_m"]
                    g_min = gm / (config.WALK_SPEED_MPS * 60)
                    gain  = (beta["walk"] * g_min
                             - beta["ride"] * t_info["dwell_pen_min"])

        elif lever == "speed" and cand_id in lk_speed:
            info = lk_speed[cand_id]
            if tid in info.get("trip_ids", []):
                gain = beta["ride"] * info["delta_ride_min"]

        dD_new = max(dD - gain, -20.0)
        delta_prop_sum += sigmoid(dD_new) - sigmoid(dD)
        n_valid += 1

    dp_mean = delta_prop_sum / n_valid if n_valid > 0 else 0.0
    uc      = dp_mean * dem

    # Costo del candidato
    if lever == "freq":
        cost = lk_freq.get(cand_id, {}).get("cost_eur", 0)
    elif lever == "sync":
        row  = cands_sync[cands_sync["candidate_id"] == cand_id]
        cost = float(row.iloc[0]["cost_eur"]) if len(row) > 0 else 0
    elif lever == "stops":
        cost = lk_stops.get(cand_id, {}).get("cost_eur", config.C_STOP_CAPEX_EUR)
    elif lever == "speed":
        cost = lk_speed.get(cand_id, {}).get("cost_eur", 0)
    else:
        cost = 0.0

    return uc, cost, dp_mean

# ─────────────────────────────────────────────────────────
# Per ogni corridoio: trova il BEST candidato per ogni leva
# ─────────────────────────────────────────────────────────
print("\n  Analisi per-corridoio per-leva (best candidate isolato)...")

ce_rows = []   # cost-efficiency rows

for _, crow in top.iterrows():
    cid  = crow["corridor_id"]
    rank = int(crow["rank"])
    dem  = int(crow["demand"])
    h_ix = float(crow["h_index"])
    p_b  = float(res_ga[res_ga["rank"]==rank]["prop_initial"].values[0])

    for lever, cands_df in [("freq",  cands_freq),
                             ("sync",  cands_sync),
                             ("stops", cands_stops),
                             ("speed", cands_speed)]:
        if len(cands_df) == 0:
            continue

        best_uc   = 0.0
        best_cpu  = float("inf")
        best_cid  = None
        best_cost = 0.0
        best_dp   = 0.0

        for cand_id in cands_df["candidate_id"].tolist():
            uc, cost, dp = eval_single_candidate(cid, cand_id, lever)
            if uc > 1e-6 and cost > 0:
                cpu_cand = cost / uc
                if cpu_cand < best_cpu:
                    best_cpu  = cpu_cand
                    best_uc   = uc
                    best_cost = cost
                    best_cid  = cand_id
                    best_dp   = dp

        if best_cid is not None:
            ce_rows.append({
                "rank":            rank,
                "corridor_id":     cid,
                "time_range":      crow["time_range"],
                "demand":          dem,
                "h_index":         round(h_ix, 2),
                "prop_baseline":   round(p_b, 4),
                "lever":           lever,
                "best_candidate":  best_cid,
                "users_converted": round(best_uc, 3),
                "cost_eur":        round(best_cost, 0),
                "cost_per_user_eur": round(best_cpu, 0),
                "delta_prop":      round(best_dp, 4),
                "roi_social":      round(best_uc * config.SOCIAL_VALUE_PER_USER_EUR - best_cost, 0),
            })

ce_df = pd.DataFrame(ce_rows)
ce_df.to_csv(os.path.join(config.OUTPUT_DIR, "cost_efficiency_by_corridor.csv"), index=False)

print(f"  Righe analisi costo-efficienza: {len(ce_df)}")
print()
print(f"  {'C':>3} {'Leva':>6} {'Dem':>4} {'ΔProp':>7} "
      f"{'Users':>6} {'Costo €':>9} {'€/utente':>9} {'ROI soc. €':>11}")
print("  " + "-"*65)
for _, r in ce_df.sort_values(["rank","lever"]).iterrows():
    print(f"  C{int(r["rank"]):>2} {r["lever"]:>6} {int(r["demand"]):>4} "
          f"{r["delta_prop"]:>+7.3f} {r["users_converted"]:>6.2f} "
          f"{r["cost_eur"]:>9,.0f} {r["cost_per_user_eur"]:>9,.0f} "
          f"{r["roi_social"]:>+11,.0f}")

# ─────────────────────────────────────────────────────────
# FIGURA H — Cost-Efficiency Bubble Chart (paper-ready)
# ─────────────────────────────────────────────────────────
LEVER_COLORS = {"freq": "#E74C3C", "sync": "#9B59B6",
                "stops": "#27AE60", "speed": "#2980B9"}
LEVER_LABELS = {"freq":  r"$y^{freq}$ — Frequenza",
                "sync":  r"$y^{sync}$ — Sincronizzazione",
                "stops": r"$y^{stops}$ — Nuove fermate",
                "speed": r"$y^{speed}$ — Velocità commerciale"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ── H1: Scatter bubble — utenti convertiti vs costo ──────
for lever, color in LEVER_COLORS.items():
    sub = ce_df[ce_df["lever"] == lever]
    if len(sub) == 0:
        continue
    sc = ax1.scatter(
        sub["cost_eur"] / 1000,
        sub["users_converted"],
        s=sub["demand"] * 30,         # area bolla proporzionale alla domanda
        c=color, alpha=0.75, edgecolors="white", linewidths=0.8,
        label=LEVER_LABELS[lever], zorder=4
    )
    # Etichetta corridoio
    for _, r in sub.iterrows():
        ax1.annotate(f"C{int(r['rank'])}",
                     (r["cost_eur"]/1000, r["users_converted"]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7.5, color=color, fontweight="bold")

# Iso-CPu lines (€/utente costante)
x_range = np.linspace(0.1, ce_df["cost_eur"].max()/1000 * 1.1, 200)
for cpu_line in [5000, 20000, 50000, 100000]:
    y_line = x_range * 1000 / cpu_line
    ax1.plot(x_range, y_line, "k--", lw=0.7, alpha=0.3)
    idx = len(x_range)//2
    ax1.text(x_range[idx], y_line[idx]+0.02,
             f"€{cpu_line//1000}k/ut.", fontsize=6.5, color="grey", alpha=0.7)

ax1.set_xlabel("Costo intervento  [k€]", fontsize=11)
ax1.set_ylabel("Utenti convertiti  [n]", fontsize=11)
ax1.set_title("H1 — Costo-efficienza per corridoio e leva\n"
              "(dimensione bolla ∝ domanda del corridoio)",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)

# ── H2: Heatmap €/utente per (corridoio × leva) ──────────
levers_order = ["freq", "sync", "stops", "speed"]
corr_order   = [f"C{r}" for r in sorted(top["rank"].astype(int))]
matrix       = np.full((len(corr_order), len(levers_order)), np.nan)
roi_matrix   = np.full((len(corr_order), len(levers_order)), np.nan)

for _, r in ce_df.iterrows():
    ci = corr_order.index(f"C{int(r['rank'])}")
    li = levers_order.index(r["lever"])
    matrix[ci, li]    = r["cost_per_user_eur"]
    roi_matrix[ci, li] = r["roi_social"]

# Maschera celle senza dati
masked = np.ma.array(matrix, mask=np.isnan(matrix))

# Normalizza: verde = basso €/utente (efficiente), rosso = alto
cmap  = matplotlib.cm.RdYlGn_r
vmax  = np.nanpercentile(matrix[~np.isnan(matrix)], 90) if not np.all(np.isnan(matrix)) else 1
im = ax2.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

ax2.set_xticks(range(len(levers_order)))
ax2.set_xticklabels(
    [r"$y^{freq}$", r"$y^{sync}$", r"$y^{stops}$", r"$y^{speed}$"],
    fontsize=10)
ax2.set_yticks(range(len(corr_order)))
ax2.set_yticklabels(
    [f"{l}\n{top[top['rank']==int(l[1:])]['time_range'].values[0]}"
     for l in corr_order],
    fontsize=8)

for ci in range(len(corr_order)):
    for li in range(len(levers_order)):
        val = matrix[ci, li]
        if not np.isnan(val):
            roi = roi_matrix[ci, li]
            roi_str = f"\n(ROI €{roi/1000:+.0f}k)" if not np.isnan(roi) else ""
            ax2.text(li, ci, f"€{val/1000:.0f}k{roi_str}",
                     ha="center", va="center", fontsize=7,
                     color="black" if val < vmax*0.6 else "white",
                     fontweight="bold")
        else:
            ax2.text(li, ci, "n.d.", ha="center", va="center",
                     fontsize=7, color="#BBBBBB")

cbar = plt.colorbar(im, ax=ax2, shrink=0.85)
cbar.set_label("€ per utente convertito", fontsize=9)
ax2.set_title("H2 — Costo per utente convertito  [€/utente]\n"
              "(verde = più efficiente, rosso = meno efficiente)",
              fontsize=10, fontweight="bold")

fig.suptitle("Analisi costo-efficienza degli interventi PT  —  Cagliari",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "fig_H_cost_efficiency.png"),
            bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\n  Salvato: fig_H_cost_efficiency.png")

# ─────────────────────────────────────────────────────────
# FIGURA I — Tabella IV heatmap (style paper)
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis("off")

ga_res = res_ga.sort_values("rank")
col_labels = ["Corridoio", "Ora", "Dem.", "H-idx",
              "ΔD medio", "Prop₀", "Propf",
              "ΔProp", "Utenti conv.", "Fase dom.", "Leve"]

table_data = []
for _, r in ga_res.iterrows():
    table_data.append([
        f"C{int(r['rank'])}: {r['corridor_id'].split('→')[0]}→{r['corridor_id'].split('→')[1].split(' ')[0]}",
        r["time_range"],
        str(int(r["demand"])),
        f"{r['h_index']:.2f}",
        f"{r['avg_delta_d']:.1f}",
        f"{r['prop_initial']:.3f}",
        f"{r['prop_final']:.3f}",
        f"{r['delta_prop']:+.3f}",
        f"{r['users_converted']:.2f}",
        r["dominant_phase"],
        r.get("levers_activated", "—"),
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

# Colorazione header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2C3E50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Colorazione righe alternate + ΔProp
for i, (_, r) in enumerate(ga_res.iterrows(), 1):
    bg = "#F8F9FA" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(bg)
    dp = r["delta_prop"]
    if dp > 0.4:
        tbl[(i, 7)].set_facecolor("#D5F5E3")
    elif dp < 0.05:
        tbl[(i, 7)].set_facecolor("#FADBD8")

ax.set_title("Tabella IV — Impatto degli interventi ottimizzati sui corridoi prioritari\n"
             f"(Budget €{sol['total_cost_eur']:,.0f} | "
             f"€{sol['cost_per_user_eur']:,.0f}/utente | "
             f"{sol['users_converted']:.1f} utenti convertiti totali)",
             fontsize=10, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "fig_I_table4.png"),
            bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Salvato: fig_I_table4.png")

# ─────────────────────────────────────────────────────────
# Snippet LaTeX — Tabella IV
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SNIPPET LaTeX — Tabella IV")
print("="*60)
latex = r"""\begin{table}[!t]
\caption{Impact of the Optimized Interventions on Priority Corridors}
\label{tab:results}
\centering
\small
\begin{tabular}{llccccccl}
\toprule
\textbf{Corridor} & \textbf{Time} & \textbf{Dem.} & \textbf{H} &
\textbf{$\Delta D$} & \textbf{$\text{Prop}_0$} & \textbf{$\text{Prop}_f$} &
\textbf{$\Delta\text{Prop}$} & \textbf{Lever} \\
\midrule
"""
for _, r in ga_res.iterrows():
    oc = r["corridor_id"].split("→")[0]
    dc = r["corridor_id"].split("→")[1].split(" ")[0]
    lev = r.get("levers_activated","—").replace("y^","$y^{").replace("+","$+$y^{")
    latex += (f"C{int(r['rank'])}: {oc}$\\rightarrow${dc} & "
              f"{r['time_range']} & "
              f"{int(r['demand'])} & "
              f"{r['h_index']:.2f} & "
              f"{r['avg_delta_d']:.1f} & "
              f"{r['prop_initial']:.3f} & "
              f"{r['prop_final']:.3f} & "
              f"{r['delta_prop']:+.3f} & "
              f"{r['dominant_phase']} \\\\\n")

latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item """ + (f"Total cost: \\euro{sol['total_cost_eur']:,.0f}. "
             f"Cost per converted user: \\euro{sol['cost_per_user_eur']:,.0f}. "
             f"Social value net of costs: \\euro{sol['social_value_net_eur']:,.0f}. "
             f"$P_{{max}}={sol['sigmoid_params']['p_max']:.3f}$, "
             f"$g={sol['sigmoid_params']['g']:.4f}$, "
             f"$\\Delta D_{{flex}}={sol['sigmoid_params']['d0_flex']:.1f}$.") + r"""
\end{tablenotes}
\end{table}"""

print(latex)
print()
print(f"  Tutti i file salvati in: {config.OUTPUT_DIR}/")
