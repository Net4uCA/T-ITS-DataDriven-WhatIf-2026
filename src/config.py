"""
config.py — Parametri di configurazione del framework What-If v2
Allineato con il paper v2: "Bridging the Public-Private Discomfort Gap"
"""

# ---------------------------------------------------------------------------
# File di input (percorsi relativi rispetto alla directory del progetto)
# ---------------------------------------------------------------------------
FILE_TRIPS   = "spostamenti_combinati_solo_auto_autopasseggero_autobus_moto_motopasseggero.csv"
FILE_ALT     = "alternative_otp_car_carpasseggero_moto_motopasseggero_bus_filtered_completo.csv"
FILE_OFFSETS = "time_offsets_auto_autobus_moto_ecc.csv"

# ---------------------------------------------------------------------------
# Classificazione delle modalità
# ---------------------------------------------------------------------------
MODES_PV = {"Auto", "Auto (passeggero)", "Moto", "Moto (passeggero)"}   # veicolo privato
MODES_PT = {"Autobus"}                                                     # trasporto pubblico

# ---------------------------------------------------------------------------
# Bounding box area di studio (Cagliari metropolitana)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 39.15, 39.30
LON_MIN, LON_MAX =  8.95,  9.20

# ---------------------------------------------------------------------------
# Value of Time — coefficienti β (Eq. 2, paper v2)
# Unità: adimensionale rispetto al tempo a bordo (β^ride = 1.0 baseline)
# ---------------------------------------------------------------------------
BETA = {
    "walk":  2.00,   # camminare pesa 2× rispetto a stare seduti
    "wait":  2.50,   # aspettare alla fermata pesa 2.5×
    "ride":  1.00,   # tempo a bordo — baseline
    "trans": 1.75,   # attesa al trasbordo pesa 1.75×
}

# ---------------------------------------------------------------------------
# Parametri di parsing e filtering
# ---------------------------------------------------------------------------
TRANSFER_WAIT_CAP_S = 1800     # 30 min — cap per i transfer wait outlier (sec)
WALK_SPEED_MPS      = 1.39     # ~5 km/h — velocità pedonale media (m/s)
                                # usata SOLO per eventuale cross-check distanze

# ---------------------------------------------------------------------------
# Spazio-temporale (corridoi, Eq. 6-7)
# ---------------------------------------------------------------------------
CELL_SIZE_M   = 500            # lato cella griglia (metri)
TIME_SLOT_MIN = 10             # durata slot temporale (minuti)
MIN_DEMAND    = 3              # domanda minima per corridoio candidato: num. spostamenti

# ---------------------------------------------------------------------------
# Sigmoid — valori iniziali e bounds per curve_fit (Eq. 6)
# ---------------------------------------------------------------------------
SIGMOID_P0     = [1.0, 0.05, 20.0]      # [P_max, g, D0_flex]
SIGMOID_BOUNDS = ([0.5, 0.001, -50],    # lower bounds
                  [1.0, 10.0,   200])   # upper bounds
N_BINS_SIGMOID = 30                     # numero di bin per la calibrazione

# ---------------------------------------------------------------------------
# Corridoi prioritari
# ---------------------------------------------------------------------------
N_PRIORITY = 10    # top-N corridoi selezionati per l'analisi what-if

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output_v2"
