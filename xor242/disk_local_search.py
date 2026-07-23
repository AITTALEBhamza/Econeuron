"""
disk_local_search.py
====================
Equivalent Python du script Cadence cadence_sa_direct.py
Utilise les tables de caracterisation pour simuler le reseau
sans lancer Spectre — beaucoup plus rapide.

Architecture : 2 -> 8 -> 8 -> 1
Probleme     : classification disque r=0.5
Algorithme   : local search synapse par synapse (comme l'original)

Contraintes respectees :
- INH limites a W=0.700 max (comme dans Cadence)
- Memes candidats EXC/INH que le script Cadence
- Meme topologie, memes cas d'entrainement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 0. CHARGEMENT DES TABLES DE CARACTERISATION
# =========================================================

def read_cadence_rows(csv_path):
    df = pd.read_csv(csv_path)
    data_cols = [c for c in df.columns if c not in ['Parameter', 'Unnamed: 1']]
    def get_row(row_name):
        row = df.loc[df['Parameter'] == row_name, data_cols]
        if row.empty:
            raise ValueError("'{}' not found in {}".format(row_name, csv_path))
        return pd.to_numeric(row.iloc[0], errors='coerce').to_numpy(dtype=float)
    return get_row

inp = read_cadence_rows('input_min_conso.csv')
INPUT_V_AXIS = inp('Vex')
INPUT_F_AXIS = inp('Fspikes')

neu = read_cadence_rows('neuron.csv')
NEURON_I_AXIS = neu('Iex')
NEURON_F_AXIS = neu('Fspikes')

exc = read_cadence_rows('excitation20.csv')
EXC_W2    = exc('W2')
EXC_FPRE  = exc('Fpre')
EXC_IPOST = exc('Ipost')

inh = read_cadence_rows('inhibition20.csv')
INH_W2    = inh('W2')
INH_FPRE  = inh('Fpre')
INH_IPOST = inh('Ipost')

print("Tables chargees:")
print("  Input  V: [{:.0f}, {:.0f}] mV -> F: [{:.1f}, {:.1f}] kHz".format(
    INPUT_V_AXIS[0]*1e3, INPUT_V_AXIS[-1]*1e3,
    INPUT_F_AXIS[0]/1e3, INPUT_F_AXIS[-1]/1e3))
print("  Neuron I: [{:.0f}, {:.0f}] pA -> F: [{:.1f}, {:.1f}] kHz".format(
    NEURON_I_AXIS[0]*1e12, NEURON_I_AXIS[-1]*1e12,
    NEURON_F_AXIS[0]/1e3, NEURON_F_AXIS[-1]/1e3))
print("  EXC W2: [{:.3f}, {:.3f}] um  Fpre: [{:.0f}, {:.0f}] Hz".format(
    EXC_W2.min(), EXC_W2.max(),
    EXC_FPRE.min(), EXC_FPRE.max()))
print("  INH W2: [{:.3f}, {:.3f}] um  Fpre: [{:.0f}, {:.0f}] Hz".format(
    INH_W2.min(), INH_W2.max(),
    INH_FPRE.min(), INH_FPRE.max()))

# =========================================================
# 1. INTERPOLATION
# =========================================================

def interp1(x, xp, fp):
    """Interpolation 1D clampee."""
    return float(np.interp(x, xp, fp))


def build_grid(W_all, F_all, Y_all):
    """Construit grille 2D (W, F) -> Y pour interpolation."""
    Wu = np.unique(W_all)
    Fu = np.unique(F_all)
    g  = np.full((len(Wu), len(Fu)), np.nan)
    for i, w in enumerate(Wu):
        for j, f in enumerate(Fu):
            mask = np.isclose(W_all, w) & np.isclose(F_all, f)
            if mask.any():
                g[i, j] = np.mean(Y_all[mask])
    return Wu, Fu, g


def interp2(w, f, Wa, Fa, G):
    """Interpolation bilineaire 2D."""
    w = np.clip(w, Wa[0], Wa[-1])
    f = np.clip(f, Fa[0], Fa[-1])
    wi = np.searchsorted(Wa, w, side='right') - 1
    fi = np.searchsorted(Fa, f, side='right') - 1
    wi = min(wi, len(Wa) - 2)
    fi = min(fi, len(Fa) - 2)
    tx = (w - Wa[wi]) / (Wa[wi+1] - Wa[wi]) if Wa[wi+1] != Wa[wi] else 0.
    ty = (f - Fa[fi]) / (Fa[fi+1] - Fa[fi]) if Fa[fi+1] != Fa[fi] else 0.
    return (G[wi,   fi]   * (1-tx) * (1-ty) +
            G[wi+1, fi]   * tx     * (1-ty) +
            G[wi,   fi+1] * (1-tx) * ty     +
            G[wi+1, fi+1] * tx     * ty)


# Construire les grilles
EXC_WA, EXC_FA, EXC_GRID = build_grid(EXC_W2, EXC_FPRE, EXC_IPOST)
INH_WA, INH_FA, INH_GRID = build_grid(INH_W2, INH_FPRE, INH_IPOST)

print("\nGrilles construites:")
print("  EXC grid: {}x{}".format(len(EXC_WA), len(EXC_FA)))
print("  INH grid: {}x{}".format(len(INH_WA), len(INH_FA)))

# =========================================================
# 2. SIMULATION RESEAU
# =========================================================

def v_to_freq(v_mv):
    """Tension d'entree (mV) -> frequence de spike (Hz)."""
    return interp1(v_mv / 1000.0, INPUT_V_AXIS, INPUT_F_AXIS)


def current_to_freq(i_a):
    """Courant d'entree (A) -> frequence de spike (Hz)."""
    return interp1(i_a, NEURON_I_AXIS, NEURON_F_AXIS)


def synapse_current(mode, w_um, f_pre_hz):
    """
    Courant post-synaptique d'une synapse.
    mode='EXC' -> courant positif
    mode='INH' -> courant negatif
    mode='MIN' -> 0
    """
    if mode == 'EXC':
        return interp2(w_um, f_pre_hz, EXC_WA, EXC_FA, EXC_GRID)
    elif mode == 'INH':
        return -abs(interp2(w_um, f_pre_hz, INH_WA, INH_FA, INH_GRID))
    else:
        return 0.0


def forward(vx_mv, vy_mv, weights):
    """
    Passe avant complete du reseau 2->8->8->1.
    Retourne la frequence de sortie en Hz.
    """
    # Entrees -> frequences
    fx = v_to_freq(vx_mv)
    fy = v_to_freq(vy_mv)

    # Courants C1
    i_c1 = np.zeros(N_C1)
    for i in range(N_C1):
        e_vx = "Vx_c1_e1{}".format(i+1)
        i_vx = "Vx_c1_i1{}".format(i+1)
        e_vy = "Vy_c1_e2{}".format(i+1)
        i_vy = "Vy_c1_i2{}".format(i+1)

        mode_vx, w_vx = get_mode_w(weights, e_vx, i_vx)
        mode_vy, w_vy = get_mode_w(weights, e_vy, i_vy)

        i_c1[i] = (synapse_current(mode_vx, w_vx, fx) +
                   synapse_current(mode_vy, w_vy, fy))

    # Frequences C1
    f_c1 = np.array([current_to_freq(max(0., ic)) for ic in i_c1])

    # Courants C2
    i_c2 = np.zeros(N_C2)
    for j in range(N_C2):
        for i in range(N_C1):
            e_n = "c1_c2_e{}{}".format(i+1, j+1)
            i_n = "c1_c2_i{}{}".format(i+1, j+1)
            mode_c, w_c = get_mode_w(weights, e_n, i_n)
            i_c2[j] += synapse_current(mode_c, w_c, f_c1[i])

    # Frequences C2
    f_c2 = np.array([current_to_freq(max(0., ic)) for ic in i_c2])

    # Courant sortie
    i_out = 0.0
    for i in range(N_C2):
        e_n = "c2_s_e{}1".format(i+1)
        i_n = "c2_s_i{}1".format(i+1)
        mode_s, w_s = get_mode_w(weights, e_n, i_n)
        i_out += synapse_current(mode_s, w_s, f_c2[i])

    return current_to_freq(max(0., i_out))


def get_mode_w(weights, e_name, i_name):
    """Retourne (mode, valeur) pour une synapse."""
    e = weights[e_name]
    i = weights[i_name]
    if e > W_MIN and abs(i - W_MIN) < 1e-9:
        return 'EXC', e
    if i > W_MIN and abs(e - W_MIN) < 1e-9:
        return 'INH', i
    return 'MIN', W_MIN

# =========================================================
# 3. TOPOLOGIE ET POIDS
# =========================================================
W_MIN = 0.135

EXC_CANDIDATES = [0.135, 0.270, 0.405, 0.540, 0.675, 0.810,
                  1.080, 1.350, 1.620, 2.000, 2.500, 3.000]
INH_CANDIDATES = [0.135, 0.270, 0.405, 0.540, 0.675, 0.700]  # max 0.700 !

N_C1  = 8
N_C2  = 8
N_OUT = 1

SYNAPSES = []
for i in range(1, N_C1+1):
    SYNAPSES.append(("Vx_c1", "Vx_c1_e1{}".format(i), "Vx_c1_i1{}".format(i)))
for i in range(1, N_C1+1):
    SYNAPSES.append(("Vy_c1", "Vy_c1_e2{}".format(i), "Vy_c1_i2{}".format(i)))
for i in range(1, N_C1+1):
    for j in range(1, N_C2+1):
        SYNAPSES.append(("c1_c2", "c1_c2_e{}{}".format(i,j),
                                  "c1_c2_i{}{}".format(i,j)))
for i in range(1, N_C2+1):
    SYNAPSES.append(("c2_s", "c2_s_e{}1".format(i), "c2_s_i{}1".format(i)))

N_SYNAPSES   = len(SYNAPSES)
WEIGHT_NAMES = [e for _,e,_ in SYNAPSES] + [iv for _,_,iv in SYNAPSES]

print("\nTopologie:")
print("  Synapses totales : {}".format(N_SYNAPSES))
print("  Parametres poids : {}".format(len(WEIGHT_NAMES)))


def default_weights():
    return {k: W_MIN for k in WEIGHT_NAMES}


def apply_synapse_state(weights, syn_idx, mode, value):
    w = weights.copy()
    _, e_name, i_name = SYNAPSES[syn_idx]
    if mode == 'EXC':
        w[e_name] = float(value)
        w[i_name] = W_MIN
    elif mode == 'INH':
        w[e_name] = W_MIN
        w[i_name] = float(value)
    else:
        w[e_name] = W_MIN
        w[i_name] = W_MIN
    return w


def synapse_state(weights, e_name, i_name):
    e = weights[e_name]
    i = weights[i_name]
    if abs(e - W_MIN) < 1e-9 and abs(i - W_MIN) < 1e-9:
        return 'MIN', W_MIN
    if e > W_MIN:
        return 'EXC', e
    return 'INH', i


def weights_key(weights):
    return tuple(round(weights[k], 6) for k in WEIGHT_NAMES)


def make_all_neighbors(current):
    """Tous les voisins a une synapse de distance."""
    neighbors = []
    seen = {weights_key(current)}
    cid  = 0
    for syn_idx, (syn_name, e_name, i_name) in enumerate(SYNAPSES):
        states = [('MIN', W_MIN)]
        for w in EXC_CANDIDATES[1:]:
            states.append(('EXC', float(w)))
        for w in INH_CANDIDATES[1:]:
            states.append(('INH', float(w)))
        for mode, value in states:
            w2 = apply_synapse_state(current, syn_idx, mode, value)
            k  = weights_key(w2)
            if k not in seen:
                seen.add(k)
                neighbors.append((cid, syn_name, mode, value, w2))
                cid += 1
    return neighbors

# =========================================================
# 4. CAS D'ENTRAINEMENT — DISQUE
# =========================================================

FAST_CASES = [
    ( 15,  0),
    (-15,  0),
    ( 85,  0),
    (-85,  0),
]

FULL_TRAIN_CASES = [
    # Interieur r~0.15
    ( 15,  0), (  8, 13), ( -8, 13),
    (-15,  0), ( -8,-13), (  8,-13),
    # Exterieur r~0.85
    ( 85,  0), ( 43, 74), (-43, 74),
    (-85,  0), (-43,-74), ( 43,-74),
]

MIN_GOOD_MARGIN_HZ = 500.0
N_TRAIN = 200   # points dataset complet pour evaluation finale
N_TEST  = 100
DISK_RADIUS = 0.5
MARGIN_EXCLUSION = 0.05
SEED = 42


def label_from_inputs(vx_mv, vy_mv):
    x = 2.0 * (vx_mv/1000.0 - (-0.1)) / 0.2 - 1.0
    y = 2.0 * (vy_mv/1000.0 - (-0.1)) / 0.2 - 1.0
    return +1 if x*x + y*y < DISK_RADIUS**2 else -1


def xy_to_vmv(x, y):
    """Coordonnee [-1,1] -> tension mV."""
    vx = (-0.1 + (x+1)*0.1) * 1000.0
    vy = (-0.1 + (y+1)*0.1) * 1000.0
    return vx, vy


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        batch = rng.uniform(-1, 1, (n*3, 2))
        ok = np.abs(np.sqrt(batch[:,0]**2 + batch[:,1]**2) - DISK_RADIUS) > MARGIN_EXCLUSION
        pts.extend(batch[ok].tolist())
    return np.array(pts[:n])

# =========================================================
# 5. SCORE
# =========================================================

def find_best_threshold(freqs, labels):
    f_min, f_max = min(freqs), max(freqs)
    if f_max <= f_min:
        return (f_min + f_max) / 2.0
    best_thr, best_acc = (f_min+f_max)/2.0, -1.0
    for thr in np.linspace(f_min, f_max, 50):
        inside  = [f for f,l in zip(freqs,labels) if l==+1]
        outside = [f for f,l in zip(freqs,labels) if l==-1]
        ai = np.mean([f > thr for f in inside])  if inside  else 0.
        ao = np.mean([f <= thr for f in outside]) if outside else 0.
        acc = 0.5 * (ai + ao)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def evaluate_weights(weights, cases):
    """
    Evalue un jeu de poids sur les cas donnes.
    Retourne dict avec score, accuracy, details.
    """
    freqs  = []
    labels = []
    details = []

    for vx, vy in cases:
        fout = forward(vx, vy, weights)
        lab  = label_from_inputs(vx, vy)
        freqs.append(fout)
        labels.append(lab)
        details.append({'vx': vx, 'vy': vy, 'fout': fout, 'label': lab})

    thr = find_best_threshold(freqs, labels)

    margins = []
    ok_list = []
    for d in details:
        margin = d['label'] * (d['fout'] - thr)
        ok     = margin >= MIN_GOOD_MARGIN_HZ
        d['margin'] = margin
        d['ok']     = ok
        d['thr']    = thr
        margins.append(margin)
        ok_list.append(1.0 if ok else 0.0)

    accuracy    = float(np.mean(ok_list))
    mean_margin = float(np.mean(margins))
    min_margin  = float(np.min(margins))
    bad_penalty = sum(max(0., MIN_GOOD_MARGIN_HZ - m)
                      for m in margins if m < MIN_GOOD_MARGIN_HZ)
    score = (1_000_000.0 * accuracy
             + 5.0 * min_margin
             + 0.1 * mean_margin
             - 10.0 * bad_penalty)

    return {
        'score':       score,
        'accuracy':    accuracy,
        'mean_margin': mean_margin,
        'min_margin':  min_margin,
        'threshold':   thr,
        'details':     details,
    }

# =========================================================
# 6. LOCAL SEARCH
# =========================================================

N_RESTARTS  = 4
MAX_PASSES  = 8
EVAL_CACHE  = {}


def local_search_one_restart(restart_idx, rng, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("RESTART {}/{}".format(restart_idx+1, N_RESTARTS))

    # Init : W_MIN
    current = default_weights()
    result  = evaluate_weights(current, FULL_TRAIN_CASES)
    best    = result
    best_w  = current

    if verbose:
        print("Init (W_MIN): score={:.1f} acc={:.3f} "
              "min_m={:.1f} thr={:.0f}".format(
                  result['score'], result['accuracy'],
                  result['min_margin'], result['threshold']))

    for p in range(MAX_PASSES):
        if verbose:
            print("\n--- Pass {}/{} | score={:.1f} acc={:.3f}".format(
                p+1, MAX_PASSES, best['score'], best['accuracy']))

        neighbors = make_all_neighbors(current)
        if verbose:
            print("  {} voisins a evaluer...".format(len(neighbors)))

        t0 = time.time()

        # Fast screen sur FAST_CASES
        fast_scores = []
        for lid, sname, mode, val, w2 in neighbors:
            key = weights_key(w2)
            if key in EVAL_CACHE:
                r = EVAL_CACHE[key]
            else:
                r = evaluate_weights(w2, FAST_CASES)
                EVAL_CACHE[key] = r
            fast_scores.append((r['score'], lid, sname, mode, val, w2))

        fast_scores.sort(key=lambda x: -x[0])
        top_k = fast_scores[:TOP_K]

        if verbose:
            print("  Fast screen done ({:.1f}s) — top {} -> full eval".format(
                time.time()-t0, TOP_K))

        # Full eval
        best_candidate = None
        for score_fast, lid, sname, mode, val, w2 in top_k:
            key = (weights_key(w2), 'full')
            if key in EVAL_CACHE:
                r = EVAL_CACHE[key]
            else:
                r = evaluate_weights(w2, FULL_TRAIN_CASES)
                EVAL_CACHE[key] = r
            if best_candidate is None or r['score'] > best_candidate[0]:
                best_candidate = (r['score'], sname, mode, val, w2, r)

        if best_candidate is None:
            break

        sc, sname, mode, val, w2, r = best_candidate

        if verbose:
            print("  Best neighbor: {}={}({:.3f}) "
                  "score={:.1f} acc={:.3f} min_m={:.1f}".format(
                      sname, mode, val,
                      r['score'], r['accuracy'], r['min_margin']))

        if r['score'] > best['score']:
            current = w2
            best    = r
            best_w  = w2
            if verbose:
                print("  *** AMELIORATION! score={:.1f} acc={:.3f} "
                      "thr={:.0f}".format(
                          best['score'], best['accuracy'],
                          best['threshold']))
        else:
            if verbose:
                print("  Pas d'amelioration — arret.")
            break

    return best, best_w


# =========================================================
# 7. LANCEMENT
# =========================================================

TOP_K = 12  # top K apres fast screen pour full eval

print("\n" + "="*60)
print("LOCAL SEARCH — 2-8-8-1 DISQUE")
print("Synapses     : {}".format(N_SYNAPSES))
print("N_RESTARTS   : {}".format(N_RESTARTS))
print("MAX_PASSES   : {}".format(MAX_PASSES))
print("TOP_K        : {}".format(TOP_K))
print("Fast cases   : {}".format(FAST_CASES))
print("Full cases   : {}".format(FULL_TRAIN_CASES))
print("MIN_MARGIN   : {} Hz".format(MIN_GOOD_MARGIN_HZ))
print("="*60)

rng = np.random.RandomState(SEED)
global_best   = None
global_best_w = None

t_start = time.time()

for r in range(N_RESTARTS):
    result, weights = local_search_one_restart(r, rng)
    if global_best is None or result['score'] > global_best['score']:
        global_best   = result
        global_best_w = weights

print("\n" + "="*60)
print("GLOBAL BEST")
print("score={:.1f}  acc={:.3f}  min_m={:.1f}  thr={:.0f}".format(
    global_best['score'], global_best['accuracy'],
    global_best['min_margin'], global_best['threshold']))
print("Temps total: {:.1f}s".format(time.time() - t_start))

for d in global_best['details']:
    print("  Vx={:>4}mV Vy={:>4}mV | FOUT={:>8.1f} Hz label={:+d} | "
          "margin={:>8.1f} | {}".format(
              d['vx'], d['vy'], d['fout'], d['label'],
              d['margin'], 'OK' if d['ok'] else 'BAD'))

# =========================================================
# 8. EVALUATION COMPLETE SUR DATASET
# =========================================================
print("\n" + "="*60)
print("EVALUATION SUR DATASET COMPLET")

X_tr = make_dataset(N_TRAIN, SEED)
X_te = make_dataset(N_TEST,  SEED+1)

def eval_dataset(X, weights, thr):
    freqs  = []
    labels = []
    for x, y in X:
        vx, vy = xy_to_vmv(x, y)
        fout = forward(vx, vy, weights)
        lab  = 1 if x**2 + y**2 < DISK_RADIUS**2 else -1
        freqs.append(fout)
        labels.append(lab)
    preds = [1 if f > thr else -1 for f in freqs]
    acc   = np.mean([p==l for p,l in zip(preds, labels)])
    inside  = [f for f,l in zip(freqs,labels) if l==+1]
    outside = [f for f,l in zip(freqs,labels) if l==-1]
    acc_in  = np.mean([f > thr for f in inside])  if inside  else 0.
    acc_out = np.mean([f <= thr for f in outside]) if outside else 0.
    return acc, 0.5*(acc_in+acc_out), freqs, labels, preds

thr = global_best['threshold']
acc_tr, bal_tr, f_tr, l_tr, p_tr = eval_dataset(X_tr, global_best_w, thr)
acc_te, bal_te, f_te, l_te, p_te = eval_dataset(X_te, global_best_w, thr)

print("Train: acc={:.3f}  bal_acc={:.3f}".format(acc_tr, bal_tr))
print("Test : acc={:.3f}  bal_acc={:.3f}".format(acc_te, bal_te))
print("Seuil: {:.0f} Hz".format(thr))

# =========================================================
# 9. VISUALISATION
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Local Search 2-8-8-1 Disque\n'
             'bal_acc train={:.3f}  test={:.3f}  seuil={:.0f}Hz'.format(
                 bal_tr, bal_te, thr), fontweight='bold')

# Carte de decision
ax = axes[0]
res = 40
xs  = np.linspace(-1, 1, res)
XX, YY = np.meshgrid(xs, xs)
Fout_grid = np.zeros_like(XX)
for i in range(res):
    for j in range(res):
        vx, vy = xy_to_vmv(XX[i,j], YY[i,j])
        Fout_grid[i,j] = forward(vx, vy, global_best_w)

pred_grid = (Fout_grid > thr).astype(int)
ax.contourf(XX, YY, pred_grid, levels=[-0.5,0.5,1.5],
            colors=['#EF535044','#42A5F544'])
ax.contour(XX, YY, Fout_grid - thr, levels=[0], colors='k', linewidths=2)
ax.add_patch(plt.Circle((0,0), DISK_RADIUS, color='white',
                          fill=False, lw=2, ls='--'))
ax.scatter(X_te[np.array(l_te)==-1, 0],
           X_te[np.array(l_te)==-1, 1],
           s=8, c='#EF5350', alpha=0.6, label='Ext')
ax.scatter(X_te[np.array(l_te)==+1, 0],
           X_te[np.array(l_te)==+1, 1],
           s=8, c='#42A5F5', alpha=0.6, label='Int')
ax.set_title('Carte de decision (test)')
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Distribution Fout
ax = axes[1]
f_in  = [f for f,l in zip(f_te, l_te) if l==+1]
f_out = [f for f,l in zip(f_te, l_te) if l==-1]
bins  = np.linspace(min(f_te)-1000, max(f_te)+1000, 40)
ax.hist(np.array(f_in)/1e3,  bins=bins/1e3, alpha=0.6,
        color='#42A5F5', label='Interieur')
ax.hist(np.array(f_out)/1e3, bins=bins/1e3, alpha=0.6,
        color='#EF5350', label='Exterieur')
ax.axvline(thr/1e3, color='k', lw=2, ls='--',
           label='Seuil {:.1f}kHz'.format(thr/1e3))
ax.set_xlabel('FOUT (kHz)')
ax.set_title('Distribution FOUT (test)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Carte FOUT brute
ax = axes[2]
cf = ax.contourf(XX, YY, Fout_grid/1e3, levels=50, cmap='plasma')
plt.colorbar(cf, ax=ax, label='FOUT (kHz)')
ax.contour(XX, YY, Fout_grid - thr, levels=[0],
           colors='white', linewidths=2)
ax.add_patch(plt.Circle((0,0), DISK_RADIUS, color='cyan',
                          fill=False, lw=2))
ax.set_title('FOUT brute')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('disk_local_search_result.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardee: disk_local_search_result.png")

# =========================================================
# 10. EXPORT POIDS
# =========================================================
rows = []
for syn_name, e_name, i_name in SYNAPSES:
    mode, val = synapse_state(global_best_w, e_name, i_name)
    rows.append({
        'synapse': syn_name,
        'e_param': e_name,
        'i_param': i_name,
        'mode':    mode,
        'width_um': val,
        'e_value': global_best_w[e_name],
        'i_value': global_best_w[i_name],
    })

df_w = pd.DataFrame(rows)
df_w.to_csv('weights_disk_local_search.csv', index=False)
print("\nPoids exportes: weights_disk_local_search.csv")
print(df_w[df_w['mode'] != 'MIN'].to_string(index=False))
