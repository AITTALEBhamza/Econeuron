import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

np.set_printoptions(precision=4, suppress=True)

# =========================================================
# FILES
# =========================================================
INPUT_CSV   = "input_min_conso.csv"
NEURON_CSV  = "neuron.csv"
SYN_EXC_CSV = "excitation.csv"
SYN_INH_CSV = "inhibition.csv"

# =========================================================
# GLOBAL HYPERPARAMETERS
# =========================================================
SEED             = 42
RHO              = 0.55
N_TRAIN_UNIFORM  = 700
N_TRAIN_BOUNDARY = 1400
BOUNDARY_SIGMA_R = 0.05
N_TEST           = 500

# ---- architecture of each branch ----
USE_SYMMETRIC_INPUTS = False
BRANCH_HIDDEN_SIZES  = [6, 4, 6]   # example: [4,4], [6,6], [8,6,8]

# ---- search ----
N_RESTARTS_PER_BRANCH = 6
MAX_PASSES            = 4
MARGIN_SCALE          = 5e-6
FREQ_MERGE_TOL        = 1000.0

# ---- hardware search options ----
ALLOW_ZERO_SYNAPSES            = True
ZERO_SYNAPSE_INIT_PROB         = 0.20
USE_SMALLER_WIDTHS_DEEP_LAYERS = True
DEEP_EXC_MAX                   = 0.7717
DEEP_INH_MAX                   = 0.5194

# ---- score penalty ----
IMBALANCE_PENALTY_SCALE = 0.15

# ---- branch B residual weighting ----
ERROR_BOOST        = 3.0
LOW_MARGIN_BOOST   = 2.0
MARGIN_TARGET_HZ   = 4000.0
MAX_SAMPLE_WEIGHT  = 6.0

# ---- alpha search for branch combination ----
ALPHA_CANDIDATES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

# =========================================================
# PROGRESS BAR
# =========================================================
def progress_bar(current, total, prefix="", suffix="", bar_len=30):
    filled = int(bar_len * current / total)
    bar    = "█" * filled + "░" * (bar_len - filled)
    pct    = int(100 * current / total)
    line   = f"\r  {prefix}[{bar}] {current}/{total} {pct}%  {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# =========================================================
# CSV READING
# =========================================================
def read_cadence_rows(csv_path):
    df = pd.read_csv(csv_path)

    if "Parameter" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'Parameter' column")

    data_cols = [c for c in df.columns if c not in ["Parameter", "Unnamed: 1"]]

    def get_row(row_name, allow_non_numeric=False):
        row = df.loc[df["Parameter"] == row_name, data_cols]
        if row.empty:
            raise ValueError(f"Row '{row_name}' not found in {csv_path}")

        values = pd.to_numeric(row.iloc[0], errors="coerce").to_numpy(dtype=float)

        if (not allow_non_numeric) and np.isnan(values).any():
            bad_idx = np.where(np.isnan(values))[0]
            raise ValueError(
                f"Row '{row_name}' in {csv_path} contains non-numeric values at {bad_idx[:10]}"
            )
        return values

    return get_row

def get_clean_rows(reader, row_names):
    rows    = [reader(name, allow_non_numeric=True) for name in row_names]
    stacked = np.vstack(rows)
    mask    = np.all(np.isfinite(stacked), axis=0)
    return [r[mask] for r in rows], mask

# =========================================================
# INTERPOLATION
# =========================================================
def interp_clip(x, xp, fp):
    return np.interp(np.asarray(x, dtype=float), xp, fp, left=fp[0], right=fp[-1])

def interp2d_bilinear_vec(x, y, x_axis, y_axis, z_grid):
    x  = np.clip(np.asarray(x, dtype=float), x_axis[0], x_axis[-1])
    y  = np.clip(np.asarray(y, dtype=float), y_axis[0], y_axis[-1])

    ix = np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, len(x_axis) - 2)
    iy = np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, len(y_axis) - 2)

    x1 = x_axis[ix]
    x2 = x_axis[ix + 1]
    y1 = y_axis[iy]
    y2 = y_axis[iy + 1]

    tx = np.where(x2 == x1, 0.0, (x - x1) / (x2 - x1))
    ty = np.where(y2 == y1, 0.0, (y - y1) / (y2 - y1))

    return (
        z_grid[ix,   iy  ] * (1 - tx) * (1 - ty) +
        z_grid[ix+1, iy  ] * tx       * (1 - ty) +
        z_grid[ix,   iy+1] * (1 - tx) * ty       +
        z_grid[ix+1, iy+1] * tx       * ty
    )

def strictly_monotonic_branch(x, y):
    im    = int(np.argmax(y))
    x_m   = np.asarray(x[:im+1], dtype=float)
    y_m   = np.asarray(y[:im+1], dtype=float)
    order = np.argsort(x_m)
    return x_m[order], y_m[order]

def merge_close_values(values, tol):
    vals   = np.sort(np.asarray(values, dtype=float))
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([np.mean(g) for g in groups], dtype=float)

def snap_to_axis(values, axis):
    return np.array([axis[np.argmin(np.abs(axis - v))]
                     for v in np.asarray(values, dtype=float)])

def build_grid_tolerant(W_all, F_all, Y_all, freq_tol=1000.0):
    W_unique = np.unique(W_all)
    F_axis   = merge_close_values(np.unique(F_all), tol=freq_tol)
    F_snap   = snap_to_axis(F_all, F_axis)
    z_grid   = np.empty((len(W_unique), len(F_axis)))

    for i, w in enumerate(W_unique):
        for j, f in enumerate(F_axis):
            mask = np.isclose(W_all, w) & np.isclose(F_snap, f)
            vals = Y_all[mask]
            if len(vals) == 0:
                raise ValueError(f"Missing point for W={w}, Fbin={f}")
            z_grid[i, j] = np.mean(vals)

    return W_unique, F_axis, z_grid

# =========================================================
# LOAD HARDWARE TABLES
# =========================================================
inp = read_cadence_rows(INPUT_CSV)
INPUT_V_AXIS, INPUT_F_AXIS = strictly_monotonic_branch(inp("Vex"), inp("Fspikes"))

neuro = read_cadence_rows(NEURON_CSV)
NEURON_I_AXIS, NEURON_F_AXIS = strictly_monotonic_branch(neuro("Iex"), neuro("Fspikes"))

syn_exc = read_cadence_rows(SYN_EXC_CSV)
(W2_EXC, FPRE_EXC, IPOST_EXC), _ = get_clean_rows(syn_exc, ["W2", "Fpre", "Ipost"])
W_EXC_AXIS, F_EXC_AXIS, IPOST_EXC_GRID = build_grid_tolerant(
    W2_EXC, FPRE_EXC, IPOST_EXC, freq_tol=FREQ_MERGE_TOL
)

syn_inh = read_cadence_rows(SYN_INH_CSV)
(W2_INH, FPRE_INH, IPOST_INH), _ = get_clean_rows(syn_inh, ["W2", "Fpre", "Ipost"])
W_INH_AXIS, F_INH_AXIS, IPOST_INH_GRID = build_grid_tolerant(
    W2_INH, FPRE_INH, IPOST_INH, freq_tol=FREQ_MERGE_TOL
)

V_MIN = float(INPUT_V_AXIS[0])
V_MAX = float(INPUT_V_AXIS[-1])

# =========================================================
# ARCHITECTURE
# =========================================================
INPUT_DIM          = 4 if USE_SYMMETRIC_INPUTS else 2
BRANCH_LAYER_SIZES = [INPUT_DIM] + BRANCH_HIDDEN_SIZES + [2]

# =========================================================
# HARDWARE PRIMITIVE FUNCTIONS
# =========================================================
def input_f_of_v(Vin_V):
    return interp_clip(Vin_V, INPUT_V_AXIS, INPUT_F_AXIS)

def neuron_f_of_i(Iin_A):
    return interp_clip(Iin_A, NEURON_I_AXIS, NEURON_F_AXIS)

# =========================================================
# LAYER CANDIDATES
# =========================================================
def get_layer_candidates(layer_idx):
    exc_axis = W_EXC_AXIS
    inh_axis = W_INH_AXIS

    if USE_SMALLER_WIDTHS_DEEP_LAYERS and layer_idx >= 1:
        exc_axis = exc_axis[exc_axis <= DEEP_EXC_MAX]
        inh_axis = inh_axis[inh_axis <= DEEP_INH_MAX]

        if len(exc_axis) == 0:
            exc_axis = W_EXC_AXIS
        if len(inh_axis) == 0:
            inh_axis = W_INH_AXIS

    cand = []
    if ALLOW_ZERO_SYNAPSES:
        cand.append((0, 0.0))
    cand += [(+1, float(w)) for w in exc_axis]
    cand += [(-1, float(w)) for w in inh_axis]
    return cand

ALL_CANDIDATES_PER_LAYER = [
    get_layer_candidates(l)
    for l in range(len(BRANCH_LAYER_SIZES) - 1)
]

# =========================================================
# BATCHED HARDWARE FORWARD PASS
# =========================================================
def layer_forward_batch(Fin_batch, sign_matrix, width_matrix):
    B, N = Fin_batch.shape
    M    = sign_matrix.shape[1]

    F = np.broadcast_to(Fin_batch[:, :, None],    (B, N, M))
    W = np.broadcast_to(width_matrix[None, :, :], (B, N, M))
    S = np.broadcast_to(sign_matrix[None, :, :],  (B, N, M))

    Iexc = interp2d_bilinear_vec(W, F, W_EXC_AXIS, F_EXC_AXIS, IPOST_EXC_GRID)
    Iinh = interp2d_bilinear_vec(W, F, W_INH_AXIS, F_INH_AXIS, IPOST_INH_GRID)

    Iconn = np.where(S == 1, Iexc,
             np.where(S == -1, -Iinh, 0.0))

    Itot = np.sum(Iconn, axis=1)
    return neuron_f_of_i(Itot)

def network_forward_batch(Vin_batch, sign_matrices, width_matrices):
    F = input_f_of_v(Vin_batch)
    for sign_mat, width_mat in zip(sign_matrices, width_matrices):
        F = layer_forward_batch(F, sign_mat, width_mat)
    return F

# =========================================================
# DATASET
# =========================================================
def make_uniform_dataset(n, rho=RHO, seed=0):
    rng    = np.random.default_rng(seed)
    X      = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    labels = (X[:, 0]**2 + X[:, 1]**2 <= rho**2).astype(np.int32)
    return X, labels

def make_boundary_dataset(n, rho=RHO, sigma_r=0.08, seed=0):
    rng    = np.random.default_rng(seed)
    theta  = rng.uniform(0, 2*np.pi, n)
    r      = rho + rng.normal(0, sigma_r, n)
    X      = np.column_stack([r*np.cos(theta), r*np.sin(theta)]).astype(np.float32)
    mask   = np.all(np.abs(X) <= 1.0, axis=1)
    X      = X[mask]
    labels = (X[:, 0]**2 + X[:, 1]**2 <= rho**2).astype(np.int32)
    return X, labels

def xy_to_voltage(X):
    X = np.asarray(X, dtype=np.float32)

    if USE_SYMMETRIC_INPUTS:
        X_hw = np.column_stack([X[:, 0], -X[:, 0], X[:, 1], -X[:, 1]])
    else:
        X_hw = X

    return (V_MIN + (X_hw + 1.0) * (V_MAX - V_MIN) / 2.0).astype(np.float32)

def make_datasets():
    X_u, y_u = make_uniform_dataset(N_TRAIN_UNIFORM, rho=RHO, seed=SEED)
    X_b, y_b = make_boundary_dataset(
        N_TRAIN_BOUNDARY, rho=RHO, sigma_r=BOUNDARY_SIGMA_R, seed=SEED + 1
    )

    X_train = np.vstack([X_u, X_b])
    y_train = np.hstack([y_u, y_b])

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    X_test, y_test = make_uniform_dataset(N_TEST, rho=RHO, seed=SEED + 2)

    V_train = xy_to_voltage(X_train)
    V_test  = xy_to_voltage(X_test)

    print(f"Train: {len(y_train)} | inside={y_train.sum()} ({y_train.mean()*100:.0f}%)  outside={(1-y_train).sum()} ({(1-y_train).mean()*100:.0f}%)")
    print(f"Test : {len(y_test)}  | inside={y_test.sum()} ({y_test.mean()*100:.0f}%)  outside={(1-y_test).mean()*100:.0f}%)")

    return X_train, y_train, X_test, y_test, V_train, V_test

# =========================================================
# METRICS
# =========================================================
def normalize_weights_per_class(sample_weights, y_true):
    w = np.asarray(sample_weights, dtype=float).copy()
    y = np.asarray(y_true, dtype=int)

    for cls in [0, 1]:
        mask = (y == cls)
        if np.any(mask):
            m = np.mean(w[mask])
            if m > 0:
                w[mask] /= m
    return w

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights)
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * values) / denom)

def compute_true_margin(Fout, y_true):
    y_true = np.asarray(y_true, dtype=int)
    idx = np.arange(len(y_true))
    true_f = Fout[idx, y_true]
    other_f = Fout[idx, 1 - y_true]
    return true_f - other_f

def evaluate_outputs(Fout, y_true, sample_weights=None):
    y_true = np.asarray(y_true, dtype=int)
    preds  = np.argmax(Fout, axis=1)

    if sample_weights is None:
        sample_weights = np.ones(len(y_true), dtype=float)
    else:
        sample_weights = normalize_weights_per_class(sample_weights, y_true)

    mask0 = (y_true == 0)
    mask1 = (y_true == 1)

    w0 = sample_weights[mask0]
    w1 = sample_weights[mask1]

    acc0 = weighted_mean((preds[mask0] == 0).astype(float), w0) if np.any(mask0) else 0.0
    acc1 = weighted_mean((preds[mask1] == 1).astype(float), w1) if np.any(mask1) else 0.0
    bal_acc = 0.5 * (acc0 + acc1)

    margin0 = weighted_mean(Fout[mask0, 0] - Fout[mask0, 1], w0) if np.any(mask0) else 0.0
    margin1 = weighted_mean(Fout[mask1, 1] - Fout[mask1, 0], w1) if np.any(mask1) else 0.0
    bal_margin = 0.5 * (margin0 + margin1)

    imbalance_penalty = IMBALANCE_PENALTY_SCALE * abs(acc0 - acc1)
    score = bal_acc + MARGIN_SCALE * bal_margin - imbalance_penalty

    return {
        "preds": preds,
        "Fout": Fout,
        "acc0": acc0,
        "acc1": acc1,
        "bal_acc": bal_acc,
        "bal_margin": bal_margin,
        "score": score,
    }

def evaluate_network(Vin_batch, y_true, sign_matrices, width_matrices, sample_weights=None):
    Fout = network_forward_batch(Vin_batch, sign_matrices, width_matrices)
    return evaluate_outputs(Fout, y_true, sample_weights=sample_weights)

# =========================================================
# RANDOM NETWORK
# =========================================================
def random_network(layer_sizes, rng):
    signs, widths = [], []

    for layer_idx, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if ALLOW_ZERO_SYNAPSES:
            s = rng.choice(
                [-1, 0, 1],
                size=(n_in, n_out),
                p=[(1 - ZERO_SYNAPSE_INIT_PROB) / 2,
                   ZERO_SYNAPSE_INIT_PROB,
                   (1 - ZERO_SYNAPSE_INIT_PROB) / 2]
            )
        else:
            s = rng.choice([-1, 1], size=(n_in, n_out))

        w = np.zeros((n_in, n_out))
        candidates = ALL_CANDIDATES_PER_LAYER[layer_idx]
        exc_only = [c[1] for c in candidates if c[0] == 1]
        inh_only = [c[1] for c in candidates if c[0] == -1]

        for i in range(n_in):
            for j in range(n_out):
                if s[i, j] == 1:
                    w[i, j] = rng.choice(exc_only)
                elif s[i, j] == -1:
                    w[i, j] = rng.choice(inh_only)
                else:
                    w[i, j] = 0.0

        signs.append(s)
        widths.append(w)

    return signs, widths

# =========================================================
# COORDINATE SEARCH FOR ONE BRANCH
# =========================================================
def coordinate_search_branch(
    V_train, y_train, V_val, y_val,
    layer_sizes,
    sample_weights_train=None,
    n_restarts=N_RESTARTS_PER_BRANCH,
    max_passes=MAX_PASSES,
    seed=SEED,
    branch_name="A"
):
    rng = np.random.default_rng(seed)

    all_conns = [(l, i, j)
                 for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
                 for i in range(n_in)
                 for j in range(n_out)]
    n_conns = len(all_conns)

    best_signs_g  = None
    best_widths_g = None
    best_val_score = -1e99
    best_train_score = -1e99
    best_history = []

    for restart in range(n_restarts):
        print(f"\n{'='*70}")
        print(f"BRANCH {branch_name} | RESTART {restart+1}/{n_restarts}")

        signs, widths = random_network(layer_sizes, rng)

        tr = evaluate_network(V_train, y_train, signs, widths, sample_weights=sample_weights_train)
        va = evaluate_network(V_val,   y_val,   signs, widths, sample_weights=None)

        history = [{
            "pass": 0,
            "tr_bal_acc": tr["bal_acc"],
            "tr_score": tr["score"],
            "va_bal_acc": va["bal_acc"],
            "va_score": va["score"],
        }]

        print(f"Init | train_bal_acc={tr['bal_acc']:.4f} train_score={tr['score']:.6f} | "
              f"val_bal_acc={va['bal_acc']:.4f} val_score={va['score']:.6f}")

        t0 = time.time()

        for pass_idx in range(max_passes):
            improved = False
            conn_order = list(all_conns)
            rng.shuffle(conn_order)

            for conn_num, (l, i, j) in enumerate(conn_order):
                best_s = signs[l][i, j]
                best_w = widths[l][i, j]
                best_c = tr["score"]

                for cand_s, cand_w in ALL_CANDIDATES_PER_LAYER[l]:
                    if cand_s == signs[l][i, j] and np.isclose(cand_w, widths[l][i, j]):
                        continue

                    old_s, old_w = signs[l][i, j], widths[l][i, j]
                    signs[l][i, j]  = cand_s
                    widths[l][i, j] = cand_w

                    cand_tr = evaluate_network(
                        V_train, y_train, signs, widths,
                        sample_weights=sample_weights_train
                    )

                    if cand_tr["score"] > best_c:
                        best_c = cand_tr["score"]
                        best_s = cand_s
                        best_w = cand_w

                    signs[l][i, j]  = old_s
                    widths[l][i, j] = old_w

                if best_c > tr["score"]:
                    signs[l][i, j]  = best_s
                    widths[l][i, j] = best_w
                    tr = evaluate_network(
                        V_train, y_train, signs, widths,
                        sample_weights=sample_weights_train
                    )
                    improved = True

                suffix = (f"pass {pass_idx+1}/{max_passes}  "
                          f"tr_bal_acc={tr['bal_acc']:.4f}  "
                          f"score={tr['score']:.6f}  "
                          f"{time.time()-t0:.0f}s")
                progress_bar(conn_num + 1, n_conns, prefix=f"Branch {branch_name} ", suffix=suffix)

            va = evaluate_network(V_val, y_val, signs, widths, sample_weights=None)
            history.append({
                "pass": pass_idx + 1,
                "tr_bal_acc": tr["bal_acc"],
                "tr_score": tr["score"],
                "va_bal_acc": va["bal_acc"],
                "va_score": va["score"],
            })

            print(f"Pass {pass_idx+1} | "
                  f"train_bal_acc={tr['bal_acc']:.4f} train_score={tr['score']:.6f} | "
                  f"val_bal_acc={va['bal_acc']:.4f} val_score={va['score']:.6f}")

            if not improved:
                print("Converged")
                break

        if (va["score"] > best_val_score or
                (np.isclose(va["score"], best_val_score) and tr["score"] > best_train_score)):
            best_signs_g   = [s.copy() for s in signs]
            best_widths_g  = [w.copy() for w in widths]
            best_val_score = va["score"]
            best_train_score = tr["score"]
            best_history   = history

    return best_signs_g, best_widths_g, best_history

# =========================================================
# BUILD AUTOMATIC WEIGHTS FOR BRANCH B
# =========================================================
def build_residual_weights_from_branch_A(Fout_A, y_true):
    y_true = np.asarray(y_true, dtype=int)
    preds_A = np.argmax(Fout_A, axis=1)
    true_margin = compute_true_margin(Fout_A, y_true)

    weights = np.ones(len(y_true), dtype=float)

    # hard errors
    weights += ERROR_BOOST * (preds_A != y_true).astype(float)

    # low-margin samples
    margin_deficit = np.clip((MARGIN_TARGET_HZ - true_margin) / MARGIN_TARGET_HZ, 0.0, None)
    weights += LOW_MARGIN_BOOST * margin_deficit

    weights = np.clip(weights, 1.0, MAX_SAMPLE_WEIGHT)
    weights = normalize_weights_per_class(weights, y_true)
    return weights

# =========================================================
# SEARCH BEST ALPHA FOR BRANCH COMBINATION
# =========================================================
def search_best_alpha(FA_train, FB_train, y_train, FA_val, FB_val, y_val, alpha_candidates):
    best_alpha = None
    best_train = None
    best_val = None

    for alpha in alpha_candidates:
        F_train = FA_train + alpha * FB_train
        F_val   = FA_val   + alpha * FB_val

        tr = evaluate_outputs(F_train, y_train, sample_weights=None)
        va = evaluate_outputs(F_val,   y_val,   sample_weights=None)

        print(f"alpha={alpha:>4.2f} | "
              f"train_bal_acc={tr['bal_acc']:.4f} train_score={tr['score']:.6f} | "
              f"val_bal_acc={va['bal_acc']:.4f} val_score={va['score']:.6f}")

        if best_val is None or va["score"] > best_val["score"]:
            best_alpha = alpha
            best_train = tr
            best_val = va

    return best_alpha, best_train, best_val

# =========================================================
# VISUALIZATION
# =========================================================
def print_model_info():
    print("=== HARDWARE ===")
    print(f"  Input  V: {V_MIN*1e3:.0f} mV -> {V_MAX*1e3:.0f} mV  "
          f"F: {INPUT_F_AXIS[0]*1e-3:.1f} -> {INPUT_F_AXIS[-1]*1e-3:.1f} kHz")
    print(f"  Neuron I: {NEURON_I_AXIS[0]*1e12:.1f} pA -> {NEURON_I_AXIS[-1]*1e12:.0f} pA  "
          f"F: {NEURON_F_AXIS[0]*1e-3:.1f} -> {NEURON_F_AXIS[-1]*1e-3:.1f} kHz")
    print(f"  Exc widths (um): {W_EXC_AXIS}")
    print(f"  Inh widths (um): {W_INH_AXIS}")

def print_network(signs, widths, name="NETWORK"):
    names = (["Input->H1"]
             + [f"H{k}->H{k+1}" for k in range(1, len(signs)-1)]
             + [f"H{len(signs)-1}->Output"])
    print(f"\n=== {name} ===")
    for k, (s, w, lname) in enumerate(zip(signs, widths, names)):
        print(f"  Layer {k+1} [{lname}] {s.shape[0]}->{s.shape[1]}"
              f"  exc={int(np.sum(s==1))} inh={int(np.sum(s==-1))} off={int(np.sum(s==0))}")
        print(f"    signs:\n{s}")
        print(f"    widths (um):\n{w}")

def plot_dataset(X, y, rho=RHO, title="Dataset"):
    theta = np.linspace(0, 2*np.pi, 400)
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], s=5, alpha=0.5, label="Outside")
    plt.scatter(X[y==1, 0], X[y==1, 1], s=5, alpha=0.5, label="Inside")
    plt.plot(rho*np.cos(theta), rho*np.sin(theta), "r--", lw=2, label="True boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weights_histogram(sample_weights, y_true):
    plt.figure(figsize=(7, 4))
    plt.hist(sample_weights[y_true == 0], bins=30, alpha=0.7, label="class 0")
    plt.hist(sample_weights[y_true == 1], bins=30, alpha=0.7, label="class 1")
    plt.xlabel("Residual sample weight for Branch B")
    plt.ylabel("Count")
    plt.title("Automatic weights built from Branch A")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_history(history, title="Search history"):
    if not history:
        return
    passes   = [h["pass"] for h in history]
    tr_accs  = [h["tr_bal_acc"] for h in history]
    va_accs  = [h["va_bal_acc"] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(passes, tr_accs, "o-", label="Train balanced acc")
    plt.plot(passes, va_accs, "s-", label="Validation/Test balanced acc")
    plt.xlabel("Pass")
    plt.ylabel("Balanced accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_results(label, metrics):
    print(f"\n=== {label} ===")
    print(f"  acc class 0 (outside) = {metrics['acc0']:.4f}")
    print(f"  acc class 1 (inside)  = {metrics['acc1']:.4f}")
    print(f"  balanced accuracy     = {metrics['bal_acc']:.4f}")
    print(f"  balanced margin       = {metrics['bal_margin']*1e-3:.2f} kHz")
    print(f"  score                 = {metrics['score']:.6f}")

def plot_combined_decision_map(signs_A, widths_A, signs_B, widths_B, alpha, rho=RHO, resolution=160):
    xs     = np.linspace(-1, 1, resolution)
    ys     = np.linspace(-1, 1, resolution)
    XX, YY = np.meshgrid(xs, ys)
    grid_X = np.column_stack([XX.ravel(), YY.ravel()])
    grid_V = xy_to_voltage(grid_X)

    FA = network_forward_batch(grid_V, signs_A, widths_A)
    FB = network_forward_batch(grid_V, signs_B, widths_B)
    Ftot = FA + alpha * FB

    pred   = np.argmax(Ftot, axis=1).reshape(XX.shape)
    margin = (Ftot[:, 1] - Ftot[:, 0]).reshape(XX.shape) * 1e-3

    theta  = np.linspace(0, 2*np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.contourf(XX, YY, pred, levels=[-0.5, 0.5, 1.5],
                colors=["#f4a3a3", "#a3d9a3"], alpha=0.7)
    ax.contour(XX, YY, margin, levels=[0], colors="k", linewidths=2)
    ax.plot(rho*np.cos(theta), rho*np.sin(theta), "r--", lw=2, label="True boundary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Combined decision (alpha={alpha:.2f})")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    vmax = max(abs(float(margin.min())), abs(float(margin.max())), 1.0)
    cf = ax.contourf(XX, YY, margin, levels=50, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    plt.colorbar(cf, ax=ax, label="F_inside - F_outside (kHz)")
    ax.contour(XX, YY, margin, levels=[0], colors="k", linewidths=2)
    ax.plot(rho*np.cos(theta), rho*np.sin(theta), "r--", lw=2, label="True boundary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Combined frequency margin")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print_model_info()

    print("\n=== SETTINGS ===")
    print(f"USE_SYMMETRIC_INPUTS       = {USE_SYMMETRIC_INPUTS}")
    print(f"BRANCH_LAYER_SIZES         = {BRANCH_LAYER_SIZES}")
    print(f"N_RESTARTS_PER_BRANCH      = {N_RESTARTS_PER_BRANCH}")
    print(f"MAX_PASSES                 = {MAX_PASSES}")
    print(f"MARGIN_SCALE               = {MARGIN_SCALE}")
    print(f"IMBALANCE_PENALTY_SCALE    = {IMBALANCE_PENALTY_SCALE}")
    print(f"BOUNDARY_SIGMA_R           = {BOUNDARY_SIGMA_R}")
    print(f"ERROR_BOOST                = {ERROR_BOOST}")
    print(f"LOW_MARGIN_BOOST           = {LOW_MARGIN_BOOST}")
    print(f"MARGIN_TARGET_HZ           = {MARGIN_TARGET_HZ}")
    print(f"ALPHA_CANDIDATES           = {ALPHA_CANDIDATES}")

    print("\n=== DATASET ===")
    X_train, y_train, X_test, y_test, V_train, V_test = make_datasets()

    plot_dataset(X_train, y_train, title="Train (uniform + boundary)")
    plot_dataset(X_test,  y_test,  title="Test (uniform)")

    # -----------------------------------------------------
    # BRANCH A
    # -----------------------------------------------------
    print("\n=== SEARCH BRANCH A ===")
    print(f"Branch A architecture: {BRANCH_LAYER_SIZES}")
    signs_A, widths_A, hist_A = coordinate_search_branch(
        V_train, y_train, V_test, y_test,
        layer_sizes=BRANCH_LAYER_SIZES,
        sample_weights_train=None,
        n_restarts=N_RESTARTS_PER_BRANCH,
        max_passes=MAX_PASSES,
        seed=SEED,
        branch_name="A"
    )

    print_network(signs_A, widths_A, name="BRANCH A")
    metrics_A_train = evaluate_network(V_train, y_train, signs_A, widths_A)
    metrics_A_test  = evaluate_network(V_test,  y_test,  signs_A, widths_A)
    print_results("BRANCH A - TRAIN", metrics_A_train)
    print_results("BRANCH A - TEST",  metrics_A_test)
    plot_history(hist_A, title="Branch A — coordinate search history")

    # -----------------------------------------------------
    # BUILD AUTOMATIC RESIDUAL WEIGHTS FOR BRANCH B
    # -----------------------------------------------------
    print("\n=== AUTOMATIC RESIDUAL WEIGHTS FOR BRANCH B ===")
    sample_weights_B = build_residual_weights_from_branch_A(metrics_A_train["Fout"], y_train)
    print(f"weights min={sample_weights_B.min():.4f}  max={sample_weights_B.max():.4f}  mean={sample_weights_B.mean():.4f}")
    plot_weights_histogram(sample_weights_B, y_train)

    # -----------------------------------------------------
    # BRANCH B
    # -----------------------------------------------------
    print("\n=== SEARCH BRANCH B (weighted on A residuals) ===")
    signs_B, widths_B, hist_B = coordinate_search_branch(
        V_train, y_train, V_test, y_test,
        layer_sizes=BRANCH_LAYER_SIZES,
        sample_weights_train=sample_weights_B,
        n_restarts=N_RESTARTS_PER_BRANCH,
        max_passes=MAX_PASSES,
        seed=SEED + 1000,
        branch_name="B"
    )

    print_network(signs_B, widths_B, name="BRANCH B")
    metrics_B_train = evaluate_network(V_train, y_train, signs_B, widths_B)
    metrics_B_test  = evaluate_network(V_test,  y_test,  signs_B, widths_B)
    print_results("BRANCH B - TRAIN", metrics_B_train)
    print_results("BRANCH B - TEST",  metrics_B_test)
    plot_history(hist_B, title="Branch B — coordinate search history")

    # -----------------------------------------------------
    # COMBINE BRANCH A + BRANCH B
    # -----------------------------------------------------
    print("\n=== ALPHA SEARCH FOR A + alpha*B ===")
    FA_train = metrics_A_train["Fout"]
    FA_test  = metrics_A_test["Fout"]
    FB_train = metrics_B_train["Fout"]
    FB_test  = metrics_B_test["Fout"]

    best_alpha, best_comb_train, best_comb_test = search_best_alpha(
        FA_train, FB_train, y_train,
        FA_test,  FB_test,  y_test,
        ALPHA_CANDIDATES
    )

    print(f"\nBest alpha = {best_alpha:.4f}")
    print_results("COMBINED - TRAIN", best_comb_train)
    print_results("COMBINED - TEST",  best_comb_test)

    plot_combined_decision_map(signs_A, widths_A, signs_B, widths_B, best_alpha, rho=RHO)