"""Rewrite cells 40, 41, 42 for SPO+ Option B + direct Option C training."""
import json
from pathlib import Path

NB = Path("DCOPF_v3.ipynb")
nb = json.loads(NB.read_text())


def set_cell_source(cell_id, src):
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            c["source"] = src.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            return
    raise KeyError(cell_id)


# ---------------------------------------------------------------- cell 40
CELL_40 = '''# =============================================================================
# Phase 3 setup:  trend+seasonal offset AND KNN scenarios for the training set.
#
# `mw_offset_per_t_train`  : (N_train,)        trend+seasonal at each train ts
#                            -- needed to convert the NN's stationary-space
#                               prediction back to MW for the LP oracle.
#
# `training_scenarios`     : (N_train, K, N_BUS)  KNN scenarios.
#     For each training instance i we find the K nearest neighbours in feature
#     space (excluding self), take their CAISO-MW residuals (true minus
#     trend+seasonal), and re-add the trend+seasonal at i.  This produces
#     covariate-conditional realisations of demand at timestamp i -- they
#     describe the demand distribution we'll *actually* face under any
#     stage-1 commitment.  Used for stage-2 of the stochastic LP in both
#     Option B (expected-cost SPO+) and Option C (direct decision-focused).
# =============================================================================
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- 1. Trend + seasonal offset at each training timestamp -----------------
idx_tr = train_data.index
mw_offset_per_t_train = (
    load_trend_5m.reindex(idx_tr).values
    + sum(s.reindex(idx_tr).values for s in load_seasons_5m.values())
)
print(f"mw_offset_per_t_train shape: {mw_offset_per_t_train.shape}")
print(f"  range: [{mw_offset_per_t_train.min():.1f}, {mw_offset_per_t_train.max():.1f}] MW")

# --- 2. KNN-based training scenarios ---------------------------------------
K_SCEN_TRAIN = 20

# K+1 because the first neighbour of any point is itself (zero distance).
nbrs = NearestNeighbors(n_neighbors=K_SCEN_TRAIN + 1).fit(X_tr)
_, idx_knn = nbrs.kneighbors(X_tr)
neighbor_idx = idx_knn[:, 1:]                         # (N_train, K)  drop self

# CAISO-MW residual at each training timestamp (= load - trend - seasonal).
y_caiso_resid_tr = y_tr * load_scaler.scale_[0] + load_scaler.mean_[0]   # (N_train,)

# Scenario s at instance i = trend(i) + residual from i's s-th nearest neighbour.
caiso_scens = (
    mw_offset_per_t_train[:, None]                    # (N_train, 1)
    + y_caiso_resid_tr[neighbor_idx]                  # (N_train, K)
)
caiso_scens = np.maximum(caiso_scens, 0.0)            # no negative system load

# Distribute system MW across nodes proportionally (same recipe as cell 33).
nodal_share = NODAL_BASE_MW / MEDIAN_LOAD_TRAIN       # (N_BUS,)
training_scenarios = caiso_scens[:, :, None] * nodal_share[None, None, :]   # (N_train, K, N_BUS)

print(f"training_scenarios shape: {training_scenarios.shape}")
print(f"  per-instance scenario spread (mean std of total MW): "
      f"{training_scenarios.sum(axis=2).std(axis=1).mean():.1f} MW")
'''
set_cell_source("659180d1", CELL_40)


# ---------------------------------------------------------------- cell 41
CELL_41 = '''# =============================================================================
# Subsample training data for SPO+ / direct decision-focused training.
#
# 5-minute data is highly redundant; both methods need 1-2 LP solves per
# sample, so we stratified-sample N_SPO_SAMPLES instances spanning the full
# operating range instead of training on every step.
# =============================================================================
import numpy as np
import pandas as pd

N_SPO_SAMPLES = 500
RNG_SUB = np.random.default_rng(99)

hours_tr = pd.DatetimeIndex(train_data.index).hour
unique_hours = np.unique(hours_tr)
samples_per_hour = max(1, N_SPO_SAMPLES // len(unique_hours))

sub_idx = []
for h in unique_hours:
    mask = np.where(hours_tr == h)[0]
    n_pick = min(samples_per_hour, len(mask))
    sub_idx.extend(RNG_SUB.choice(mask, size=n_pick, replace=False).tolist())

RNG_SUB.shuffle(sub_idx)
sub_idx = sorted(sub_idx[:N_SPO_SAMPLES])

X_tr_spo   = X_tr[sub_idx]
y_tr_spo   = y_tr[sub_idx]
offset_spo = mw_offset_per_t_train[sub_idx]
scen_spo   = training_scenarios[sub_idx]      # (N_SPO_SAMPLES, K, N_BUS)

print(f"SPO+ training subset: {len(sub_idx)} / {len(X_tr)} samples "
      f"({100*len(sub_idx)/len(X_tr):.1f}%)")
print(f"  Samples per hour:    ~{samples_per_hour}")
print(f"  Hour coverage:       {len(unique_hours)} hours")
print(f"  K scenarios/sample:  {scen_spo.shape[1]}")
'''
set_cell_source("8f3c87ad", CELL_41)


# ---------------------------------------------------------------- cell 42
CELL_42 = '''# =============================================================================
# Decision-focused training -- two methods sharing one training loop.
# =============================================================================
# method = "spo_plus"  (Model 3a)  Expected-cost SPO+ (Elmachtoub-Grigas)
#     Subgradient w.r.t. Pd_hat:
#         g = 2 * ( LMP*(Pd_true) - LMP*(2*Pd_hat - Pd_true) )
#     Both LP solves use the SAME stochastic LP family (KNN scenarios), so the
#     SPO+ convex-surrogate guarantee holds.  2 LP solves per sample.
#
# method = "direct"    (Model 3b)  Direct decision-focused
#     Loss   L(Pd_hat) = StochLP(Pd_hat, KNN)         [perfect-foresight term
#                                                      drops out -- constant in Pd_hat]
#     Gradient g = LMP*(Pd_hat)
#     1 LP solve per sample.  Loss is non-convex but matches the realised-cost
#     intuition: "stage-1 from prediction, stage-2 facing real scenarios."
#
# In both cases the surrogate-loss trick `(Pd_hat * g).sum().backward()` routes
# the LP-derived gradient through the affine MW conversion and into the NN
# weights (since d/dPd_hat <Pd_hat, g> = g).
#
# Skip rule: if the stage-1 slacks (shed1, curt1) are active in any LP solve
# the corresponding LMPs are pinned to the penalty constants (VOLL, c_curt)
# and carry no useful gradient -- we skip the sample.
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class SPONet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),       # predicts system load in stationary space
        )

    def forward(self, x):
        return self.net(x).squeeze()


def _slack_saturated(res, tol):
    """True if the LP's stage-1 slacks are active -> LMPs are not informative."""
    return res["shed1"].sum() + res["curt1"].sum() > tol


def _solve(Pd_np, scens):
    """Stochastic LP at demand Pd_np with KNN scenarios -- returns the result dict."""
    return build_and_solve_affine(Pd_np, scens, return_lmps=True, verbose=False)


def train_decision_focused_net(
    X_tr, y_tr_resid, trend_season_mw, scenarios,
    load_scaler, nodal_base, median_load,
    method,                 # "spo_plus" or "direct"
    epochs=10, mse_warmup=5, saturation_tol=1e-3, lr=1e-3, seed=0,
):
    """
    X_tr            : (N, p)            features
    y_tr_resid      : (N,)              true load in stationary (residual) space
    trend_season_mw : (N,)              trend+seasonal MW at each training ts
    scenarios       : (N, K, N_BUS)     KNN nodal-MW scenarios per training ts
    """
    assert method in ("spo_plus", "direct"), method
    torch.manual_seed(seed)
    net = SPONet(X_tr.shape[1])
    optimizer = optim.Adam(net.parameters(), lr=lr)

    X_tensor    = torch.tensor(X_tr, dtype=torch.float32)
    y_tensor    = torch.tensor(y_tr_resid, dtype=torch.float32)
    scaler_mean  = float(load_scaler.mean_[0])
    scaler_scale = float(load_scaler.scale_[0])
    nodal_factor_t  = torch.tensor(nodal_base / median_load, dtype=torch.float32)
    nodal_factor_np = nodal_base / median_load

    # ---------- Phase A: MSE warm-up (fast, no LP solves) ------------------
    print(f"MSE warm-up: {mse_warmup} epochs")
    mse_loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=128, shuffle=True,
    )
    net.train()
    for ep in range(mse_warmup):
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mse_loss_fn(net(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"  MSE epoch {ep+1}/{mse_warmup}  loss={ep_loss/len(X_tensor):.5f}")
    print(f"MSE warm-up complete -- switching to {method!r} loss\\n")

    # ---------- Phase B: decision-focused fine-tuning ----------------------
    net.train()
    for epoch in range(epochs):
        print(f"{method} epoch {epoch+1}/{epochs}")
        n_skipped = 0

        for i in tqdm(range(len(X_tensor))):
            x_i = X_tensor[i]

            # Forward pass: NN -> stationary -> CAISO MW -> nodal MW
            pred_stat     = net(x_i)
            pred_caiso_mw = pred_stat * scaler_scale + scaler_mean + trend_season_mw[i]
            Pd_hat        = pred_caiso_mw * nodal_factor_t                  # (N_BUS,) tensor

            # True nodal demand (numpy)
            true_caiso_mw = float(y_tr_resid[i]) * scaler_scale + scaler_mean + trend_season_mw[i]
            Pd_true_np    = true_caiso_mw * nodal_factor_np                 # (N_BUS,) ndarray
            scens_i       = scenarios[i]                                    # (K, N_BUS)

            # ---- LP oracle (no autograd) ----
            with torch.no_grad():
                Pd_hat_np = Pd_hat.numpy()
                try:
                    if method == "spo_plus":
                        res_true = _solve(Pd_true_np, scens_i)
                        if _slack_saturated(res_true, saturation_tol):
                            n_skipped += 1; continue
                        Pd_spo_np = 2.0 * Pd_hat_np - Pd_true_np
                        res_spo = _solve(Pd_spo_np, scens_i)
                        if _slack_saturated(res_spo, saturation_tol):
                            n_skipped += 1; continue
                        g_np = 2.0 * (res_true["LMP_1"] - res_spo["LMP_1"])
                    else:   # method == "direct"
                        res_pred = _solve(Pd_hat_np, scens_i)
                        if _slack_saturated(res_pred, saturation_tol):
                            n_skipped += 1; continue
                        g_np = res_pred["LMP_1"]            # = grad_{Pd_hat} L
                except Exception:
                    n_skipped += 1; continue

                g_tensor = torch.tensor(g_np, dtype=torch.float32)

            # Backprop via surrogate-loss trick: d/dPd_hat <Pd_hat, g> = g.
            optimizer.zero_grad()
            (Pd_hat * g_tensor).sum().backward()
            optimizer.step()

        print(f"  Skipped {n_skipped}/{len(X_tensor)} samples "
              f"(LP failure or stage-1 slack saturation)")

    return net


# ---- Train Model 3a: Expected-cost SPO+ -----------------------------------
print("=" * 72)
print("Model 3a -- Expected-cost SPO+ (Elmachtoub-Grigas, RHS-uncertainty form)")
print("=" * 72)
spo_model = train_decision_focused_net(
    X_tr_spo, y_tr_spo, offset_spo, scen_spo,
    load_scaler, NODAL_BASE_MW, MEDIAN_LOAD_TRAIN,
    method="spo_plus", epochs=10, mse_warmup=5,
)

# ---- Train Model 3b: Direct decision-focused ------------------------------
print("\\n" + "=" * 72)
print("Model 3b -- Direct decision-focused (gradient = LMP at Pd_hat)")
print("=" * 72)
direct_model = train_decision_focused_net(
    X_tr_spo, y_tr_spo, offset_spo, scen_spo,
    load_scaler, NODAL_BASE_MW, MEDIAN_LOAD_TRAIN,
    method="direct", epochs=10, mse_warmup=5,
)
'''
set_cell_source("73cec1b5", CELL_42)

NB.write_text(json.dumps(nb, indent=1))
print("OK -- cells 40, 41, 42 rewritten with shared training loop.")
