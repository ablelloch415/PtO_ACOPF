"""
DC OPF Predict+Optimize — Hu, Lee & Lee (AAAI 2023)
Linear predictor | scipy LP solver | Lemma 2 gradient (no PyTorch)
"""
import numpy as np
from scipy.optimize import linprog

# ── 3-bus system ──────────────────────────────────────────────────────────────
B      = np.array([[15.,-10.,-5.], [-10.,18.,-8.], [-5.,-8.,13.]], float)
COST   = np.array([25., 35.])          # linear gen cost [$/MWh]
PG_MIN = np.array([0.1, 0.0])
PG_MAX = np.array([2.0, 1.5])
PD_NOM = np.array([0.0, 0.8, 1.2])    # nominal loads per bus

# ── DC OPF: scipy LP ──────────────────────────────────────────────────────────
# Variables x = [Pg0, Pg1, th1, th2]  (th0 = 0, slack bus)
# Power balance at each bus:  Pg_bus_i - sum_j B[i,j]*th_j = Pd_i
# With th0 = 0: Pg_bus_i - B[i,1]*th1 - B[i,2]*th2 = Pd_i
_A_eq = np.array([
    [1, 0, -B[0,1], -B[0,2]],   # bus 0: Pg0 present
    [0, 1, -B[1,1], -B[1,2]],   # bus 1: Pg1 present
    [0, 0, -B[2,1], -B[2,2]],   # bus 2: no generator
])
_c    = [COST[0], COST[1], 0., 0.]
_bnds = [(PG_MIN[0], PG_MAX[0]), (PG_MIN[1], PG_MAX[1]),
         (None, None), (None, None)]

def solve_dcopf(Pd: np.ndarray) -> np.ndarray:
    """Returns optimal Pg [shape 2]."""
    res = linprog(_c, A_eq=_A_eq, b_eq=Pd, bounds=_bnds, method='highs')
    return res.x[:2] if res.success else (PG_MIN + PG_MAX) / 2

def gen_cost(Pg: np.ndarray) -> float:
    return float(COST @ Pg)

# ── Post-hoc correction function (Eq. 10, covering LP version) ───────────────
def correct(Pg_hat: np.ndarray, Pd_true: np.ndarray) -> np.ndarray:
    """
    Scale Pg up by λ ≥ 1 so total supply meets total true demand.
    λ = max{λ ≥ 1 | scaled dispatch stays within Pg_max}.
    """
    lam = max(1.0, Pd_true[1:].sum() / (Pg_hat.sum() + 1e-8))
    lam = min(lam, float(PG_MAX.sum()))
    return np.clip(lam * Pg_hat, PG_MIN, PG_MAX)

# ── Lemma 2: ∂Pg*/∂Pd ────────────────────────────────────────────────────────
def sensitivity(Pg: np.ndarray, mu: float = 1e-3, pen: float = 80.0) -> np.ndarray:
    """
    Approximates ∂Pg*/∂Pd using the log-barrier Hessian (Lemma 2).

    At the interior-point optimum, the implicit function theorem gives:
        H · ∂Pg*/∂Pd + C^T = 0  →  ∂Pg*/∂Pd = −H⁻¹ C^T

    H  = ∂²f/∂Pg²  : barrier terms + soft balance penalty  (2×2 diagonal)
    C  = ∂²f/∂Pd∂Pg: coupling through balance penalty       (3×2)
    """
    H = np.diag(  mu / (Pg - PG_MIN + 1e-8)**2      # lower barrier curvature
                + mu / (PG_MAX - Pg  + 1e-8)**2      # upper barrier curvature
                + 2 * pen)                            # from penalty (Pg_bus - Pd)²

    # ∂²f/∂Pd_i ∂Pg_j = -2·pen where gen j sits on bus i
    C = np.zeros((3, 2))
    C[0, 0] = -2 * pen    # gen 0 at bus 0
    C[1, 1] = -2 * pen    # gen 1 at bus 1
    # bus 2 has no generator → C[2,:] = 0

    return -np.linalg.inv(H) @ C.T   # shape (2, 3)

# ── Linear predictor: Pd_hat = W @ features + bias ───────────────────────────
def predict(W: np.ndarray, bias: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.clip(A @ W.T + bias, 0.05, 2.5)   # shape (n, 3)

# ── Training ──────────────────────────────────────────────────────────────────
def train(A_tr, Pd_tr, use_regret: bool,
          sigma: float = 0.5, epochs: int = 60, lr: float = 5e-3, bs: int = 32):

    W    = np.zeros((3, A_tr.shape[1]))
    bias = PD_NOM.copy()

    for ep in range(1, epochs + 1):
        perm = np.random.permutation(len(A_tr))
        for i in range(0, len(A_tr), bs):
            idx    = perm[i:i+bs]
            A_b    = A_tr[idx]
            Pd_b   = Pd_tr[idx]
            Pd_hat = predict(W, bias, A_b)   # (bs, 3)

            if not use_regret:
                # ── MSE baseline ──────────────────────────────────────────────
                # Standard supervised learning: no LP involved in gradient
                err   = Pd_hat - Pd_b                      # (bs, 3)
                W    -= lr * 2 * (err.T @ A_b) / len(idx)  # (3, n_feat)
                bias -= lr * 2 * err.mean(0)               # (3,)

            else:
                # ── IntOpt-C: post-hoc regret gradient ───────────────────────
                # Full chain rule (Eq. 8):
                #   ∂Regret/∂W = ∂Regret/∂Pg* · ∂Pg*/∂Pd_hat · ∂Pd_hat/∂W
                #                 (from cost)     (Lemma 2)       (= features)
                dW    = np.zeros_like(W)
                dbias = np.zeros_like(bias)

                for j in range(len(idx)):
                    Pg_hat  = solve_dcopf(Pd_hat[j])
                    Pg_corr = correct(Pg_hat, Pd_b[j])

                    # ∂Regret/∂Pg_hat  (gradient of Eq. 6 w.r.t. Pg_hat)
                    # cost(Pg_corr) term: Pg_corr = λ·Pg_hat → contributes λ·COST
                    # penalty term:      σ·COST·sign(Pg_hat - Pg_corr)
                    lam    = np.clip(Pd_b[j, 1:].sum() / (Pg_hat.sum() + 1e-8),
                                     1.0, float(PG_MAX.sum()))
                    dR_dPg = lam * COST + sigma * COST * np.sign(Pg_hat - Pg_corr)

                    # Lemma 2: ∂Pg*/∂Pd_hat  →  (2, 3)
                    # Chain rule: ∂Regret/∂Pd_hat = dR_dPg @ (∂Pg*/∂Pd)  →  (3,)
                    dR_dPd  = dR_dPg @ sensitivity(Pg_hat)

                    # ∂Pd_hat/∂W = features → outer product gives (3, n_feat)
                    dW    += np.outer(dR_dPd, A_b[j]) / len(idx)
                    dbias += dR_dPd / len(idx)

                W    -= lr * dW
                bias -= lr * dbias

        if ep % 20 == 0:
            print(f"  ep {ep:3d}/{epochs}")

    return W, bias

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(W, bias, A_te, Pd_te):
    Pd_hat = predict(W, bias, A_te)
    regret = np.mean([
        max(0., gen_cost(solve_dcopf(Pd_hat[i])) - gen_cost(solve_dcopf(Pd_te[i])))
        for i in range(len(A_te))
    ])
    mse = np.mean((Pd_hat - Pd_te) ** 2)
    return regret, mse

# ── Synthetic data ────────────────────────────────────────────────────────────
def make_data(n: int, n_feat: int = 6, seed: int = 0):
    rng = np.random.RandomState(seed)
    A   = rng.randn(n, n_feat).astype(float)
    Pd  = np.clip(
        PD_NOM + A @ (rng.randn(n_feat, 3) * 0.04) + rng.randn(n, 3) * 0.07,
        0.05, 2.2
    )
    return A, Pd

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    A_tr, Pd_tr = make_data(500, seed=0)
    A_te, Pd_te = make_data(150, seed=1)

    results = {}
    for label, use_regret, sigma in [
        ("MSE baseline",    False, 0.0),
        ("IntOpt-C  σ=0",   True,  0.0),
        ("IntOpt-C  σ=0.5", True,  0.5),
    ]:
        print(f"\n── {label} ──")
        W, bias = train(A_tr, Pd_tr, use_regret, sigma)
        results[label] = evaluate(W, bias, A_te, Pd_te)

    print("\n" + "=" * 50)
    print(f"  {'Method':<20}  {'Regret':>10}  {'MSE':>10}")
    print("  " + "-" * 46)
    for label, (regret, mse) in results.items():
        print(f"  {label:<20}  {regret:>10.4f}  {mse:>10.4f}")
    print()
    print("  IntOpt-C should have lower regret than MSE,")
    print("  at the cost of higher MSE (per paper Figs 1-5).")