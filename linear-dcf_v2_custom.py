import numpy as np
from scipy.optimize import linprog

# ── 3-bus system ──────────────────────────────────────────────────────────────
B      = np.array([[15.,-10.,-5.], [-10.,18.,-8.], [-5.,-8.,13.]], float)
COST   = np.array([25., 35.])          # linear gen cost [$/MWh]
PG_MIN = np.array([0.1, 0.0])
PG_MAX = np.array([2.0, 1.5])
PD_NOM = np.array([0.0, 0.8, 1.2])    # nominal loads per bus

# ── NEW: AGC Participation Factors ────────────────────────────────────────────
# Represents the physical ramping distribution (must sum to 1.0)
# Gen 0 handles 60% of the imbalance, Gen 1 handles 40%
A_G = np.array([0.6, 0.4]) 

# ── DC OPF: scipy LP ──────────────────────────────────────────────────────────
_A_eq = np.array([
    [1, 0, -B[0,1], -B[0,2]],   
    [0, 1, -B[1,1], -B[1,2]],   
    [0, 0, -B[2,1], -B[2,2]],   
])
_c    = [COST[0], COST[1], 0., 0.]
_bnds = [(PG_MIN[0], PG_MAX[0]), (PG_MIN[1], PG_MAX[1]),
         (None, None), (None, None)]

def solve_dcopf(Pd: np.ndarray) -> np.ndarray:
    res = linprog(_c, A_eq=_A_eq, b_eq=Pd, bounds=_bnds, method='highs')
    return res.x[:2] if res.success else (PG_MIN + PG_MAX) / 2

def gen_cost(Pg: np.ndarray) -> float:
    return float(COST @ Pg)

# ── MODIFIED: AGC Additive Correction ─────────────────────────────────────────
def correct(Pg_hat: np.ndarray, Pd_true: np.ndarray) -> np.ndarray:
    """
    Adjust Pg_hat using additive AGC participation factors (A_G) based on 
    the total system imbalance (delta).
    """
    # delta > 0 means under-prediction (need more power)
    # delta < 0 means over-prediction (need to ramp down)
    delta = Pd_true[1:].sum() - Pg_hat.sum()
    
    # Additive scaling based on physical participation
    Pg_corr = Pg_hat + A_G * delta
    
    return np.clip(Pg_corr, PG_MIN, PG_MAX)

# ── Lemma 2: ∂Pg*/∂Pd ────────────────────────────────────────────────────────
def sensitivity(Pg: np.ndarray, mu: float = 1e-3, pen: float = 80.0) -> np.ndarray:
    H = np.diag(  mu / (Pg - PG_MIN + 1e-8)**2      
                + mu / (PG_MAX - Pg  + 1e-8)**2      
                + 2 * pen)                            
    C = np.zeros((3, 2))
    C[0, 0] = -2 * pen    
    C[1, 1] = -2 * pen    
    return -np.linalg.inv(H) @ C.T   

# ── Linear predictor ─────────────────────────────────────────────────────────
def predict(W: np.ndarray, bias: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.clip(A @ W.T + bias, 0.05, 2.5)   

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
            Pd_hat = predict(W, bias, A_b)

            if not use_regret:
                err   = Pd_hat - Pd_b                      
                W    -= lr * 2 * (err.T @ A_b) / len(idx)  
                bias -= lr * 2 * err.mean(0)               
            else:
                dW    = np.zeros_like(W)
                dbias = np.zeros_like(bias)

                for j in range(len(idx)):
                    Pg_hat  = solve_dcopf(Pd_hat[j])
                    Pg_corr = correct(Pg_hat, Pd_b[j])

                    # Total system imbalance
                    delta = Pd_b[j, 1:].sum() - Pg_hat.sum()

                    # ── MODIFIED: Regret Gradient for AGC Formulation ──
                    # 1. Gradient of the economic base cost: C^T(Pg_hat + A_G*delta)
                    avg_agc_cost = np.dot(COST, A_G)
                    dBase_dPg = COST - avg_agc_cost 

                    # 2. Gradient of your specific Penalty: sum(A_G * C * |Pg_hat - Pg_corr|)
                    # Since |Pg_hat - Pg_corr| = A_G * |delta|, 
                    # Penalty = sum(C * A_G^2) * |delta|
                    penalty_weight = np.sum((A_G ** 2) * COST)
                    dPen_dPg = -sigma * penalty_weight * np.sign(delta)

                    dR_dPg = dBase_dPg + dPen_dPg

                    # Chain backward through Lemma 2
                    dR_dPd  = dR_dPg @ sensitivity(Pg_hat)

                    dW    += np.outer(dR_dPd, A_b[j]) / len(idx)
                    dbias += dR_dPd / len(idx)

                W    -= lr * dW
                bias -= lr * dbias

        if ep % 20 == 0:
            print(f"  ep {ep:3d}/{epochs}")

    return W, bias

# ── MODIFIED: Evaluation ──────────────────────────────────────────────────────
def evaluate(W, bias, A_te, Pd_te):
    Pd_hat = predict(W, bias, A_te)
    regret = []
    
    # Updated to evaluate the actual cost of the *corrected* generation
    for i in range(len(A_te)):
        Pg_hat = solve_dcopf(Pd_hat[i])
        Pg_corr = correct(Pg_hat, Pd_te[i])
        Pg_opt = solve_dcopf(Pd_te[i])
        
        actual_cost = gen_cost(Pg_corr)
        opt_cost = gen_cost(Pg_opt)
        regret.append(max(0., actual_cost - opt_cost))
        
    mse = np.mean((Pd_hat - Pd_te) ** 2)
    return np.mean(regret), mse

# ── Synthetic data & Main ─────────────────────────────────────────────────────
def make_data(n: int, n_feat: int = 6, seed: int = 0):
    rng = np.random.RandomState(seed)
    A   = rng.randn(n, n_feat).astype(float)
    Pd  = np.clip(
        PD_NOM + A @ (rng.randn(n_feat, 3) * 0.04) + rng.randn(n, 3) * 0.07,
        0.05, 2.2
    )
    return A, Pd

if __name__ == "__main__":
    np.random.seed(42)
    A_tr, Pd_tr = make_data(500, seed=0)
    A_te, Pd_te = make_data(150, seed=1)

    results = {}
    for label, use_regret, sigma in [
        ("MSE baseline",    False, 0.0),
        ("IntOpt-C  sigma=0",   True,  0.0),
        ("IntOpt-C  sigma=0.5", True,  0.5),
    ]:
        print(f"\n── {label} ──")
        W, bias = train(A_tr, Pd_tr, use_regret, sigma)
        results[label] = evaluate(W, bias, A_te, Pd_te)

    print("\n" + "=" * 50)
    print(f"  {'Method':<20}  {'Regret':>10}  {'MSE':>10}")
    print("  " + "-" * 46)
    for label, (regret, mse) in results.items():
        print(f"  {label:<20}  {regret:>10.4f}  {mse:>10.4f}")