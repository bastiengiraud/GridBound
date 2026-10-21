import numpy as np

from certflow.nn import Linear, MinMaxDenormalize, MinMaxNormalize, ReLU, Sequential, Tanh
from certflow.power import DispatchController, PFVerifier, case9, default_constraints, from_matpower
from certflow.sets import low_rank_load_set


def dispatch_net(grid, act="relu", width=24, seed=0, nsamp=2000):
    """Small dispatch NN for tests: random hidden layers, output layer least-squares fitted to
    Pg = nominal * total-load ratio, Vg = nominal + 0.02 * (ratio - 1)."""
    rng = np.random.default_rng(seed)
    Pd0, Qd0 = grid.names["load_pq"]
    pg0, vg0 = grid.names["gen_pg"], grid.names["gen_vg"]
    x0 = np.r_[Pd0, Qd0]
    X = x0 * rng.uniform(0.8, 1.2, (nsamp, 1)) * (1 + 0.05 * rng.standard_normal((nsamp, len(x0))))
    ratio = X[:, : len(Pd0)].sum(1) / Pd0.sum()
    Y = np.hstack([np.outer(ratio, pg0), vg0[None, :] + 0.02 * (ratio[:, None] - 1)])
    xmn, xmx, ymn, ymx = X.min(0), X.max(0), Y.min(0), Y.max(0)
    A = {"relu": ReLU, "tanh": Tanh}[act]
    layers = [MinMaxNormalize(xmn, xmx)]
    H = (X - xmn) / (xmx - xmn + 1e-8)
    din = X.shape[1]
    for _ in range(2):
        W = rng.normal(0, np.sqrt(2.0 / din), (width, din))
        b = rng.normal(0, 0.1, width)
        layers += [Linear(W, b), A()]
        H = A().forward(H @ W.T + b)
        din = width
    Yn = (Y - ymn) / (ymx - ymn + 1e-8)
    sol = np.linalg.lstsq(np.hstack([H, np.ones((nsamp, 1))]), Yn, rcond=None)[0]
    layers += [Linear(sol[:-1].T, sol[-1]), MinMaxDenormalize(ymn, ymx)]
    return Sequential(layers)


def case9_verifier(frac=0.05, act="relu", grid=None, constraints=None, K=3, seed=0):
    grid = grid or from_matpower(case9())
    Pd0, Qd0 = grid.names["load_pq"]
    x0 = np.r_[Pd0, Qd0]
    rng = np.random.default_rng(42)
    Gd = np.hstack([np.ones((len(x0), 1)), rng.standard_normal((len(x0), K - 1))])
    Gd /= np.abs(Gd).sum(1).mean()
    ctrl = DispatchController(dispatch_net(grid, act, seed=seed), grid.ng, vg_clip=(0.9, 1.1))
    U = low_rank_load_set(x0, Gd, frac)
    return PFVerifier(grid, ctrl, U, constraints if constraints is not None else default_constraints(grid))
