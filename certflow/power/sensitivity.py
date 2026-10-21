"""Control sensitivities via the implicit function theorem through the converged power flow.

For the control offset d = [dVg (ng, pu) ; dPg (ng, MW)] at a fixed load, one sparse factorisation of the
reduced PF Jacobian gives dz/dd = -J^-1 dr/dd, and from it the first-order change of every bus quantity.
Constraints read what they need (see Constraint.restoration_rows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pf import jac, jac_blocks


@dataclass
class ControlLinearization:
    Vm: np.ndarray      # (N,) base |V|
    th: np.ndarray      # (N,) base angle
    V: np.ndarray       # (N,) complex base voltage
    dVm: np.ndarray     # (N, 2ng) d|V| / d[dVg, dPg]
    dth: np.ndarray     # (N, 2ng) d theta / d[...]
    dP: np.ndarray      # (N, 2ng) d P_inj (pu) / d[...]
    dQ: np.ndarray      # (N, 2ng) d Q_inj (pu) / d[...]
    dV: np.ndarray      # (N, 2ng) complex dV / d[...]
    mismatch: float = 0.0


def linearize(grid, Vm, th):
    """Sensitivities at the converged state (Vm, th) of ``grid`` (columns: [Vg_1..Vg_ng, Pg_1..Pg_ng])."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    G, B = grid.G, grid.B
    ns = np.asarray(grid.ns)
    pq = np.asarray(grid.pq)
    pv = np.asarray(grid.pv)
    N, ng, sn = grid.N, grid.ng, grid.sn
    na = len(ns)
    H, Nn, M, L = jac_blocks(G, B, th, Vm)
    lu = splu(csc_matrix(jac(G, B, ns, pq, th, Vm)))
    out = {}
    for which in ("vg", "pg"):
        if which == "pg":
            R = np.zeros((na + len(pq), ng))
            R[:na, :] = -grid.Cg[ns] / sn
            dVpv = np.zeros((len(pv), ng))
        else:
            dVpv = grid.Cgv[pv]
            R = np.vstack([Nn[np.ix_(ns, pv)], L[np.ix_(pq, pv)]]) @ dVpv
        dZ = lu.solve(R)
        dTh, dVpq = dZ[:na], dZ[na:]
        dVm = np.zeros((N, ng))
        dVm[pq] = dVpq
        dVm[pv] = dVpv
        dth = np.zeros((N, ng))
        dth[ns] = dTh
        dP = H[:, ns] @ dTh + Nn[:, pq] @ dVpq
        dQ = M[:, ns] @ dTh + L[:, pq] @ dVpq
        if which == "vg":
            dP = dP + Nn[:, pv] @ dVpv
            dQ = dQ + L[:, pv] @ dVpv
        out[which] = (dVm, dth, dP, dQ)
    dVm, dth, dP, dQ = (np.hstack([out["vg"][i], out["pg"][i]]) for i in range(4))
    c, s = np.cos(th), np.sin(th)
    dV = (c[:, None] * dVm - (Vm * s)[:, None] * dth) + 1j * (s[:, None] * dVm + (Vm * c)[:, None] * dth)
    return ControlLinearization(Vm, th, Vm * np.exp(1j * th), dVm, dth, dP, dQ, dV)
