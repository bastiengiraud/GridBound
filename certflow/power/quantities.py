"""Observable quantities as Taylor models over the certified state (same symbols as the inputs).

Everything here keeps the correlation with the input symbols, so the quantities can be fed to a
downstream network (e.g. an RL policy that observes power-flow results) without decorrelation.
All values in per unit (multiply by grid.sn for MW / MVAr / MVA).
"""
from __future__ import annotations

import numpy as np

from ..tm import TM, add, maximum, mul, scale, sqrt, sub, matmul_sparse
from .pf import vmag2_tm


def vmag2(enc, buses):
    return vmag2_tm(enc.Vr.take(buses), enc.Vi.take(buses))


def vmag(enc, buses):
    """|V| at buses (correlated sqrt of vr^2 + vi^2)."""
    return sqrt(vmag2(enc, buses), nonneg=True)


def injections(enc, buses):
    """Net bus injections (P, Q) incl. uncertain shunts."""
    return enc.P.take(buses), enc.Q.take(buses)


def admittance_power(enc, buses, y, factor: TM = None):
    """Power ABSORBED by admittances y (pu) at ``buses``, optionally scaled by a TM factor (e.g. 1 + delta):
    S = conj(y * factor) |V|^2  ->  (P, Q)."""
    y = np.asarray(y, complex)
    V2 = vmag2(enc, buses)
    if factor is not None:
        V2 = mul(factor, V2)
    return scale(y.real, V2), scale(-y.imag, V2)


def branch_end_flows(grid, enc, branches, end="from"):
    """(P, Q) entering the branch at its ``end`` terminal (incl. the branch shunt), S = V_end conj(I_end)."""
    branches = np.asarray(branches, int)
    Y = grid.Yf if end == "from" else grid.Yt
    bus = grid.f[branches] if end == "from" else grid.t[branches]
    Ys = Y[branches]
    import scipy.sparse as sp
    Gm, Bm = sp.csr_matrix(Ys.real), sp.csr_matrix(Ys.imag)
    Ire = sub(matmul_sparse(Gm, enc.Vr), matmul_sparse(Bm, enc.Vi))
    Iim = add(matmul_sparse(Gm, enc.Vi), matmul_sparse(Bm, enc.Vr))
    vr, vi = enc.Vr.take(bus), enc.Vi.take(bus)
    P = add(mul(vr, Ire), mul(vi, Iim))
    Q = sub(mul(vi, Ire), mul(vr, Iim))
    return P, Q


def branch_apparent_power(grid, enc, branches):
    """max(|S_from|, |S_to|) per branch (pu MVA) as a TM -- e.g. the PSS/E / MATPOWER loading definition."""
    out = []
    for end in ("from", "to"):
        P, Q = branch_end_flows(grid, enc, branches, end)
        out.append(sqrt(add(mul(P, P), mul(Q, Q)), nonneg=True))
    return maximum(out[0], out[1])


def branch_current_magnitude(grid, enc, branches):
    """max(|I_from|, |I_to|) per branch (pu) as a TM."""
    branches = np.asarray(branches, int)
    import scipy.sparse as sp
    out = []
    for Y in (grid.Yf, grid.Yt):
        Ys = Y[branches]
        Gm, Bm = sp.csr_matrix(Ys.real), sp.csr_matrix(Ys.imag)
        Ire = sub(matmul_sparse(Gm, enc.Vr), matmul_sparse(Bm, enc.Vi))
        Iim = add(matmul_sparse(Gm, enc.Vi), matmul_sparse(Bm, enc.Vr))
        out.append(sqrt(add(mul(Ire, Ire), mul(Iim, Iim)), nonneg=True))
    return maximum(out[0], out[1])
