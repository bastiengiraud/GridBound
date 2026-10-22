"""Monte-Carlo validation of a certificate: sample the input set, run the DEPLOYED controller and the AC power
flow (independent polar Newton), and check every sampled quantity against the certified bands.  Also gives the
empirical (Monte-Carlo) band, e.g. to measure how tight the certified bands are."""
from __future__ import annotations

import numpy as np

from .pf import inj_pt


def point_values(ver, cons, Pg, Vg, Pd, Qd, th, Vm):
    """Per-constraint point values in the band units (mirrors Constraint.evaluate)."""
    g = ver.grid
    sn = g.sn
    P, Q = inj_pt(g.G, g.B, th, Vm)
    Pdb, Qdb = g.Cl @ Pd / sn, g.Cl @ Qd / sn
    V = Vm * np.exp(1j * th)
    out = {}
    for c in cons:
        n = c.name
        if n == "V":
            out[n] = Vm[c.elements]
        elif n == "Pg":
            out[n] = Pg / sn
        elif n == "Qg":
            out[n] = Q[c.elements] + Qdb[c.elements]
        elif n == "slack":
            s = c.slk
            out[n] = np.array([P[s] + Pdb[s], Q[s] + Qdb[s]])
        elif n == "branch":
            out[n] = np.maximum(np.abs(c.Yf @ V) ** 2, np.abs(c.Yt @ V) ** 2)
        elif n == "angle":
            out[n] = th[c.f] - th[c.t]
        elif hasattr(c, "point"):
            out[n] = c.point(ver, dict(Pg=Pg, Vg=Vg, Pd=Pd, Qd=Qd, th=th, Vm=Vm, P=P, Q=Q, V=V))
    return out


def monte_carlo(ver, cert, n=200, seed=0, corners=0.25, offset=None, tol=1e-9):
    """Sample ``n`` points of the certificate's e-box (a fraction ``corners`` at vertices) and check that every
    converged sample lies inside every certified band.  Returns dict with per-constraint worst excursion
    (<= tol means sound), the empirical min/max per element, coverage counts and non-converged count."""
    lo, hi = cert.ebox
    rng = np.random.default_rng(seed)
    E = rng.uniform(lo, hi, (n, len(lo)))
    k = int(round(corners * n))
    if k:
        E[:k] = np.where(rng.random((k, len(lo))) < 0.5, lo, hi)
    off = cert.offset if offset is None else offset
    worst = {c.name: -np.inf for c in cert.constraints}
    emp = {}
    nconv = 0
    for e in E:
        x = ver.U.point(e)
        Pd, Qd = x[: ver.nl], x[ver.nl:]
        Pg, Vg = ver.ctrl.forward(x, off)
        th, Vm, mis = ver.centre_pf(Pg, Vg, Pd, Qd, iters=100)
        if mis > 1e-8:
            continue
        nconv += 1
        vals = point_values(ver, cert.constraints, Pg, Vg, Pd, Qd, th, Vm)
        for name, v in vals.items():
            b = cert.bands.get(name)
            if b is None or not np.size(v):
                continue
            ex = float(np.max(np.maximum(b.lo - v, v - b.hi)))
            worst[name] = max(worst[name], ex)
            lo_, hi_ = emp.get(name, (np.full(np.shape(v), np.inf), np.full(np.shape(v), -np.inf)))
            emp[name] = (np.minimum(lo_, v), np.maximum(hi_, v))
    sound = all(w <= tol for w in worst.values() if np.isfinite(w))
    return dict(sound=sound, worst=worst, empirical=emp, n_converged=nconv, n=n)
