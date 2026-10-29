"""Certified AC power flow for UNCERTAIN SPECIFICATIONS (no controller in the loop).

    enclose_pf(grid, P_spec(e), Q_spec(e), V_set(e), [shunt(e)])  ->  PFEnclosure

proves (Krawczyk) that for every e the power flow has a unique solution near the linearisation point and
returns the state as Taylor models over the same symbols.  The controller verifier (PFVerifier) and the
policy (RL) verification both build on it; :mod:`certflow.power.quantities` turns the enclosure into
observable quantities (|V|, injections, branch flows, shunt powers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..krawczyk import krawczyk
from ..tm import TM, bounds, concat, mul
from .pf import RectPFSystem, Shunt, newton


@dataclass
class PFEnclosure:
    ok: bool
    rho: float
    score: np.ndarray            # per-symbol residual mass (split heuristic)
    th_s: np.ndarray             # linearisation point
    Vm_s: np.ndarray
    system: RectPFSystem = None
    kr: object = None
    Vr: TM = None                # full-bus rectangular voltage enclosure
    Vi: TM = None
    P: TM = None                 # full-bus net injections (pu), including uncertain shunts
    Q: TM = None
    mismatch: float = 0.0


def centre_newton(grid, Psp, Qsp, Vpin, warm, shunt: Shunt = None, iters=60):
    """Point PF (polar Newton) at fixed specs; uncertain shunts at their centre value."""
    G, B = (grid.G, grid.B) if shunt is None else shunt.effective_GB(grid.G, grid.B)
    return newton(G, B, grid.ns, grid.pq, grid.pv, Psp, Qsp, Vpin, warm[0], warm[1], iters=iters)


def enclose_pf(grid, P_spec: TM, Q_spec: TM, V_set: TM, warm, shunt: Shunt = None, centre_tol=1e-6,
               corr_max_m=1700, iters=60, centre_iters=2) -> PFEnclosure:
    """Certify the PF over the uncertainty carried by the TMs.

    P_spec, Q_spec : full-bus (N rows) net specified injections in pu (generation - constant-power load)
    V_set          : PV-bus voltage-magnitude setpoints (rows in grid.pv order)
    warm           : (theta, |V|) Newton start for the linearisation point
    shunt          : optional uncertain shunt admittances (e.g. admittance loads)
    centre_iters   : Newton steps (in Taylor-model arithmetic) for the Krawczyk centre function: 1 = affine centre
                     (as in the paper), 2 = quadratic centre -- a much smaller remainder, which matters for quantities
                     that amplify it (e.g. currents of short lines with a large admittance)"""
    ns, pq, pv = grid.ns, grid.pq, grid.pv
    Vpin = np.zeros(grid.N)
    Vpin[pv] = V_set[0]
    th_s, Vm_s, mis = centre_newton(grid, P_spec[0], Q_spec[0], Vpin, warm, shunt, iters)
    enc = PFEnclosure(False, np.inf, None, th_s, Vm_s, mismatch=mis)
    if mis > centre_tol or (len(pq) and Vm_s[pq].min() < 0.5):
        return enc
    S = concat([P_spec.take(ns), Q_spec.take(pq), mul(V_set, V_set)])
    sysm = RectPFSystem(grid.G, grid.B, ns, pq, pv, Vm_s * np.cos(th_s), Vm_s * np.sin(th_s), shunt=shunt)
    try:
        kr = krawczyk(sysm, S, sysm.z0, corr_max_m=corr_max_m, centre_iters=centre_iters)
    except np.linalg.LinAlgError:                    # singular linearisation (e.g. voltage collapse): not certified
        enc.system, enc.score = sysm, np.ones(S.C.shape[-1])
        return enc
    enc.system, enc.kr, enc.ok, enc.rho, enc.score = sysm, kr, kr.ok, kr.rho, kr.score
    if kr.ok:
        enc.Vr, enc.Vi = sysm.state(kr.enclosure())
        enc.P, enc.Q = sysm.injections(enc.Vr, enc.Vi)
    return enc


def point_pf(grid, Pspec, Qspec, Vpin, warm, shunt_d=None, shunt: Shunt = None, iters=100):
    """Plain point PF (for Monte-Carlo validation): shunt values ``shunt_d`` (default: centre)."""
    G, B = (grid.G, grid.B) if shunt is None else shunt.effective_GB(grid.G, grid.B, shunt_d)
    return newton(G, B, grid.ns, grid.pq, grid.pv, Pspec, Qspec, Vpin, warm[0], warm[1], iters=iters)


def enclose_pf_split(grid, spec_fn, lo, hi, reduce=None, max_leaves=64, centre_tol=1e-6):
    """Certified PF over an e-box with adaptive splitting (bisect where Krawczyk fails).

    spec_fn(grid, lo, hi) -> dict(P=TM, Q=TM, V=TM, shunt=Shunt or None, warm=(theta, |V|))  for the sub-box
    reduce(enc, lo, hi)   -> compact payload kept per leaf (default: the PFEnclosure itself)
    Returns (leaves, ok): payloads of the certified leaves; ok False if the leaf cap was hit (abstain)."""
    from ..split import adaptive_split, bisect
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    K = len(lo)

    def leaf(a, b):
        s = spec_fn(grid, a, b)
        enc = enclose_pf(grid, s["P"], s["Q"], s["V"], s["warm"], shunt=s.get("shunt"), centre_tol=centre_tol)
        if not enc.ok:
            sc = enc.score[:K] if enc.score is not None else None
            return False, None, sc
        return True, (reduce(enc, a, b) if reduce else enc), None

    def choose(a, b, score):
        s = np.ones(K) if score is None else np.asarray(score, float)
        return list(bisect(a, b, int(np.argmax(s * (b - a)))))

    return adaptive_split(leaf, lo, hi, choose, max_leaves=max_leaves)
