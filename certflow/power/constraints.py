"""Pluggable operational constraints.

A constraint turns a certified leaf (state enclosure) into a :class:`Band` (rigorous per-element range), and
knows its limits, its acceptance check, its split sensitivity and its rows in the restoration QP:

    evaluate(leaf)                    -> Band(lo, hi, sens)         (sens = |d q / d e_j| per input direction)
    ok(band)                          -> bool                       (within limits up to its tolerance)
    infeasibility(band)               -> float  (>= 0, pu yardstick; sums across constraints)
    width_excess(band)                -> float  (> 0: the range is wider than its window -> an offset cannot fix it)
    split_candidates(band)            -> [(violation, weight_vector_K), ...]
    restoration_rows(band, lin, ctx)  -> [RowBlock]                 (linearised in the control offset d)
    predict_infeasibility(band, lin, d)

Adding a new limit family (transformer loading, voltage unbalance, ...) = adding one subclass.
Units: voltage pu, P/Q pu (sn base), branch |I|^2 pu^2, angles rad.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fp import CONFIG, odn, oup
from ..tm import TM, add, arad, bounds, dev, div, matmul, matmul_sparse, mul, sub
from .pf import vmag2_tm

BIG = 1e9


def _fin(x, big=BIG):
    return np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=big, neginf=-big)


@dataclass
class Band:
    lo: np.ndarray
    hi: np.ndarray
    sens: np.ndarray = None            # (n, K) split weights |d q/d e_j| (optional)
    extra: dict = field(default_factory=dict)

    @staticmethod
    def union(bands):
        bands = [b for b in bands if b is not None]
        if not bands:
            return None
        lo = np.min([b.lo for b in bands], 0)
        hi = np.max([b.hi for b in bands], 0)
        ss = [b.sens for b in bands if b.sens is not None]
        sens = np.max(ss, 0) if ss else None
        return Band(lo, hi, sens)


@dataclass
class RowBlock:
    """Linear constraint block  A d <= b  on the control step d = [dVg (ng, pu); dPg (ng, MW)]."""
    A: np.ndarray
    b: np.ndarray
    window: np.ndarray                 # per-row limit window (row scaling -> fraction of window)
    hard: bool = False
    weight: float = 1.0                # slack penalty priority (soft rows)


class Constraint:
    name = "constraint"
    unit = ""

    def __init__(self, tol=0.0):
        self.tol = float(tol)
        self.lim_lo = None
        self.lim_hi = None
        self.elements = None           # element ids (bus / gen / branch indices)

    # -- to implement ----------------------------------------------------------------------------------
    def evaluate(self, leaf) -> Band:
        raise NotImplementedError

    def restoration_rows(self, band, lin, ctx):
        return []

    def predict_shift(self, band, lin, d):
        """Linear prediction of the band centre shift for control step d (None = not predictable)."""
        return None

    # -- generic -----------------------------------------------------------------------------------------
    def excess(self, band):
        """Per-element amount outside the limits (<= 0 inside)."""
        return np.maximum(self.lim_lo - band.lo, band.hi - self.lim_hi)

    def ok(self, band) -> bool:
        lo_ok = band.lo >= oup(oup(self.lim_lo) - self.tol)
        hi_ok = band.hi <= odn(odn(self.lim_hi) + self.tol)
        return bool(lo_ok.all() and hi_ok.all())

    def infeasibility(self, band) -> float:
        return max(0.0, float(self.excess(band).max())) if band.lo.size else 0.0

    def width_excess(self, band) -> float:
        if not band.lo.size:
            return -np.inf
        return float(((band.hi - band.lo) - (self.lim_hi - self.lim_lo)).max())

    def split_candidates(self, band):
        if band.sens is None:
            return []
        v = np.maximum(self.excess(band) - self.tol, 0.0)
        return [(float(v[k]), band.sens[k]) for k in np.where(v > 1e-9)[0]]

    def predict_infeasibility(self, band, lin, d):
        sh = self.predict_shift(band, lin, d)
        if sh is None:
            return self.infeasibility(band)
        return self.infeasibility(Band(band.lo + sh, band.hi + sh))

    def describe(self, band, sn=1.0):
        ex = self.excess(band) if band.lo.size else np.zeros(0)
        k = int(np.argmax(ex)) if ex.size else -1
        return dict(name=self.name, ok=self.ok(band), worst=(float(ex[k]) if k >= 0 else 0.0),
                    element=(int(self.elements[k]) if k >= 0 else None))


# ==================================================================================================
# voltage magnitude
# ==================================================================================================

class VoltageMagnitude(Constraint):
    """|V| in [vmin, vmax] at PQ buses (from the certified state, correlated |V|^2) and PV buses (= the
    clipped setpoint range, exact).  Default elements: pq + pv buses (slack |V| is fixed)."""
    name = "V"
    unit = "pu"

    def __init__(self, grid, vmin, vmax, tol=1e-4, buses=None):
        super().__init__(tol)
        pq, pv = list(grid.pq), list(grid.pv)
        self.pq_buses = pq if buses is None else [b for b in pq if b in set(buses)]
        self.pv_buses = pv if buses is None else [b for b in pv if b in set(buses)]
        self.elements = np.array(self.pq_buses + self.pv_buses, int)
        vmin = np.broadcast_to(np.asarray(vmin, float), (grid.N,))
        vmax = np.broadcast_to(np.asarray(vmax, float), (grid.N,))
        self.lim_lo = vmin[self.elements].copy()
        self.lim_hi = vmax[self.elements].copy()

    def evaluate(self, leaf):
        K = leaf.K
        pq = self.pq_buses
        V2 = vmag2_tm(leaf.Vr.take(pq), leaf.Vi.take(pq))
        lo2, hi2 = dev(V2)
        vlo = odn(np.sqrt(np.maximum(odn(V2[0] + lo2), 0.0)))
        vhi = oup(np.sqrt(oup(V2[0] + hi2)))
        sens_pq = np.abs(V2[1][:, :K]) / (2.0 * np.maximum(leaf.Vm_s[pq], 1e-3)[:, None])
        Vpr = matmul(leaf.grid.Cgv, leaf.y_raw.take(slice(leaf.ng, 2 * leaf.ng)))     # pre-clip PV setpoints
        pvb = self.pv_buses
        vgc = Vpr[0][pvb]
        vgr = arad(Vpr)[pvb]
        clo, chi = leaf.vg_clip
        vpvlo = np.clip(odn(vgc - vgr), clo, chi)
        vpvhi = np.clip(oup(vgc + vgr), clo, chi)
        sens_pv = np.abs(Vpr[1][pvb, :K])
        return Band(np.concatenate([vlo, vpvlo]), np.concatenate([vhi, vpvhi]), np.vstack([sens_pq, sens_pv]))

    def infeasibility(self, band):
        if not band.lo.size:
            return 0.0
        return max(0.0, float((self.lim_lo - band.lo).max()) - self.tol, float((band.hi - self.lim_hi).max()) - self.tol)

    def ok(self, band):
        viol = max(float(oup(self.lim_lo - band.lo).max()), float(oup(band.hi - self.lim_hi).max()))
        return viol <= self.tol

    def restoration_rows(self, band, lin, ctx):
        S = lin.dVm[self.elements]
        Vc = 0.5 * (band.lo + band.hi)
        w = 0.5 * (band.hi - band.lo)
        win = self.lim_hi - self.lim_lo
        return [RowBlock(-S, Vc - (self.lim_lo + w), win), RowBlock(S, (self.lim_hi - w) - Vc, win)]

    def predict_shift(self, band, lin, d):
        return lin.dVm[self.elements] @ d


# ==================================================================================================
# generator active / reactive power, slack
# ==================================================================================================

class GenActivePower(Constraint):
    """Pg in [Pmin, Pmax] per generator (Pg is the dispatch itself -> exact NN reachable set).  Hard rows in
    the restoration (actuator range)."""
    name = "Pg"
    unit = "pu"

    def __init__(self, grid, tol=1e-3):
        super().__init__(tol)
        self.sn = grid.sn
        self.elements = np.arange(grid.ng)
        self.lim_lo = _fin(grid.gen_pmin) / grid.sn
        self.lim_hi = _fin(grid.gen_pmax) / grid.sn
        self.ng = grid.ng

    def evaluate(self, leaf):
        Pg = leaf.y.take(slice(0, leaf.ng))
        lo, hi = bounds(Pg)
        sn = self.sn
        return Band(odn(lo / sn), oup(hi / sn), np.abs(Pg[1][:, : leaf.K]) / sn)

    def restoration_rows(self, band, lin, ctx):
        ng, sn = self.ng, self.sn
        I = np.eye(ng)
        Z = np.zeros((ng, ng))
        b0 = band.lo * sn - self.lim_lo * sn
        b1 = self.lim_hi * sn - band.hi * sn
        b0, b1 = (np.where(b > -self.tol * sn, np.maximum(b, 0.0), b) for b in (b0, b1))
        win = np.maximum(ctx.pg_range, 1.0)
        return [RowBlock(np.hstack([Z, -I]), b0, win, hard=True), RowBlock(np.hstack([Z, I]), b1, win, hard=True)]

    def predict_shift(self, band, lin, d):
        return d[self.ng:] / self.sn


class GenReactivePower(Constraint):
    """Generator reactive OUTPUT at PV buses = Q injection + bus load (correlated TM sum), in [Qmin, Qmax]
    (limits summed over the bus's generators)."""
    name = "Qg"
    unit = "pu"

    def __init__(self, grid, tol=1e-3):
        super().__init__(tol)
        self.sn = grid.sn
        self.elements = np.array(grid.pv, int)
        qmn = grid.Cg @ (_fin(grid.gen_qmin) / grid.sn)
        qmx = grid.Cg @ (_fin(grid.gen_qmax) / grid.sn)
        self.lim_lo = qmn[self.elements]
        self.lim_hi = qmx[self.elements]

    def evaluate(self, leaf):
        Qg = add(leaf.Q.take(self.elements), leaf.qd_bus_pu.take(self.elements))
        lo, hi = bounds(Qg)
        return Band(lo, hi, np.abs(Qg[1][:, : leaf.K]))

    def restoration_rows(self, band, lin, ctx):
        S = lin.dQ[self.elements]
        win = np.maximum(self.lim_hi - self.lim_lo, 1e-3)
        return [RowBlock(-S, band.lo - self.lim_lo, win), RowBlock(S, self.lim_hi - band.hi, win)]

    def predict_shift(self, band, lin, d):
        return lin.dQ[self.elements] @ d


class SlackPower(Constraint):
    """Slack active and reactive output (injection + bus load) within the slack limits.  Elements: [P, Q]."""
    name = "slack"
    unit = "pu"

    def __init__(self, grid, tol=1e-3):
        super().__init__(tol)
        self.sn = grid.sn
        self.slk = grid.slk
        self.elements = np.array([grid.slk, grid.slk])
        pmn, pmx, qmn, qmx = grid.slack_lims
        self.lim_lo = np.array([pmn, qmn], float) / grid.sn
        self.lim_hi = np.array([pmx, qmx], float) / grid.sn

    def evaluate(self, leaf):
        s = [self.slk]
        Ps = add(leaf.P.take(s), leaf.pd_bus_pu.take(s))
        Qs = add(leaf.Q.take(s), leaf.qd_bus_pu.take(s))
        (plo, phi), (qlo, qhi) = bounds(Ps), bounds(Qs)
        return Band(np.r_[plo, qlo], np.r_[phi, qhi], np.vstack([np.abs(Ps[1][:, : leaf.K]), np.abs(Qs[1][:, : leaf.K])]))

    def restoration_rows(self, band, lin, ctx):
        sn = self.sn
        rows = []
        for i, dq in enumerate((lin.dP, lin.dQ)):
            S = (dq[self.slk] * sn)[None, :]
            win = np.array([max((self.lim_hi[i] - self.lim_lo[i]) * sn, 1.0)])
            rows += [RowBlock(-S, np.array([(band.lo[i] - self.lim_lo[i]) * sn]), win),
                     RowBlock(S, np.array([(self.lim_hi[i] - band.hi[i]) * sn]), win)]
        return rows

    def predict_shift(self, band, lin, d):
        return np.array([lin.dP[self.slk] @ d, lin.dQ[self.slk] @ d])


# ==================================================================================================
# branch current (thermal) and angle difference
# ==================================================================================================

_IZ_NGRID = 512
_IZ_TH = np.linspace(0.0, np.pi, _IZ_NGRID, endpoint=False)
_IZ_COS = np.cos(_IZ_TH)
_IZ_SIN = np.sin(_IZ_TH)


def zono_imax(Ire: TM, Iim: TM):
    """Sound upper bound of max |I| over the box, treating I = I0 + sum_k g_k e_k as a COMPLEX zonotope
    (support function over a direction grid + Lipschitz padding) plus the nonlinear part's magnitude."""
    a0x, a0y = Ire[0], Iim[0]
    Cx, Cy = Ire[1], Iim[1]
    gmax = np.full(Cx.shape[0], -np.inf)
    for s in range(0, _IZ_NGRID, 64):
        cos = _IZ_COS[s:s + 64, None, None]
        sin = _IZ_SIN[s:s + 64, None, None]
        p = Cx[None] * cos + Cy[None] * sin
        h = np.abs(a0x[None] * cos[..., 0] + a0y[None] * sin[..., 0]) + np.abs(p).sum(2)
        gmax = np.maximum(gmax, h.max(0))
    Lip = np.sqrt(a0x ** 2 + a0y ** 2) + np.sqrt(Cx ** 2 + Cy ** 2).sum(1)
    Amax = gmax + Lip * (np.pi / _IZ_NGRID * 0.5)
    eta = np.sqrt((np.abs(Ire[2]).sum((1, 2)) + Ire[3]) ** 2 + (np.abs(Iim[2]).sum((1, 2)) + Iim[3]) ** 2)
    if CONFIG.rigorous:
        Amax = oup(Amax)
        eta = oup(eta)
    return oup(Amax + eta)


def branch_current_tms(blocks, Vr, Vi):
    Gm, Bm = blocks
    Ire = sub(matmul_sparse(Gm, Vr), matmul_sparse(Bm, Vi))
    Iim = add(matmul_sparse(Gm, Vi), matmul_sparse(Bm, Vr))
    return Ire, Iim


class BranchCurrent(Constraint):
    """|I_from|, |I_to| <= rating (current-based thermal limit, squared).  Band = (0, max over ends of the
    certified |I|^2) in pu^2; the acceptance tolerance is relative: loading <= 1 + tol."""
    name = "branch"
    unit = "pu^2"

    def __init__(self, grid, tol=1e-4, weight=None, hard=True):
        super().__init__(tol)
        r2 = grid.rate2
        self.elements = np.where(np.isfinite(r2))[0]
        self.r2 = r2[self.elements]
        self.lim_lo = np.zeros(len(self.elements))
        self.lim_hi = self.r2.copy()
        Gf, Bf, Gt, Bt = grid.branch_blocks
        e = self.elements
        self.blocks_f = (Gf[e], Bf[e])
        self.blocks_t = (Gt[e], Bt[e])
        self.Yf = grid.Yf[e]
        self.Yt = grid.Yt[e]
        self.weight = (0.12 if weight is None else float(weight))   # pu yardstick for infeasibility (voltage window)
        self.hard = hard

    def _end(self, blocks, Vr, Vi, K):
        Ire, Iim = branch_current_tms(blocks, Vr, Vi)
        I2 = add(mul(Ire, Ire), mul(Iim, Iim))
        dv = oup(I2[0] + dev(I2)[1])
        im = zono_imax(Ire, Iim)
        return np.minimum(dv, oup(im * im)), np.abs(I2[1][:, :K])

    def evaluate(self, leaf):
        if not len(self.elements):
            return Band(np.zeros(0), np.zeros(0), np.zeros((0, leaf.K)))
        sf, wf = self._end(self.blocks_f, leaf.Vr, leaf.Vi, leaf.K)
        st, wt = self._end(self.blocks_t, leaf.Vr, leaf.Vi, leaf.K)
        return Band(np.zeros_like(sf), np.maximum(sf, st), np.maximum(wf, wt))

    def loading(self, band):
        """Certified loading |I| / rating (rounded up)."""
        return oup(np.sqrt(oup(band.hi / self.r2)))

    def ok(self, band):
        if not band.hi.size:
            return True
        return bool((oup(band.hi / self.r2) <= odn(odn(1.0 + self.tol) ** 2)).all())

    def excess(self, band):
        return self.loading(band) - 1.0

    def infeasibility(self, band):
        if not band.hi.size:
            return 0.0
        return self.weight * max(0.0, float(self.loading(band).max()) - (1.0 + self.tol))

    def width_excess(self, band):
        return -np.inf                                      # one-sided limit: no window

    def split_candidates(self, band):
        if not band.hi.size:
            return []
        ld = self.loading(band)
        over = np.where(ld > 1.0 + self.tol)[0]
        return [(self.weight * float(ld[k] - (1.0 + self.tol)), band.sens[k] / max(float(band.hi[k]), 1e-12))
                for k in over]

    def _sens(self, lin, rows):
        out = []
        for Y in (self.Yf, self.Yt):
            Ysub = Y[rows]
            I0 = Ysub @ lin.V
            dI = Ysub @ lin.dV
            out.append((np.abs(I0) ** 2, 2 * np.real(np.conj(I0)[:, None] * dI)))
        return out

    def restoration_rows(self, band, lin, ctx):
        over = np.where(band.hi > self.r2)[0]                   # over the STRICT rating (target ignores tol)
        if not len(over):
            return []
        rows = []
        for I2_0, S in self._sens(lin, over):
            mg = np.maximum(band.hi[over] - I2_0, 0.0)          # reserve the certified box spread
            rows.append(RowBlock(S, self.r2[over] - I2_0 - mg, np.maximum(self.r2[over], 1e-6), hard=self.hard))
        return rows

    def predict_shift(self, band, lin, d):
        over = np.where(band.hi > self.r2)[0]
        sh = np.zeros(len(self.elements))
        if len(over):
            (_, Sf), (_, St) = self._sens(lin, over)
            sh[over] = np.maximum(Sf @ d, St @ d)
        return sh

    def predict_infeasibility(self, band, lin, d):
        sh = self.predict_shift(band, lin, d)
        return self.infeasibility(Band(band.lo, np.maximum(band.hi + sh, 0.0)))

    def describe(self, band, sn=1.0):
        if not band.hi.size:
            return dict(name=self.name, ok=True, worst=0.0, element=None)
        ld = self.loading(band)
        k = int(np.argmax(ld))
        return dict(name=self.name, ok=self.ok(band), worst=float(ld[k] - 1.0), element=int(self.elements[k]),
                    max_loading=float(ld[k]), n_over=int((ld > 1.0 + self.tol).sum()))


def bus_angle_bands(Vr: TM, Vi: TM):
    """Per-bus angle range from the box [vr +- hw] x [vi +- hw] (corner rule; [-pi, pi] if the box leaves the
    right half-plane)."""
    hr = arad(Vr)
    hi_ = arad(Vi)
    vr_lo, vr_hi = odn(Vr[0] - hr), oup(Vr[0] + hr)
    vi_lo, vi_hi = odn(Vi[0] - hi_), oup(Vi[0] + hi_)
    ang = np.stack([np.arctan2(b, a) for a in (vr_lo, vr_hi) for b in (vi_lo, vi_hi)], 0)
    lo = ang.min(0)
    hi = ang.max(0)
    pad = 1e-12 * np.maximum(np.abs(lo), np.abs(hi)) + 1e-13      # atan2 is not guaranteed correctly rounded
    ok = vr_lo > 0
    return np.where(ok, lo - pad, -np.pi), np.where(ok, hi + pad, np.pi)


class AngleDifference(Constraint):
    """ANGMIN <= theta_f - theta_t <= ANGMAX on branches with an active limit (interval on the bus angle bands)."""
    name = "angle"
    unit = "rad"

    def __init__(self, grid, tol=1e-6):
        super().__init__(tol)
        self.elements = np.where(grid.angle_active)[0]
        self.f = grid.f[self.elements]
        self.t = grid.t[self.elements]
        self.lim_lo = np.deg2rad(np.asarray(grid.angmin, float)[self.elements])
        self.lim_hi = np.deg2rad(np.asarray(grid.angmax, float)[self.elements])

    def evaluate(self, leaf):
        tlo, thi = bus_angle_bands(leaf.Vr, leaf.Vi)
        return Band(odn(tlo[self.f] - thi[self.t]), oup(thi[self.f] - tlo[self.t]), None)

    def ok(self, band):
        return not bool(((band.lo < self.lim_lo - self.tol) | (band.hi > self.lim_hi + self.tol)).any())

    def restoration_rows(self, band, lin, ctx):
        over = np.where((band.lo < self.lim_lo - self.tol) | (band.hi > self.lim_hi + self.tol))[0]
        if not len(over):
            return []
        f, t = self.f[over], self.t[over]
        adc = lin.th[f] - lin.th[t]
        S = lin.dth[f] - lin.dth[t]
        rlo = np.maximum(adc - band.lo[over], 0.0)
        rhi = np.maximum(band.hi[over] - adc, 0.0)
        win = np.maximum(self.lim_hi[over] - self.lim_lo[over], 1e-3)
        return [RowBlock(-S, adc - (self.lim_lo[over] + rlo), win),
                RowBlock(S, (self.lim_hi[over] - rhi) - adc, win)]

    def predict_shift(self, band, lin, d):
        return (lin.dth[self.f] - lin.dth[self.t]) @ d


def default_constraints(grid, vmin=None, vmax=None, v_tol=1e-4, lim_tol=1e-3, br_tol=1e-4, branch=True,
                        angle=True):
    """The default constraint set: |V|, gen P, gen Q, slack P/Q, branch current, angle difference."""
    vmin = grid.vmin if vmin is None else vmin
    vmax = grid.vmax if vmax is None else vmax
    cs = [VoltageMagnitude(grid, vmin, vmax, tol=v_tol), GenActivePower(grid, tol=lim_tol),
          GenReactivePower(grid, tol=lim_tol), SlackPower(grid, tol=lim_tol)]
    if branch:
        w = float(np.max(np.broadcast_to(vmax, (grid.N,))) - np.min(np.broadcast_to(vmin, (grid.N,))))
        cs.append(BranchCurrent(grid, tol=br_tol, weight=w))
    if angle:
        cs.append(AngleDifference(grid))
    return cs
