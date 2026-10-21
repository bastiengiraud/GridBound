"""Certified N-1 contingency screening over an uncertainty box.

For every outage (branch removed from the grid) the power flow is certified over the whole box with
:func:`certflow.power.pfcert.enclose_pf_split`, and every monitored branch's loading is enclosed from BOTH sides,
so overloads are classified as certain (lower bound > rating everywhere) or possible (upper bound > rating
somewhere).  Outages that island the grid are certain power-flow failures; outages whose power flow cannot be
certified are possible failures.  Model agnostic: the uncertainty enters only through ``spec_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quantities as qty
from .pfcert import enclose_pf_split


@dataclass
class ContingencyResult:
    outage: int                       # branch index removed
    islanded: bool = False            # certain PF failure (disconnected network)
    formal: bool = True               # PF certified over the whole box (False: cannot exclude a PF failure)
    loading_lo: np.ndarray = None     # per monitored branch, fraction of rating (nan: out of service / outaged)
    loading_hi: np.ndarray = None
    nleaf: int = 0

    @property
    def over_certain(self):
        return np.nan_to_num(self.loading_lo, nan=0.0) > 1.0

    @property
    def over_possible(self):
        return np.nan_to_num(self.loading_hi, nan=0.0) > 1.0


@dataclass
class N1Report:
    results: list = field(default_factory=list)
    monitored: np.ndarray = None

    def count_bounds(self, fail_weight=0):
        """Certified [lower, upper] bound of  sum_outages (#overloaded branches) + fail_weight * #PF failures
        over the box (the lower bound holds for EVERY point of the box, the upper for none exceeding it)."""
        lo = hi = 0
        for r in self.results:
            if r.islanded:
                lo += fail_weight
                hi += fail_weight
            elif not r.formal:
                hi += fail_weight + len(self.monitored)
            else:
                lo += int(r.over_certain.sum())
                hi += int(r.over_possible.sum())
        return lo, hi

    @property
    def secure(self) -> bool:
        """Certified N-1 secure: no PF failure and no possible overload for any outage."""
        return self.count_bounds(fail_weight=1)[1] == 0

    def worst_loading(self):
        vals = [np.nanmax(r.loading_hi) for r in self.results if r.formal and not r.islanded and r.loading_hi is not None]
        return max(vals) if vals else np.nan


def branch_loading_bounds(grid, enc, branches, rating_mva, measure="S", K=None):
    """Two-sided certified loading (fraction of rating) of ``branches`` on a PF enclosure, plus (if K is given) the
    per-input-direction sensitivity |d loading / d e_j| (split weights).
    measure "S": max(|S_from|, |S_to|) / rating;  "I": max(|I_from|, |I_to|) / rating (rating in MVA at 1 pu)."""
    if measure == "S":
        tm = qty.branch_apparent_power(grid, enc, branches)
    else:
        tm = qty.branch_current_magnitude(grid, enc, branches)
    lo, hi = tm.bounds()
    r = np.asarray(rating_mva, float)[branches] / grid.sn
    out = (np.maximum(lo, 0) / r, hi / r)
    if K is not None:
        out += (np.abs(tm.C[:, :K]) / r[:, None],)
    return out


def _outage_bounds(g2, spec_fn, lo, hi, on, rating_mva, measure, max_leaves, refine):
    """Union of certified loading bounds over [lo, hi] for one outage.  Leaves where some branch is UNDECIDED
    (lower bound <= rating < upper bound) are bisected further, up to ``refine`` extra splits, along the
    direction that drives the undecided loadings most."""
    from ..split import bisect
    K = len(lo)

    def red(enc, a, b):
        return branch_loading_bounds(g2, enc, on, rating_mva, measure, K=K) + (a, b)

    stack = [(np.asarray(lo, float), np.asarray(hi, float))]
    parts, budget, nleaf = [], int(refine), 0
    while stack:
        a, b = stack.pop()
        leaves, ok = enclose_pf_split(g2, spec_fn, a, b, reduce=red, max_leaves=max_leaves)
        if not ok or not leaves:
            return None, nleaf
        for llo, lhi, sens, a2, b2 in leaves:
            und = (llo <= 1.0) & (lhi > 1.0)
            if und.any() and budget > 0:
                budget -= 1
                j = int(np.argmax(sens[und].sum(0) * (b2 - a2)))
                stack += list(bisect(a2, b2, j))
            else:
                parts.append((llo, lhi))
                nleaf += 1
    return parts, nleaf


def n1_analysis(grid, spec_fn, lo, hi, outages, rating_mva, monitored=None, measure="S", max_leaves=32,
                refine=32) -> N1Report:
    """Certified N-1 screening.

    grid       : base (pre-contingency) Grid
    spec_fn    : spec_fn(grid, lo, hi) -> dict(P, Q, V, shunt, warm) over the e-box (see enclose_pf_split)
    outages    : branch indices to remove one at a time (skipped if already out of service)
    rating_mva : per-branch rating (MVA), inf = unmonitored
    monitored  : branch indices to monitor (default: all with a finite rating)
    refine     : extra bisections per outage spent on boxes where a loading is undecided (0 = none)"""
    rating_mva = np.asarray(rating_mva, float)
    mon = np.where(np.isfinite(rating_mva))[0] if monitored is None else np.asarray(monitored, int)
    rep = N1Report(monitored=mon)
    for l in outages:
        if not grid.status[l]:
            continue
        st = grid.status.copy()
        st[l] = False
        g2 = grid.with_branch_status(st)
        if g2.islanded():
            rep.results.append(ContingencyResult(int(l), islanded=True))
            continue
        on = np.array([b for b in mon if g2.status[b]], int)
        parts, nleaf = _outage_bounds(g2, spec_fn, lo, hi, on, rating_mva, measure, max_leaves, refine)
        llo = np.full(len(mon), np.nan)
        lhi = np.full(len(mon), np.nan)
        if parts:
            pos = np.searchsorted(mon, on)
            llo[pos] = np.min([p[0] for p in parts], 0)
            lhi[pos] = np.max([p[1] for p in parts], 0)
        rep.results.append(ContingencyResult(int(l), False, parts is not None, llo, lhi, nleaf))
    return rep
