"""Certified N-1 contingency screening over an uncertainty box.

For every outage (branch removed from the grid) the power flow is certified over the whole box with
:func:`certflow.power.pfcert.enclose_pf_split`, and every monitored branch's loading is enclosed from BOTH sides,
so overloads are classified as certain (lower bound > rating everywhere) or possible (upper bound > rating
somewhere).  Optionally the same two-sided classification is applied to every bus voltage magnitude (``vlim``) and
to the slack bus's net active injection (``slack``).  Outages that island the grid are certain power-flow
failures; outages whose power flow cannot be certified are possible failures.  Model agnostic: the uncertainty
enters only through ``spec_fn``.
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
    v_lo: np.ndarray = None           # per bus |V| bounds (pu), if voltage limits were requested
    v_hi: np.ndarray = None
    slack_lo: float = None            # slack-bus net active injection bounds (MW), if slack limits were requested
    slack_hi: float = None
    vlim: tuple = None                # (vmin, vmax) pu used for the voltage classification
    slack_lim: tuple = None           # (pmin, pmax) MW used for the slack classification

    @property
    def over_certain(self):
        return np.nan_to_num(self.loading_lo, nan=0.0) > 1.0

    @property
    def over_possible(self):
        return np.nan_to_num(self.loading_hi, nan=0.0) > 1.0

    @property
    def v_certain(self):
        """Buses whose |V| is outside the limits for EVERY point of the box."""
        if self.v_lo is None:
            return np.zeros(0, bool)
        return (self.v_hi < self.vlim[0]) | (self.v_lo > self.vlim[1])

    @property
    def v_possible(self):
        """Buses whose |V| can leave the limits somewhere in the box."""
        if self.v_lo is None:
            return np.zeros(0, bool)
        return (self.v_lo < self.vlim[0]) | (self.v_hi > self.vlim[1])

    @property
    def slack_certain(self):
        return self.slack_lo is not None and (self.slack_hi < self.slack_lim[0] or self.slack_lo > self.slack_lim[1])

    @property
    def slack_possible(self):
        return self.slack_lo is not None and (self.slack_lo < self.slack_lim[0] or self.slack_hi > self.slack_lim[1])


@dataclass
class N1Report:
    results: list = field(default_factory=list)
    monitored: np.ndarray = None
    nbus: int = 0
    vlim: tuple = None                # voltage limits screened (None: voltages not screened)
    slack_lim: tuple = None           # slack limits screened (None: slack not screened)

    @property
    def kinds(self):
        """The violation kinds this report screened."""
        return ("overload",) + (("voltage",) if self.vlim is not None else ()) + (("slack",) if self.slack_lim is not None else ())

    def count_bounds(self, fail_weight=0, kinds=("overload",)):
        """Certified [lower, upper] bound of  sum_outages (#violations of the given kinds) + fail_weight * #PF failures
        over the box (the lower bound holds for EVERY point of the box, the upper for none exceeding it).
        kinds: "overload" (#overloaded branches), "voltage" (#buses outside vlim), "slack" (1 if the slack leaves
        slack_lim); the default is the overload count only (e.g. an RL environment's N-1 metric)."""
        kinds = tuple(k for k in kinds if k in self.kinds)
        lo = hi = 0
        for r in self.results:
            if r.islanded:
                lo += fail_weight
                hi += fail_weight
            elif not r.formal:
                hi += (fail_weight + ("overload" in kinds) * len(self.monitored) + ("voltage" in kinds) * self.nbus
                       + ("slack" in kinds))
            else:
                if "overload" in kinds:
                    lo += int(r.over_certain.sum())
                    hi += int(r.over_possible.sum())
                if "voltage" in kinds:
                    lo += int(r.v_certain.sum())
                    hi += int(r.v_possible.sum())
                if "slack" in kinds:
                    lo += int(r.slack_certain)
                    hi += int(r.slack_possible)
        return lo, hi

    @property
    def secure(self) -> bool:
        """Certified N-1 secure: no PF failure and no possible violation of any screened kind for any outage."""
        return self.count_bounds(fail_weight=1, kinds=self.kinds)[1] == 0

    def worst_loading(self):
        vals = [np.nanmax(r.loading_hi) for r in self.results if r.formal and not r.islanded and r.loading_hi is not None]
        return max(vals) if vals else np.nan

    def voltage_range(self):
        """(min, max) certified |V| over all buses and all non-islanded outages (nan if not screened)."""
        rs = [r for r in self.results if r.formal and not r.islanded and r.v_lo is not None]
        return (min(float(r.v_lo.min()) for r in rs), max(float(r.v_hi.max()) for r in rs)) if rs else (np.nan, np.nan)

    def slack_range(self):
        """(min, max) certified slack injection (MW) over all non-islanded outages (nan if not screened)."""
        rs = [r for r in self.results if r.formal and not r.islanded and r.slack_lo is not None]
        return (min(r.slack_lo for r in rs), max(r.slack_hi for r in rs)) if rs else (np.nan, np.nan)


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


def _outage_bounds(g2, spec_fn, lo, hi, on, rating_mva, measure, max_leaves, refine, vlim=None, slack=None):
    """Union of certified bounds over [lo, hi] for one outage: branch loadings, and optionally all bus |V| (vlim) and
    the slack bus's net active injection (slack = (bus, pmin_mw, pmax_mw)).  Leaves where some quantity is
    UNDECIDED (its interval straddles a limit) are bisected further, up to ``refine`` extra splits, along the
    direction that drives the undecided quantities most (sensitivities normalised by the limit window)."""
    from ..split import bisect
    K = len(lo)
    allb = list(range(g2.N))

    def red(enc, a, b):
        llo, lhi, sens = branch_loading_bounds(g2, enc, on, rating_mva, measure, K=K)
        out = dict(l=(llo, lhi), ls=sens, a=a, b=b)
        if vlim is not None:
            vm = qty.vmag(enc, allb)
            vlo, vhi = vm.bounds()
            win = np.maximum(np.broadcast_to(np.asarray(vlim[1], float) - np.asarray(vlim[0], float), vlo.shape), 1e-9)
            out.update(v=(vlo, vhi), vs=np.abs(vm.C[:, :K]) / win[:, None])
        if slack is not None:
            ps = enc.P.take([slack[0]])
            plo, phi = ps.bounds()
            out.update(s=(float(plo[0]) * g2.sn, float(phi[0]) * g2.sn),
                       ss=np.abs(ps.C[:, :K]) * g2.sn / max(slack[2] - slack[1], 1e-9))
        return out

    def undecided(L):
        llo, lhi = L["l"]
        rows = [L["ls"][(llo <= 1.0) & (lhi > 1.0)]]
        if vlim is not None:
            vlo, vhi = L["v"]
            u = ((vlo < vlim[0]) & (vhi >= vlim[0])) | ((vhi > vlim[1]) & (vlo <= vlim[1]))
            rows.append(L["vs"][u])
        if slack is not None:
            plo, phi = L["s"]
            if (plo < slack[1] <= phi) or (phi > slack[2] >= plo):
                rows.append(L["ss"])
        rows = [r for r in rows if len(r)]
        return np.vstack(rows).sum(0) if rows else None

    stack = [(np.asarray(lo, float), np.asarray(hi, float))]
    parts, budget, nleaf = [], int(refine), 0
    while stack:
        a, b = stack.pop()
        leaves, ok = enclose_pf_split(g2, spec_fn, a, b, reduce=red, max_leaves=max_leaves)
        if not ok or not leaves:
            return None, nleaf
        for L in leaves:
            sc = undecided(L)
            if sc is not None and budget > 0:
                budget -= 1
                j = int(np.argmax(sc * (L["b"] - L["a"])))
                stack += list(bisect(L["a"], L["b"], j))
            else:
                parts.append(L)
                nleaf += 1
    return parts, nleaf


def n1_analysis(grid, spec_fn, lo, hi, outages, rating_mva, monitored=None, measure="S", max_leaves=32,
                refine=32, vlim=None, slack=None) -> N1Report:
    """Certified N-1 screening.

    grid       : base (pre-contingency) Grid
    spec_fn    : spec_fn(grid, lo, hi) -> dict(P, Q, V, shunt, warm) over the e-box (see enclose_pf_split)
    outages    : branch indices to remove one at a time (skipped if already out of service)
    rating_mva : per-branch rating (MVA), inf = unmonitored
    monitored  : branch indices to monitor (default: all with a finite rating)
    refine     : extra bisections per outage spent on boxes where a quantity is undecided (0 = none)
    vlim       : (vmin, vmax) pu, scalars or per-bus arrays -> also screen |V| at EVERY bus after each outage
    slack      : (bus, pmin_mw, pmax_mw) -> also screen the net active injection of the slack bus (generation
                 minus constant-power load at that bus) after each outage"""
    rating_mva = np.asarray(rating_mva, float)
    mon = np.where(np.isfinite(rating_mva))[0] if monitored is None else np.asarray(monitored, int)
    if vlim is not None:                                       # scalars or per-bus arrays
        vlim = tuple(float(v) if np.ndim(v) == 0 else np.asarray(v, float) for v in vlim)
    rep = N1Report(monitored=mon, nbus=grid.N, vlim=vlim,
                   slack_lim=None if slack is None else (float(slack[1]), float(slack[2])))
    for l in outages:
        if not grid.status[l]:
            continue
        st = grid.status.copy()
        st[l] = False
        g2 = grid.with_branch_status(st)
        if g2.islanded():
            rep.results.append(ContingencyResult(int(l), islanded=True, vlim=rep.vlim, slack_lim=rep.slack_lim))
            continue
        on = np.array([b for b in mon if g2.status[b]], int)
        parts, nleaf = _outage_bounds(g2, spec_fn, lo, hi, on, rating_mva, measure, max_leaves, refine, vlim, slack)
        r = ContingencyResult(int(l), False, parts is not None, np.full(len(mon), np.nan), np.full(len(mon), np.nan),
                              nleaf, vlim=rep.vlim, slack_lim=rep.slack_lim)
        if parts:
            where = {int(b): k for k, b in enumerate(mon)}       # 'monitored' need not be sorted
            pos = np.array([where[int(b)] for b in on], int)
            r.loading_lo[pos] = np.min([p["l"][0] for p in parts], 0)
            r.loading_hi[pos] = np.max([p["l"][1] for p in parts], 0)
            if vlim is not None:
                r.v_lo = np.min([p["v"][0] for p in parts], 0)
                r.v_hi = np.max([p["v"][1] for p in parts], 0)
            if slack is not None:
                r.slack_lo = min(p["s"][0] for p in parts)
                r.slack_hi = max(p["s"][1] for p in parts)
        rep.results.append(r)
    return rep
