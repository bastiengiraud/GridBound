"""Power-flow verification of an NN controller over an input set (the paper's pipeline, Fig. 1).

For every e-space leaf box:
  1. input zonotope  x(e)                                    (certflow.sets)
  2. controller reach  y(e) = [Pg; Vg]  (NN DeepZ + clip)    (certflow.power.controller)
  3. specified injections S(e) and PV setpoints as TMs
  4. Krawczyk: unique PF solution z*(e) in x1(e) +- r          (certflow.krawczyk + RectPFSystem)
  5. every constraint evaluates its rigorous band on the state enclosure
Leaves that fail (rho >= 1) are bisected along the residual-dominant directions; the certificate is the union.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fp import odn, oup
from ..sets import AffineSet
from ..split import adaptive_split, split_children, split_dims
from ..tm import TM, add, const, div, matmul, pad_symbols, scale, sub
from .constraints import Band, default_constraints
from .controller import DispatchController
from .grid import Grid
from .pf import newton
from .pfcert import enclose_pf


@dataclass
class VerifierOptions:
    max_leaves: int = 512          # cap on Krawczyk rho-tree leaves (abstain beyond)
    split_by_score: bool = True    # bisect residual-dominant directions (else widest)
    split_tau: float = 0.5         # multi-direction split: all directions with score >= tau * max ...
    split_m: int = 4               # ... at most this many at once
    newton_iters: int = 60
    centre_tol: float = 1e-6       # centre PF mismatch above which a leaf is split
    corr_max_m: int = 1700         # correlated Krawczyk remainder up to this state size


@dataclass
class LeafEnclosure:
    """Everything certified on one leaf (big: Taylor models over all buses)."""
    grid: Grid
    lo: np.ndarray
    hi: np.ndarray
    K: int
    ng: int
    vg_clip: tuple
    x: TM = None
    y_raw: TM = None
    y: TM = None
    th_s: np.ndarray = None
    Vm_s: np.ndarray = None
    ok: bool = False
    rho: float = np.inf
    score: np.ndarray = None
    Vr: TM = None
    Vi: TM = None
    P: TM = None
    Q: TM = None
    pd_bus_pu: TM = None
    qd_bus_pu: TM = None
    kr: object = None


@dataclass
class LeafResult:
    """Compact, picklable result of one leaf (what workers send back)."""
    lo: np.ndarray
    hi: np.ndarray
    ok: bool
    rho: float
    score: np.ndarray
    bands: dict = None
    nsym: int = 0


@dataclass
class Certificate:
    formal: bool                       # every leaf proven (existence + uniqueness of the PF solution)
    leaves: list                       # LeafResult per leaf
    bands: dict                        # constraint name -> union Band
    constraints: list
    ebox: tuple
    offset: tuple = None
    rho: float = np.inf
    sn: float = 100.0
    status: dict = field(default_factory=dict)

    @property
    def nleaf(self):
        return len(self.leaves)

    @property
    def safe(self) -> bool:
        """FORMALLY SAFE: proven PF solution AND every constraint within limits over the whole box."""
        return self.formal and all(self.status.values())

    def constraint(self, name):
        return next(c for c in self.constraints if c.name == name)

    def infeasibility(self) -> float:
        if not self.formal:
            return np.inf
        return float(sum(c.infeasibility(self.bands[c.name]) for c in self.constraints))

    def width_excess(self) -> float:
        if not self.formal:
            return np.inf
        return float(max(c.width_excess(self.bands[c.name]) for c in self.constraints))

    def summary(self) -> str:
        head = (f"{'FORMAL' if self.formal else 'ABSTAIN'}  {'SAFE' if self.safe else 'CAN-VIOLATE'}  "
                f"leaves={self.nleaf}  rho={self.rho:.3f}")
        if not self.formal:
            return head
        parts = []
        for c in self.constraints:
            d = c.describe(self.bands[c.name], self.sn)
            if c.name == "branch":
                parts.append(f"branch max {d.get('max_loading', 0) * 100:.1f}%" + ("" if d["ok"] else f" ({d['n_over']} over)"))
            else:
                b = self.bands[c.name]
                if not b.lo.size:
                    continue
                parts.append(f"{c.name} {'ok' if d['ok'] else 'VIOL'}" + ("" if d["ok"] else f" (worst +{d['worst']:.4g} {c.unit} @ {d['element']})"))
        return head + "\n  " + " | ".join(parts)


class PFVerifier:
    """Verify an NN dispatch controller over an input set with the AC power flow.

    grid        : :class:`Grid`
    controller  : :class:`DispatchController` (or any object with reach/forward/ng/vg_clip)
    input_set   : :class:`certflow.sets.AffineSet` over the controller input x = [Pd (nl); Qd (nl)] in MW/MVAr
    constraints : list of :class:`Constraint` (default: the paper's set with the grid's own voltage limits)
    warm_start  : (theta, |V|) used to start every centre Newton solve (default: grid.V0, else flat)"""

    def __init__(self, grid: Grid, controller: DispatchController, input_set: AffineSet, constraints=None,
                 warm_start=None, options: VerifierOptions = None):
        self.grid = grid
        self.ctrl = controller
        self.U = input_set
        self.cons = list(constraints) if constraints is not None else default_constraints(grid)
        self.opt = options or VerifierOptions()
        if warm_start is None:
            if grid.V0 is not None:
                warm_start = (np.angle(grid.V0), np.abs(grid.V0))
            else:
                warm_start = (np.zeros(grid.N), np.ones(grid.N))
        self.th0, self.Vm0 = np.asarray(warm_start[0], float), np.asarray(warm_start[1], float)
        self.nl = grid.nl
        if input_set.n != 2 * grid.nl:
            raise ValueError(f"input set has {input_set.n} entries; expected [Pd; Qd] = {2 * grid.nl}")

    # ------------------------------------------------------------------ point evaluation
    def centre_pf(self, Pg, Vg, Pd, Qd, iters=None):
        g = self.grid
        sn = g.sn
        Psp = (g.Cg @ Pg - g.Cl @ Pd) / sn + g.sgen_p / sn
        Qsp = -(g.Cl @ Qd) / sn + g.sgen_q / sn
        Vpin = g.Cgv @ Vg
        return newton(g.G, g.B, g.ns, g.pq, g.pv, Psp, Qsp, Vpin, self.th0, self.Vm0,
                      iters=iters or self.opt.newton_iters)

    def operating_point(self, e=None, offset=None, iters=80):
        """Deployed controller + AC PF at input point e (default: set centre).  Returns dict."""
        e = np.zeros(self.U.K) if e is None else np.asarray(e, float)
        x = self.U.point(e)
        Pd, Qd = x[: self.nl], x[self.nl:]
        Pg, Vg = self.ctrl.forward(x, offset)
        th, Vm, mis = self.centre_pf(Pg, Vg, Pd, Qd, iters=iters)
        return dict(x=x, Pd=Pd, Qd=Qd, Pg=Pg, Vg=Vg, th=th, Vm=Vm, mismatch=mis)

    # ------------------------------------------------------------------ one leaf
    def enclose(self, lo, hi, offset=None) -> LeafEnclosure:
        g = self.grid
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        ec = 0.5 * (lo + hi)
        h = 0.5 * (hi - lo)
        ng, nl, N, sn = g.ng, self.nl, g.N, g.sn
        ns, pq, pv = g.ns, g.pq, g.pv
        K = self.U.K
        leaf = LeafEnclosure(grid=g, lo=lo, hi=hi, K=K, ng=ng, vg_clip=self.ctrl.vg_clip)
        x = self.U.sub_tm(ec, h)
        y_raw, y = self.ctrl.reach(x, offset)
        Ks = y[1].shape[1]
        leaf.x, leaf.y_raw, leaf.y = x, y_raw, y
        xK = pad_symbols(x, Ks, Ks)
        Pd_tm = xK.take(slice(0, nl))
        Qd_tm = xK.take(slice(nl, 2 * nl))
        Pg = y.take(slice(0, ng))
        Vg = y.take(slice(ng, 2 * ng))
        Pdb = matmul(g.Cl, Pd_tm)
        Qdb = matmul(g.Cl, Qd_tm)
        snv = np.full(N, sn)
        Pbus = add(div(sub(matmul(g.Cg, Pg), Pdb), snv), const(g.sgen_p / sn, Pdb))
        Qbus = add(div(scale(-np.ones(N), Qdb), snv), const(g.sgen_q / sn, Qdb))
        Vset = matmul(g.Cgv, Vg).take(pv)
        enc = enclose_pf(g, Pbus, Qbus, Vset, (self.th0, self.Vm0), centre_tol=self.opt.centre_tol,
                         corr_max_m=self.opt.corr_max_m, iters=self.opt.newton_iters)
        leaf.th_s, leaf.Vm_s, leaf.kr = enc.th_s, enc.Vm_s, enc.kr
        if enc.system is None:
            leaf.score = h                                 # centre PF failed -> split the widest direction
            return leaf
        leaf.ok, leaf.rho, leaf.score = enc.ok, enc.rho, enc.score[:K]
        if not enc.ok:
            return leaf
        leaf.Vr, leaf.Vi, leaf.P, leaf.Q = enc.Vr, enc.Vi, enc.P, enc.Q
        leaf.pd_bus_pu = div(Pdb, snv)
        leaf.qd_bus_pu = div(Qdb, snv)
        return leaf

    def leaf(self, lo, hi, offset=None) -> LeafResult:
        """Certify one leaf and evaluate all constraints on it (compact result)."""
        L = self.enclose(lo, hi, offset)
        if not L.ok:
            return LeafResult(L.lo, L.hi, False, L.rho, L.score)
        bands = {c.name: c.evaluate(L) for c in self.cons}
        return LeafResult(L.lo, L.hi, True, L.rho, L.score, bands, int(L.y[1].shape[1]))

    def _leaf_task(self, lo, hi, offset=None):        # adaptive_split protocol: (ok, payload, score)
        r = self.leaf(lo, hi, offset)
        return r.ok, r, r.score

    # ------------------------------------------------------------------ box certificate
    def _choose_split(self, lo, hi, score):
        dims = split_dims(score, hi - lo, tau=self.opt.split_tau, m_max=self.opt.split_m,
                          use_score=self.opt.split_by_score)
        return split_children(lo, hi, dims)

    def certify(self, lo=None, hi=None, offset=None, pool=None) -> Certificate:
        """Certificate over the e-box [lo, hi] (default: the whole input set) for the controller + offset.
        ``pool``: an :class:`certflow.split.ObjectPool` holding this verifier, to certify leaves in parallel."""
        lo0, hi0 = self.U.full_box() if lo is None else (np.asarray(lo, float), np.asarray(hi, float))
        if pool is None:
            leaves, ok_all = adaptive_split(lambda a, b: self._leaf_task(a, b, offset), lo0, hi0,
                                            self._choose_split, max_leaves=self.opt.max_leaves)
        else:
            leaves, ok_all = adaptive_split(None, lo0, hi0, self._choose_split, max_leaves=self.opt.max_leaves,
                                            pool=pool, method="_leaf_task", extra=(offset,))
        formal = bool(ok_all and leaves)
        bands = {c.name: Band.union([lf.bands[c.name] for lf in leaves]) for c in self.cons} if leaves else {}
        status = {c.name: c.ok(bands[c.name]) for c in self.cons} if formal else {}
        rho = max((lf.rho for lf in leaves), default=np.inf)
        return Certificate(formal, leaves, bands, self.cons, (lo0, hi0), offset, rho, self.grid.sn, status)
