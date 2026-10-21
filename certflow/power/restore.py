"""Verified safety restoration (paper Sec. IV): constant offset d = (dPg, dVg) on the controller output, found by a
sequential trust-region QP on the constraints' linearised rows and RE-CERTIFIED after every step; if one offset
cannot cover the box, the box is bisected (violation-targeted) and each part gets its own offset
(piecewise-constant offset controller).  Algorithms 2 and 3.

The QP is assembled generically from the constraints' :class:`RowBlock` s, so new constraint families take
part in the restoration without changes here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..split import ObjectPool, bisect
from .constraints import RowBlock, _fin
from .sensitivity import linearize


# ==================================================================================================
# QP primitives (exact min-norm via Lawson-Hanson LDP / NNLS)
# ==================================================================================================

def ldp_solve(Gm, hv, sq):
    """EXACT weighted min-norm  min ||sqrt(w) x||^2  s.t.  Gm x <= hv  (sq = sqrt(w) or None).
    Returns (x, Gm x - hv)."""
    from scipy.optimize import nnls
    m, n = Gm.shape
    Gs = Gm if sq is None else Gm / sq[None, :]
    E = np.vstack([(-Gs).T, (-hv)[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    u, _ = nnls(E, f)
    r = E @ u - f
    y = np.zeros(n) if (np.linalg.norm(r) < 1e-12 or abs(r[n]) < 1e-12) else -r[:n] / r[n]
    x = y if sq is None else y / sq
    return x, Gm @ x - hv


def qp_minnorm(Gm, hv, wcol=None):
    """Weighted min-norm QP  min sum_i w_i x_i^2  s.t.  Gm x <= hv, by constraint generation (exact).
    Returns (x, max residual) -- residual > 0 means infeasible."""
    Gm = np.asarray(Gm, float)
    hv = np.asarray(hv, float)
    m, n = Gm.shape
    sq = None if wcol is None else np.sqrt(np.asarray(wcol, float))
    if m <= n + 1:
        x, resid = ldp_solve(Gm, hv, sq)
        return x, float(resid.max())
    tol = 1e-7
    W = np.flatnonzero(hv < tol)
    if W.size == 0:
        W = np.array([int(np.argmin(hv))], dtype=int)
    x = np.zeros(n)
    resid = -hv.copy()
    for _ in range(20):
        if W.size >= m:
            x, resid = ldp_solve(Gm, hv, sq)
            break
        x, rW = ldp_solve(Gm[W], hv[W], sq)
        resid = Gm @ x - hv
        if float(rW.max()) > tol:
            break
        viol = np.flatnonzero(resid > tol)
        if viol.size == 0:
            break
        W = np.union1d(W, viol)
    return x, float(resid.max())


def qp_besteffort(soft_rows, soft_h, hard_rows, hard_h, n, tr, soft_w=None, dcost=1.0):
    """Bounded best-effort step when the exact QP is infeasible: minimise the (heavily penalised) soft-row
    slack subject to the hard rows and the trust region |dx| <= tr.  Always feasible."""
    tr = np.asarray(tr, float)
    wrow = (np.ones(np.vstack(soft_rows).shape[0]) if soft_w is None
            else np.concatenate([np.full(np.atleast_2d(r).shape[0], float(wv)) for r, wv in zip(soft_rows, soft_w)]))
    Gs = np.vstack(soft_rows)
    hs = np.concatenate([np.atleast_1d(h) for h in soft_h])
    keep = hs <= np.abs(Gs) @ tr + 1e-9                     # rows that can never bind inside the TR -> drop
    Gs, hs, wrow = Gs[keep], hs[keep], wrow[keep]
    k = Gs.shape[0]
    if k == 0:
        return np.zeros(n)
    blocks = [np.hstack([Gs, -np.eye(k)]), np.hstack([np.zeros((k, n)), -np.eye(k)])]
    hv = [hs, np.zeros(k)]
    if hard_rows:
        Gh = np.vstack(hard_rows)
        hh = np.concatenate([np.atleast_1d(h) for h in hard_h])
        blocks.append(np.hstack([Gh, np.zeros((Gh.shape[0], k))]))
        hv.append(hh)
    I = np.eye(n)
    blocks += [np.hstack([I, np.zeros((n, k))]), np.hstack([-I, np.zeros((n, k))])]
    hv += [tr, tr]
    x, _ = qp_minnorm(np.vstack(blocks), np.concatenate(hv), wcol=np.concatenate([dcost * np.ones(n), 1e6 * wrow]))
    return x[:n]


# ==================================================================================================
# restoration
# ==================================================================================================

@dataclass
class RestoreOptions:
    max_it: int = 12              # SLP iterations per box
    eps_vg: float = 0.1           # objective weight on the (scaled) voltage step: prefer voltage control
    tr_v0: float = 0.04           # adaptive trust region on |dVg| (pu): initial / min / max
    tr_min: float = 0.01
    tr_max: float = 0.30
    tr_good: float = 0.75
    tr_poor: float = 0.25
    tr_grow: float = 2.0
    tr_shrink: float = 0.25
    tr_p_frac: float = 0.25       # |dPg| <= frac * (Pmax - Pmin) per best-effort step
    stall: float = 0.10           # stop when an iteration cuts the infeasibility by less than this fraction
    width_tol: float = 5e-4       # a range wider than its window by more than this -> needs a split
    feasibility_first: bool = True  # best-effort step ignores ||d|| (tiny cost) -> reach feasibility first
    max_split: int = 8            # depth of the piecewise (box-split) restoration


@dataclass
class RestoreResult:
    lo: np.ndarray
    hi: np.ndarray
    offset: tuple                  # (dPg MW, dVg pu)
    cert0: object                  # certificate without offset (or at the warm start)
    cert: object                   # best certificate found (with ``offset``)
    restored: bool
    needs_split: bool
    n_cert: int = 0
    t_total: float = 0.0
    seed: str = None
    history: list = field(default_factory=list)


@dataclass
class _Ctx:
    pg_range: np.ndarray


class Restorer:
    """Restore a verifier's box to FORMALLY SAFE with a (piecewise) constant output offset.

    seeds : optional dict  name -> callable(x_centre) -> (Pg, Vg)  of reference dispatches (e.g. an AC-OPF
            solver) used as alternative starting offsets d0 = seed(x_c) - controller(x_c)  (Algorithm 3)."""

    def __init__(self, verifier, options: RestoreOptions = None, seeds=None):
        self.ver = verifier
        self.opt = options or RestoreOptions()
        self.seeds = dict(seeds or {})
        g = verifier.grid
        self.ng = g.ng
        self.vmin_c, self.vmax_c = verifier.ctrl.vg_clip
        pmn = np.nan_to_num(np.asarray(g.gen_pmin, float), nan=-1e4, posinf=1e4, neginf=-1e4)
        pmx = np.nan_to_num(np.asarray(g.gen_pmax, float), nan=1e4, posinf=1e4, neginf=-1e4)
        self.pg_range = np.maximum(pmx - pmn, 1.0)
        self.csc = np.concatenate([(self.vmax_c - self.vmin_c) * np.ones(self.ng), self.pg_range])

    # ------------------------------------------------------------------ one QP step
    def _step(self, C, lin, Vg_cur, tr_v):
        ng = self.ng
        o = self.opt
        rows = []
        for c in self.ver.cons:
            rows += c.restoration_rows(C.bands[c.name], lin, _Ctx(self.pg_range))
        I, Z = np.eye(ng), np.zeros((ng, ng))
        vw = np.full(ng, self.vmax_c - self.vmin_c)
        rows += [RowBlock(np.hstack([-I, Z]), Vg_cur - self.vmin_c, vw, hard=True),     # actuator: Vg in its band
                 RowBlock(np.hstack([I, Z]), self.vmax_c - Vg_cur, vw, hard=True)]

        def scl(rb):
            w = np.maximum(np.abs(np.broadcast_to(np.atleast_1d(rb.window), (rb.A.shape[0],))), 1e-12)
            return (np.atleast_2d(rb.A) * self.csc[None, :]) / w[:, None], np.atleast_1d(rb.b) / w

        soft = [scl(rb) + (rb.weight,) for rb in rows if not rb.hard]
        hard = [scl(rb) for rb in rows if rb.hard]
        Gm = np.vstack([r[0] for r in soft] + [r[0] for r in hard])
        hv = np.concatenate([r[1] for r in soft] + [r[1] for r in hard])
        hv = np.nan_to_num(hv, nan=0.0, posinf=1e12, neginf=-1e12)
        d, res = qp_minnorm(Gm, hv, wcol=np.concatenate([o.eps_vg * np.ones(ng), np.ones(ng)]))
        ok = res <= 1e-6
        if not ok:
            tr = np.concatenate([tr_v * np.ones(ng), o.tr_p_frac * self.pg_range]) / self.csc
            sh = [np.nan_to_num(r[1], nan=0.0, posinf=1e12, neginf=-1e12) for r in soft]
            hh = [np.nan_to_num(r[1], nan=0.0, posinf=1e12, neginf=-1e12) for r in hard]
            d = qp_besteffort([r[0] for r in soft], sh, [r[0] for r in hard], hh, 2 * ng, tr,
                              soft_w=[r[2] for r in soft], dcost=(1e-6 if o.feasibility_first else 1.0))
        return d * self.csc, ok

    def _predict(self, C, lin, d):
        return float(sum(c.predict_infeasibility(C.bands[c.name], lin, d) for c in self.ver.cons))

    # ------------------------------------------------------------------ one box (Algorithm 2)
    def restore_box(self, lo=None, hi=None, warm=None, seed=None) -> RestoreResult:
        ver, o, ng = self.ver, self.opt, self.ng
        lo, hi = ver.U.full_box() if lo is None else (np.asarray(lo, float), np.asarray(hi, float))
        t0 = time.perf_counter()
        ec = 0.5 * (lo + hi)
        op = ver.operating_point(ec)
        Pg0, Vg0, Pd0, Qd0 = op["Pg"], op["Vg"], op["Pd"], op["Qd"]
        dPg = np.zeros(ng)
        dVg = np.zeros(ng)
        warm0 = warm is not None
        if warm is not None:
            if warm[0] is not None:
                dPg = np.asarray(warm[0], float).copy()
            if warm[1] is not None:
                dVg = np.asarray(warm[1], float).copy()
        elif seed is not None:
            Pgs, Vgs = self.seeds[seed](op["x"])
            dPg = np.asarray(Pgs, float) - Pg0
            dVg = np.clip(np.asarray(Vgs, float), self.vmin_c, self.vmax_c) - Vg0
            warm0 = True
        trv = o.tr_v0
        tv_pred = None
        hit = False
        prev_tv = np.inf
        best_key, best = None, None
        C0 = None
        n_cert = 0
        hist = []
        for it in range(o.max_it + 1):
            use = bool(it) or warm0
            C = ver.certify(lo, hi, offset=((dPg.copy(), dVg.copy()) if use else None))
            n_cert += 1
            if it == 0:
                C0 = C
            tv = C.infeasibility()
            feas = C.safe
            hist.append(tv)
            if it > 0 and tv_pred is not None:                     # adaptive trust region
                pred_red = prev_tv - tv_pred
                act_red = prev_tv - tv
                rho = act_red / pred_red if pred_red > 1e-9 else (1.0 if act_red > 0 else 0.0)
                if rho < o.tr_poor:
                    trv = max(o.tr_min, trv * o.tr_shrink)
                elif rho > o.tr_good and hit:
                    trv = min(o.tr_max, trv * o.tr_grow)
            key = (0 if feas else 1, tv)
            if best_key is None or key < best_key:
                best_key, best = key, (dPg.copy(), dVg.copy(), C)
            stall = it > 0 and tv >= prev_tv * (1.0 - o.stall)
            wdom = (not feas) and C.width_excess() > o.width_tol
            if feas or it == o.max_it or stall or wdom or not C.formal:
                break
            prev_tv = tv
            Vg_cur = np.clip(Vg0 + dVg, self.vmin_c, self.vmax_c)
            Pg_cur = Pg0 + dPg
            th, Vm, mis = ver.centre_pf(Pg_cur, Vg_cur, Pd0, Qd0, iters=80)
            lin = linearize(ver.grid, Vm, th)
            d, ok = self._step(C, lin, Vg_cur, trv)
            dVg = dVg + d[:ng]
            dPg = dPg + d[ng:]
            hit = (not ok) and float(np.abs(d[:ng]).max()) >= 0.99 * trv
            tv_pred = self._predict(C, lin, d)
        dPg, dVg, C = best
        restored = bool(C.safe)
        needs_split = (not restored) and C.width_excess() > o.width_tol
        return RestoreResult(lo, hi, (dPg, dVg), C0, C, restored, needs_split, n_cert,
                             time.perf_counter() - t0, seed, hist)

    # ------------------------------------------------------------------ split choice (violation-targeted)
    def split_choice(self, C, hw):
        """Direction j maximising sqrt(sum_e (violation_e * |dq_e/de_j| * hw_j)^2) over all over-limit
        elements e of all constraints.  Returns (j, score)."""
        cands = []
        for c in self.ver.cons:
            cands += c.split_candidates(C.bands[c.name])
        if not cands:
            return int(np.argmax(hw)), 0.0
        Cm = np.array([v * np.abs(w) * hw for v, w in cands])
        sc = np.sqrt((Cm ** 2).sum(0))
        j = int(np.argmax(sc))
        return j, float(sc[j])

    # ------------------------------------------------------------------ piecewise restoration (Algorithm 3)
    def restore_box_task(self, lo, hi, warm=None, seed=None):
        return self.restore_box(lo, hi, warm=warm, seed=seed)

    def restore_split(self, max_split=None, nproc=1):
        """Restore the whole input set: try one offset (min-norm seed, then each registered seed); if none
        closes, bisect (violation-targeted) and restore each part, children warm-started from the best
        ancestor offset, with keep-best-ancestor monotonicity.  Returns (root, leaves) where every leaf is a
        RestoreResult over its own e-box (their union is the input set)."""
        max_split = self.opt.max_split if max_split is None else max_split
        lo0, hi0 = self.ver.U.full_box()
        tries = [self.restore_box(lo0, hi0, seed=None)]
        if tries[0].restored:
            return tries[0], [tries[0]]
        for name in self.seeds:
            r = self.restore_box(lo0, hi0, seed=name)
            if r.restored:
                return r, [r]
            tries.append(r)
        root = min(tries, key=lambda r: r.cert.infeasibility())
        if max_split <= 0:
            return root, [root]
        infeas = lambda r: r.cert.infeasibility()

        def best_of(r, anc):
            if anc is None or infeas(r) <= infeas(anc):
                return r
            return _AncestorView(anc, r.lo, r.hi)

        def plan(r, br, depth):
            if br.restored or r.restored or depth >= max_split:
                return None
            j, sc = self.split_choice(r.cert, 0.5 * (r.hi - r.lo)) if r.cert.formal else (int(np.argmax(r.hi - r.lo)), 1.0)
            return None if sc <= 1e-15 else j

        j0, _ = self.split_choice(root.cert, 0.5 * (hi0 - lo0))
        a, b = bisect(lo0, hi0, j0)
        queue = [(a[0], a[1], 1, root), (b[0], b[1], 1, root)]
        leaves = []
        pool = ObjectPool(self, nproc) if nproc and nproc > 1 else None
        try:
            if pool is None:
                while queue:
                    lo, hi, depth, anc = queue.pop()
                    r = self.restore_box(lo, hi, warm=anc.offset, seed=root.seed)
                    br = best_of(r, anc)
                    j = plan(r, br, depth)
                    if j is None:
                        leaves.append(br)
                    else:
                        (a, b) = bisect(lo, hi, j)
                        queue += [(b[0], b[1], depth + 1, br), (a[0], a[1], depth + 1, br)]
            else:
                from concurrent.futures import FIRST_COMPLETED, wait
                pend = {pool.submit("restore_box_task", lo, hi, anc.offset, root.seed): (lo, hi, depth, anc)
                        for lo, hi, depth, anc in queue}
                while pend:
                    done, _ = wait(list(pend), return_when=FIRST_COMPLETED)
                    for fut in done:
                        lo, hi, depth, anc = pend.pop(fut)
                        r = fut.result()
                        br = best_of(r, anc)
                        j = plan(r, br, depth)
                        if j is None:
                            leaves.append(br)
                        else:
                            for (cl, ch) in bisect(lo, hi, j):
                                pend[pool.submit("restore_box_task", cl, ch, br.offset, root.seed)] = (cl, ch, depth + 1, br)
        finally:
            if pool is not None:
                pool.close()
        leaves.sort(key=lambda r: tuple(np.r_[r.lo, r.hi]))
        return root, leaves


class _AncestorView:
    """An ancestor's (sound, larger-box) certificate and offset reported for a sub-box."""

    def __init__(self, anc, lo, hi):
        self.__dict__.update(anc.__dict__)
        self.lo, self.hi = lo, hi
        self.inherited = True


@dataclass
class PiecewiseOffsetController:
    """Deployable restored controller: the NN plus the certified constant offset of the e-box containing the
    operating point (paper Sec. IV-C)."""
    leaves: list            # RestoreResult (lo, hi, offset)

    def offset(self, e):
        e = np.asarray(e, float)
        for r in self.leaves:
            if (e >= r.lo - 1e-9).all() and (e <= r.hi + 1e-9).all():
                return r.offset
        raise ValueError("point outside the restored set")

    @property
    def safe(self):
        return all(r.restored for r in self.leaves)
