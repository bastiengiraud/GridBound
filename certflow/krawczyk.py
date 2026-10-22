"""Validated Krawczyk existence/uniqueness proof for parameterised implicit systems

        F(z) = S(e),        e in [-1, 1]^K,

with a JACOBIAN THAT IS AFFINE IN z (any bilinear/quadratic residual, e.g. the AC power flow in rectangular
coordinates).  For such systems the Jacobian's state-derivatives H_j = dJ/dz_j are constant, which gives the
correlated remainder  rem(r) = sum_j |M H_j| r_j.

Given a linearisation point z0, the solver builds the Newton warm step  x1(e) = z0 - M (F(z0) - S(e))  as a
Taylor model, bounds the residual corr >= |M (F(x1(e)) - S(e))| and the slope majorant
Kb(r) >= |I - M J(xi)| over xi in x1(e) +- r, and runs a monotone inflation loop on r (starting from r = 0).  If
rho(Kb) < 1 and r >= (I - Kb)^-1 corr, then for EVERY e there is a unique solution z*(e) in x1(e) +- r.
In rigorous mode the float candidate radius is re-checked a posteriori with all rounding errors included.

The application provides the system through the :class:`AffineJacobianSystem` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .fp import CONFIG, gam, upr
from .tm import TM, arad, mag, matmul, nt, sub


class AffineJacobianSystem(Protocol):
    """What the Krawczyk solver needs from an application (see certflow.power.pfsystem for an example)."""

    m: int                      # state dimension
    jac_entry_terms: int        # max number of float terms in one Jacobian entry (rounding-error count)

    def residual(self, x: TM) -> TM:
        """F(x) for a state Taylor model x (m rows)."""

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Point Jacobian dF/dz at z, dense (m, m)."""

    def jacobian_tm(self, x: TM):
        """Jacobian Taylor model at state TM x: (J0 (m,m), JlinC (m,m,K), quad_src, R_rad (m,m)).
        quad_src is opaque; it is passed back to quad_sum / quad_weights."""

    def lin_sum(self, M, JlinC) -> np.ndarray:
        """sum_k |M JlinC[:,:,k]|."""

    def quad_sum(self, M, quad_src) -> np.ndarray:
        """sum over quadratic coefficient pairs of |M J_quad|  (exact streamed form)."""

    def hessian_structure(self):
        """Sparse structure of dJ/dz_j (see :func:`structure_from_triplets`), or None (ball remainder).
        Either one structure (H_j constant), or a LIST of ``(structure, scale)`` terms when the Jacobian's
        state-slope depends on the parameters: dJ/dz_j = sum_t H^t_j s^t_j(e) with |s^t_j(e)| <= scale^t_j
        (scale None = 1)."""

    def quad_weights(self, quad_src):
        """Per-state weight w_j of the quadratic Jacobian generators (len m), or a list aligned with the
        terms of hessian_structure()."""

    def jac_majorant(self, x: TM, r: np.ndarray) -> np.ndarray:
        """Entrywise majorant of |J(xi)| over xi in x +- r (floating-point error accounting)."""


def mdotsp(M, Jac):
    """M @ Jac exploiting Jac's sparsity:  (Jac_csr^T @ M^T)^T."""
    from scipy.sparse import csr_matrix
    return np.asarray(csr_matrix(Jac).T @ M.T).T


def structure_from_triplets(rr, cc, jj, vv, m):
    """Sparse selectors for  |M H_j|  given the nonzeros (row rr, col cc, value vv) of every H_j (index jj):
    returns (rows, inv, vals, cols_u, j_u, ShT, ScT) such that  |M H_j|[:, c]  for every populated (c, j) is
    abs(ShT @ (M[:, rows] * vals).T).T  and  sum_j |M H_j| w_j = ScT @ (MHcol * w[j_u]).T."""
    from scipy.sparse import csr_matrix
    rr, cc, jj, vv = (np.asarray(a) for a in (rr, cc, jj, vv))
    ucol, inv = np.unique(cc * m + jj, return_inverse=True)
    ccu = ucol // m
    jju = ucol % m
    nnz = len(rr)
    ncol = len(ucol)
    ShT = csr_matrix((np.ones(nnz), (inv, np.arange(nnz))), shape=(ncol, nnz))
    ScT = csr_matrix((np.ones(ncol), (ccu, np.arange(ncol))), shape=(m, ncol))
    return rr, inv, vv, ccu, jju, ShT, ScT


def specrad(A, iters=150, tol=1e-10, v0=None):
    """Rigorous UPPER bound on the Perron root of an entrywise-NONNEGATIVE matrix (power iteration +
    Collatz-Wielandt  rho <= max_i (Av)_i / v_i  for any v > 0).  Returns (rho, v)."""
    n = A.shape[0]
    v = v0.copy() if (v0 is not None and v0.shape == (n,)) else np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            return 0.0, v
        v = w / nw
        if abs(nw - lam) <= tol * (nw + 1e-30):
            break
        lam = nw
    vp = v + 1e-12
    return float(np.max((A @ vp) / vp)), v


@dataclass
class KrawczykResult:
    ok: bool                    # existence + uniqueness proven
    rho: float                  # (upper bound of) spectral radius of the slope majorant
    x1: TM                      # Newton warm-step state polynomial x1(e)
    r: Optional[np.ndarray]     # verified radius (None if not ok)
    corr: np.ndarray            # residual radius  sup_e |M (F(x1(e)) - S(e))|
    score: np.ndarray           # per-symbol residual mass (split heuristic: bisect the top symbols)
    M: np.ndarray               # preconditioner (inverse point Jacobian)

    def enclosure(self) -> TM:
        """Certified state enclosure  z*(e) in x1(e) +- r  as a Taylor model."""
        if not self.ok:
            raise ValueError("no certificate (rho >= 1 or rigorous check failed)")
        return TM(self.x1.c0, self.x1.C, self.x1.Q, self.x1.R + self.r)


def krawczyk(system: AffineJacobianSystem, S: TM, z0: np.ndarray, corr_max_m: int = 1700,
             max_iter: int = 100) -> KrawczykResult:
    """Prove that F(z) = S(e) has a unique solution near z0 for every e, and enclose it.

    S        : TM of the specified right-hand side (m rows) over the symbols e (input + NN-relaxation symbols)
    z0       : linearisation point (typically the point solution at the box centre)
    corr_max_m : use the correlated remainder sum_j |M H_j| r_j up to this state size (memory O(m * nnz));
                 beyond it, the Jacobian TM is re-evaluated with r as an independent ball."""
    m = system.m
    Ks = S[1].shape[1]
    Kq = S[2].shape[-1]
    I = np.eye(m)
    M = np.linalg.inv(system.jacobian(z0))      # any M is valid; an inexact inverse shows up in |I - M J0|
    x0 = TM(z0.copy(), np.zeros((m, Ks)), np.zeros((m, Kq, Kq)), np.zeros(m))
    Lz = matmul(-M, sub(system.residual(x0), S))
    x1 = TM(z0 + Lz[0], Lz[1], Lz[2], np.zeros(m))           # chosen centre FUNCTION p(e) (no remainder)
    dc = matmul(M, sub(system.residual(x1), S))
    corr = np.abs(dc[0]) + arad(dc)                             # sup_e |M (F(p(e)) - S(e))| incl. the constant part
    if CONFIG.rigorous:
        corr = upr(corr, 2)
    score = np.abs(dc[1]).sum(0)
    _qsc = np.abs(dc[2]).sum((0, 2))
    score[: _qsc.shape[0]] = score[: _qsc.shape[0]] + _qsc
    J0, JlinC0, JQ0, R0 = system.jacobian_tm(x1)
    c0z = np.abs(I - mdotsp(M, J0))
    hs = system.hessian_structure() if m <= corr_max_m else None
    use_corr = hs is not None
    base_lin = system.lin_sum(M, JlinC0)
    base_rem = np.abs(M) @ R0
    if use_corr:
        terms = hs if isinstance(hs, list) else [(hs, None)]
        wq = system.quad_weights(JQ0)
        wq = wq if isinstance(wq, list) else [wq]
        prep = []                                                # (MHcol, Hju, ScT, scale) per term
        base_quad = 0
        for (Hrr, Hinv, Hvv, Hcu, Hju, ShT, ScT), scale in terms:
            contrib = M[:, Hrr] * Hvv[None, :]
            MHcol = np.abs(np.asarray(ShT @ contrib.T).T)        # |M H_j| on the populated (col, j) pairs
            prep.append((MHcol, Hju, ScT, scale))
        for (MHcol, Hju, ScT, scale), w in zip(prep, wq):
            base_quad = base_quad + np.asarray(ScT @ (MHcol * w[Hju][None, :]).T).T
    else:
        base_quad = system.quad_sum(M, JQ0)

    def _Kb(rr):
        if use_corr:                                             # r as CORRELATED generators: sum_j |M H_j| s_j r_j
            rem2 = 0
            for MHcol, Hju, ScT, scale in prep:
                rs = rr if scale is None else scale * rr
                rem2 = rem2 + np.asarray(ScT @ (MHcol * rs[Hju][None, :]).T).T
            return c0z + base_lin + base_quad + (base_rem + rem2)
        _, JlinC, JQ, R_rad = system.jacobian_tm(TM(x1[0], x1[1], x1[2], x1[3] + rr))
        return c0z + system.lin_sum(M, JlinC) + system.quad_sum(M, JQ) + np.abs(M) @ R_rad

    def _verify(rn, pv_hint=None):
        """Rigorous a-posteriori check (epsilon-inflation, self-consistent radius) including the rounding of
        every Kb term:  Kr(rh) = up(Kb(rh)) + up(gamma (|M| Jmaj + I));  accept r' = (I - Kr)^-1 (corr + eta)
        when r' <= rh and corr + Kr r' < r' strictly.  Returns (r', rho_upper) or None."""
        nA = Ks * Ks + Ks + 4 * m + 64
        absM = np.abs(M)
        tiny = np.finfo(float).tiny

        def _Kr(rh):
            Kf = _Kb(rh)
            Jmaj = system.jac_majorant(x1, rh)
            nB = int(np.count_nonzero(Jmaj, axis=0).max()) + system.jac_entry_terms
            Eb = upr(gam(nB) * upr(mdotsp(absM, Jmaj) + I, nB), 4)
            return upr(Kf, nA) + Eb

        cmax = float(corr.max())
        rmax = float(rn.max())
        if not (np.isfinite(cmax) and cmax > 0):
            return None
        for dl in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
            eta = upr(0.5 * dl * (corr + 1e-3 * cmax), 2)
            rh = upr(rn * (1.0 + dl) + dl * 1e-3 * rmax, 4) + tiny
            for _k in range(10):
                Kr = _Kr(rh)
                try:
                    rp = np.linalg.solve(I - Kr, corr + eta)
                except np.linalg.LinAlgError:
                    break
                if not (np.isfinite(rp).all() and (rp > 0).all()):
                    break
                KrR = upr(Kr @ rp, m + 4)
                lhs = upr(corr + KrR, 4)
                if (rp <= rh).all() and (lhs < rp).all():
                    rho_up = float(upr(KrR / rp, 4).max())
                    if pv_hint is not None and pv_hint.shape == rp.shape:
                        vp = np.abs(pv_hint) + 1e-12 * np.abs(pv_hint).max() + tiny
                        rho_up = min(rho_up, float(upr(upr(Kr @ vp, m + 4) / vp, 4).max()))
                    return rp, rho_up
                rh = np.maximum(rh, upr(rp * (1.0 + 0.5 * dl), 2))
        return None

    def _fail(rho):
        return KrawczykResult(False, rho, x1, None, corr, score, M)

    r = np.zeros(m)
    rho = np.inf
    pvec = None
    for _it in range(max_iter):
        Kb = _Kb(r)
        rho, pvec = specrad(Kb, v0=pvec)
        if rho >= 1:
            return _fail(rho)
        rn = corr.copy()
        tk = corr.copy()
        for _nk in range(400):                                    # (I - Kb)^-1 corr via the Neumann series
            tk = Kb @ tk
            rn = rn + tk
            if tk.max() <= 1e-12 * (rn.max() + 1e-30):
                break
        else:
            rn = np.linalg.solve(I - Kb, corr)
        if (rn <= r + 1e-12 * (r + 1e-9)).all():                 # converged from below (no feedback double-count)
            if CONFIG.rigorous:
                ver = _verify(rn, pvec)
                if ver is None:
                    return _fail(rho)
                rn, rho = ver
            return KrawczykResult(True, rho, x1, rn, corr, score, M)
        r = np.maximum(r, rn)                                     # monotone inflation from below
    return _fail(rho)
