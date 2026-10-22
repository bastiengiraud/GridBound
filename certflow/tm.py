"""Degree-2 Taylor models over shared noise symbols.

    x(e) = c0 + C e + e^T Q e + [-R, R],      e in [-1, 1]^K

A TM is stored as the 4-tuple ``(c0, C, Q, R)`` (a :class:`TM` named tuple) of arrays with a common leading
shape (usually ``(n,)`` for a vector of n quantities):  c0 ``(n,)``, C ``(n, K)``, Q ``(n, Kq, Kq)``, R ``(n,)``.
C may carry more symbols than Q (``K >= Kq``): the extra symbols are linear-only.

Every operation keeps the quadratic part exactly, bounds degree >= 3 terms into R, and (in rigorous mode,
see :mod:`certflow.fp`) adds its own floating-point rounding error to R, so the result is a sound enclosure
of the exact-real quantity.  Affine zonotopes are the special case Q = 0, R = 0.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .fp import CONFIG, ETA, U, gam, odn, oup, upr


class TM(NamedTuple):
    c0: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    # ---------------------------------------------------------------- construction
    @staticmethod
    def affine(c0, C, R=None, kq=None):
        """Affine zonotope c0 + C e (+ optional remainder R); Q = 0 of width ``kq`` (default: all symbols)."""
        c0 = np.asarray(c0, float)
        C = np.asarray(C, float)
        K = C.shape[-1]
        kq = K if kq is None else kq
        Q = np.zeros(c0.shape + (kq, kq))
        R = np.zeros_like(c0) if R is None else np.asarray(R, float) * np.ones_like(c0)
        return TM(c0, C, Q, R)

    @staticmethod
    def const(c, like):
        return const(c, like)

    # ---------------------------------------------------------------- queries
    @property
    def n_sym(self) -> int:
        return self.C.shape[-1]

    def dev(self):
        """Rigorous (lo, hi) enclosure of x - c0 over the box."""
        return dev(self)

    def bounds(self):
        """Rigorous (lo, hi) enclosure of x over the box (outward rounded)."""
        lo, hi = dev(self)
        return odn(self.c0 + lo), oup(self.c0 + hi)

    def rad(self):
        """max |x - c0| over the box."""
        return arad(self)

    def take(self, idx):
        """Sub-vector (rows ``idx``)."""
        return TM(self.c0[idx], self.C[idx], self.Q[idx], self.R[idx])

    def eval(self, e):
        """Polynomial part at a point e (length K); R is NOT included (use for testing)."""
        e = np.asarray(e, float)
        kq = self.Q.shape[-1]
        q = np.einsum("...ab,a,b->...", self.Q, e[:kq], e[:kq]) if kq else 0.0
        return self.c0 + self.C @ e + q

    # ---------------------------------------------------------------- operators
    def __add__(self, o):  # type: ignore[override]
        return add(self, o)

    def __sub__(self, o):
        return sub(self, o)

    def __mul__(self, o):  # type: ignore[override]
        return mul(self, o)


def concat(tms, axis=0):
    """Stack TMs along the leading axis (symbol widths are zero-padded to the widest)."""
    K = max(t[1].shape[-1] for t in tms)
    kq = max(t[2].shape[-1] for t in tms)
    tms = [pad_symbols(t, K, kq) for t in tms]
    return TM(*(np.concatenate([t[c] for t in tms], axis) for c in range(4)))


def pad_symbols(a, K, kq=None):
    """Zero-pad the linear (and quadratic) symbol dimension of ``a`` up to K (resp. kq)."""
    c0, C, Q, R = a
    kq = Q.shape[-1] if kq is None else kq
    if C.shape[-1] < K:
        C = np.concatenate([C, np.zeros(C.shape[:-1] + (K - C.shape[-1],))], -1)
    if Q.shape[-1] < kq:
        Qn = np.zeros(Q.shape[:-2] + (kq, kq))
        Qn[..., : Q.shape[-1], : Q.shape[-1]] = Q
        Q = Qn
    return TM(c0, C, Q, R)


# ==================================================================================================
# primitives
# ==================================================================================================

def mag(a):
    """Per-entry max |polynomial part| over the box: |c0| + sum|C| + sum|Q| (R not included)."""
    return np.abs(a[0]) + np.abs(a[1]).sum(-1) + np.abs(a[2]).sum((-1, -2))


def nt(a):
    """Number of coefficients per TM entry (length of the sums inside ``mag``)."""
    return 1 + a[1].shape[-1] + a[2].shape[-1] ** 2


def const(c, like):
    """Constant TM shaped like ``like`` (exact, R = 0)."""
    return TM(np.asarray(c, float) * np.ones_like(like[0]), np.zeros_like(like[1]), np.zeros_like(like[2]),
              np.zeros_like(like[3]))


def dev(x):
    """Rigorous enclosure (lo, hi) of x - c0 over the box.

    Per-symbol parabolas (diagonal Q_kk e_k^2 + C_k e_k, including the vertex when it lies inside [-1,1])
    plus the off-diagonal cross terms bounded by sum_{k != j} |Q_kj| and the remainder R."""
    c0, C, Q, R = x
    dq = np.diagonal(Q, axis1=-2, axis2=-1)
    if C.shape[-1] != dq.shape[-1]:                       # extra LINEAR-ONLY symbols -> pure linear range
        dq = np.concatenate([dq, np.zeros(dq.shape[:-1] + (C.shape[-1] - dq.shape[-1],))], -1)
    fm = dq - C
    fp = dq + C
    lo = np.minimum(fm, fp)
    hi = np.maximum(fm, fp)
    with np.errstate(divide="ignore", invalid="ignore"):
        es = -C / (2 * dq)
        fv = -(C * C) / (4 * dq)
    ins = (np.abs(es) <= 1) & (dq != 0)
    lo = np.where(ins, np.minimum(lo, fv), lo)
    hi = np.where(ins, np.maximum(hi, fv), hi)
    S1 = np.abs(Q).sum((-1, -2))
    S2 = np.abs(dq).sum(-1)
    off = S1 - S2                                         # sum_{k != j} |Q_kj|
    lo_s = lo.sum(-1)
    hi_s = hi.sum(-1)
    if not CONFIG.rigorous:
        return lo_s - off - R, hi_s + off + R
    Kc = C.shape[-1]
    n = Kc * Kc + Kc + 16
    g = gam(n)
    fvi = np.where(ins, np.abs(np.nan_to_num(fv)), 0.0)
    err = upr(g * (np.abs(lo).sum(-1) + np.abs(hi).sum(-1) + np.abs(dq).sum(-1) + np.abs(C).sum(-1) + fvi.sum(-1)
                   + 2.0 * (S1 + S2) + R + np.abs(lo_s) + np.abs(hi_s)), n)
    return odn(lo_s - off - R - err), oup(hi_s + off + R + err)


def arad(x):
    """max |x - c0| over the box."""
    lo, hi = dev(x)
    return np.maximum(np.abs(lo), np.abs(hi))


def bounds(x):
    """Rigorous (lo, hi) of x over the box."""
    lo, hi = dev(x)
    return odn(x[0] + lo), oup(x[0] + hi)


def _align(a, b):
    """Zero-pad two TMs to a common symbol width (symbols are shared by position: new symbols are appended)."""
    if a[1].shape[-1] == b[1].shape[-1] and a[2].shape[-1] == b[2].shape[-1]:
        return a, b
    K = max(a[1].shape[-1], b[1].shape[-1])
    kq = max(a[2].shape[-1], b[2].shape[-1])
    return pad_symbols(a, K, kq), pad_symbols(b, K, kq)


def add(a, b):
    a, b = _align(a, b)
    R = a[3] + b[3]
    if CONFIG.rigorous:
        R = upr(R + U * (mag(a) + mag(b)), nt(a) + 4)
    return TM(a[0] + b[0], a[1] + b[1], a[2] + b[2], R)


def sub(a, b):
    a, b = _align(a, b)
    R = a[3] + b[3]
    if CONFIG.rigorous:
        R = upr(R + U * (mag(a) + mag(b)), nt(a) + 4)
    return TM(a[0] - b[0], a[1] - b[1], a[2] - b[2], R)


def scale(mat, a):
    """Elementwise constant (broadcast over the leading shape) times TM."""
    R = np.abs(mat) * a[3]
    if CONFIG.rigorous:
        R = upr(R + U * np.abs(mat) * mag(a), nt(a) + 4)
    return TM(mat * a[0], mat[..., None] * a[1], mat[..., None, None] * a[2], R)


def div(a, bc):
    """TM divided by a (positive or negative, nonzero) elementwise constant."""
    R = a[3] / np.abs(bc)
    if CONFIG.rigorous:
        R = upr((a[3] + U * mag(a)) / np.abs(bc), nt(a) + 6)
    return TM(a[0] / bc, a[1] / bc[..., None], a[2] / bc[..., None, None], R)


def matmul(M, a):
    """Dense constant matrix M @ TM a."""
    R = np.tensordot(np.abs(M), a[3], (1, 0))
    if CONFIG.rigorous:
        n = int(np.count_nonzero(M, axis=1).max()) if M.size else 0   # zero terms add exactly
        R = upr(R + gam(n + 1) * np.tensordot(np.abs(M), mag(a), (1, 0)), n + nt(a) + 4)
    return TM(np.tensordot(M, a[0], (1, 0)), np.tensordot(M, a[1], (1, 0)), np.tensordot(M, a[2], (1, 0)), R)


def matmul_sparse(Msp, a):
    """SPARSE (scipy) constant matrix Msp @ TM a."""
    Q = a[2]
    if Q.size:
        m0, k1, k2 = Q.shape
        Qo = (Msp @ Q.reshape(m0, k1 * k2)).reshape(Msp.shape[0], k1, k2)
    else:
        Qo = np.zeros((Msp.shape[0], Q.shape[1], Q.shape[2]))
    R = abs(Msp) @ a[3]
    if CONFIG.rigorous:
        _c = Msp.tocsr()
        n = int(np.diff(_c.indptr).max()) if _c.shape[0] else 0
        R = upr(R + gam(n + 1) * (abs(Msp) @ mag(a)), n + nt(a) + 4)
    return TM(Msp @ a[0], Msp @ a[1], Qo, R)


def mul(a, b):
    """Elementwise TM product: keep degree 2 exactly, bound degree >= 3 terms into R."""
    a, b = _align(a, b)
    a0, Ca, Qa, Ra = a
    b0, Cb, Qb, Rb = b
    c0 = a0 * b0
    C = a0[..., None] * Cb + b0[..., None] * Ca
    rla = np.abs(Ca).sum(-1)
    rlb = np.abs(Cb).sum(-1)
    kq = Qa.shape[-1]
    if kq < Ca.shape[-1]:                                 # linear-only tail symbols: products touching them -> R
        Caq, Cbq = Ca[..., :kq], Cb[..., :kq]
        outer = Caq[..., :, None] * Cbq[..., None, :]
        Q = a0[..., None, None] * Qb + b0[..., None, None] * Qa + 0.5 * (outer + np.swapaxes(outer, -1, -2))
        rla_t = np.abs(Ca[..., kq:]).sum(-1)
        rlb_t = np.abs(Cb[..., kq:]).sum(-1)
        excess = rla_t * rlb + np.abs(Caq).sum(-1) * rlb_t
    else:
        outer = Ca[..., :, None] * Cb[..., None, :]
        Q = a0[..., None, None] * Qb + b0[..., None, None] * Qa + 0.5 * (outer + np.swapaxes(outer, -1, -2))
        excess = 0.0
    qa = np.abs(Qa).sum((-1, -2))
    qb = np.abs(Qb).sum((-1, -2))
    pam = np.abs(a0) + rla + qa
    pbm = np.abs(b0) + rlb + qb
    R = rla * qb + qa * rlb + qa * qb + pam * Rb + pbm * Ra + Ra * Rb + excess
    if CONFIG.rigorous:
        n = 3 * nt(a) + 40
        R = upr(R + gam(6) * pam * pbm, n)
    return TM(c0, C, Q, R)


def relu(a):
    """DeepZ ReLU on a TM: each crossing component gets its OWN new error symbol (correlation preserved)."""
    c0, C, Q, R = a
    lo, hi = dev(a)
    vlo = c0 + lo
    vhi = c0 + hi
    if CONFIG.rigorous:
        vlo = odn(vlo)
        vhi = oup(vhi)
    cr = (vlo < 0) & (vhi > 0)
    lam = np.where(cr, vhi / np.maximum(vhi - vlo, 1e-30), (vlo >= 0).astype(float))
    mu = np.where(cr, -0.5 * lam * vlo, 0.0)
    if CONFIG.rigorous:                                   # relaxation valid for ANY lam in [0,1]
        lam = np.where(cr, np.clip(lam, 0.0, 1.0), lam)
        mu = np.where(cr, upr(0.5 * np.maximum(-lam * vlo, (1.0 - lam) * vhi), 6), 0.0)
    c0n = lam * c0 + mu
    Cn = lam[:, None] * C
    Qn = lam[:, None, None] * Q
    Rn = lam * R
    idx = np.where(cr)[0]
    Kc = C.shape[1]
    if CONFIG.rigorous:
        Rn = np.where(cr, upr(lam * R + gam(3) * (lam * mag(a) + mu), nt(a) + 10), lam * R)
    if len(idx):
        nc = np.zeros((len(c0), len(idx)))
        nc[idx, np.arange(len(idx))] = mu[idx]
        Cn = np.concatenate([Cn, nc], 1)
        Qp = np.zeros((len(c0), Kc + len(idx), Kc + len(idx)))
        Qp[:, :Kc, :Kc] = Qn
        Qn = Qp
    return TM(c0n, Cn, Qn, Rn)


def clip(y, lo, hi):
    """Componentwise clip of TM y into [lo, hi] via two correlated ReLUs: max(min(y, hi), lo).
    Exact (pass-through) for components that never reach a bound."""
    r1 = relu(sub(const(hi, y), y))
    v1 = sub(const(hi, r1), r1)                           # min(y, hi) = hi - relu(hi - y)
    r2 = relu(sub(v1, const(lo, v1)))
    return add(const(lo, r2), r2)                         # max(., lo) = lo + relu(. - lo)


def add_const(a, c):
    """TM + exact float constant vector c (rounding of the centre add goes to R)."""
    c = np.asarray(c, float)
    R = a[3]
    if CONFIG.rigorous:
        R = upr(R + U * (np.abs(a[0]) + np.abs(c)), 4)
    return TM(a[0] + c, a[1], a[2], R)


def compose(x, f, df, d2_range, lib_ulps=8.0):
    """Elementwise univariate function of a TM:  f(x)  with  x = c + (x - c),  c = x.c0:

        f(x) = f(c) + f'(c) (x - c) + 0.5 f''(xi) (x - c)^2,     xi in [lo, hi] = range of x.

    The linear term keeps x's full (degree-2, correlated) structure; the Lagrange term is bounded with the
    range of f'' (``d2_range(lo, hi) -> (m2lo, m2hi)``, rigorous) and goes to the centre/remainder.
    ``f``, ``df`` are evaluated in float; their library error (``lib_ulps`` ulps) is bounded into R."""
    c0, C, Q, R = x
    lo, hi = bounds(x)
    dlo, dhi = dev(x)
    rad2 = np.maximum(dlo * dlo, dhi * dhi)
    fc = f(c0)
    dfc = df(c0)
    m2lo, m2hi = d2_range(lo, hi)
    rlo = 0.5 * np.minimum(m2lo, 0.0) * rad2
    rhi = 0.5 * np.maximum(m2hi, 0.0) * rad2
    mid = 0.5 * (rlo + rhi)
    half = 0.5 * (rhi - rlo)
    out_R = np.abs(dfc) * R + half
    if CONFIG.rigorous:
        rad = np.sqrt(rad2)
        err = (lib_ulps * U * (np.abs(fc) + np.abs(dfc) * rad)          # f(c), f'(c) library error
               + U * np.abs(dfc) * mag(TM(np.zeros_like(c0), C, Q, R))     # coefficient scaling
               + 2 * U * (np.abs(fc) + np.abs(mid) + np.abs(rlo) + np.abs(rhi)))
        out_R = upr(out_R + err, nt(x) + 12)
    return TM(fc + mid, dfc[..., None] * C, dfc[..., None, None] * Q, out_R)


def sqrt(x, nonneg=False, max_ratio=4.0):
    """sqrt of a TM.  Uses the correlated Taylor composition where the range is positive (kept unless it is more
    than ``max_ratio`` times wider than the plain interval -- correlation usually matters more downstream than
    width), else the interval [sqrt(lo), sqrt(hi)] as centre + remainder.
    nonneg=True asserts that the exact quantity is >= 0 (e.g. a squared magnitude) so a range reaching
    below 0 is clipped at 0; otherwise such a range raises."""
    lo, hi = bounds(x)
    if np.any(lo <= 0) and not nonneg:
        raise ValueError("tm.sqrt: range not strictly positive (pass nonneg=True for a squared magnitude)")
    lo0 = np.maximum(lo, 0.0)
    slo = odn(np.sqrt(lo0) * (1 - 4 * U))
    shi = oup(np.sqrt(hi) * (1 + 4 * U))
    iv = TM(0.5 * (slo + shi), np.zeros_like(x[1]), np.zeros_like(x[2]), oup(0.5 * (shi - slo) * (1 + 4 * U)))
    pos = lo > 0
    if not pos.any():
        return iv

    def d2(lo_, hi_):                                       # f'' = -x^{-3/2}/4: most negative at lo, nearest 0 at hi
        return -0.25 * lo_ ** -1.5 * (1 + 16 * U), -0.25 * hi_ ** -1.5 * (1 - 16 * U)
    safe = TM(np.where(pos, x[0], 1.0), x[1] * pos[..., None], x[2] * pos[..., None, None], np.where(pos, x[3], 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tmv = compose(safe, np.sqrt, lambda c: 0.5 / np.sqrt(c), d2)
    tlo, thi = bounds(tmv)
    use = pos & np.isfinite(thi - tlo) & ((thi - tlo) <= max_ratio * (shi - slo))
    return TM(np.where(use, tmv[0], iv[0]), np.where(use[..., None], tmv[1], 0.0),
              np.where(use[..., None, None], tmv[2], 0.0), np.where(use, tmv[3], iv[3]))


def maximum(a, b):
    """Elementwise max(a, b) = a + relu(b - a) (correlated; exact where one side always dominates)."""
    return add(a, relu(sub(b, a)))


__all__ = ["TM", "compose", "sqrt", "maximum", "concat", "pad_symbols", "mag", "nt", "const", "dev", "arad", "bounds", "add", "sub", "scale",
           "div", "matmul", "matmul_sparse", "mul", "relu", "clip", "add_const", "ETA"]
