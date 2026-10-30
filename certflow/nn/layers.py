"""Layer-wise abstract transformers (DeepZ-style signed-generator zonotopes with rigorous rounding control).

The reachable set of a layer's output is carried as a :class:`ZonoState`

    exact-real value  in  c0 + C e + [-err, err]                 (e in [-1,1]^K, shared symbols)
    deployed value    in  exact-real value + [-dep, dep]          (forward error of the deployed float pass)
    hmag              >=  max |exact-real value|                  (used to bound the deployed rounding)

Affine layers act exactly on (c0, C); activations relax each neuron over its rigorous pre-activation interval
[lo, hi] as  act(x) in lam x + mu1 + mu2 e_new  and append ONE new signed generator per relaxed neuron, so the
relaxation error keeps its own symbol all the way to the output (it can cancel downstream).  All float
roundings of this pass are bounded into ``err``; ``dep`` bounds the deployed network's own arithmetic
(unit roundoff ``ctx.deploy_u``, e.g. float32).  With rigorous mode off (fp.set_rigorous(False)) the classic
non-rigorous DeepZ formulas are used.

To add a new activation, subclass :class:`Layer` and implement ``reach`` (and ``forward``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fp import CONFIG, U, gam, odn, oup, upr
from ..tm import TM, mag, nt


@dataclass
class ReachContext:
    deploy_u: float = U         # unit roundoff of the DEPLOYED network arithmetic (2^-53 float64, 2^-24 float32)
    fold_below: float = 0.0     # relaxation errors mu2 <= fold_below go to err instead of a new symbol

    def gd(self, n):
        return gam(n, self.deploy_u)


@dataclass
class ZonoState:
    c0: np.ndarray
    C: np.ndarray
    err: np.ndarray
    dep: np.ndarray
    hmag: np.ndarray

    @property
    def n(self):
        return self.c0.shape[0]

    def interval(self):
        """Rigorous pre-activation interval [lo, hi] of the EXACT network value."""
        if not CONFIG.rigorous:
            rad = np.abs(self.C).sum(1)
            return self.c0 - rad, self.c0 + rad
        rad = upr(np.abs(self.C).sum(1), self.C.shape[1] + 2)
        lo = odn(odn(self.c0 - rad) - self.err)
        hi = oup(oup(self.c0 + rad) + self.err)
        return lo, hi


# ==================================================================================================
# entry / exit between TaylorModel and ZonoState
# ==================================================================================================

def enter_identity(x: TM, ctx: ReachContext) -> ZonoState:
    """Generic entry: the input TM's quadratic part and remainder become interval error; the deployed input
    is the input cast to the deployed dtype (exact for float64)."""
    K = x.C.shape[1]
    c0 = np.array(x.c0, float)
    C = np.array(x.C, float)
    if not CONFIG.rigorous:
        z = np.zeros_like(c0)
        return ZonoState(c0, C, z, z.copy(), np.abs(c0) + np.abs(C).sum(1))
    xq = np.abs(x.Q).sum((-1, -2)) + x.R
    err = upr(xq, nt(x) + 2)
    xin = upr(mag(x) + x.R, nt(x) + 2)
    dep = upr(ctx.deploy_u * xin, 2) if ctx.deploy_u > U else np.zeros_like(c0)
    hmag = upr(np.abs(c0) + np.abs(C).sum(1) + err, K + 4)
    return ZonoState(c0, C, err, dep, hmag)


def exit_identity(s: ZonoState, ctx: ReachContext) -> TM:
    Kt = s.C.shape[1]
    if not CONFIG.rigorous:
        return TM(s.c0, s.C, np.zeros((s.n, Kt, Kt)), np.zeros(s.n))
    return TM(s.c0, s.C, np.zeros((s.n, Kt, Kt)), upr(s.err + s.dep, 2))


# ==================================================================================================
# layers
# ==================================================================================================

class Layer:
    """Base class.  ``reach`` maps a ZonoState to a ZonoState; ``forward`` evaluates on concrete inputs
    (float64, batch-first or a single vector)."""

    def reach(self, s: ZonoState, ctx: ReachContext) -> ZonoState:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__ + "()"


class Linear(Layer):
    """y = W x + b  (W stored as (out, in), like torch)."""

    def __init__(self, W, b=None):
        self.W = np.asarray(W, float)
        self.b = np.zeros(self.W.shape[0]) if b is None else np.asarray(b, float)

    def __repr__(self):
        return f"Linear({self.W.shape[1]}->{self.W.shape[0]})"

    def forward(self, x):
        return x @ self.W.T + self.b

    def reach(self, s, ctx):
        W, b = self.W, self.b
        if not CONFIG.rigorous:
            c0 = W @ s.c0 + b
            C = W @ s.C
            return ZonoState(c0, C, s.err, s.dep, np.abs(c0) + np.abs(C).sum(1))
        aW = np.abs(W)
        ab = np.abs(b)
        n = W.shape[1]
        pm = np.abs(s.c0) + np.abs(s.C).sum(1)
        dep = upr(aW @ s.dep + ctx.gd(n + 2) * (aW @ upr(s.hmag + s.dep, 2) + ab), n + 8)
        err = upr(aW @ s.err + gam(n + 2) * (aW @ pm + ab), n + s.C.shape[1] + 8)
        c0 = W @ s.c0 + b
        C = W @ s.C
        hmag = upr(np.abs(c0) + np.abs(C).sum(1) + err, C.shape[1] + 4)
        return ZonoState(c0, C, err, dep, hmag)


def _append_generators(C, coef, idx):
    if len(idx):
        nc = np.zeros((C.shape[0], len(idx)))
        nc[idx, np.arange(len(idx))] = coef[idx]
        C = np.concatenate([C, nc], 1)
    return C


class ReLU(Layer):
    """DeepZ ReLU (minimal-area parallelogram; one new symbol per crossing neuron)."""

    def forward(self, x):
        return np.maximum(x, 0.0)

    def reach(self, s, ctx):
        lo, hi = s.interval()
        cr = (lo < 0) & (hi > 0)
        if not CONFIG.rigorous:
            lam = np.where(cr, hi / np.maximum(hi - lo, 1e-30), (lo >= 0).astype(float))
            mu = np.where(cr, -0.5 * lam * lo, 0.0)
            c0 = lam * s.c0 + mu
            C = _append_generators(lam[:, None] * s.C, mu, np.where(cr)[0])
            return ZonoState(c0, C, s.err, s.dep, np.maximum(hi, 0.0))
        lam = np.where(cr, np.clip(hi / np.maximum(hi - lo, 1e-300), 0.0, 1.0), (lo >= 0).astype(float))
        # mu = 0.5 max(-lam lo, (1-lam) hi) makes relu(x) in lam x + mu +- mu valid for ANY lam in [0,1]
        mu = np.where(cr, upr(0.5 * np.maximum(-lam * lo, (1.0 - lam) * hi), 6), 0.0)
        pm = np.abs(s.c0) + np.abs(s.C).sum(1)
        c0 = lam * s.c0 + mu
        C = lam[:, None] * s.C
        err = np.where(cr, upr(lam * s.err + gam(3) * (lam * pm + mu), C.shape[1] + 10), lam * s.err)
        C = _append_generators(C, mu, np.where(cr)[0])
        hmag = upr(np.maximum(hi, 0.0), 2)                 # exact relu output in [0, max(hi,0)]
        return ZonoState(c0, C, err, s.dep, hmag)          # dep unchanged: relu is 1-Lipschitz and exact in float


class _SShaped(Layer):
    """Monotone increasing, bounded activation whose derivative is even and unimodal with its maximum at 0
    (tanh, sigmoid).  DeepZ relaxation (Singh et al. 2018): slope lam = min(f'(lo), f'(hi)) makes the residual
    g(x) = f(x) - lam x nondecreasing on [lo, hi], so its range is [g(lo), g(hi)] exactly; the relaxation is
    f(x) in lam x + mu1 + mu2 e_new with mu1 = (g(lo)+g(hi))/2, mu2 = (g(hi)-g(lo))/2.

    Rigorous mode: lam is shrunk by (1 - 64u) so it stays <= the true minimum derivative despite libm rounding
    (assumes the math library's exp/cosh/tanh are accurate to a few ulp, as for glibc/numpy), and the endpoint
    residuals are padded outward."""

    f_abs_max = 1.0           # sup |f|
    lib_ulps = 8.0            # assumed accuracy (ulps) of the math library / deployed kernel for f

    def f(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def df(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x):
        return self.f(x)

    def reach(self, s, ctx):
        lo, hi = s.interval()
        with np.errstate(over="ignore"):
            dlo = self.df(lo)
            dhi = self.df(hi)
        lam = np.minimum(dlo, dhi)
        lam = np.where(np.isfinite(lam), lam, 0.0)
        flo = self.f(lo)
        fhi = self.f(hi)
        if not CONFIG.rigorous:
            glo = flo - lam * lo
            ghi = fhi - lam * hi
            mu1 = 0.5 * (glo + ghi)
            mu2 = 0.5 * (ghi - glo)
            c0 = lam * s.c0 + mu1
            C = lam[:, None] * s.C
            new = mu2 > ctx.fold_below
            err = s.err + np.where(new, 0.0, mu2)
            C = _append_generators(C, mu2, np.where(new)[0])
            return ZonoState(c0, C, err, s.dep, np.maximum(np.abs(flo), np.abs(fhi)))
        lam = np.maximum(lam * (1.0 - 64.0 * U), 0.0)
        tol = self.lib_ulps * U
        plo = upr(tol * np.abs(flo) + 2.0 * U * (np.abs(flo) + np.abs(lam * lo)), 6) + 8 * 2.0 ** -1074
        phi = upr(tol * np.abs(fhi) + 2.0 * U * (np.abs(fhi) + np.abs(lam * hi)), 6) + 8 * 2.0 ** -1074
        glo = odn(odn(flo - lam * lo) - plo)
        ghi = oup(oup(fhi - lam * hi) + phi)
        mu1 = 0.5 * (glo + ghi)
        mu2 = upr(0.5 * (ghi - glo) + U * (np.abs(glo) + np.abs(ghi)), 6)   # also absorbs the rounding of mu1
        pm = np.abs(s.c0) + np.abs(s.C).sum(1)
        c0 = lam * s.c0 + mu1
        C = lam[:, None] * s.C
        err = upr(lam * s.err + gam(3) * (lam * pm + np.abs(mu1)), C.shape[1] + 10)
        new = mu2 > ctx.fold_below
        err = np.where(new, err, upr(err + mu2, 2))
        C = _append_generators(C, mu2, np.where(new)[0])
        hmag = np.minimum(self.f_abs_max, upr(np.maximum(np.abs(flo), np.abs(fhi)) * (1.0 + tol), 2))
        # deployed: f is 1-Lipschitz for tanh (1/4 for sigmoid) -> dep passes through, plus the kernel rounding
        dep = upr(self.lipschitz * s.dep + self.lib_ulps * ctx.deploy_u * self.f_abs_max, 4)
        return ZonoState(c0, C, err, dep, hmag)


class Tanh(_SShaped):
    lipschitz = 1.0

    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1.0 / np.cosh(x) ** 2


class Sigmoid(_SShaped):
    lipschitz = 0.25

    def f(self, x):
        return 0.5 * (1.0 + np.tanh(0.5 * x))   # = 1/(1+exp(-x)), overflow-free

    def df(self, x):
        return 1.0 / (2.0 + 2.0 * np.cosh(x))


class MinMaxNormalize(Layer):
    """Input layer x -> (x - xmn) / (xmx - xmn + eps)  (fused entry: reads the input TM directly).
    Includes the deployed-dtype normalization error."""

    is_entry = True

    def __init__(self, xmn, xmx, eps=1e-8):
        self.xmn = np.asarray(xmn, float)
        self.xmx = np.asarray(xmx, float)
        self.eps = float(eps)

    def forward(self, x):
        return (x - self.xmn) / (self.xmx - self.xmn + self.eps)

    def enter(self, x: TM, ctx: ReachContext) -> ZonoState:
        xmn, xmx = self.xmn, self.xmx
        den = xmx - xmn + self.eps
        if not CONFIG.rigorous:
            c0 = (x[0] - xmn) / den
            C = x[1] / den[:, None]
            z = np.zeros_like(c0)
            return ZonoState(c0, C, z, z.copy(), np.abs(c0) + np.abs(C).sum(1))
        K = x[1].shape[1]
        aden = np.abs(xmx) + np.abs(xmn) + self.eps
        xin = upr(mag(x) + x[3], nt(x) + 2)                   # max |input| over the input TM
        xq = np.abs(x[2]).sum((-1, -2)) + x[3]                # input quadratic part + remainder -> error
        c0 = (x[0] - xmn) / den
        C = x[1] / den[:, None]
        err = upr((xq + gam(4) * (np.abs(x[0]) + np.abs(xmn) + np.abs(x[1]).sum(-1))) / den
                  + gam(3) * aden * (xin + np.abs(xmn)) / (den * den), K + 20)
        den_lo = den - ctx.gd(4) * aden                       # deployed fl(fl(xmx-xmn)+eps), lower bound
        with np.errstate(divide="ignore"):
            dep = np.where(den_lo > 0, upr((xin + np.abs(xmn)) * ctx.gd(8) * (1.0 + aden / den) / den_lo, 16), np.inf)
        hmag = upr(np.abs(c0) + np.abs(C).sum(1) + err, K + 4)
        return ZonoState(c0, C, err, dep, hmag)

    def reach(self, s, ctx):  # only valid as the first layer
        raise RuntimeError("MinMaxNormalize must be the first layer of a network")


class MinMaxDenormalize(Layer):
    """Output layer y -> y (ymx - ymn) + ymn  (fused exit to a TaylorModel)."""

    is_exit = True

    def __init__(self, ymn, ymx):
        self.ymn = np.asarray(ymn, float)
        self.ymx = np.asarray(ymx, float)

    def forward(self, y):
        return y * (self.ymx - self.ymn) + self.ymn

    def exit(self, s: ZonoState, ctx: ReachContext) -> TM:
        d = self.ymx - self.ymn
        Kt = s.C.shape[1]
        if not CONFIG.rigorous:
            return TM(s.c0 * d + self.ymn, s.C * d[:, None], np.zeros((s.n, Kt, Kt)), np.zeros(s.n))
        ad = np.abs(d)
        pm = np.abs(s.c0) + np.abs(s.C).sum(1)
        om = upr(pm + s.err, Kt + 4)
        R = (upr(ad * s.err + gam(5) * (ad * pm + np.abs(self.ymn)), Kt + 8)
             + upr(ad * s.dep + ctx.gd(5) * (ad * upr(om + s.dep, 2) + np.abs(self.ymn)), 8))
        return TM(s.c0 * d + self.ymn, s.C * d[:, None], np.zeros((s.n, Kt, Kt)), upr(R, 2))

    def reach(self, s, ctx):
        raise RuntimeError("MinMaxDenormalize must be the last layer of a network")


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
