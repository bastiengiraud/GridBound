"""Input sets: affine zonotopes  x(e) = c + G e,  e in [-1, 1]^K, and their e-space sub-boxes.

Verification runs over boxes in e-space (``lo <= e <= hi``); a sub-box with centre ec and half-widths h is the
zonotope  c + G ec + (G diag h) e'.  :meth:`AffineSet.sub_tm` returns it as a Taylor model whose remainder
covers the floating-point rounding of the centre/generator computation, so the union of sub-boxes provably
covers the set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fp import CONFIG, gam, upr
from .tm import TM


@dataclass
class AffineSet:
    c: np.ndarray                        # centre (n,)
    G: np.ndarray                        # generators (n, K)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.c = np.asarray(self.c, float)
        self.G = np.asarray(self.G, float)

    @property
    def K(self) -> int:
        return self.G.shape[1]

    @property
    def n(self) -> int:
        return self.c.shape[0]

    def full_box(self):
        return np.full(self.K, -1.0), np.full(self.K, 1.0)

    def point(self, e):
        return self.c + self.G @ np.asarray(e, float)

    def sample(self, n, rng=None, corners=0.0):
        """n random points in e-space (uniform; a fraction ``corners`` drawn from the vertices)."""
        rng = np.random.default_rng(rng)
        E = rng.uniform(-1, 1, (n, self.K))
        if corners:
            k = int(round(corners * n))
            E[:k] = rng.choice([-1.0, 1.0], (k, self.K))
        return E

    def sub_center(self, ec):
        return self.c + self.G @ ec

    def sub_tm(self, ec, h, extra_R=None) -> TM:
        """Taylor model of the sub-box {c + G (ec + h e')}: centre c + G ec, generators G diag(h)."""
        ec = np.asarray(ec, float)
        h = np.asarray(h, float)
        K = self.K
        xc = self.c + self.G @ ec
        Gs = self.G * h[None, :]
        R = np.zeros(self.n) if extra_R is None else np.asarray(extra_R, float)
        if CONFIG.rigorous:
            R = R + upr(gam(K + 8) * (np.abs(self.c) + np.abs(self.G) @ (np.abs(ec) + h)), K + 8)
        return TM(xc, Gs, np.zeros((self.n, K, K)), R)


def low_rank_load_set(x0, Gdirs, frac, meta=None) -> AffineSet:
    """Low-rank deterministic load set:  x = x0 * (1 + frac * Gdirs e),  e in [-1, 1]^K.

    x0    : nominal stacked loads [Pd; Qd] (2 nl,)
    Gdirs : (2 nl, K) normalised mode matrix
    frac  : robustness radius r_f"""
    x0 = np.asarray(x0, float)
    Gdirs = np.asarray(Gdirs, float)
    m = dict(x0=x0, Gdirs=Gdirs, frac=float(frac))
    if meta:
        m.update(meta)
    return AffineSet(x0, frac * (Gdirs * x0[:, None]), m)


def box_set(lo, hi, meta=None) -> AffineSet:
    """Axis-aligned box  lo <= x <= hi  as an affine set (one symbol per coordinate with nonzero width)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    hw = 0.5 * (hi - lo)
    idx = np.where(hw > 0)[0]
    G = np.zeros((len(c), len(idx)))
    G[idx, np.arange(len(idx))] = hw[idx]
    return AffineSet(c, G, dict(meta or {}, axes=idx))
