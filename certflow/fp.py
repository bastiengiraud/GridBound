"""Rigorous floating-point error bounds (a-priori forward error analysis).

Every IEEE-754 double operation satisfies fl(a op b) = (a op b)(1 + d), |d| <= u = 2^-53 (round to nearest),
plus an absolute underflow term <= 2^-1074.  A length-n dot product satisfies
|fl(x.y) - x.y| <= gamma_n |x|.|y| with gamma_n = n u / (1 - n u)  (Higham, "Accuracy and Stability of
Numerical Algorithms", ch. 2-3).

Every Taylor-model / zonotope operation in certflow adds the rounding error of its own coefficient
computations to the interval remainder R, so the enclosures stay sound for the exact-real quantity.
Rigorous mode is ON by default; switch it off only for quick experiments (``set_rigorous(False)``).
"""
from __future__ import annotations

import contextlib

import numpy as np

U = 2.0 ** -53          # unit roundoff of float64
ETA = 2.0 ** -1074      # smallest subnormal (absolute underflow term)
U32 = 2.0 ** -24        # unit roundoff of float32 (deployed-network arithmetic, e.g. torch default)


class _Config:
    rigorous: bool = True


CONFIG = _Config()


def rigorous() -> bool:
    """True when floating-point rounding is bounded into the enclosures (default)."""
    return CONFIG.rigorous


def set_rigorous(on: bool) -> None:
    CONFIG.rigorous = bool(on)


@contextlib.contextmanager
def rigorous_mode(on: bool):
    """Temporarily switch rigorous floating-point accounting on/off."""
    old = CONFIG.rigorous
    CONFIG.rigorous = bool(on)
    try:
        yield
    finally:
        CONFIG.rigorous = old


def gam(n, u=U):
    """gamma_n = n u / (1 - n u)."""
    n = float(n)
    return n * u / (1.0 - n * u)


def upr(x, n):
    """Rigorous UPPER bound of a NONNEGATIVE float expression built from <= n roundings of nonnegative terms."""
    return x * (1.0 + 2.0 * gam(n + 4)) + (n + 4) * ETA


def odn(x):
    """Round outward (down) by one ulp after one correctly-rounded op."""
    return np.nextafter(x, -np.inf)


def oup(x):
    """Round outward (up) by one ulp after one correctly-rounded op."""
    return np.nextafter(x, np.inf)
