"""Adaptive input splitting (branch-and-bound over e-space boxes) and a Windows-safe process pool.

``adaptive_split`` is the generic driver used for the Krawczyk rho-tree: evaluate a box, and if it fails,
bisect it along the direction(s) chosen from its score.  The worker side receives the verifier object ONCE
(pickled in the pool initializer), so it runs under both 'fork' (Linux) and 'spawn' (Windows/macOS).
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np


def split_dims(score, wid, tau=0.5, m_max=4, use_score=True):
    """Directions to bisect: every direction whose score is within ``tau`` of the max (at most ``m_max``,
    ranked by score); falls back to the widest splittable direction."""
    wid = np.asarray(wid, float)
    splittable = wid > 1e-9
    if not splittable.any():
        return [int(np.argmax(wid))]
    widest = int(np.argmax(np.where(splittable, wid, -np.inf)))
    if not use_score or score is None:
        return [widest]
    sc = np.where(splittable, np.asarray(score, float), -np.inf)
    smax = float(sc.max())
    if not np.isfinite(smax) or smax <= 0:
        return [widest]
    order = np.argsort(-sc, kind="stable")
    dims = [int(k) for k in order if splittable[k] and sc[k] >= tau * smax][: max(1, m_max)]
    return dims or [widest]


def split_children(lo, hi, dims):
    """The 2^len(dims) sub-boxes from bisecting each dim at its midpoint."""
    mids = {d: 0.5 * (lo[d] + hi[d]) for d in dims}
    out = []
    for bits in range(1 << len(dims)):
        a = np.array(lo, float)
        b = np.array(hi, float)
        for i, d in enumerate(dims):
            if (bits >> i) & 1:
                a[d] = mids[d]
            else:
                b[d] = mids[d]
        out.append((a, b))
    return out


def bisect(lo, hi, d):
    mid = 0.5 * (lo[d] + hi[d])
    a_hi = np.array(hi, float)
    a_hi[d] = mid
    b_lo = np.array(lo, float)
    b_lo[d] = mid
    return (np.array(lo, float), a_hi), (b_lo, np.array(hi, float))


# ==================================================================================================
# process pool (the worker holds ONE pickled object and calls its methods)
# ==================================================================================================

_WORKER = {"obj": None}


def _init_worker(blob, rigorous=True):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from . import fp
    fp.set_rigorous(rigorous)                    # spawn-started workers do not inherit module state
    _WORKER["obj"] = pickle.loads(blob)


def _call(method, args, kwargs):
    return getattr(_WORKER["obj"], method)(*args, **kwargs)


class ObjectPool:
    """Process pool whose workers each hold a copy of ``obj``; ``submit(method, *args)`` runs obj.method.

    Works with the 'spawn' start method (Windows/macOS): guard your script with ``if __name__ == "__main__":``
    and make sure certflow is importable in the workers (installed, or on PYTHONPATH)."""

    def __init__(self, obj, nproc):
        self.nproc = int(nproc)
        from . import fp
        self.ex = ProcessPoolExecutor(max_workers=self.nproc, initializer=_init_worker,
                                      initargs=(pickle.dumps(obj), fp.rigorous()))

    def submit(self, method, *args, **kwargs):
        return self.ex.submit(_call, method, args, kwargs)

    def close(self):
        self.ex.shutdown(cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def adaptive_split(evaluate, lo0, hi0, choose_split, max_leaves=512, pool=None, method=None, extra=()):
    """Generic adaptive bisection.

    evaluate(lo, hi) -> (ok, payload, score)          (serial)   or
    pool.submit(method, lo, hi, *extra) -> same       (parallel; the pool's object implements ``method``)
    choose_split(lo, hi, score) -> list of (lo, hi) children

    Returns (leaves, ok_all): payloads of the boxes that succeeded; ok_all False if the leaf cap was hit."""
    leaves = []
    if pool is None:
        stack = [(np.array(lo0, float), np.array(hi0, float))]
        while stack:
            if len(leaves) + len(stack) > max_leaves:
                return leaves, False
            lo, hi = stack.pop()
            ok, payload, score = evaluate(lo, hi)
            if ok:
                leaves.append(payload)
            else:
                stack += choose_split(lo, hi, score)
        return leaves, True
    pend = {}
    ok_all = True
    capped = False
    f = pool.submit(method, np.array(lo0, float), np.array(hi0, float), *extra)
    pend[f] = (np.array(lo0, float), np.array(hi0, float))
    while pend:
        done, _ = wait(list(pend), return_when=FIRST_COMPLETED)
        for fut in done:
            lo, hi = pend.pop(fut)
            ok, payload, score = fut.result()
            if ok:
                leaves.append(payload)
            elif capped or len(leaves) + len(pend) + 1 > max_leaves:
                ok_all = False
                capped = True
            else:
                for a, b in choose_split(lo, hi, score):
                    pend[pool.submit(method, a, b, *extra)] = (a, b)
    return leaves, ok_all
