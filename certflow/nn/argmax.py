"""Discrete (argmax) outputs: which actions can a policy choose over an input set?

The deterministic action is argmax_a l_a(x) over the (unmasked) logits.  Over an input set the chosen action
is piecewise constant.  Action a can be chosen somewhere in the set only if, for every competitor b,
sup_e (l_a - l_b)(e) >= 0.  The difference is formed as a CORRELATED Taylor model (shared symbols), which is
far tighter than comparing independent logit intervals.  Ties (difference exactly 0) count as possible, so
the reachable set is sound for any tie-breaking rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fp import CONFIG, U, odn, oup, upr
from ..tm import TM, dev, mag, nt


def pairwise_diff_upper(logits: TM, idx_a, idx_b):
    """Rigorous upper bounds of sup_e (l_a - l_b) for all a in idx_a, b in idx_b -> (|A|, |B|) array."""
    a = logits.take(np.asarray(idx_a))
    b = logits.take(np.asarray(idx_b))
    c0 = a.c0[:, None] - b.c0[None, :]
    C = a.C[:, None, :] - b.C[None, :, :]
    Q = a.Q[:, None] - b.Q[None, :]
    R = a.R[:, None] + b.R[None, :]
    if CONFIG.rigorous:
        R = upr(R + U * (mag(a)[:, None] + mag(b)[None, :]), nt(a) + 4)
    D = TM(c0, C, Q, R)
    _, hi = dev(D)
    return oup(c0 + hi)


@dataclass
class ArgmaxResult:
    actions: np.ndarray          # actions that can be chosen somewhere in the input set (sorted)
    lo: np.ndarray               # rigorous logit lower bounds (all actions)
    hi: np.ndarray               # rigorous logit upper bounds
    valid: np.ndarray            # mask used

    @property
    def certain(self) -> bool:
        """True if one single action is chosen over the whole input set."""
        return len(self.actions) == 1

    @property
    def action(self):
        return int(self.actions[0]) if self.certain else None


def reachable_argmax(logits: TM, mask=None) -> ArgmaxResult:
    """Sound over-approximation of the set of argmax actions over the input set.

    logits : TM of the network logits (e.g. ``net.reach(x)``)
    mask   : boolean array of valid actions (masked logits are excluded, as in sb3_contrib MaskableCategorical
             when at least one action is valid)."""
    n = logits.c0.shape[0]
    valid = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    lo, hi = dev(logits)
    lo = odn(logits.c0 + lo)
    hi = oup(logits.c0 + hi)
    cand = np.where(valid)[0]
    if len(cand) <= 1:
        return ArgmaxResult(cand, lo, hi, valid)
    # 1) interval pruning: a is dominated if hi_a < max_b lo_b
    best_lo = lo[cand].max()
    cand = cand[hi[cand] >= best_lo]
    # 2) correlated pairwise pruning: a is out if some valid b has sup(l_a - l_b) < 0
    if len(cand) > 1:
        comp = np.where(valid)[0]
        ub = pairwise_diff_upper(logits, cand, comp)
        self_mask = cand[:, None] == comp[None, :]
        ub = np.where(self_mask, np.inf, ub)
        cand = cand[(ub >= 0).all(1)]
    return ArgmaxResult(np.sort(cand), lo, hi, valid)


def ambiguity_split_score(logits: TM, actions, K=None):
    """Per-input-symbol score for splitting an ambiguous box: sum over competing pairs (a,b) of |C_a - C_b|
    on the first K symbols (the input directions).  Bisecting the top-scoring direction shrinks the
    logit-difference ranges that keep several actions alive."""
    actions = np.asarray(actions)
    C = logits.C[actions]
    if K is not None:
        C = C[:, :K]
    if len(actions) < 2:
        return np.zeros(C.shape[1])
    D = np.abs(C[:, None, :] - C[None, :, :])
    return D.sum((0, 1))
