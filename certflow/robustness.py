"""Robustness radii of a (discrete) policy to observation noise, e.g. measurement noise around an operating point.

    noisy observation   o + eta,   |eta_i| <= eps * scale_i     (independent bounded noise per entry)

Two bounds bracket the true robustness radius eps*:
  * certified (lower bound): the largest eps for which the requirement is PROVEN for every noise in the box
    (reachable argmax over the noisy observation set; bisection, every reported value is itself certified);
  * falsified (upper bound): the smallest eps at which a concrete noise vector breaking the requirement was found
    (a linearisation-guided attack on the logit margins + random vertices).

Model agnostic: the requirement is a predicate on the certified set of reachable actions, e.g.
"only the nominal action" or "every reachable action is certified safe" (see :func:`robustness_radius`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nn.argmax import argmax_heads_point, reachable_argmax, reachable_argmax_heads
from .tm import TM, pad_symbols


def add_noise(x: TM, radius) -> TM:
    """x + eta with |eta_i| <= radius_i: one NEW independent symbol per entry with radius_i > 0."""
    radius = np.broadcast_to(np.asarray(radius, float), x.c0.shape)
    idx = np.where(radius > 0)[0]
    K = x.C.shape[1]
    y = pad_symbols(x, K + len(idx), x.Q.shape[-1])
    C = y.C.copy()
    C[idx, K + np.arange(len(idx))] = radius[idx]
    return TM(y.c0, C, y.Q, y.R)


@dataclass
class RadiusResult:
    certified: float                 # requirement proven for every noise with |eta| <= certified * scale
    falsified: float = np.inf        # a concrete noise of this size breaks the requirement (inf: none found)
    counterexample: dict = None      # dict(eps, action, eta) for the falsified radius
    history: list = field(default_factory=list)    # (eps, certified?, actions) evaluated during the bisection
    nominal: int = None              # the noise-free action (None if the noise-free observation set is ambiguous)
    eps_max: float = np.inf          # largest noise level tried
    holds_at_zero: bool = True       # False: the requirement is not proven even without noise (certified = 0 is vacuous)

    @property
    def bracket(self):
        return self.certified, self.falsified


def _linear_logits(policy, o, idx, s):
    """Logits at o and their linear coefficients on the noise symbols (one per noisy entry) over the box o +- s.
    Over a small box this is the local gradient of the logits w.r.t. the noisy entries (times s)."""
    n = len(o)
    x = add_noise(TM(o, np.zeros((n, 0)), np.zeros((n, 0, 0)), np.zeros(n)), s)
    lg = policy.reach(x)
    return lg.C[:, : len(idx)]


def argmax_attack(policy, obs, scale, eps, target, mask=None, n_random=64, seed=0, bad=None, iters=8, heads=None):
    """Look for eta (|eta| <= eps * scale) such that the policy picks a BAD action at obs + eta.

    bad(action) -> bool  : concrete test of the picked action (default: action != target).  E.g. "the action is
                           unsafe at the true state", checked with a plain power flow.
    Candidates: (1) for every bad competitor b, the vertex of the noise box that pushes (l_b - l_target) up along
    the logits' sensitivity over the whole box; (2) the same step iterated from the current point with the local
    sensitivity (a projected sign-gradient attack, ``iters`` steps); (3) ``n_random`` random vertices.
    Returns (action, eta) for the first success, or None."""
    obs = np.asarray(obs, float)
    s = eps * np.asarray(scale, float)
    n_out = policy.out_dim if policy.out_dim else len(policy.forward(obs))
    valid = np.ones(n_out, bool) if mask is None else np.asarray(mask, bool)
    is_bad = (lambda a: a != target) if bad is None else bad

    def act(o):
        lg = policy.forward(o)
        if heads is not None:
            return argmax_heads_point(lg, heads, valid)
        return int(np.argmax(np.where(valid, lg, -np.inf)))

    idx = np.where(s > 0)[0]
    if not len(idx):
        return None
    if heads is None:                                   # (competitor logit, logit it must beat)
        pairs = [(b, target) for b in np.where(valid)[0] if b != target and (bad is None or is_bad(int(b)))]
    else:
        off = np.r_[0, np.cumsum(heads)[:-1]]
        tl = [o + i for o, i in zip(off, np.unravel_index(int(target), heads))]
        pairs = [(o + b, t) for o, h, t in zip(off, heads, tl) for b in range(h) if valid[o + b] and o + b != t]
    Cn = _linear_logits(policy, obs, idx, s)
    for b, t in pairs:                                  # (1) one vertex per bad competitor, global sensitivity
        eta = np.zeros_like(obs)
        eta[idx] = np.sign(Cn[b] - Cn[t]) * s[idx]
        a = act(obs + eta)
        if is_bad(a):
            return a, eta
    small = 1e-3 * s                                    # (2) iterated sign steps with the local sensitivity
    for b, t in pairs:
        eta = np.zeros_like(obs)
        for k in range(iters):
            cur = act(obs + eta)
            if is_bad(cur):
                return cur, eta
            Cl = _linear_logits(policy, obs + eta, idx, small)
            step = np.sign(Cl[b] - Cl[t])
            eta[idx] = np.clip(eta[idx] + step * s[idx] * (2.0 / (k + 2)), -s[idx], s[idx])
        a = act(obs + eta)
        if is_bad(a):
            return a, eta
    rng = np.random.default_rng(seed)                   # (3) random vertices
    for _ in range(n_random):
        eta = np.zeros_like(obs)
        eta[idx] = rng.choice([-1.0, 1.0], len(idx)) * s[idx]
        a = act(obs + eta)
        if is_bad(a):
            return a, eta
    return None


def bisect_radius(certify, eps_max, tol=1e-3, falsify=None):
    """Largest certified eps in [0, eps_max] by bisection (assumes the requirement is monotone in eps, which holds
    for the true sets; every returned value has been certified itself).  ``certify(eps) -> (ok, info)``,
    ``falsify(eps) -> counterexample or None`` (optional, gives the upper bound)."""
    hist = []
    ok, info = certify(eps_max)
    hist.append((eps_max, ok, info))
    res = RadiusResult(eps_max if ok else 0.0, np.inf, None, hist, eps_max=eps_max)
    if not ok:                                     # the bisection assumes the requirement holds at 0: check it
        ok0, info0 = certify(0.0)
        hist.append((0.0, ok0, info0))
        res.holds_at_zero = bool(ok0)
    if not ok and res.holds_at_zero:
        lo, hi = 0.0, eps_max
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            ok, info = certify(mid)
            hist.append((mid, ok, info))
            lo, hi = (mid, hi) if ok else (lo, mid)
        res.certified = lo
    if falsify is not None:                        # smallest breaking eps: bisect the attack between certified and eps_max
        ce = falsify(eps_max)
        if ce is not None:
            a, b, best = res.certified, eps_max, (eps_max, ce)
            while b - a > tol:
                m = 0.5 * (a + b)
                c = falsify(m)
                if c is not None:
                    b, best = m, (m, c)
                else:
                    a = m
            res.falsified = best[0]
            res.counterexample = dict(eps=best[0], action=best[1][0], eta=best[1][1])
    return res


def robustness_radius(policy, obs_tm, scale, mask=None, require="same", eps_max=0.2, tol=1e-3, obs_point=None,
                      bad_action=None, heads=None):
    """Robustness radius of a masked-argmax policy to observation noise |eta| <= eps * scale.

    obs_tm     : TM of the (noise-free) observation -- a constant TM for a single operating point, or a set
                 (e.g. a certified power-flow observation over a load box)
    scale      : per-entry noise scale (0 = that entry is exact, e.g. the agent's own setpoints)
    require    : "same"   -> the reachable action set is exactly the nominal action {a0}
                 callable -> require(actions: set) -> bool, e.g. "every reachable action is certified safe"
    bad_action : concrete test bad_action(a) -> bool used for the falsification (upper bound); defaults to
                 "a != nominal" for require="same".  For a safety requirement pass e.g. "a is unsafe at the true
                 state" (a plain power flow): the falsified radius is then a real noise vector that makes the agent
                 pick a really unsafe action.
    obs_point  : a concrete observation for the falsification attack (default: obs_tm's centre)
    heads      : MultiDiscrete policy (sizes of the logit blocks); actions are then flat joint indices"""
    am = (lambda lg: reachable_argmax(lg, mask)) if heads is None else (lambda lg: reachable_argmax_heads(lg, heads, mask))
    nominal = am(policy.reach(obs_tm))
    a0 = int(nominal.actions[0]) if nominal.certain else None
    pred = (lambda acts: acts == {a0}) if require == "same" else require

    def certify(eps):
        acts = set(int(a) for a in am(policy.reach(add_noise(obs_tm, eps * np.asarray(scale)))).actions)
        return pred(acts), sorted(acts)

    falsify = None
    o = obs_tm.c0 if obs_point is None else np.asarray(obs_point, float)
    if a0 is not None and (require == "same" or bad_action is not None):
        bad = None if require == "same" and bad_action is None else bad_action
        falsify = lambda eps: argmax_attack(policy, o, scale, eps, a0, mask, bad=bad, heads=heads)
    if a0 is not None and bad_action is not None and bad_action(a0):     # already broken without any noise
        ok, info = certify(0.0)
        res = RadiusResult(0.0, 0.0, dict(eps=0.0, action=a0, eta=np.zeros_like(o)), [(0.0, ok, info)],
                           eps_max=eps_max, holds_at_zero=False)
    else:
        res = bisect_radius(certify, eps_max, tol, falsify)
    res.nominal = a0
    return res
