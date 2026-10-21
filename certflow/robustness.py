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

from .nn.argmax import reachable_argmax
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

    @property
    def bracket(self):
        return self.certified, self.falsified


def argmax_attack(policy, obs, scale, eps, target, mask=None, n_random=64, seed=0):
    """Look for eta (|eta| <= eps * scale) with argmax(policy(obs + eta)) != target.  Uses the reachable logits'
    linear coefficients on the noise symbols to push every competitor's margin down (one candidate per
    competitor), plus random vertices of the noise box.  Returns (action, eta) or None."""
    obs = np.asarray(obs, float)
    s = eps * np.asarray(scale, float)
    valid = np.ones(policy.out_dim if policy.out_dim else len(policy.forward(obs)), bool) if mask is None else np.asarray(mask, bool)

    def act(o):
        lg = policy.forward(o)
        return int(np.argmax(np.where(valid, lg, -np.inf)))

    x = add_noise(TM(obs, np.zeros((len(obs), 0)), np.zeros((len(obs), 0, 0)), np.zeros(len(obs))), s)
    idx = np.where(s > 0)[0]
    lg = policy.reach(x)
    Cn = lg.C[:, : len(idx)]                       # logit sensitivity to each noise symbol (first symbols = noise)
    cands = []
    for b in np.where(valid)[0]:
        if b == target:
            continue
        d = np.sign(Cn[target] - Cn[b])            # decrease (l_target - l_b)
        eta = np.zeros_like(obs)
        eta[idx] = -d * s[idx]
        cands.append(eta)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        eta = np.zeros_like(obs)
        eta[idx] = rng.choice([-1.0, 1.0], len(idx)) * s[idx]
        cands.append(eta)
    for eta in cands:
        a = act(obs + eta)
        if a != target:
            return a, eta
    return None


def bisect_radius(certify, eps_max, tol=1e-3, falsify=None):
    """Largest certified eps in [0, eps_max] by bisection (assumes the requirement is monotone in eps, which holds
    for the true sets; every returned value has been certified itself).  ``certify(eps) -> (ok, info)``,
    ``falsify(eps) -> counterexample or None`` (optional, gives the upper bound)."""
    hist = []
    ok, info = certify(eps_max)
    hist.append((eps_max, ok, info))
    if ok:
        return RadiusResult(eps_max, np.inf, None, hist)
    lo, hi = 0.0, eps_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        ok, info = certify(mid)
        hist.append((mid, ok, info))
        lo, hi = (mid, hi) if ok else (lo, mid)
    res = RadiusResult(lo, np.inf, None, hist)
    if falsify is not None:                        # smallest breaking eps: bisect the attack between lo and eps_max
        ce = falsify(eps_max)
        if ce is not None:
            a, b, best = lo, eps_max, (eps_max, ce)
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


def robustness_radius(policy, obs_tm, scale, mask=None, require="same", eps_max=0.2, tol=1e-3, obs_point=None):
    """Robustness radius of a masked-argmax policy to observation noise |eta| <= eps * scale.

    obs_tm    : TM of the (noise-free) observation -- a constant TM for a single operating point, or a set
                (e.g. a certified power-flow observation over a load box)
    require   : "same"   -> the reachable action set is exactly the nominal action {a0}
                callable -> require(actions: set) -> bool, e.g. "every reachable action is certified safe"
    obs_point : a concrete observation for the falsification attack (default: obs_tm's centre)"""
    nominal = reachable_argmax(policy.reach(obs_tm), mask)
    a0 = int(nominal.actions[0]) if nominal.certain else None
    pred = (lambda acts: acts == {a0}) if require == "same" else require

    def certify(eps):
        acts = set(int(a) for a in reachable_argmax(policy.reach(add_noise(obs_tm, eps * np.asarray(scale))), mask).actions)
        return pred(acts), sorted(acts)

    falsify = None
    if require == "same" and a0 is not None:
        o = obs_tm.c0 if obs_point is None else obs_point
        falsify = lambda eps: argmax_attack(policy, o, scale, eps, a0, mask)
    res = bisect_radius(certify, eps_max, tol, falsify)
    res.nominal = a0
    return res
