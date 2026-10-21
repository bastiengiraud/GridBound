"""Verification of a DISCRETE policy (e.g. an RL agent with masked argmax actions) over a box of exogenous
inputs, by adaptive partitioning.  Model agnostic: the application supplies three callbacks.

    observe(lo, hi)                 -> Observation    the policy input as a Taylor model over the e-box
                                                      (e.g. a certified power flow of the current state)
    mask(lo, hi, obs)               -> bool array     valid actions on this box (optional)
    check_action(a, lo, hi, obs)    -> ActionCheck    is taking action a safe for every input of the box?
                                                      (e.g. a certified post-action power flow)

For every leaf box the verifier proves which actions the policy can choose (``reachable_argmax`` on the
policy's reachable logits), checks every reachable action, and bisects when the observation cannot be
certified, several actions compete, or a reachable action cannot be proven safe -- up to ``max_depth``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nn.argmax import ArgmaxResult, ambiguity_split_score, reachable_argmax
from .split import bisect


@dataclass
class Observation:
    ok: bool                       # observation map certified on the box
    tm: object = None              # TM of the policy input (the first K symbols are the box directions)
    info: dict = field(default_factory=dict)
    score: np.ndarray = None       # split hint (per box direction) when not ok


@dataclass
class ActionCheck:
    safe: bool                     # proven safe on the whole box
    formal: bool = True            # the proof machinery succeeded (False: could not decide)
    info: dict = field(default_factory=dict)
    score: np.ndarray = None       # split hint (per box direction)


@dataclass
class PolicyLeaf:
    lo: np.ndarray
    hi: np.ndarray
    depth: int
    observed: bool
    argmax: ArgmaxResult = None
    checks: dict = field(default_factory=dict)
    obs_info: dict = field(default_factory=dict)

    @property
    def actions(self):
        return None if self.argmax is None else self.argmax.actions

    @property
    def status(self) -> str:
        if not self.observed:
            return "UNKNOWN"
        if all(self.checks[int(a)].safe for a in self.actions):
            return "SAFE"
        if any(not self.checks[int(a)].formal for a in self.actions):
            return "UNDECIDED"
        return "CAN-VIOLATE"

    def volume(self, lo0, hi0):
        return float(np.prod((self.hi - self.lo) / (hi0 - lo0)))


@dataclass
class PolicyReport:
    lo: np.ndarray
    hi: np.ndarray
    leaves: list
    complete: bool                 # the leaf cap was not hit

    def share(self, pred):
        return sum(L.volume(self.lo, self.hi) for L in self.leaves if pred(L))

    def action_sets(self):
        """{tuple(actions): volume share} over the observed leaves."""
        out = {}
        for L in self.leaves:
            if L.observed:
                k = tuple(int(a) for a in L.actions)
                out[k] = out.get(k, 0.0) + L.volume(self.lo, self.hi)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def reachable_actions(self):
        acts = set()
        for L in self.leaves:
            if L.observed:
                acts |= set(int(a) for a in L.actions)
        return sorted(acts)

    def status_shares(self):
        return {s: self.share(lambda L, s=s: L.status == s) for s in ("SAFE", "CAN-VIOLATE", "UNDECIDED", "UNKNOWN")}

    @property
    def safe(self) -> bool:
        return self.complete and all(L.status == "SAFE" for L in self.leaves)


class DiscretePolicyVerifier:
    """Adaptive partition of an e-box into leaves with certified action sets and per-action safety.

    policy        : NNReachable producing logits
    K             : number of box directions (the first K symbols of the observation TM)
    refine        : "safe"   -> split until every reachable action is proven safe (or max_depth)
                    "single" -> additionally split until a single action is reachable
    max_depth     : maximum number of bisections along any path
    max_leaves    : cap on the number of leaves (the report is then flagged incomplete)"""

    def __init__(self, policy, observe, K, mask=None, check_action=None, refine="safe", max_depth=8,
                 max_leaves=4000):
        self.policy = policy
        self.observe = observe
        self.K = int(K)
        self.mask = mask
        self.check_action = check_action
        self.refine = refine
        self.max_depth = int(max_depth)
        self.max_leaves = int(max_leaves)

    def evaluate(self, lo, hi, depth=0) -> PolicyLeaf:
        obs = self.observe(lo, hi)
        leaf = PolicyLeaf(np.array(lo, float), np.array(hi, float), depth, bool(obs.ok), obs_info=obs.info)
        if not obs.ok:
            leaf._score = obs.score
            return leaf
        logits = self.policy.reach(obs.tm)
        m = self.mask(lo, hi, obs) if callable(self.mask) else self.mask
        leaf.argmax = reachable_argmax(logits, m)
        scores = []
        if len(leaf.argmax.actions) > 1:
            scores.append(ambiguity_split_score(logits, leaf.argmax.actions, K=self.K))
        if self.check_action is not None:
            for a in leaf.argmax.actions:
                c = self.check_action(int(a), lo, hi, obs)
                leaf.checks[int(a)] = c
                if (not c.safe) and c.score is not None:
                    scores.append(np.asarray(c.score, float)[: self.K])
        else:
            leaf.checks = {int(a): ActionCheck(True, True) for a in leaf.argmax.actions}
        leaf._score = sum(s / (np.max(s) + 1e-300) for s in scores) if scores else None
        return leaf

    def _done(self, leaf):
        if leaf.depth >= self.max_depth:
            return True
        if leaf.status != "SAFE":
            return False
        return not (self.refine == "single" and len(leaf.actions) > 1)

    def verify(self, lo=None, hi=None) -> PolicyReport:
        lo0 = -np.ones(self.K) if lo is None else np.asarray(lo, float)
        hi0 = np.ones(self.K) if hi is None else np.asarray(hi, float)
        stack = [(lo0.copy(), hi0.copy(), 0)]
        leaves = []
        complete = True
        while stack:
            if len(leaves) + len(stack) > self.max_leaves:
                complete = False
                leaves += [self.evaluate(a, b, d) for a, b, d in stack]
                break
            lo, hi, d = stack.pop()
            leaf = self.evaluate(lo, hi, d)
            if self._done(leaf):
                leaves.append(leaf)
                continue
            w = hi - lo
            sc = getattr(leaf, "_score", None)
            j = int(np.argmax(sc * w)) if sc is not None and np.any(sc * w > 0) else int(np.argmax(w))
            (a_lo, a_hi), (b_lo, b_hi) = bisect(lo, hi, j)
            stack += [(b_lo, b_hi, d + 1), (a_lo, a_hi, d + 1)]
        return PolicyReport(lo0, hi0, leaves, complete)


# ==================================================================================================
# multi-step reachability
# ==================================================================================================

@dataclass
class StepNode:
    """One node of the k-step reachability tree: the policy's certified behaviour from ``state`` for every
    exogenous input in the box [lo, hi], and (if depth < steps) one child per reachable action."""
    state: object
    lo: np.ndarray
    hi: np.ndarray
    depth: int
    report: PolicyReport = None
    children: dict = field(default_factory=dict)       # action -> StepNode

    @property
    def safe(self) -> bool:
        """Every reachable action at this node, and recursively in every child, is certified safe."""
        return self.report.safe and all(c.safe for c in self.children.values())

    def walk(self):
        yield self
        for c in self.children.values():
            yield from c.walk()

    def paths(self, prefix=()):
        """All reachable action sequences (tuples) of the tree."""
        acts = self.report.reachable_actions()
        if not self.children:
            return [prefix + (a,) for a in acts]
        out = []
        for a in acts:
            kids = [c for k, c in self.children.items() if (k[0] if isinstance(k, tuple) else k) == a]
            if not kids:
                out.append(prefix + (a,))
            for c in kids:
                out += [p for p in c.paths(prefix + (a,)) if p not in out]
        return out


class MultiStepVerifier:
    """k-step reachability of a discrete policy under exogenous disturbances.

    problem(state, lo, hi)  -> dict(observe=, mask=, check=, K=)   callbacks for :class:`DiscretePolicyVerifier`
                               over e in [-1, 1]^K, where e maps AFFINELY onto the box [lo, hi]
                               (x = lo + (e + 1)/2 (hi - lo));  ``check`` judges the next step's safety
    transition(state, a)    -> the next state after action a (discrete/controllable part)
    propagate(lo, hi)       -> (lo', hi')  box of the next exogenous input given the current one
                               (e.g. widened by the per-step disturbance)
    merge                   -> "hull": one child per action, over the bounding box of all leaves where the
                               action is reachable (cheap, conservative); "leaf": one child per (action, leaf)

    Soundness: the exogenous input is re-boxed at every step (correlation across steps is dropped), so the
    tree over-approximates every reachable trajectory of length ``steps``."""

    def __init__(self, policy, problem, transition, propagate, steps=2, merge="hull", **verifier_kwargs):
        self.policy = policy
        self.problem = problem
        self.transition = transition
        self.propagate = propagate
        self.steps = int(steps)
        self.merge = merge
        self.kw = verifier_kwargs

    def _node(self, state, lo, hi, depth):
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        p = self.problem(state, lo, hi)
        V = DiscretePolicyVerifier(self.policy, p["observe"], p["K"], p.get("mask"), p.get("check"), **self.kw)
        node = StepNode(state, lo, hi, depth, V.verify())
        if depth + 1 >= self.steps:
            return node
        to_phys = lambda e: lo + 0.5 * (np.asarray(e) + 1.0) * (hi - lo)
        for a in node.report.reachable_actions():
            boxes = [(to_phys(L.lo), to_phys(L.hi)) for L in node.report.leaves
                     if L.observed and a in set(int(x) for x in L.actions)]
            nxt = self.transition(state, a)
            if self.merge == "leaf":
                for i, (bl, bh) in enumerate(boxes):
                    node.children[(a, i)] = self._node(nxt, *self.propagate(bl, bh), depth + 1)
            else:
                bl = np.min([b[0] for b in boxes], 0)
                bh = np.max([b[1] for b in boxes], 0)
                node.children[a] = self._node(nxt, *self.propagate(bl, bh), depth + 1)
        return node

    def verify(self, state, lo, hi) -> StepNode:
        return self._node(state, lo, hi, 0)
