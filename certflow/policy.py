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

from .nn.argmax import ArgmaxResult, ambiguity_split_score, reachable_argmax, reachable_argmax_heads
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
    max_leaves    : cap on the number of leaves (the report is then flagged incomplete)
    heads         : MultiDiscrete policy: sizes of the consecutive logit blocks (one argmax each); actions are then
                    flat joint indices (see certflow.nn.argmax.reachable_argmax_heads)
    stop_when_decided : stop splitting a leaf with a single reachable action whose check is decided (safe or not);
                    right when ``check_action`` does not depend on the leaf box (e.g. a fixed next-step set)"""

    def __init__(self, policy, observe, K, mask=None, check_action=None, refine="safe", max_depth=8,
                 max_leaves=4000, heads=None, stop_when_decided=False):
        self.policy = policy
        self.observe = observe
        self.K = int(K)
        self.mask = mask
        self.check_action = check_action
        self.refine = refine
        self.max_depth = int(max_depth)
        self.max_leaves = int(max_leaves)
        self.heads = heads
        self.stop_when_decided = bool(stop_when_decided)

    def evaluate(self, lo, hi, depth=0) -> PolicyLeaf:
        obs = self.observe(lo, hi)
        leaf = PolicyLeaf(np.array(lo, float), np.array(hi, float), depth, bool(obs.ok), obs_info=obs.info)
        if not obs.ok:
            leaf._score = obs.score
            return leaf
        logits = self.policy.reach(obs.tm)
        m = self.mask(lo, hi, obs) if callable(self.mask) else self.mask
        if self.heads is None:
            leaf.argmax = reachable_argmax(logits, m)
            amb = leaf.argmax.actions if len(leaf.argmax.actions) > 1 else []
        else:
            leaf.argmax = reachable_argmax_heads(logits, self.heads, m)
            off = np.r_[0, np.cumsum(self.heads)[:-1]]
            per = np.unravel_index(leaf.argmax.actions, self.heads) if len(leaf.argmax.actions) else []
            amb = [o + i for o, ii in zip(off, per) for i in np.unique(ii)] if len(leaf.argmax.actions) > 1 else []
        scores = []
        if len(amb) > 1:
            scores.append(ambiguity_split_score(logits, np.asarray(amb), K=self.K))
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
        if (self.stop_when_decided and leaf.observed and len(leaf.actions) == 1
                and leaf.checks[int(leaf.actions[0])].formal):
            return True                     # one action, decided: splitting cannot change a box-independent check
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
    exogenous input of the node's box, and (if depth < steps) one child per reachable action.

    track="rebox":   [lo, hi] is a box of the exogenous input itself.
    track="history": [lo, hi] is a box in the symbol space of the affine input set ``U`` (x = U.c + U.G e); the
                     symbols of earlier steps are kept, so ``U`` grows by the new step's symbols at every depth."""
    state: object
    lo: np.ndarray
    hi: np.ndarray
    depth: int
    report: PolicyReport = None
    children: dict = field(default_factory=dict)       # action (or (action, leaf index)) -> StepNode
    U: object = None                                     # history mode: the affine input set of this step
    seconds: float = 0.0                                 # time spent verifying this node (excluding children)

    @property
    def safe(self) -> bool:
        """Every reachable action at this node, and recursively in every child, is certified safe."""
        return self.report.safe and all(c.safe for c in self.children.values())

    def input_box(self):
        """Bounding box of the exogenous input covered by this node."""
        if self.U is None:
            return self.lo, self.hi
        ec, h = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        c = self.U.c + self.U.G @ ec
        r = np.abs(self.U.G) @ h
        return c - r, c + r

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

    def stats(self):
        """Tree size and cost: nodes, policy leaves (boxes), total verification time (s), max symbol count."""
        nodes = list(self.walk())
        return dict(nodes=len(nodes), leaves=sum(len(n.report.leaves) for n in nodes),
                    seconds=sum(n.seconds for n in nodes),
                    max_symbols=max(len(n.lo) for n in nodes))


class MultiStepVerifier:
    """k-step reachability of a discrete policy under exogenous disturbances.

    Two ways to carry the uncertainty from one step to the next (``track``):

    "rebox" (default)  the next step's input is a fresh box: the bounding box of the leaves where the action is
                       reachable, widened by ``propagate``.  Cheap (the dimension stays K), but the correlation
                       between steps is dropped.
        problem(state, lo, hi) -> dict(observe=, mask=, check=, K=)   callbacks for :class:`DiscretePolicyVerifier`
                                  over e in [-1, 1]^K, where e maps affinely onto the box [lo, hi]
        propagate(lo, hi)      -> (lo', hi')  box of the next exogenous input given the current one

    "history"          the input of every step is an affine set  x = c + G e  over ALL symbols so far: the next
                       step keeps the region of the earlier symbols where the action is reachable and appends
                       NEW symbols for the new uncertainty (``extend``).  The correlation between steps is kept
                       (the only over-approximation of earlier steps is their leaf boxes in symbol space); the
                       dimension grows by the number of new symbols every step.
        problem(state, U)      -> dict(observe=, mask=, check=)       callbacks over U's symbol space
                                  (e in [-1, 1]^U.K; a sub-box [lo, hi] is U.sub_tm(centre, half-width)).  ``check``
                                  should judge the action over the NEXT step's input, i.e. extend(U) on the
                                  sub-box x the full range of the new symbols.
        extend(U)              -> U'  the same set with the next step's uncertainty appended as new generator
                                  columns (e.g. AffineSet(U.c, [U.G, G_step]))

    transition(state, a)    -> the next state after action a (discrete/controllable part)
    merge                   -> "hull": one child per action, over the bounding box (of the inputs, resp. of the
                               symbols) of all leaves where the action is reachable; "leaf": one child per
                               (action, leaf)

    Soundness: in both modes the tree over-approximates every reachable trajectory of length ``steps``."""

    def __init__(self, policy, problem, transition, propagate=None, steps=2, merge="hull", track="rebox",
                 extend=None, **verifier_kwargs):
        if track not in ("rebox", "history"):
            raise ValueError("track must be 'rebox' or 'history'")
        if track == "rebox" and propagate is None:
            raise ValueError("track='rebox' needs propagate(lo, hi)")
        if track == "history" and extend is None:
            raise ValueError("track='history' needs extend(U)")
        self.policy = policy
        self.problem = problem
        self.transition = transition
        self.propagate = propagate
        self.extend = extend
        self.steps = int(steps)
        self.merge = merge
        self.track = track
        self.kw = verifier_kwargs

    def _boxes(self, report, a):
        return [(L.lo, L.hi) for L in report.leaves if L.observed and a in set(int(x) for x in L.actions)]

    def _node(self, state, lo, hi, depth, U=None):
        import time
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        t0 = time.perf_counter()
        if self.track == "rebox":
            p = self.problem(state, lo, hi)
            V = DiscretePolicyVerifier(self.policy, p["observe"], p["K"], p.get("mask"), p.get("check"), **self.kw)
            node = StepNode(state, lo, hi, depth, V.verify())
        else:
            p = self.problem(state, U)
            V = DiscretePolicyVerifier(self.policy, p["observe"], U.K, p.get("mask"), p.get("check"), **self.kw)
            node = StepNode(state, lo, hi, depth, V.verify(lo, hi), U=U)
        node.seconds = time.perf_counter() - t0
        if depth + 1 >= self.steps:
            return node
        if self.track == "rebox":
            to_phys = lambda e: lo + 0.5 * (np.asarray(e) + 1.0) * (hi - lo)
            for a in node.report.reachable_actions():
                boxes = [(to_phys(bl), to_phys(bh)) for bl, bh in self._boxes(node.report, a)]
                nxt = self.transition(state, a)
                if self.merge == "leaf":
                    for i, (bl, bh) in enumerate(boxes):
                        node.children[(a, i)] = self._node(nxt, *self.propagate(bl, bh), depth + 1)
                else:
                    bl = np.min([b[0] for b in boxes], 0)
                    bh = np.max([b[1] for b in boxes], 0)
                    node.children[a] = self._node(nxt, *self.propagate(bl, bh), depth + 1)
            return node
        U2 = self.extend(U)
        r = U2.K - U.K
        grow = lambda bl, bh: (np.r_[bl, -np.ones(r)], np.r_[bh, np.ones(r)])
        for a in node.report.reachable_actions():
            boxes = self._boxes(node.report, a)
            nxt = self.transition(state, a)
            if self.merge == "leaf":
                for i, (bl, bh) in enumerate(boxes):
                    node.children[(a, i)] = self._node(nxt, *grow(bl, bh), depth + 1, U2)
            else:
                bl = np.min([b[0] for b in boxes], 0)
                bh = np.max([b[1] for b in boxes], 0)
                node.children[a] = self._node(nxt, *grow(bl, bh), depth + 1, U2)
        return node

    def verify(self, state, lo=None, hi=None, U=None) -> StepNode:
        """rebox: verify(state, lo, hi) over the input box;  history: verify(state, U=U[, lo, hi]) over U's
        symbol box (default [-1, 1]^U.K)."""
        if self.track == "history":
            if U is None:
                raise ValueError("track='history': pass the initial input set U")
            lo = -np.ones(U.K) if lo is None else lo
            hi = np.ones(U.K) if hi is None else hi
            return self._node(state, lo, hi, 0, U)
        return self._node(state, lo, hi, 0)
