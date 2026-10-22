"""Generic discrete-policy verifier on a toy problem (no power system)."""
import numpy as np

from certflow import tm as T
from certflow.nn import Linear, Sequential, Tanh
from certflow.policy import ActionCheck, DiscretePolicyVerifier, Observation
from certflow.sets import AffineSet
from certflow.tm import TM

rng = np.random.default_rng(0)


def test_policy_partition_sound():
    net = Sequential([Linear(rng.normal(size=(12, 2))), Tanh(), Linear(rng.normal(size=(5, 12)))])
    S = AffineSet(np.zeros(2), np.eye(2))
    observe = lambda lo, hi: Observation(True, S.sub_tm(0.5 * (lo + hi), 0.5 * (hi - lo)))
    unsafe = {3}
    check = lambda a, lo, hi, obs: ActionCheck(a not in unsafe, True, {})
    R = DiscretePolicyVerifier(net, observe, 2, None, check, refine="single", max_depth=8).verify()
    assert abs(sum(L.volume(R.lo, R.hi) for L in R.leaves) - 1.0) < 1e-12        # a partition
    for e in rng.uniform(-1, 1, (2000, 2)):
        a = int(np.argmax(net.forward(S.point(e))))
        leaf = next(L for L in R.leaves if np.all(e >= L.lo) and np.all(e <= L.hi))
        assert a in set(leaf.actions.tolist())
        if a in unsafe:
            assert leaf.status == "CAN-VIOLATE"


def test_sqrt_and_maximum_sound():
    for _ in range(3):
        x = TM(1 + rng.normal(size=5) * 0.6, rng.normal(size=(5, 3)) * 0.4, rng.normal(size=(5, 3, 3)) * 0.1, np.zeros(5))
        y = T.sqrt(x, nonneg=True)
        b = TM(rng.normal(size=5), rng.normal(size=(5, 3)), np.zeros((5, 3, 3)), np.zeros(5))
        mx = T.maximum(x, b)
        (lo, hi), (ml, mh) = y.bounds(), mx.bounds()
        for e in rng.uniform(-1, 1, (3000, 3)):
            v = x.eval(e)
            s = np.sqrt(np.maximum(v, 0))
            assert np.all(np.where(v >= 0, (s >= lo) & (s <= hi), True))
            w = np.maximum(v, b.eval(e))
            assert np.all((w >= ml) & (w <= mh))


def test_multistep_tree_sound():
    """Two-step reachability on a toy system: x' = x + shift(a) + noise; every simulated action pair is certified."""
    from certflow.policy import MultiStepVerifier
    net = Sequential([Linear(rng.normal(size=(10, 2))), Tanh(), Linear(rng.normal(size=(4, 10)))])
    shifts = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [-0.3, -0.3]])
    m = 0.05

    def problem(state, lo, hi):
        S = AffineSet(0.5 * (lo + hi) + state, np.diag(0.5 * (hi - lo)))
        obs = lambda a, b: Observation(True, S.sub_tm(0.5 * (a + b), 0.5 * (b - a)))
        return dict(observe=obs, K=2)

    MV = MultiStepVerifier(net, problem, lambda s, a: s + shifts[a], lambda lo, hi: (lo - m, hi + m), steps=2,
                           refine="single", max_depth=6)
    root = MV.verify(np.zeros(2), -0.2 * np.ones(2), 0.2 * np.ones(2))
    paths = set(root.paths())
    for _ in range(1000):
        x1 = rng.uniform(-0.2, 0.2, 2)
        a1 = int(np.argmax(net.forward(x1)))
        x2 = x1 + rng.uniform(-m, m, 2)
        a2 = int(np.argmax(net.forward(x2 + shifts[a1])))
        assert (a1, a2) in paths


def test_multistep_history_sound():
    """History tracking (new symbols per step, correlated step change): every simulated action pair is certified,
    and the tracked tree has no more reachable sequences than the re-boxed one on this example."""
    from certflow.policy import MultiStepVerifier
    net = Sequential([Linear(rng.normal(size=(10, 2))), Tanh(), Linear(rng.normal(size=(4, 10)))])
    shifts = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [-0.3, -0.3]])
    g_step = np.array([[0.08], [0.08]])                  # common-mode step change: both inputs move together

    def problem(state, U):
        obs = lambda a, b: Observation(True, AffineSet(U.c + state, U.G).sub_tm(0.5 * (a + b), 0.5 * (b - a)))
        return dict(observe=obs)

    U0 = AffineSet(np.zeros(2), 0.2 * np.eye(2))
    extend = lambda U: AffineSet(U.c, np.hstack([U.G, g_step]))
    MV = MultiStepVerifier(net, problem, lambda s, a: s + shifts[a], steps=2, track="history", extend=extend,
                           refine="single", max_depth=6)
    root = MV.verify(np.zeros(2), U=U0)
    assert root.stats()["max_symbols"] == 3
    paths = set(root.paths())
    for _ in range(1000):
        e = rng.uniform(-1, 1, 3)
        x1 = U0.point(e[:2])
        a1 = int(np.argmax(net.forward(x1)))
        x2 = x1 + g_step[:, 0] * e[2]
        a2 = int(np.argmax(net.forward(x2 + shifts[a1])))
        assert (a1, a2) in paths

    def problem_box(state, lo, hi):                       # the same system, re-boxed every step
        S = AffineSet(0.5 * (lo + hi) + state, np.diag(0.5 * (hi - lo)))
        return dict(observe=lambda a, b: Observation(True, S.sub_tm(0.5 * (a + b), 0.5 * (b - a))), K=2)
    MR = MultiStepVerifier(net, problem_box, lambda s, a: s + shifts[a], lambda lo, hi: (lo - 0.08, hi + 0.08),
                           steps=2, refine="single", max_depth=6)
    rbox = MR.verify(np.zeros(2), -0.2 * np.ones(2), 0.2 * np.ones(2))
    assert paths <= set(rbox.paths())


def test_robustness_radius_bracket():
    """Certified radius <= falsified radius; no sampled noise below the certified radius changes the action."""
    from certflow.robustness import robustness_radius
    net = Sequential([Linear(rng.normal(size=(16, 5))), Tanh(), Linear(rng.normal(size=(6, 16)))])
    o = rng.normal(size=5)
    x = TM(o, np.zeros((5, 0)), np.zeros((5, 0, 0)), np.zeros(5))
    scale = np.abs(o) + 0.1
    r = robustness_radius(net, x, scale, eps_max=2.0, tol=1e-3)
    a0 = int(np.argmax(net.forward(o)))
    assert r.nominal == a0 and r.certified <= r.falsified
    for _ in range(3000):
        eta = rng.uniform(-1, 1, 5) * scale * r.certified
        assert int(np.argmax(net.forward(o + eta))) == a0
    if np.isfinite(r.falsified):
        ce = r.counterexample
        assert int(np.argmax(net.forward(o + ce["eta"]))) == ce["action"] != a0
        assert np.all(np.abs(ce["eta"]) <= ce["eps"] * scale + 1e-12)
