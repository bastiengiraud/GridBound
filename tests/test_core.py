"""Soundness of the Taylor-model arithmetic and the NN transformers (sampling)."""
import numpy as np

from certflow import tm as T
from certflow.fp import U32
from certflow.nn import Linear, ReLU, Sequential, Sigmoid, Tanh, reachable_argmax
from certflow.sets import AffineSet
from certflow.tm import TM

rng = np.random.default_rng(0)


def _rand(n, K):
    return TM(rng.normal(size=n), rng.normal(size=(n, K)), rng.normal(size=(n, K, K)) * 0.3, np.zeros(n))


def _samples(K, N=3000):
    E = rng.uniform(-1, 1, (N, K))
    E[: N // 4] = rng.choice([-1.0, 1.0], (N // 4, K))
    return E


def _assert_encloses(tmx, fn, K0, N=3000):
    lo, hi = tmx.bounds()
    for e in _samples(tmx.C.shape[1], N):
        v = fn(e[:K0])
        assert np.all(v >= lo) and np.all(v <= hi), (v, lo, hi)


def test_tm_ops():
    a, b = _rand(5, 4), _rand(5, 4)
    M = rng.normal(size=(3, 5))
    _assert_encloses(T.add(a, b), lambda e: a.eval(e) + b.eval(e), 4)
    _assert_encloses(T.sub(a, b), lambda e: a.eval(e) - b.eval(e), 4)
    _assert_encloses(T.mul(a, b), lambda e: a.eval(e) * b.eval(e), 4)
    _assert_encloses(T.mul(T.mul(a, b), a), lambda e: a.eval(e) ** 2 * b.eval(e), 4)
    _assert_encloses(T.matmul(M, a), lambda e: M @ a.eval(e), 4)
    _assert_encloses(T.clip(a, -0.3, 0.6), lambda e: np.clip(a.eval(e), -0.3, 0.6), 4)
    import scipy.sparse as sp
    Ms = sp.csr_matrix(np.where(np.abs(M) > 0.7, M, 0.0))
    _assert_encloses(T.matmul_sparse(Ms, a), lambda e: Ms @ a.eval(e), 4)


def test_dev_is_tight_on_vertex():
    # x = e^2 - e on [-1, 1]: range [-0.25, 2]
    x = TM(np.zeros(1), -np.ones((1, 1)), np.ones((1, 1, 1)), np.zeros(1))
    lo, hi = x.bounds()
    assert abs(lo[0] + 0.25) < 1e-12 and abs(hi[0] - 2.0) < 1e-12


def _net(act, din=4, dout=3, w=16):
    return Sequential([Linear(rng.normal(size=(w, din))), act(), Linear(rng.normal(size=(w, w)) * 0.5, rng.normal(size=w)),
                       act(), Linear(rng.normal(size=(dout, w)))])


def test_nn_reach_sound():
    for act in (ReLU, Tanh, Sigmoid):
        net = _net(act)
        x = TM.affine(rng.normal(size=4), rng.normal(size=(4, 3)) * 0.4)
        _assert_encloses(net.reach(x), lambda e: net.forward(x.eval(e)), 3)


def test_float32_deployment_enclosed():
    net = _net(Tanh)
    net.ctx.deploy_u = U32
    for L in net.layers:
        if isinstance(L, Linear):
            L.W = L.W.astype(np.float32).astype(float)
            L.b = L.b.astype(np.float32).astype(float)
    x = TM.affine(rng.normal(size=4).astype(np.float32).astype(float), np.zeros((4, 2)))
    x = TM(x.c0, (rng.normal(size=(4, 2)) * 0.3).astype(np.float32).astype(float), x.Q, x.R)
    lo, hi = net.reach(x).bounds()

    def f32(v):
        h = v.astype(np.float32)
        for L in net.layers:
            if isinstance(L, Linear):
                h = (L.W.astype(np.float32) @ h + L.b.astype(np.float32)).astype(np.float32)
            else:
                h = np.tanh(h).astype(np.float32)
        return h.astype(float)
    for e in _samples(2, 1000):
        v = f32(x.eval(e))
        assert np.all(v >= lo) and np.all(v <= hi)


def test_argmax_sound():
    net = _net(Tanh, din=4, dout=8)
    S = AffineSet(rng.normal(size=4), rng.normal(size=(4, 2)) * 0.3)
    mask = np.ones(8, bool)
    mask[[1, 5]] = False
    res = reachable_argmax(net.reach(S.sub_tm(np.zeros(2), np.ones(2))), mask)
    seen = set()
    for e in _samples(2, 3000):
        y = net.forward(S.point(e))
        seen.add(int(np.argmax(np.where(mask, y, -1e8))))
    assert seen <= set(res.actions.tolist()), (seen, res.actions)
