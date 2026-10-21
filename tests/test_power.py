"""End-to-end power-flow verification on MATPOWER case9 (no pandapower needed)."""
import numpy as np

from certflow.power import (BranchCurrent, Restorer, case9, default_constraints, from_matpower, monte_carlo)
from tests.helpers import case9_verifier


def test_certify_relu_sound():
    ver = case9_verifier(0.05, "relu")
    C = ver.certify()
    assert C.formal and C.rho < 1
    mc = monte_carlo(ver, C, n=150)
    assert mc["sound"], mc["worst"]


def test_certify_tanh_sound():
    ver = case9_verifier(0.05, "tanh")
    C = ver.certify()
    assert C.formal
    mc = monte_carlo(ver, C, n=150, seed=1)
    assert mc["sound"], mc["worst"]


def test_split_union_sound():
    ver = case9_verifier(0.05, "relu")
    lo, hi = ver.U.full_box()
    C1 = ver.certify()
    mid = 0.5 * (lo + hi)
    a_hi = hi.copy(); a_hi[0] = mid[0]
    b_lo = lo.copy(); b_lo[0] = mid[0]
    Ca, Cb = ver.certify(lo, a_hi), ver.certify(b_lo, hi)
    for n in ("V", "Qg", "slack"):
        assert np.all(np.minimum(Ca.bands[n].lo, Cb.bands[n].lo) >= C1.bands[n].lo - 1e-6)


def test_restoration_makes_box_safe():
    grid = from_matpower(case9())
    grid.gen_qmax[:] = [8.0, -2.0]                      # tight reactive limits -> the NN violates them
    grid.gen_qmin[:] = [-30.0, -30.0]
    ver = case9_verifier(0.05, "relu", grid=grid)
    C0 = ver.certify()
    assert C0.formal and not C0.safe
    R = Restorer(ver)
    root, leaves = R.restore_split(max_split=3)
    assert all(r.restored for r in leaves)
    for r in leaves:
        mc = monte_carlo(ver, r.cert, n=100)
        assert mc["sound"], mc["worst"]
        q = mc["empirical"]["Qg"]
        assert np.all(q[1] <= grid.gen_qmax[[0, 1]] / grid.sn + 1e-3)


def test_topology_switch():
    mpc = case9()
    g = from_matpower(mpc)
    s = g.status.copy(); s[4] = False                   # open line 6-7
    g2 = g.with_branch_status(s)
    mpc["branch"][4, 10] = 0
    assert np.abs(g2.Ybus - from_matpower(mpc).Ybus).max() < 1e-12
    assert not g2.islanded()
    ver = case9_verifier(0.03, "relu", grid=g2, constraints=default_constraints(g2))
    C = ver.certify()
    assert C.formal
    assert monte_carlo(ver, C, n=100)["sound"]
    br = C.constraint("branch")
    assert 4 not in br.elements                          # open line not monitored
