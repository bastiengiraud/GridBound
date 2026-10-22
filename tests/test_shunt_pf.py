"""Uncertain shunt (admittance-load) power flow + quantity Taylor models: sampling soundness on case9."""
import numpy as np

from certflow.power import Shunt, case9, enclose_pf, from_matpower, point_pf, quantities as qty
from certflow.power.pf import inj_pt
from certflow.sets import AffineSet
from certflow.tm import TM


def _setup(w=0.1, centre=0.0):
    mpc = case9()
    lb = np.where(mpc["bus"][:, 2] != 0)[0]
    S0 = mpc["bus"][lb, 2] + 1j * mpc["bus"][lb, 3]
    mpc["bus"][lb, 4] = S0.real            # loads -> constant admittance at 1 pu (MW / MVAr at V = 1)
    mpc["bus"][lb, 5] = -S0.imag
    mpc["bus"][lb, 2:4] = 0
    g = from_matpower(mpc)
    y = np.conj(S0) / g.sn                 # nominal load admittance (pu); the uncertain part is y * delta
    U = AffineSet(np.full(3, centre), w * np.eye(3))      # delta in centre +- w (one symbol per load)
    return g, lb, y, U


def _const(v, K):
    v = np.asarray(v, float)
    return TM(v, np.zeros((len(v), K)), np.zeros((len(v), K, K)), np.zeros(len(v)))


def _specs(g, x):
    K = x.C.shape[1]
    Pg = np.zeros(g.N)
    np.add.at(Pg, g.gen_bus, g.names["gen_pg"] / g.sn)
    return _const(Pg, K), _const(np.zeros(g.N), K), _const(g.names["gen_vg"], K)


def test_shunt_pf_and_quantities_sound():
    g, lb, y, U = _setup()
    x = U.sub_tm(np.zeros(3), np.ones(3))
    sh = Shunt(lb, y, x)
    P, Q, Vs = _specs(g, x)
    enc = enclose_pf(g, P, Q, Vs, (np.zeros(g.N), np.ones(g.N)), shunt=sh)
    assert enc.ok and enc.rho < 1
    Vm_tm = qty.vmag(enc, list(range(g.N)))
    S_tm = qty.branch_apparent_power(g, enc, np.arange(g.L))
    Pl, Ql = qty.admittance_power(enc, lb, y, x)       # delta part only (factor = delta)
    bnd = [t.bounds() for t in (Vm_tm, S_tm, Pl, Ql, enc.P, enc.Q)]
    rng = np.random.default_rng(0)
    E = np.vstack([rng.uniform(-1, 1, (150, 3)), rng.choice([-1.0, 1.0], (40, 3))])
    for e in E:
        d = U.point(e)
        Vpin = np.zeros(g.N)
        Vpin[g.pv] = g.names["gen_vg"]
        th, Vm, mis = point_pf(g, P[0], Q[0], Vpin, (np.zeros(g.N), np.ones(g.N)), shunt_d=d, shunt=sh)
        assert mis < 1e-9
        V = Vm * np.exp(1j * th)
        Sf = V[g.f] * np.conj(g.Yf @ V)
        St = V[g.t] * np.conj(g.Yt @ V)
        Gd, Bd = sh.effective_GB(g.G, g.B, d)
        Pi, Qi = inj_pt(Gd, Bd, th, Vm)
        vals = [Vm, np.maximum(np.abs(Sf), np.abs(St)), (y * d).real * 0 + np.real(np.conj(y * d)) * Vm[lb] ** 2,
                np.imag(np.conj(y * d)) * Vm[lb] ** 2, Pi, Qi]
        for (lo, hi), v in zip(bnd, vals):
            assert np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)


def test_n1_bounds_sound():
    """Certified N-1 loading bounds contain the sampled post-contingency loadings (case9, admittance loads)."""
    from certflow.power import n1_analysis
    g, lb, y, U = _setup(w=0.08)
    rating = np.full(g.L, 100.0)
    spec = lambda gg, a, b: dict(**dict(zip(("P", "Q", "V"), _specs(gg, U.sub_tm(0.5 * (a + b), 0.5 * (b - a))))),
                                 shunt=Shunt(lb, y, U.sub_tm(0.5 * (a + b), 0.5 * (b - a))),
                                 warm=(np.zeros(gg.N), np.ones(gg.N)))
    rep = n1_analysis(g, spec, -np.ones(3), np.ones(3), outages=range(g.L), rating_mva=rating,
                      vlim=(0.95, 1.05), slack=(g.slk, 0.0, 250.0))
    assert rep.kinds == ("overload", "voltage", "slack")
    lo_all, hi_all = rep.count_bounds(fail_weight=1, kinds=rep.kinds)
    assert 0 <= lo_all <= hi_all and rep.secure == (hi_all == 0)
    rng = np.random.default_rng(1)
    for r in rep.results:
        if r.islanded:
            st = g.status.copy(); st[r.outage] = False
            assert g.with_branch_status(st).islanded()
            continue
        assert r.formal
        st = g.status.copy(); st[r.outage] = False
        g2 = g.with_branch_status(st)
        sh = Shunt(lb, y, U.sub_tm(np.zeros(3), np.ones(3)))
        for e in rng.uniform(-1, 1, (40, 3)):
            Vpin = np.zeros(g.N); Vpin[g.pv] = g.names["gen_vg"]
            P, Q, _ = _specs(g2, U.sub_tm(np.zeros(3), np.ones(3)))
            th, Vm, mis = point_pf(g2, P[0], Q[0], Vpin, (np.zeros(g.N), np.ones(g.N)), shunt_d=U.point(e), shunt=sh)
            V = Vm * np.exp(1j * th)
            S = np.maximum(np.abs(V[g2.f] * np.conj(g2.Yf @ V)), np.abs(V[g2.t] * np.conj(g2.Yt @ V))) * g.sn / rating
            m = np.isfinite(r.loading_hi)
            assert np.all(S[rep.monitored][m] <= r.loading_hi[m] + 1e-12)
            assert np.all(S[rep.monitored][m] >= r.loading_lo[m] - 1e-12)
            assert np.all(Vm >= r.v_lo - 1e-12) and np.all(Vm <= r.v_hi + 1e-12)          # every bus |V|
            Gd, Bd = sh.effective_GB(g2.G, g2.B, U.point(e))
            Ps = inj_pt(Gd, Bd, th, Vm)[0][g.slk] * g.sn                                  # slack injection (MW)
            assert r.slack_lo - 1e-9 <= Ps <= r.slack_hi + 1e-9
