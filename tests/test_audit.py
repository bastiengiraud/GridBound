"""certflow.audit: pandapower grid model, RL auditor (next-step safety, measurement noise), dispatch wrapper."""
import warnings

import numpy as np

from certflow.nn import Linear, ReLU, Sequential, Tanh


def _have_pandapower():
    try:
        import pandapower  # noqa: F401
        return True
    except Exception:
        return False


def _cigre(tap_pos=-2):
    import pandapower.networks as pn
    net = pn.create_cigre_network_mv(with_der="pv_wind")
    net.trafo["tap_changer_type"] = "Ratio"
    net.trafo["tap_side"] = "hv"
    net.trafo["tap_neutral"] = 0
    net.trafo["tap_min"], net.trafo["tap_max"] = -8, 8
    net.trafo["tap_step_percent"], net.trafo["tap_step_degree"] = 1.25, 0.0
    net.trafo["tap_pos"] = tap_pos
    return net


def _model(net, limits=None):
    from certflow.audit import GridModel, Limits, act, obs
    observation = [obs.bus_vm(), obs.line_loading(), obs.trafo_loading(), obs.sgen_p(), obs.load_p(), obs.trafo_tap()]
    actions = ([act.idle()] + [act.shift_tap(t, s) for t in net.trafo.index for s in (-1, 1, 2)]
               + [act.set_sgen(8, q_mvar=q) for q in (-1.0, 1.0)])
    return GridModel(net, observation, actions, limits or Limits(vm_pu=(0.95, 1.05)))


def _random_policy(model, seed=3):
    n_obs, n_act = len(model.observation.names(model)), len(model.actions)
    rng = np.random.default_rng(seed)
    return Sequential([Linear(rng.normal(0, 0.3, (16, n_obs)), np.zeros(16)), Tanh(),
                       Linear(rng.normal(0, 0.6, (n_act, 16)), np.zeros(n_act))])


def test_pandapower_ratings_match_loading_percent():
    if not _have_pandapower():
        return
    import pandapower as pp
    from certflow.power import from_pandapower
    net = _cigre(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp.runpp(net, numba=False)
        g = from_pandapower(net)
    V = g.V0
    I = np.maximum(np.abs(g.Yf @ V), np.abs(g.Yt @ V))
    for kind, res in (("line", net.res_line), ("trafo", net.res_trafo)):
        rows = [k for k, e in enumerate(g.names["branch_element"]) if e[0] == kind]
        ids = [g.names["branch_element"][k][1] for k in rows]
        mine = I[rows] * g.sn / g.rate_a[rows] * 100
        assert np.allclose(mine, res.loading_percent.loc[ids].values, atol=1e-6), kind


def test_grid_model_matches_pandapower_and_taps_act():
    if not _have_pandapower():
        return
    m = _model(_cigre(-2))
    dev = m.check_against_pandapower(verbose=False)
    assert max(dev.values()) < 1e-6, dev
    s0 = m.state_from_net(m.net)
    s1 = m.actions.actions[2].apply(s0, m)                       # trafo 0 tap +1
    assert s1.tap[0] == s0.tap[0] + 1
    assert np.abs(m.grid(s1).Ybus - m.grid(s0).Ybus).max() > 1e-6
    net2 = _cigre(-1)                                            # pandapower at that tap agrees with the model
    assert max(m.check_against_pandapower(net2, verbose=False).values()) < 1e-6


def test_next_step_sound_and_counterexample_real():
    if not _have_pandapower():
        return
    from certflow.audit import RLAuditor
    net = _cigre(-2)
    m = _model(net)
    aud = RLAuditor(_random_policy(m), m)
    rep = aud.next_step(net, load_width=0.05, max_depth=3, n_check=60)
    c = rep.check
    assert c["action_ok"] == c["n"] > 0 and c["bounds_ok"] == c["n_post"] > 0
    assert rep.verdict in ("SAFE", "UNSAFE", "NOT PROVEN")
    for r in rep.actions:
        if r.counterexample and "quantity" in r.counterexample:   # re-check the counterexample independently
            ce = r.counterexample
            st = m.state_from_net(net)
            sim = m.simulate(m.actions.joint(r.action).apply(st, m), ce["u"], outage=ce["outage"])
            q = sim["quantities"]
            assert q["violation"][q["names"].index(ce["quantity"])] > 0
            assert not r.safe


def test_noise_radius_bracket_and_multidiscrete():
    if not _have_pandapower():
        return
    from certflow.audit import ActionSpace, GridModel, Limits, RLAuditor, act, obs
    net = _cigre(-2)
    m = _model(net)
    nr = RLAuditor(_random_policy(m), m).noise_robustness(net, eps_max=0.1, tol=2e-3, groups=False, levels=())
    assert nr.same.certified <= nr.same.falsified
    if nr.safe is not None:
        assert nr.safe.certified <= nr.safe.falsified
    heads = [[act.shift_tap(0, -1), act.idle(), act.shift_tap(0, 1)], [act.set_sgen(8, q_mvar=q) for q in (-1., 0., 1.)]]
    m2 = GridModel(net, m.observation, ActionSpace.multi(heads), Limits(vm_pu=(0.95, 1.05)))
    n_obs = len(m2.observation.names(m2))
    rng = np.random.default_rng(0)
    pol = Sequential([Linear(rng.normal(0, 0.3, (12, n_obs)), np.zeros(12)), ReLU(),
                      Linear(rng.normal(0, 0.6, (6, 12)), np.zeros(6))])
    rep = RLAuditor(pol, m2).next_step(net, load_width=0.03, max_depth=2, n_check=40)
    assert rep.check["action_ok"] == rep.check["n"] > 0


def test_verify_dispatch_case9_layouts():
    from certflow.audit import verify_dispatch
    from certflow.power import case9, from_matpower
    from tests.helpers import dispatch_net
    g = from_matpower(case9())
    net = dispatch_net(g, "relu")                                # [Pd; Qd] (MW) -> [Pg (MW); Vg]
    rep = verify_dispatch(case9(), net, load_width=0.05, n_check=100)
    assert rep.certificate.formal and rep.check["sound"]
    # same network, declared with Vg first and Pg in pu: wrap it so the declared layout is what it computes
    sn = g.sn
    W = np.zeros((4, 4))
    W[0:2, 2:4] = np.eye(2)
    W[2:4, 0:2] = np.eye(2) / sn
    from certflow.nn import with_output_layers
    swapped = with_output_layers(net, [Linear(W)])
    rep2 = verify_dispatch(case9(), swapped, load_width=0.05, outputs=("vg", "pg"), pg_units="pu", n_check=0)
    for k in rep.certificate.bands:
        assert np.allclose(rep.certificate.bands[k].hi, rep2.certificate.bands[k].hi, atol=1e-6), k


def test_n1_unsorted_monitored():
    from certflow.power import case9, from_matpower, n1_analysis
    g = from_matpower(case9())
    Pd0, Qd0 = g.names["load_pq"]
    from certflow.tm import TM

    def spec_fn(gg, a, b):
        P = (g.Cg @ g.names["gen_pg"] - g.Cl @ Pd0) / g.sn
        Q = -(g.Cl @ Qd0) / g.sn
        mk = lambda v: TM(np.asarray(v, float), np.zeros((len(v), 1)), np.zeros((len(v), 1, 1)), np.zeros(len(v)))
        return dict(P=mk(P), Q=mk(Q), V=mk(g.names["gen_vg"]), warm=(np.angle(gg.V0), np.abs(gg.V0)))
    rate = np.full(g.L, 250.0)
    a = n1_analysis(g, spec_fn, -np.ones(1), np.ones(1), [2], rate, monitored=[8, 0, 5], measure="I")
    b = n1_analysis(g, spec_fn, -np.ones(1), np.ones(1), [2], rate, monitored=[0, 5, 8], measure="I")
    ra, rb = a.results[0], b.results[0]
    assert np.allclose(ra.loading_hi[[1, 2, 0]], rb.loading_hi)


def test_multi_step_envelopes_sound():
    if not _have_pandapower():
        return
    from certflow.audit import LoadUncertainty, RLAuditor
    net = _cigre(-2)
    m = _model(net)
    aud = RLAuditor(_random_policy(m), m)
    rep = aud.multi_step(net, uncertainty=LoadUncertainty(rel=0.03, directions="common"), steps=2, step_change=0.01,
                         max_depth=1, n_check=40)
    c = rep.check
    assert c["n"] > 0 and c["seq_ok"] == c["n"] and c["env_ok"] == c["n"]
    assert len(rep.envelopes) == 3 and all(E.lo for E in rep.envelopes)
    assert all(len(p) == 2 for p in rep.paths)
    fig = rep.plot()                                   # renders without a display
    assert len(fig.axes) == 2


def test_multi_step_rebox_not_looser_than_history():
    if not _have_pandapower():
        return
    from certflow.audit import LoadUncertainty, RLAuditor
    net = _cigre(-2)
    m = _model(net)
    aud = RLAuditor(_random_policy(m), m)
    kw = dict(uncertainty=LoadUncertainty(rel=0.03, directions="common"), steps=2, step_change=0.01, max_depth=1,
              n_check=30)
    h = aud.multi_step(net, track="history", **kw)
    r = aud.multi_step(net, track="rebox", **kw)
    assert r.check["env_ok"] == r.check["n"] > 0 and r.check["seq_ok"] == r.check["n"]
    assert set(r.paths) <= set(h.paths)
    for Eh, Er in zip(h.envelopes, r.envelopes):
        for n in Er.lo:
            assert Er.lo[n] >= Eh.lo[n] - 1e-9 and Er.hi[n] <= Eh.hi[n] + 1e-9, n
    assert max(len(n.lo) for n in r.tree.walk()) == 1


def test_switching_off_line_behind_open_switch_is_not_islanding():
    """pandapower puts an empty auxiliary bus behind an open line switch; switching that line off must not count as
    the grid splitting (the bus is dropped, as pandapower does), while a real island still does."""
    if not _have_pandapower():
        return
    import copy

    import pandapower as pp
    from certflow.audit import LoadUncertainty
    net = _cigre(0)
    m = _model(net)
    st = m.state_from_net(net)
    off = st.with_("line_on", m.line_pos(12), False)            # line 12 ends at an open switch (bus 7)
    assert not m.grid(off).islanded()
    assert m.simulate(off, LoadUncertainty(rel=0.0).build(m, net).c) is not None
    net2 = copy.deepcopy(net)
    net2.line.loc[12, "in_service"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp.runpp(net2, numba=False)
    assert max(m.check_against_pandapower(net2, verbose=False).values()) < 1e-6
    assert m.screen(off, LoadUncertainty(rel=0.03, directions="common").build(m, net)).formal
    assert m.grid(st.with_("line_on", m.line_pos(0), False)).islanded()   # the feeder head: a real island


def test_bus_angle_sound_behind_phase_shifting_trafo():
    """Bus angles far from 0 (the YNd5 transformer puts the LV side near -98 deg) are enclosed soundly."""
    if not _have_pandapower():
        return
    import pandapower.networks as pn
    from certflow.audit import GridModel, LoadUncertainty, act, obs
    net = pn.example_simple()
    m = GridModel(net, [obs.bus_va()], [act.idle()])
    assert max(m.check_against_pandapower(verbose=False).values()) < 1e-8
    st = m.state_from_net(net)
    U = LoadUncertainty(rel=0.10).build(m, net)
    ok, T, _ = m.observe_set(st, U, -np.ones(U.K), np.ones(U.K))
    assert ok
    lo, hi = T.bounds()
    assert lo.min() < -90                                      # past -90 deg: the case a plain atan cannot handle
    for e in np.random.default_rng(0).uniform(-1, 1, (50, U.K)):
        va = m.simulate(st, U.point(e))["obs"]
        assert np.all(va >= lo - 1e-9) and np.all(va <= hi + 1e-9)
