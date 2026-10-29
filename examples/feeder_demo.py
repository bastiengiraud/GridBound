"""A small MV feeder and a demo agent, shared by the examples and the documentation figures.

Grid: 110/20 kV transformer with an on-load tap changer feeding a 4-bus MV line with loads (buses 2..5) and two
PV parks (buses 4, 5).  Agent: a small network trained in a few seconds to imitate a voltage rule: tap down when
voltages are high, up when they are low, absorb reactive power at the PV park in between.  Replace it with your own.
"""
import warnings

import numpy as np
import pandapower as pp

from certflow.audit import GridModel, Limits, LoadUncertainty, RLAuditor, act, normalizer_layers, obs

warnings.filterwarnings("ignore")

OBSERVATION = [obs.bus_vm(), obs.line_loading(), obs.trafo_loading(), obs.load_p(), obs.sgen_q(), obs.trafo_tap()]
ACTIONS = [act.idle(), act.shift_tap(0, +1), act.shift_tap(0, -1), act.set_sgen(1, q_mvar=-1.0),
           act.set_sgen(1, q_mvar=0.0)]


def feeder(load_scale=1.0):
    net = pp.create_empty_network(sn_mva=10.0)
    b = [pp.create_bus(net, 110.0, name="HV")] + [pp.create_bus(net, 20.0, name=f"MV{k}") for k in range(1, 6)]
    pp.create_ext_grid(net, b[0], vm_pu=1.02)
    pp.create_transformer(net, b[0], b[1], std_type="25 MVA 110/20 kV")
    net.trafo["tap_changer_type"], net.trafo["tap_side"], net.trafo["tap_neutral"] = "Ratio", "hv", 0
    net.trafo["tap_min"], net.trafo["tap_max"], net.trafo["tap_step_percent"], net.trafo["tap_pos"] = -9, 9, 1.5, 0
    net.trafo["tap_step_degree"] = 0.0
    for k in range(1, 5):
        pp.create_line(net, b[k], b[k + 1], length_km=4.0, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    for k in range(2, 6):
        pp.create_load(net, b[k], p_mw=2.0 * load_scale, q_mvar=0.6 * load_scale)
    for k in (4, 5):
        pp.create_sgen(net, b[k], p_mw=4.0, q_mvar=0.0, type="PV", min_q_mvar=-1.5, max_q_mvar=1.5)
    return net


def model(net, limits=None):
    return GridModel(net, OBSERVATION, ACTIONS, limits or Limits(vm_pu=(0.95, 1.05), max_loading_percent=100))


def train_agent(model, net, seed=0, verbose=True):
    """Imitate: max |V| > 1.035 -> tap +1 (lowers MV voltage); min |V| < 0.97 -> tap -1; max |V| > 1.025 -> PV park
    absorbs 1 MVAr; otherwise idle (or release the reactive power when voltages are low).
    Returns (torch network, normalisation layers)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    st0 = model.state_from_net(net)
    U = LoadUncertainty(rel=0.4, directions="common").build(model, net)
    X, y = [], []
    for _ in range(1500):
        st = st0.with_("tap", 0, int(rng.integers(-4, 5))).with_("sgen_q", 1, float(rng.choice([-1.0, 0.0])))
        sim = model.simulate(st, U.point(rng.uniform(-1, 1, 1)))
        if sim is None:
            continue
        vm = sim["obs"][:6]
        vmax, vmin = vm.max(), vm.min()
        lab = 1 if vmax > 1.035 else 2 if vmin < 0.97 else 3 if vmax > 1.025 else (4 if vmin < 0.985 and st.sgen_q[1] < 0 else 0)
        X.append(sim["obs"])
        y.append(lab)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y)
    mu, sd = X.mean(0), X.std(0) + 1e-2
    body = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, len(ACTIONS)))
    opt = torch.optim.Adam(body.parameters(), 5e-3)
    for _ in range(1500):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(body((X - mu) / sd), y)
        loss.backward()
        opt.step()
    if verbose:
        acc = float((body((X - mu) / sd).argmax(1) == y).float().mean())
        print(f"demo agent: imitation accuracy {100 * acc:.1f}%")
    return body, normalizer_layers(mu.numpy(), std=sd.numpy())


def auditor(net=None, limits=None, seed=0):
    """(auditor, model, net) for the feeder with the demo agent."""
    net = feeder() if net is None else net
    m = model(net, limits)
    pol, norm = train_agent(m, net, seed)
    return RLAuditor(pol, m, input_layers=norm), m, net
