"""Audit a discrete RL agent that controls a distribution grid: sgen P/Q setpoints and transformer taps.

The grid is the CIGRE MV benchmark with PV / wind sgens (pandapower), with on-load tap changers added.  The agent here
is a small network trained in a few seconds to imitate a simple voltage rule, so that the example runs on its own;
replace it with your own agent (see "YOUR AGENT" below).

    python examples/pandapower_rl_audit.py

What you declare (all in pandapower terms):
  1. the observation vector, block by block, in the order your agent reads it;
  2. the action list, in the order of your agent's outputs;
  3. the limits that define "safe".
What you get:
  mode 1  next_step          for every load within +-w of the snapshot: which actions the agent can take, and whether
                             each keeps every limit one step later (certified), with a concrete counterexample if not;
  mode 2  noise_robustness   how much measurement noise the agent tolerates before it changes its action, or before it
                             can pick an action that violates a limit.
"""
import warnings

import numpy as np
import pandapower.networks as pn

from certflow.audit import GridModel, Limits, LoadUncertainty, RLAuditor, act, obs

warnings.filterwarnings("ignore")


def make_grid():
    net = pn.create_cigre_network_mv(with_der="pv_wind")
    # on-load tap changers (pandapower >= 3 only applies tap_pos when tap_changer_type is set)
    net.trafo["tap_changer_type"] = "Ratio"
    net.trafo["tap_side"], net.trafo["tap_neutral"] = "hv", 0
    net.trafo["tap_min"], net.trafo["tap_max"], net.trafo["tap_pos"] = -8, 8, -2
    net.trafo["tap_step_percent"], net.trafo["tap_step_degree"] = 1.25, 0.0
    net.sgen["min_q_mvar"], net.sgen["max_q_mvar"] = -1.0, 1.0
    return net


net = make_grid()

# 1) what the agent observes, in order -------------------------------------------------------------------------
observation = [
    obs.bus_vm(),             # 15 bus voltages (pu)                 -- measurements: noisy in mode 2
    obs.line_loading(),       # 15 line loadings (%)
    obs.trafo_loading(),      # 2 trafo loadings (%)
    obs.load_p(),             # 18 load P (MW)
    obs.sgen_p(),             # 9 sgen P setpoints (MW)              -- the agent's own setpoints: exact
    obs.sgen_q(),             # 9 sgen Q setpoints (MVAr)
    obs.trafo_tap(),          # 2 tap positions                      -- exact
]

# 2) what the agent can do, in the order of its outputs ----------------------------------------------------------
actions = ([act.idle()]
           + [act.shift_tap(t, step) for t in net.trafo.index for step in (-1, +1)]
           + [act.set_sgen(8, q_mvar=q) for q in (-1.0, -0.5, 0.5, 1.0)]          # wind park reactive power
           + [act.set_sgen(8, p_mw=p) for p in (0.5, 1.0)])                       # wind park curtailment

# 3) what "safe" means ---------------------------------------------------------------------------------------------
limits = Limits(vm_pu=(0.95, 1.05), max_loading_percent=100)

model = GridModel(net, observation, actions, limits)
model.check_against_pandapower()          # once: the model must read the grid exactly like the agent's env does


# YOUR AGENT ----------------------------------------------------------------------------------------------------
#   policy = torch_module                      # nn.Sequential of Linear / ReLU / Tanh up to the logits
#   policy = MaskablePPO.load("agent.zip")     # SB3 / sb3-contrib model (actor logits; DQN: Q-values)
#   RLAuditor(policy, model, input_layers=normalizer_layers(vecnorm.obs_rms.mean, vecnorm.obs_rms.var, clip=10))
#                                              # if a VecNormalize wrapper normalises the observation
# Here: imitate "raise voltage (tap -1) when the minimum voltage is low, otherwise idle".
def demo_agent(model, net, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from certflow.audit import LoadUncertainty
    st = model.state_from_net(net)
    U = LoadUncertainty(rel=0.15).build(model, net)
    X, y = [], []
    for e in rng.uniform(-1, 1, (600, U.K)):
        sim = model.simulate(st, U.point(e))
        vmin = sim["obs"][:15].min()
        X.append(sim["obs"])
        y.append(1 if vmin < 0.955 else (8 if vmin < 0.965 else 0))   # tap trafo 0 -1 | sgen 8 Q=+1 MVAr | idle
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    body = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, len(actions)))
    opt = torch.optim.Adam(body.parameters(), 1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(body((X - mu) / sd), y)
        loss.backward()
        opt.step()
    from certflow.audit import normalizer_layers
    return body, normalizer_layers(mu.numpy(), std=sd.numpy())


policy, norm = demo_agent(model, net)
auditor = RLAuditor(policy, model, input_layers=norm)

# mode 1: every action the agent can take for loads within +-2 % -- safe one step later? -----------------------
# 18 independent load directions: the proof holds, but box splitting cannot separate the agent's decision regions in
# 18 dimensions, so the certified action set can be much larger than what the agent actually does
print()
print(auditor.next_step(net, load_width=0.02).summary())

# the same with a low-rank load set (as in the paper): 3 principal load modes from load data, +-3 sigma each.
# Here the "data" are synthetic daily profiles; use your own historical load measurements.
rng = np.random.default_rng(1)
p0 = net.load.p_mw.values
t = rng.uniform(0, 1, (500, 1))
profiles = p0 * (1 + 0.04 * np.sin(2 * np.pi * t) + 0.02 * np.cos(4 * np.pi * t) * (np.arange(len(p0)) % 2)
                 + 0.005 * rng.standard_normal((500, len(p0))))
modes = LoadUncertainty.from_samples(profiles, n_modes=3, coverage=3.0)
print()
print(auditor.next_step(net, uncertainty=modes).summary())

# variant: next_load="step", step_change=0.01 -> the next load is the observed one +-1 % (correlated with what the
# agent saw; adds one symbol per load, so it is slower)

# mode 2: measurement noise at the snapshot ----------------------------------------------------------------------
print()
print(auditor.noise_robustness(net, eps_max=0.10).summary())
