# certflow documentation

certflow proves whether a learned grid controller keeps the grid within its limits for **every** operating
condition in a set. Each result is one of:

- **SAFE:** proven, floating-point rounding included;
- **UNSAFE:** a concrete counterexample was found and printed;
- **NOT PROVEN:** the proof failed and no counterexample was found.

| You have | You ask | Call | Page |
|---|---|---|---|
| an NN trained on OPF data (loads → Pg, Vg) | does it keep every limit for every load in a set? | `verify_dispatch` | [dispatch.md](dispatch.md) |
| an RL agent with discrete actions | is every action it can take safe one step later? | `RLAuditor.next_step` | [rl_next_step.md](rl_next_step.md) |
| | how much sensor noise can it take? | `RLAuditor.noise_robustness` | [rl_noise.md](rl_noise.md) |
| | which states can it reach over several steps? | `RLAuditor.multi_step` | [rl_multistep.md](rl_multistep.md) |

How the proofs work: [method.md](../method.md).

## Setting up an RL audit

Describe the grid the way your agent sees it: the observation blocks and actions **in the order of the agent's
input and output vectors**.

```python
from certflow.audit import GridModel, Limits, RLAuditor, act, obs

model = GridModel(net,                                       # pandapower net
                  observation=[obs.bus_vm(), obs.line_loading(), obs.trafo_loading(), obs.load_p(),
                               obs.sgen_q(), obs.trafo_tap()],
                  actions=[act.idle(), act.shift_tap(0, +1), act.shift_tap(0, -1),
                           act.set_sgen(1, q_mvar=-1.0), act.set_sgen(1, q_mvar=0.0)],
                  limits=Limits(vm_pu=(0.95, 1.05), max_loading_percent=100))
model.check_against_pandapower()       # run once: must reproduce pandapower's results (~1e-8)
auditor = RLAuditor(policy, model)
```

**Observation blocks** (`obs.*`). Each takes `elements=[...]` (a subset or another order), `scale=` / `offset=` (a
fixed normalisation) and `noisy=` (whether mode 2 treats it as a measurement).

| Block | Values |
|---|---|
| `bus_vm`, `bus_va` | bus voltage magnitude / angle |
| `line_loading`, `trafo_loading` | `loading_percent`, as in pandapower |
| `line_p_from`, `line_q_from`, `trafo_p_hv`, `trafo_q_hv` | branch flows |
| `load_p`, `load_q`, `sgen_p`, `sgen_q`, `gen_p`, `gen_vm`, `ext_grid_p`, `ext_grid_q` | injections and setpoints |
| `trafo_tap`, `line_status`, `trafo_status` | discrete state |
| `constant(values)`, `custom(...)` | anything else |

**Actions** (`act.*`):

| Action | What it changes |
|---|---|
| `idle()` | nothing |
| `set_sgen(i, p_mw=, q_mvar=)`, `shift_sgen(i, dp_mw=, dq_mvar=)` | static generator setpoints |
| `set_gen(...)`, `shift_gen(...)` | generator P and voltage setpoints |
| `set_tap(t, pos)`, `shift_tap(t, step)` | transformer tap position |
| `switch_line(l, on)`, `switch_trafo(t, on)` | branch status |
| `combo(a, b, ...)`, `custom(fn, label)` | several changes, or your own |

- **One output per device** (MultiDiscrete): `ActionSpace.multi([[...], [...]])`.
- **Masked agents** (MaskablePPO): `ActionSpace(actions, masked=True)`.

**The agent** (`policy`) can be:

- a torch `nn.Sequential` up to the logits (or Q-values);
- a loaded stable-baselines3 / sb3-contrib model (PPO, MaskablePPO, A2C, DQN);
- a certflow `Sequential`.

If a VecNormalize wrapper normalises the observation, add
`input_layers=normalizer_layers(mean, var, clip=10.0)` to `RLAuditor`.

**What the model assumes:**

- loads are constant power;
- gens are ideal PV units;
- one `ext_grid` is the slack;
- switches keep their state. If a switching action or an outage cuts off a bus that carries nothing (pandapower's
  empty bus behind an open line switch), that bus is dropped, as pandapower does. A cut-off part with loads or
  generation counts as a violation (loss of supply);
- the guarantees concern the steady-state AC power flow, not dynamics.

pandapower ≥ 3 ignores tap positions unless `tap_changer_type` is set; the model warns about this.

## Examples and figures

All examples run in seconds to minutes, and each has its output saved next to it:

```bash
python examples/pandapower_rl_audit.py       # modes 1 and 2 on CIGRE MV with taps and sgens
python examples/multistep_reachability.py    # mode 3 on a small feeder, 3D plot
python examples/opf_dispatch_audit.py        # an NN trained on OPF data
python docs/make_figures.py                  # every figure and report in these pages (~5 min)
```

The pages use two small grids:

- **MATPOWER case9:** the dispatch NN.
- **A 6-bus MV feeder** ([`examples/feeder_demo.py`](../../examples/feeder_demo.py)): a transformer with an on-load
  tap changer, loads and two PV parks. Its demo agent is trained in seconds on a voltage rule: tap when voltages
  drift, absorb reactive power at a PV park, otherwise idle.
