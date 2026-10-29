# User guide: `certflow.audit`

`certflow.audit` is the layer you use day to day. You describe the grid, the controller and the limits in
pandapower terms, and you get one of three verdicts:

| Verdict | Meaning |
|---|---|
| **SAFE** | proven for every load (or noise vector) in the set, with floating-point rounding included |
| **UNSAFE** | a concrete load (or noise vector) was found where the controller really violates a limit; it is printed |
| **NOT PROVEN** | the proof fails somewhere, but no concrete violation was found (bounds may be loose: shrink or split the set) |

Every report also includes a Monte-Carlo cross-check of the certified claims.

| Use case | Call | Example |
|---|---|---|
| RL agent, mode 1: are all actions it can take safe one step ahead? | `RLAuditor.next_step` | [`pandapower_rl_audit.py`](../examples/pandapower_rl_audit.py) |
| RL agent, mode 2: how much measurement noise can it take? | `RLAuditor.noise_robustness` | [`pandapower_rl_audit.py`](../examples/pandapower_rl_audit.py) |
| RL agent, mode 3: which states can it reach over several steps? (3D plot) | `RLAuditor.multi_step` | [`multistep_reachability.py`](../examples/multistep_reachability.py) |
| NN trained on OPF data (loads -> Pg, Vg) | `verify_dispatch` | [`opf_dispatch_audit.py`](../examples/opf_dispatch_audit.py) |

Each example has its output saved next to it (`examples/*_output.txt`). For how each mode works, rendered results and figures, see [docs/modes/](modes/README.md).

---

## 1. RL agents with discrete actions

### 1.1 Describe the grid as the agent sees it

```python
from certflow.audit import GridModel, Limits, LoadUncertainty, RLAuditor, act, obs, ActionSpace

observation = [                     # in the order of the agent's input vector
    obs.bus_vm(),                   # res_bus.vm_pu, all buses        (measurement -> noisy in mode 2)
    obs.line_loading(),             # res_line.loading_percent
    obs.trafo_loading(),
    obs.load_p(), obs.load_q(),     # the uncertain loads
    obs.sgen_p(), obs.sgen_q(),     # setpoints                       (known exactly)
    obs.trafo_tap(),                # tap positions                   (known exactly)
]
actions = [                         # in the order of the agent's outputs
    act.idle(),
    *[act.shift_tap(t, s) for t in net.trafo.index for s in (-1, +1)],
    *[act.set_sgen(i, q_mvar=q) for i in controllable_sgens for q in (-0.5, 0.0, 0.5)],
    *[act.set_sgen(i, p_mw=p) for i in controllable_sgens for p in (0.5, 1.0)],
]
model = GridModel(net, observation, actions, Limits(vm_pu=(0.95, 1.05), max_loading_percent=100))
model.check_against_pandapower()    # run once: the model must reproduce pandapower's results at the snapshot
```

All observation blocks:

| Block | Source | Block | Source |
|---|---|---|---|
| `bus_vm`, `bus_va` | `res_bus` | `line_loading`, `trafo_loading` | `loading_percent` |
| `line_p_from`, `line_q_from` | `res_line` | `trafo_p_hv`, `trafo_q_hv` | `res_trafo` |
| `load_p`, `load_q` | `res_load` | `sgen_p`, `sgen_q` | `res_sgen` |
| `gen_p`, `gen_vm` | gen setpoints | `ext_grid_p`, `ext_grid_q` | `res_ext_grid` |
| `trafo_tap`, `line_status`, `trafo_status` | state | `constant(values)`, `custom(...)` | your own |

Every block takes `elements=[...]` (a subset or a custom order), `scale=`/`offset=` (a fixed affine normalisation)
and `noisy=` (whether mode 2 treats the block as a measurement).

Actions available: `idle`, `set_sgen` / `shift_sgen`, `set_gen` / `shift_gen`, `set_tap` / `shift_tap`,
`switch_line` / `switch_trafo`, `combo(...)` and `custom(fn, label)`.

- **One output per device (MultiDiscrete):** use `ActionSpace.multi([[head 0 actions], [head 1 actions], ...])`.
- **Masked agents:** if your agent masks invalid actions (MaskablePPO), use `ActionSpace(actions, masked=True)`.
  Leave it unmasked otherwise. Masking an agent that was not trained with a mask verifies a different policy.

### 1.2 The agent

```python
auditor = RLAuditor(policy, model)               # torch nn.Sequential up to the logits / Q-values
auditor = RLAuditor(MaskablePPO.load("agent.zip"), model)          # SB3 / sb3-contrib model (PPO, A2C, DQN)
auditor = RLAuditor(policy, model, input_layers=normalizer_layers(vn.obs_rms.mean, vn.obs_rms.var, clip=10.0))
                                                 # VecNormalize: mean/std + clip, as exact layers
```

### 1.3 Mode 1: are all actions the agent can take safe?

```python
rep = auditor.next_step(net, load_width=0.05)    # net = the snapshot (setpoints, taps, loads)
print(rep.summary())
```

- **Input set:** the agent observes the grid for every load within ±5 % of the snapshot. The power flow is
  certified over that set and the agent's observation is pushed through the network.
- **What is checked:** for every action it can take, the post-action grid is certified against the limits for
  every next load.
  - `next_load="same"` (default): the next load is anywhere in the same box.
  - `next_load="step", step_change=0.01`: the next load is the observed one ±1 %. This is tighter but slower.
- **N-1:** `Limits(n1=True)` adds every line outage (an outage that splits the grid counts as a violation).
- **Counterexamples:** for every action that is not proven safe, the auditor searches for a load where the
  deployed agent really takes it and a next load where a limit is really violated.

**Many loads? Use a low-rank load set.** Box splitting only works in a few dimensions. With one independent symbol
per load, the proof still holds, but it may not separate the agent's decision regions. Use load modes instead, as in
the paper:

```python
LoadUncertainty(rel=0.05, directions="common")                   # all loads together
LoadUncertainty.from_samples(historical_load_p, n_modes=3)       # 3 principal modes, ±3 sigma
rep = auditor.next_step(net, uncertainty=...)
```

The report shows which actions the Monte-Carlo samples actually hit. When far fewer actions are hit than certified,
the bounds are loose and a smaller set or a low-rank set will help.

### 1.4 Mode 2: how much measurement noise can the agent take?

```python
nr = auditor.noise_robustness(net, eps_max=0.2)
print(nr.summary())
```

The grid state is the snapshot. The agent sees `obs + eta` with `|eta_i| <= eps * |obs_i|` on the measurement
blocks; its own setpoints, taps and switch states are exact. The report gives:

- the **same-action** radius: guaranteed (certified) and found (a concrete noise vector that changes the action);
- the **safe-action** radius: up to which noise every action the agent can pick is certified safe, and a concrete
  noise vector that makes it pick an action that really violates a limit;
- the same-action radius per observation block;
- the actions the agent can pick at fixed noise levels.

Options:

- `noisy=["bus_vm", "line_loading"]` chooses the noisy blocks;
- `relative=False` uses absolute noise in the observation's units;
- `step_change=0.01` lets the loads move by ±1 % before the action takes effect.

### 1.5 Mode 3: which states can the agent drive the grid into over several steps?

```python
rep = auditor.multi_step(net, uncertainty=LoadUncertainty(rel=0.10, directions="common"),
                         steps=3, step_change=0.03)
print(rep.summary())
rep.plot("reachable.png")          # 3D: element x step x certified band (voltages and loadings)
```

How the load evolves:

- **Step 0:** the loads are anywhere in the initial set (here all loads together, ±10 %).
- **Each step:** the agent acts, then the loads change by up to ±`step_change` before its next action. The change
  goes along the same directions as the initial set.
- **Correlation:** the loads are tracked with new symbols per step rather than re-boxed, so the steps stay
  correlated.

The report gives:

- **every action sequence** the agent can take (the certified tree);
- **a certified envelope per step** for every bus |V| and line / transformer loading: step 0 is the grid before
  any action, step k the grid after the k-th action, each a union over all reachable sequences;
- **a Monte-Carlo check:** trajectories of the deployed agent under the same load model must follow certified
  sequences and stay inside the envelopes;
- **SAFE** if every action of every sequence is certified safe.

`rep.envelopes[k].select("vm")` returns the raw bounds, and `rep.tree` the full tree (`.paths()`, `.walk()`).

`track="rebox"` (the default) keeps the number of symbols constant; it is never looser than `track="history"` when the step change moves along the set's own directions. The horizon is limited by the branching of the tree, not by memory: see [modes/rl_multistep.md](modes/rl_multistep.md#cost-and-scaling-what-the-bottleneck-is). Runnable example:
[`examples/multistep_reachability.py`](../examples/multistep_reachability.py).

![certified reachable states over 3 steps](../examples/multistep_reachability.png)

### 1.6 What the model assumes

- Loads are constant power (ZIP shares are ignored, with a warning).
- Gens are ideal PV units (no Q limits inside the power flow).
- There is one `ext_grid` as the slack.
- Switches keep their state.
- `loading_percent` follows pandapower's definition.
- **Taps:** pandapower ≥ 3 ignores `tap_pos` unless `tap_changer_type` is set. The model warns if a tap action
  would do nothing.
- Lines that end at an open switch are handled exactly (their current is an exact function of the closed end's
  voltage).

---

## 2. NN dispatch controllers (the paper's use case)

```python
from certflow.audit import verify_dispatch
rep = verify_dispatch(net, model, load_width=0.10)              # pandapower net, MATPOWER case or Grid
rep = verify_dispatch(net, model, load_width=0.10, restore=True)  # + certified piecewise setpoint correction
print(rep.summary())
```

Declare the network's layout instead of reshaping it:

| Argument | Values |
|---|---|
| `inputs` | `"pd_qd"` (default), `"pd"`, or indices into `[Pd; Qd]` |
| `input_units` | `"MW"` or `"pu"` |
| `outputs` | `("pg", "vg")` (default), `("vg", "pg")`, `("pg",)` (Vg held at setpoint) or `("vg",)` |
| `pg_units` | `"MW"` or `"pu"` |
| `pg_gens` | which generators' Pg the network outputs (the others keep their setpoint) |
| `vg_clip` | the deployed saturation of Vg, if any |
| `load_set` | any `AffineSet` over `[Pd; Qd]`, e.g. `certflow.sets.low_rank_load_set` (paper eq. 19) |

Limits checked by default: bus |V|, generator P and Q, slack P and Q, branch current and angle difference. The
limits are exact, with zero tolerance.
