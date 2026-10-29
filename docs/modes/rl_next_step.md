# RL mode 1: next-step safety

**Question:** the agent sees the grid at some load inside a box around the current operating point. Which actions
can it take there, and does each of them keep every limit one step later, for every next load?

## API

```python
from certflow.audit import GridModel, Limits, LoadUncertainty, RLAuditor, act, obs

model = GridModel(net,
                  observation=[obs.bus_vm(), obs.line_loading(), obs.trafo_loading(), obs.load_p(),
                               obs.sgen_q(), obs.trafo_tap()],
                  actions=[act.idle(), act.shift_tap(0, +1), act.shift_tap(0, -1),
                           act.set_sgen(1, q_mvar=-1.0), act.set_sgen(1, q_mvar=0.0)],
                  limits=Limits(vm_pu=(0.95, 1.05), max_loading_percent=100))
model.check_against_pandapower()                 # once: the model must read the grid like the agent's env
auditor = RLAuditor(policy, model)               # torch module, SB3 model or certflow Sequential

rep = auditor.next_step(net, load_width=0.05)    # each load within +-5 % of the snapshot
rep = auditor.next_step(net, uncertainty=LoadUncertainty(rel=0.35, directions=D))   # low-rank load set
print(rep.summary())
rep.plot("next_step.png")                        # 2D map of the load set (two directions)
```

| Option | Meaning |
|---|---|
| `next_load="same"` (default) | the action is applied at any load in the same box |
| `next_load="step", step_change=0.01` | the next load is the observed one ±1 %; tighter, more symbols |
| `refine="safe"` / `"single"` | split until every reachable action is proven safe / until each box has one action |
| `max_depth` | the number of bisections allowed |
| `Limits(n1=True)` | also screen every line outage |
| `ActionSpace(actions, masked=True)` | the agent masks invalid actions (MaskablePPO) |
| `ActionSpace.multi([...])` | one categorical output per device (MultiDiscrete) |

## How it works

1. **Snapshot:** the setpoints, taps and switch states of `net` form the discrete state. The loads around it form
   the input set, either per load (one symbol each) or low-rank (`directions="common"`, a mode matrix, or
   `LoadUncertainty.from_samples`).
2. **Pre-action grid:** Krawczyk certifies the power flow over the box. The agent's observation vector is built
   block by block as a Taylor model in the load symbols, so every block is consistent with the same loads.
   - Lines that end at an open switch are handled exactly. Their current is an exact function of the voltage at
     the closed end.
3. **Agent:** the observation is pushed through the network, and the masked argmax gives the certified set of
   actions it can take on the box.
4. **Post-action grid:** for every reachable action, the grid after the action is certified over the next-load
   set. Then:
   - |V| at every bus, line and transformer loading and the ext_grid exchange are bounded on both sides;
   - boxes where a quantity straddles its limit are bisected further to tighten the bounds;
   - with N-1 enabled, the same is done after every line outage.
5. **Partition:** boxes are bisected where several actions compete or an action cannot be proven safe. Splitting
   stops early when the check cannot depend on the box.
6. **Counterexample search:** for every action not proven safe, the auditor looks for a load where the deployed
   agent really takes it, then for a next load where a limit is really violated. It does so along the linear
   coefficients of the violating quantity and at random vertices. A hit makes the verdict UNSAFE.
7. **Monte-Carlo check:**
   - every sampled action must lie in the certified set of its box;
   - the post-action state must lie inside the certified bounds;
   - the report lists which actions the samples actually hit. If far fewer are hit than certified, the bounds
     are loose.

## Example: the feeder, two load directions

Here the loads at buses 2–3 move together (direction e0) and the loads at buses 4–5 move together (e1), each by up
to ±35 %. `refine="single"` splits until each box has one action.

<!-- report:rl_next_step -->
```text
NEXT-STEP SAFETY AUDIT  ->  SAFE
  loads: 2 uncertain directions, low-rank: 2 load modes around the snapshot; next load: anywhere in the same box
  limits: |V| in [0.95, 1.05] pu, loading <= 100%, N-0
  partition: 63 boxes; SAFE 100.0%  CAN-VIOLATE 0.0%  UNDECIDED 0.0%  PF-NOT-CERTIFIED 0.0%
  actions the agent can take (share of the load box) and their certified effect:
     89.06%  idle                               SAFE        |V| 0.9959..1.0357 pu, max loading 65.7%
     27.34%  sgen 1 Q=-1 MVAr                   SAFE        |V| 0.9884..1.0263 pu, max loading 71.4%
      0.78%  trafo 0 tap +1                     SAFE        |V| 0.9803..1.0207 pu, max loading 66.7%
  Monte-Carlo cross-check (400 samples): agent's action in certified set 400/400, post-action state inside certified bounds 400/400
    actions the agent actually took in the samples: 2 of the 3 certified reachable (idle, sgen 1 Q=-1 MVAr); the others are possible but rare, or bound looseness
  time 10.9s
```
<!-- /report -->

![next-step audit](../figures/rl_next_step.png)

**Left: which actions the agent can take where.** The partition follows the agent's decision boundary.
- **Low loads (lower left):** voltages rise, and the agent absorbs reactive power at the PV park (green).
- **Elsewhere:** it idles (blue).
- **Hatched boxes:** both are possible there.
- **Dots:** the action the deployed agent took at random loads. Every dot lies inside the certified set of its box.

**Right: safety.** Every action the agent can take, anywhere in the box, keeps every limit for every next load, so
the verdict is SAFE.

## Scaling and limits

- **Few symbols work best:** box splitting is effective in a few dimensions. With one symbol per load the proof
  still holds, but it may not separate the agent's decision regions, so the certified action set gets larger than
  what the agent does. Use low-rank load sets for many loads, as in the paper.
- **A tight model of the environment matters:** `check_against_pandapower()` should report deviations of about
  1e-8.
  - **Taps:** pandapower ≥ 3 ignores taps unless `tap_changer_type` is set; the model warns if a tap action would
    do nothing.
  - **Loads** are constant power.
- **Steady state:** the guarantee covers the steady state after the action, not the transient.
