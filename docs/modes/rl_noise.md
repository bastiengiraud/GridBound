# RL mode 2: measurement noise

**Question:** the grid is at a known operating point, but the agent reads it through noisy sensors. How much noise
can it take before it acts differently, and before it can pick an action that violates a limit?

## API

```python
nr = auditor.noise_robustness(net, eps_max=0.10)
print(nr.summary())
nr.plot("noise.png")
```

| Option | Meaning |
|---|---|
| `noisy=None` | the measurement blocks of the observation are noisy. The agent's own setpoints, taps and switch states are exact. |
| `noisy=["bus_vm", "line_loading"]` | the noisy blocks, or a boolean mask / entry indices |
| `relative=True` | noise as a fraction of each measured value: \|η_i\| ≤ ε·\|o_i\| |
| `relative=False` | absolute noise in the observation's units; or pass an array of per-entry scales |
| `require_safe=True` | also compute the radius for "every action it can pick is safe" |
| `step_change=0.01` | the action takes effect after the loads moved by up to ±1 % |
| `levels=(0.01, 0.02, ...)` | the noise levels at which the reachable action set is listed |

## How it works

1. **Snapshot:** the grid state is exact. A plain power flow gives the true observation o.
2. **Noise:** every noisy entry gets its own new symbol, o_i + ε·s_i·η_i with η_i ∈ [−1, 1].
3. **Guaranteed radius:** bisection on ε. At each ε the observation set is pushed through the network and the masked
   argmax gives the set of actions the agent can take.
   - **"Same action":** that set must be the noise-free action alone.
   - **"Safe action":** every action in it must be certified safe, through the same post-action screening as mode
     1.
4. **Found radius:** an attack looks for a concrete noise vector that breaks the requirement. It combines one
   vertex per competing action from the logits' linear coefficients, iterated sign-gradient steps and random
   vertices. For "safe action", the action the attack produces must be really unsafe, re-simulated with a plain
   power flow.
5. **The bracket:** the true radius lies between the two, guaranteed ≤ true ≤ found. The gap measures how loose
   the bounds are.

## Example: the feeder snapshot

<!-- report:rl_noise -->
```text
MEASUREMENT-NOISE AUDIT  (noise on 15 of 18 observation entries, relative to each value; exact entries: sgen_q, trafo_tap)
  noise-free action: idle
  same action:  guaranteed up to 0.39%   |  smallest noise found that changes it: 0.54% (makes it pick sgen 1 Q=-1 MVAr)
  safe action:  guaranteed up to 10.00% (whole range)   |  smallest noise found that makes it pick an unsafe action: none found up to 10.00%
  per observation group (only that group noisy), same action:
     bus_vm           guaranteed >= 0.47%                 breaks at 0.54%
     line_loading     guaranteed >= 10.00% (whole range)  no break found
     trafo_loading    guaranteed >= 10.00% (whole range)  no break found
     load_p           guaranteed >= 10.00% (whole range)  no break found
  actions the agent can pick at fixed noise levels:
     +-  1.00%: idle | trafo 0 tap +1 | trafo 0 tap -1 | sgen 1 Q=-1 MVAr | sgen 1 Q=0 MVAr   all safe
     +-  2.00%: idle | trafo 0 tap +1 | trafo 0 tap -1 | sgen 1 Q=-1 MVAr | sgen 1 Q=0 MVAr   all safe
     +-  5.00%: idle | trafo 0 tap +1 | trafo 0 tap -1 | sgen 1 Q=-1 MVAr | sgen 1 Q=0 MVAr   all safe
     +- 10.00%: idle | trafo 0 tap +1 | trafo 0 tap -1 | sgen 1 Q=-1 MVAr | sgen 1 Q=0 MVAr   all safe
  time 1.2s
```
<!-- /report -->

![noise audit](../figures/rl_noise.png)

How to read it:

- **Same action:** this agent changes its action at about 0.5 % noise on the voltage measurements. That is a real,
  found noise vector, and the guarantee is within 0.15 percentage points of it.
- **Other groups:** the agent is insensitive to noise on loadings and loads.
- **Safe action:** every action it can pick under up to 10 % noise is still certified safe at this operating point.
  So noise can make it act differently, but not dangerously.

## Limits of the guarantee

- **One operating point:** the grid state is the snapshot. Combine with mode 1 (`step_change`) for load changes.
- **Bounded noise:** the noise is bounded and independent per entry, not adversarially correlated across time.
- **Zero values:** relative noise on an entry whose value is 0 is zero noise. Use `relative=False` or per-entry
  scales for such entries.
