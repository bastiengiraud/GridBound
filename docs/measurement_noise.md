# Measurement noise: auditing an agent's robustness to sensor errors

**Question.** The grid is at a known operating point, but the agent sees noisy measurements. How large can the noise
be before the agent may pick a different action? And up to which noise level is every action it might pick still
safe for the grid?

Both answers are computed for the trained agent as given. Nothing about its training is used. The worked example is
`run_noise_robustness` in `examples/wscc9_actor_reachability.py`, on the WSCC 9-bus agent described in
[rl_reachability.md](rl_reachability.md). The machinery is generic: [`certflow.robustness`](../certflow/robustness.py).

## How this differs from the reachability audit

Here the **grid state is exact**: the physics is known, and only what the agent sees is uncertain. Therefore:

- **No certified power flow at the input.** The true observation $o$ comes from one point power flow, and the
  uncertainty is additive noise $\eta$ on top of it.
- **The noise set is a box** with one independent symbol per measurement. It is not low-rank and not tied to the
  physics.
- **The central loop is a bisection on the noise level $\varepsilon$.** Each bisection step is one certified
  reachability computation: network reachability plus the masked argmax. A falsification search runs alongside it,
  so the result is a bracket [proven robust, shown to break] rather than a single number.
- **Certified power flows appear only in the safety check** of the actions the agent might pick under noise. They are
  evaluated at the **true** state, because that is the state the action is applied to.

## Step by step

### Step 0: the exact observation

$o$ is the 26-entry observation at the operating point, from a point power flow
(`model.simulate(st, δ)["obs"]`). It becomes a constant Taylor model, with no symbols. The action mask is computed
from the true state.

### Step 1: the noise set

$$\mathcal O(\varepsilon) = \{\, o + \eta \;:\; |\eta_i| \le \varepsilon\, s_i \,\},\qquad s_i = |o_i| \text{ for the 20 measured entries},\; s_i = 0 \text{ for the 6 line-status entries.}$$

This is an axis-aligned box. As a zonotope its generator matrix is **diagonal**: each measured entry gets its **own
new symbol**, and no symbol is shared. All 20 entries move at the same time and independently. This includes
combinations that are physically inconsistent, which is the right model for independent sensor errors. The noise is
**relative**, and one common $\varepsilon$ applies to all entries: at $\varepsilon = 1\%$, every measurement may be
off by up to 1 % of its own value.

> [`add_noise(x, ε·s)`](../certflow/robustness.py#L24) appends one symbol per entry with $s_i > 0$.

### Step 2: what the agent can do at noise level ε

1. `actor.reach(add_noise(o, ε·s))` runs network reachability through the tanh network, including the rounding of
   the deployed float32 network ([method.md, Step 3](method.md#3-network-reachability-deepz)).
2. [`reachable_argmax(logits, mask)`](../certflow/nn/argmax.py#L51) gives the certified set $A(\varepsilon)$ of
   actions the agent picks for at least one noise vector in the box.
3. The requirement is a predicate on $A(\varepsilon)$:
   - **same action:** $A(\varepsilon) = \{a_0\}$, where $a_0$ is the action without noise;
   - **safe action:** every $a \in A(\varepsilon)$ passes the one-step safety check at the true state. This is
     [`StepChecker`](../examples/wscc9_env.py#L277): a certified post-action power flow over $\delta \pm m$ against
     the grid limits listed in
     [rl_reachability.md](rl_reachability.md#what-counts-as-a-violation), with N-1 when `--require-n1` is given.
     Results are cached per (state, box), so each distinct action is certified once.

> [`robustness_radius(policy, obs_tm, scale, mask, require)`](../certflow/robustness.py#L115) with `require="same"` or any callable `require(actions) -> bool`.

### Step 3: largest certified ε (lower bound)

1. Try $\varepsilon_{max}$ (20 %). If it is certified, report "≥ 20 % (whole range)".
2. Otherwise bisect on $[0, \varepsilon_{max}]$ down to the tolerance (`tol = 5e-4`, i.e. 0.05 %). Raise the lower
   end when the midpoint is certified, lower the upper end when it is not.
3. Report the lower end. **Every reported value was itself certified**, so it is a proven lower bound on the true
   radius. This holds even if the requirement were not monotone in ε.

> [`bisect_radius(certify, eps_max, tol, falsify)`](../certflow/robustness.py#L83) returns a [`RadiusResult`](../certflow/robustness.py#L36) (`certified`, `falsified`, `counterexample`, `history`).

### Step 4: smallest breaking ε (upper bound)

A search looks for a concrete $\eta$ with $|\eta_i| \le \varepsilon s_i$ that changes the action. It tries two kinds
of candidate:

- **Targeted:** for each competing action $b$, the reachable logits' linear coefficients on the noise symbols give
  $\partial(l_{a_0} - l_b)/\partial\eta$. The candidate puts every entry at the corner of the noise box that reduces
  that margin.
- **Random:** 64 random corners of the noise box.

Each candidate goes through the network's forward pass, and any change of the masked argmax is a concrete
counterexample. A second bisection, between the certified value and $\varepsilon_{max}$, finds the smallest ε at
which the search still succeeds.

> [`argmax_attack(policy, obs, scale, ε, target, mask)`](../certflow/robustness.py#L47), used by `bisect_radius` when `require="same"`.

The true radius therefore lies in **[certified, breaking]**. A gap between the two means one of two things: the
certified set is somewhat conservative (relaxation error), or a worse noise vector exists than the search found.

### Step 5: one measurement group at a time

Steps 2–4 are repeated with $s_i = 0$ outside one group, so the other entries keep their true values:

| Group | Entries |
|---|---|
| generation | $P_{slack}, P_{g2}, P_{g3}$ |
| loads | load P (3), load Q (3) |
| PV voltages | $V_2, V_3$ |
| line loadings | 6 |
| transformer loadings | 3 |

This shows which measurements the decision is most sensitive to. The radius with all entries noisy is smaller than
any group's radius, because errors in different groups can push the decision the same way and add up.

> [`GROUPS`](../examples/wscc9_actor_reachability.py#L569) and the loop in [`run_noise_robustness`](../examples/wscc9_actor_reachability.py#L573).

### Step 6: safe-action radius and fixed noise levels

- **Safe-action radius:** Step 3 with the predicate "every action the agent might pick is safe". It is usually larger
  than the same-action radius, because a neighbouring action can be safe too. It ends at the first noise level where
  the agent might pick an action that violates a grid limit.
- **Fixed levels:** $A(\varepsilon)$ at 0.5, 1, 2, 5 and 10 %, each marked "all safe" or "NOT all safe".

## Reading the output

```
=== MEASUREMENT NOISE at after +0.5 jump at bus 8: ...
  nominal action: Pg2-45MW                                     <- a0 (no noise)
  same-action radius: certified 0.51% | breaking noise: 0.77%  <- Steps 3-4: true radius in [0.51 %, 0.77 %]
  per measurement group (only that group noisy):               <- Step 5
     loads (P, Q)   certified >= 0.86%   breaks at 1.27% -> Pg2-40MW
  safe-action radius (...): ... up to 6.60% is certified safe  <- Step 6
  certified action set at fixed noise levels:                  <- Step 6
     +- 1.0%: Pg2-45MW | Pg2-40MW | Pg2-35MW   all safe
```

Reference outputs for all operating points are in `examples/wscc9_noise_robustness_output.txt`.

## Modelling choices

- **Setpoints are noisy too.** $P_{g2}$, $P_{g3}$, $V_2$ and $V_3$ are the agent's own setpoints, which an operator
  usually knows exactly. To exclude them, pass a `scale` that is zero on those entries.
- **Worst-case noise.** The noise is bounded and independent per entry; no distribution is assumed. Correlated sensor
  errors can be modelled by a zonotope with shared generators instead of `add_noise`. The rest of the pipeline is
  unchanged.
- **True state for safety.** The mask and the safety check use the true grid state; only the agent's observation is
  noisy.
