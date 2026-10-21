# Mode 3: robustness of the agent's decision to measurement noise

**Question.** The grid is at a known operating point, but the agent sees noisy measurements. How large can the noise
be before the agent may pick a different action? Up to which noise level is every action it might pick still safe?

The worked example is `run_noise_robustness` in `examples/wscc9_actor_reachability.py`. The machinery is generic:
[`certflow.robustness`](../certflow/robustness.py).

## How it differs from the other modes

The **grid state is exact**: there is no uncertainty in the physics, only in what the agent sees. So:

- **No certified PF at the input.** The true observation $o$ comes from one point PF, and the uncertainty is the
  additive noise $\eta$ on top of it.
- **The noise set is a box with one independent symbol per measurement**, not a low-rank or physically coupled set.
- **The main new piece is the outer bisection on the noise level $\varepsilon$.** Each bisection step is a full
  certified reachability check: DeepZ plus the masked argmax. A falsification attack runs alongside it to give an
  upper bound.
- **Certified PFs appear only in the safety check** of the actions reachable under noise. They are evaluated at the
  **true** state, since that is the state the action is applied to.

So your reading is right: the core of this mode is `add_noise` plus the bisection loop `bisect_radius`. The
falsification attack turns the single certified number into a bracket.

## Step by step

### Step 0: the exact observation

$o = $ `model.simulate(st, δ)["obs"]` is a point PF at the operating point, giving all 26 entries. It becomes a
constant TM (no symbols). The mask is computed from the true state.

### Step 1: the noise set

$$\mathcal O(\varepsilon) = \{\, o + \eta \;:\; |\eta_i| \le \varepsilon\, s_i \,\},\qquad s_i = |o_i| \text{ for the 20 measured entries},\; s_i = 0 \text{ for the 6 line-status entries.}$$

This is an axis-aligned box. As a zonotope its generator matrix is **diagonal**: each measured entry gets its **own
new symbol**, and no symbol is shared. All 20 entries move at once and independently, including combinations that
are physically inconsistent. That is intended, since it models independent sensor errors. The noise is **relative**,
and one common $\varepsilon$ applies to all entries: at $\varepsilon = 1\%$ every measurement may be off by up to 1 %
of its own value.

> [`add_noise(x, ε·s)`](../certflow/robustness.py#L24): appends one symbol per entry with $s_i > 0$.

### Step 2: what the agent can do under noise level ε (one certification)

1. `actor.reach(add_noise(o, ε·s))`: DeepZ through the tanh network, including the float32 deployment error
   (as in [rl_reachability.md](rl_reachability.md), Step 4).
2. [`reachable_argmax(logits, mask)`](../certflow/nn/argmax.py#L51) gives the certified set $A(\varepsilon)$ of
   actions the agent can pick for some noise in the box.
3. The requirement is a predicate on $A(\varepsilon)$:
   - **same action:** $A(\varepsilon) = \{a_0\}$, where $a_0$ is the noise-free action;
   - **safe action:** every $a \in A(\varepsilon)$ passes the one-step check at the true state
     ([`StepChecker`](../examples/wscc9_env.py#L275): certified post-action PF over $\delta \pm m$, N-0 rule, N-1 if
     `--require-n1`). The checker caches per (state, box), so each distinct action is certified once.

> [`robustness_radius(policy, obs_tm, scale, mask, require)`](../certflow/robustness.py#L115). `require="same"` or any callable `require(actions) -> bool`.

### Step 3: bisection for the largest certified ε (lower bound)

1. Try $\varepsilon_{max}$ (20 %). If it is certified, report "≥ 20 % (whole range)".
2. Otherwise bisect on $[0, \varepsilon_{max}]$ down to the tolerance (`tol = 5e-4`, i.e. 0.05 %). Move `lo` up when
   the midpoint is certified, `hi` down when it is not.
3. Report `lo`. **Every reported value was itself certified**, so it is a proven lower bound on the true radius,
   whether or not the requirement is monotone in ε.

> [`bisect_radius(certify, eps_max, tol, falsify)`](../certflow/robustness.py#L83) returns a [`RadiusResult`](../certflow/robustness.py#L36) (`certified`, `falsified`, `counterexample`, `history`).

### Step 4: falsification for the smallest breaking ε (upper bound)

The attack looks for a concrete $\eta$ with $|\eta_i| \le \varepsilon s_i$ that changes the action:

- **Targeted candidates:** for each competitor $b$, the reachable logits' linear coefficients on the noise symbols
  give $\partial(l_{a_0} - l_b)/\partial\eta$. The candidate sets every entry at the box vertex that pushes that
  margin down.
- **Random vertices:** 64 random vertices of the noise box.

Each candidate is evaluated with the network's forward pass, and any change of the masked argmax is a
counterexample. A second bisection, between the certified value and $\varepsilon_{max}$, finds the smallest ε at
which the attack still succeeds.

> [`argmax_attack(policy, obs, scale, ε, target, mask)`](../certflow/robustness.py#L47); used by `bisect_radius` when `require="same"`.

The true radius therefore lies in **[certified, breaking]**. A gap between the two means either the certified set is
somewhat conservative (relaxation error) or a worse noise vector exists than the heuristic attack found.

### Step 5: per measurement group

The same Steps 2–4 are repeated with $s_i$ set to zero outside one group, so the other entries stay at their true
values:

| Group | Entries |
|---|---|
| generation | $P_{slack}, P_{g2}, P_{g3}$ |
| loads | load P (3), load Q (3) |
| PV voltages | $V_2, V_3$ |
| line loadings | 6 |
| transformer loadings | 3 |

The all-entries radius is smaller than every group's radius because errors in different groups can push the decision
the same way and add up.

> [`GROUPS`](../examples/wscc9_actor_reachability.py#L402) and the loop in [`run_noise_robustness`](../examples/wscc9_actor_reachability.py#L406).

### Step 6: safe-action radius and fixed noise levels

- **Safe-action radius:** Step 3 with the "every reachable action is safe" predicate. It is usually much larger than
  the same-action radius: the agent may switch to a neighbouring action that is also safe.
- **Fixed levels:** $A(\varepsilon)$ printed at 0.5, 1, 2, 5 and 10 %, each marked "all safe" or "NOT all safe".

## Reading the output

```
=== MEASUREMENT NOISE at after +0.5 jump at bus 8: ...
  nominal action: Pg2-45MW                                     <- a0 (noise-free)
  same-action radius: certified 0.51% | breaking noise: 0.77%  <- Steps 3-4: true radius in [0.51 %, 0.77 %]
  per measurement group (only that group noisy):               <- Step 5
     loads (P, Q)   certified >= 0.86%   breaks at 1.27% -> Pg2-40MW
  safe-action radius (...): ... up to 20.00% ...               <- Step 6
  certified action set at fixed noise levels:                  <- Step 6
     +- 1.0%: Pg2-45MW | Pg2-40MW | Pg2-35MW   all safe
```

## Modelling caveats

- $P_{g2}$, $P_{g3}$ and $V_2$, $V_3$ are the agent's own setpoints. A real operator knows them exactly, so noise on
  them may be unrealistic. Pass a `scale` that is zero on those entries to exclude them.
- The noise is bounded and independent per entry (worst case, no distribution). Correlated sensor errors would need a
  zonotope with shared generators instead of `add_noise`. The rest of the pipeline is unchanged.
- The mask and the safety check use the true state; only the agent's observation is noisy.
