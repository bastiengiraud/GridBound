# Mode 2: certified reachability of an RL agent over load uncertainty

**Question.** The agent is at an operating point, and the loads lie anywhere in a box around it. Which actions can
the agent take, and is each of them safe one step ahead (N-0, and optionally N-1)? Over two steps, which action
sequences can occur?

The worked example is the WSCC-9 MaskablePPO agent (`examples/wscc9_env.py`,
`examples/wscc9_actor_reachability.py`). The machinery is generic:

- [`DiscretePolicyVerifier`](../certflow/policy.py#L102)
- [`MultiStepVerifier`](../certflow/policy.py#L217)
- [`enclose_pf`](../certflow/power/pfcert.py#L43) with an uncertain [`Shunt`](../certflow/power/pf.py#L314)
- [`certflow.power.quantities`](../certflow/power/quantities.py)
- [`n1_analysis`](../certflow/power/contingency.py#L112)

## How it differs from the paper's pipeline

| | Paper (mode 1) | RL reachability (mode 2) |
|---|---|---|
| NN input | the loads $x(\varepsilon)$, an affine zonotope | the **observation**, a nonlinear function of the grid state (load P/Q at the actual voltage, slack MW, branch loadings) |
| Certified PF | **after** the NN (setpoints → state) | **before** the NN (state → observation) **and after** the action (post-action state) |
| How uncertainty enters the PF | right-hand side $S(\varepsilon)$ only | load **admittances** $Y(1+\delta)$, which enter the Jacobian too |
| NN output | continuous setpoints → bands | discrete **argmax** over masked logits → set of reachable actions |
| What is checked | bands of the controlled state within limits | safety of the **next** state for each reachable action |
| Splitting driven by | Krawczyk residual (ρ ≥ 1) | Krawczyk residual **and** action ambiguity **and** unsafe actions |

So your reading is right: **the main addition is the certified PF at the input, which couples the observation to
the load uncertainty.** Three further pieces sit on top of the paper:

- the Krawczyk extension for parameter-dependent admittances;
- the discrete-output analysis (`reachable_argmax`);
- the per-action look-ahead check, which is a second certified PF with no NN in it.

## Setup: what the environment does

- **Uncertainty:** loads at buses 5, 6, 8 are constant admittances $Y = (1+\delta)\,Y_{nom}$ with
  $Y_{nom} = \overline{S_{nom}}/|V_0|^2$, so the exogenous input is $\delta\in\mathbb R^3$. The verified set is the
  box $\delta \in \delta_c \pm w$, written as $\delta = \delta_c + \mathrm{diag}(w)\,\varepsilon$ with
  $\varepsilon\in[-1,1]^3$.
- **Discrete state:** [`OperatingState`](../examples/wscc9_env.py#L123) holds the line status, $P_{g2}, P_{g3}$ and
  $V_2, V_3$ setpoints. It is fixed within one verification. [`apply(a)`](../examples/wscc9_env.py#L129) gives the
  next state after an action, and [`mask()`](../examples/wscc9_env.py#L145) gives the valid actions.
- **Observation (26 entries):** line status (6), $P_{slack}, P_{g2}, P_{g3}$ (3), load P (3), load Q (3),
  $V_2, V_3$ (2), line loadings % (6), transformer loadings % (3). Loadings are $\max(|S_{from}|, |S_{to}|)$
  including the branch shunt.

## Step by step (one-step study, `run_case`)

### Step 1: partition loop

[`DiscretePolicyVerifier.verify`](../certflow/policy.py#L153) keeps a stack of $\varepsilon$-boxes, starting from
$[-1,1]^3$. It [evaluates](../certflow/policy.py#L123) each box (Steps 2–6), stops when the box is done
([`_done`](../certflow/policy.py#L146)), and otherwise bisects (Step 7).

> The callbacks come from [`make_problem(model, st, δc, w, checker)`](../examples/wscc9_env.py#L345):
> `observe`, `mask`, `check`.

### Step 2: certified pre-action PF over the box (coupling the state)

For the box, the load deviation is a TM, $\delta(\varepsilon)$ ([`AffineSet.sub_tm`](../certflow/sets.py#L54)).
The PF is certified for **every** $\delta$ in it with the paper's Krawczyk machinery. The only change is that the
uncertainty now sits in the admittance matrix: the bus shunt at each load bus is $y_{nom}\,\delta(\varepsilon)$, on
top of the nominal $y_{nom}$ already in $Y_{bus}$.

- **Residual:** gains $g\,\delta\,|V|^2$ and $-b\,\delta\,|V|^2$ terms (still quadratic in $z$, still exact in a
  degree-2 TM).
- **Jacobian:** its slope with respect to $z$ now depends on $\varepsilon$: $\partial J/\partial z_j = H_j +
  H^{sh}_j\,\delta(\varepsilon)$. The Krawczyk remainder (paper eq. 28) therefore gets a second term,
  $\sum_j |MH^{sh}_j|\,\max|\delta|\, r_j$, with a bounded scale.

> **In the code**
> - [`WSCC9Model.enclose`](../examples/wscc9_env.py#L212) → [`enclose_pf(..., shunt=Shunt(load_buses, y_nom, δ))`](../certflow/power/pfcert.py#L43).
> - [`Shunt`](../certflow/power/pf.py#L314) holds the admittance and the TM $\delta$.
> - [`RectPFSystem`](../certflow/power/pf.py#L340) handles it in `injections` (shunt power), `jacobian_tm` (`_jsh` terms), and [`hessian_structure`](../certflow/power/pf.py#L478), which returns `[(base, None), (shunt_structure, scale)]`.
> - [`krawczyk`](../certflow/krawczyk.py#L157) loops over these `(structure, scale)` terms in `_Kb`. This is the only extension of the core Krawczyk.
> - If the proof fails ($\rho\ge1$), the observation is *unknown* on this box and the box is split along the Krawczyk residual score (`Observation.score`).

### Step 3: the observation as a Taylor model

Every observation entry is computed from the certified state **over the same symbols $\varepsilon$**. They therefore
stay mutually consistent: a high load at bus 8 and the resulting higher loading of line 8-9 come from the same
$\varepsilon$, not from independent intervals.

| Entries | How | Function |
|---|---|---|
| line status, $P_{g2/3}$, $V_{2/3}$ | constants | `obs_tm` (`cst`) |
| $P_{slack}$ | certified injection at the slack bus | `enc.P` |
| load P, Q | $\overline{y_{nom}(1+\delta)}\,\lvert V\rvert^2$: the actual absorbed power at the actual voltage | [`admittance_power`](../certflow/power/quantities.py#L29) |
| line / transformer loading | $\max_{ends}\sqrt{P^2+Q^2}$ incl. branch shunt, ÷ rating | [`branch_end_flows`](../certflow/power/quantities.py#L39) → [`branch_apparent_power`](../certflow/power/quantities.py#L55) ([`tm.sqrt`](../certflow/tm.py#L350), [`tm.maximum`](../certflow/tm.py#L378)) |

> **In the code**: [`WSCC9Model.obs_tm`](../examples/wscc9_env.py#L221) assembles the 26-row TM. Its first 3 symbols are the box directions, followed by symbols from the Krawczyk enclosure and the nonlinear operations.

### Step 4: reachable logits

The actor (26 → 64 → 64 → 73, tanh) is pushed through with DeepZ, as in paper Step 3. The tanh relaxation adds one
symbol per relaxed neuron. The float32 deployment error of the SB3 network is included.

> [`sb3_actor(W)`](../certflow/nn/network.py#L163) → `Sequential.reach(obs_tm)`; [`_SShaped.reach`](../certflow/nn/layers.py#L194) for tanh.

### Step 5: which actions can win (masked argmax)

The mask depends only on the discrete state, so it is constant over the box. Action $a$ can be chosen somewhere in
the box only if, for every valid competitor $b$, $\sup_\varepsilon (l_a - l_b)(\varepsilon) \ge 0$. The
difference $l_a - l_b$ is formed as a **correlated** TM over the shared symbols. This is much tighter than comparing
separate logit intervals: two logits that move together can have overlapping intervals and still never swap order.

> [`reachable_argmax(logits, mask)`](../certflow/nn/argmax.py#L51): interval pruning, then [`pairwise_diff_upper`](../certflow/nn/argmax.py#L19). Ties count as reachable, so the result is sound for any tie-breaking rule.

### Step 6: one-step safety of each reachable action

For each reachable action $a$, the next state is $st' = st.\mathrm{apply}(a)$. The PF of $st'$ is certified over the
**next** load deviation $\delta' \in [\delta_{lo} - m, \delta_{hi} + m]$, where the margin $m$ is the per-step load
change. No NN is involved; this is a pure certified PF (`enclose_pf_split`, with its own splitting). The action is:

- **SAFE** if the PF is proven, no branch exceeds 100 %, and $P_{slack} \in [0, 250]$ MW for every $\delta'$;
- **CAN-VIOLATE** if the proof succeeds but a limit can be exceeded, or the topology islands the grid;
- **UNDECIDED** if the PF cannot be certified.

**N-1 (optional, reported by default):** for every line in service in $st'$, the line is removed. If that islands
the grid, it counts as a certain PF failure. Otherwise the PF is certified over the same $\delta'$-box, and every
monitored branch gets a **two-sided** loading bound $[\underline\ell, \overline\ell]$. A branch is overloaded for
sure if $\underline\ell > 1$ and possibly if $\overline\ell > 1$. Sub-boxes where some $\underline\ell \le 1 <
\overline\ell$ are bisected further (`refine`). The result is certified bounds on the environment's metric:
$\#\text{overloads} + 3\cdot\#\text{PF failures}$.

> **In the code**
> - [`StepChecker.__call__`](../examples/wscc9_env.py#L330) → [`StepChecker.certify`](../examples/wscc9_env.py#L290) → [`enclose_pf_split`](../certflow/power/pfcert.py#L74) (loading, slack, $\lvert V\rvert$ bounds per leaf).
> - N-1: [`n1_analysis`](../certflow/power/contingency.py#L112) → [`Grid.with_branch_status`](../certflow/power/grid.py#L173) / [`islanded`](../certflow/power/grid.py#L181) → [`_outage_bounds`](../certflow/power/contingency.py#L83) (refinement of undecided loadings) → [`branch_loading_bounds`](../certflow/power/contingency.py#L67).
> - Bounds: [`N1Report.count_bounds(fail_weight=3)`](../certflow/power/contingency.py#L42).
> - `--require-n1` turns a possible N-1 violation into CAN-VIOLATE.

### Step 7: when to split, and where

A box is finished when it reaches `max_depth`, or when every reachable action is SAFE and (with `refine="single"`, as
in the example) only one action is reachable. Otherwise it is bisected along the direction maximising
score × width. The score adds two normalised parts:

- **action ambiguity:** for every pair of competing actions, $\sum |C_a - C_b|$ on the box symbols. This is the
  direction that most changes which action wins.
- **unsafe actions:** the check's own split hint, when it provides one.

If the observation could not be certified, the Krawczyk residual score is used instead.

> [`DiscretePolicyVerifier.evaluate`](../certflow/policy.py#L123) (scores), [`ambiguity_split_score`](../certflow/nn/argmax.py#L78), [`verify`](../certflow/policy.py#L153) (bisection). Leaf status: [`PolicyLeaf.status`](../certflow/policy.py#L55).

### Step 8: report and Monte-Carlo cross-check

[`PolicyReport`](../certflow/policy.py#L69) gives:

- `action_sets()`: which action set holds on what share of the box;
- `status_shares()`: the SAFE / CAN-VIOLATE / UNDECIDED / UNKNOWN shares;
- `reachable_actions()`.

The example then samples the box, runs a plain point PF and the **float32** actor, and checks three things: the
observation lies inside the certified range, the chosen action is in the certified set, and the post-action state
lies inside the certified band (`monte_carlo` in `wscc9_actor_reachability.py`).

## Two-step reachability (`run_two_step`)

From the root box, every reachable action $a$ gets a child node:

1. **Boxes:** collect the leaves where $a$ is reachable and take their bounding box in $\delta$ (`merge="hull"`).
2. **Propagate:** widen that box by the per-step load change $m$. This is the same assumption the one-step safety
   check uses, so step 2 starts exactly where step 1's certificate ends.
3. **Re-verify:** run Steps 1–7 again from $st.\mathrm{apply}(a)$ over the propagated box.

The result is a tree whose root-to-leaf paths are all the 2-action sequences the agent can take. Each node carries
its certified one-step safety, and the tree is safe if every node is.

> [`MultiStepVerifier.verify`](../certflow/policy.py#L263) / [`_node`](../certflow/policy.py#L241) with callbacks from [`multistep_problem`](../examples/wscc9_env.py#L365); [`StepNode.paths()`](../certflow/policy.py#L202), [`StepNode.safe`](../certflow/policy.py#L193).

**Soundness note:** the load is re-boxed at every step. The tree over-approximates every trajectory, but it loses the
correlation between $\delta$ at step 1 and at step 2. Longer horizons work the same way (`steps=k`), and the cost
grows with the number of reachable action sequences.

## Settings used in the example

| | `--quick` | full |
|---|---|---|
| load box half-width $w$ | 0.02 | 0.05 |
| next-step load change $m$ | 0 | 0.06 |
| max bisection depth | 5 | 7 |
| Monte-Carlo samples | 100 | 300 |

Operating points: `python examples/wscc9_actor_reachability.py --list-ops`.

## Reading the output

```
=== after +0.5 jump at bus 8: ... delta = [0. 0. 0.5] +- 0.05 (next-step margin +-0.06)
  pre-action PF certified (rho 0.048); observation ranges: ...       <- Step 2-3 on the whole box
  partition: 97 boxes (complete=True); reachable actions: [...]      <- Steps 1, 5, 7
     xx.xx%  Pg2-45MW                                                 <- action set and its share of the box
  one-step safety ...: SAFE 100.0% ...                               <- Step 6, share of the box
  N-1 of the agent's actions: secure on ...                          <- Step 6, N-1
     after Pg2-45MW:  N-0: OK ...  N-1: SECURE count 0 ...           <- per reachable action
  Monte-Carlo (...): ... 300/300                                     <- Step 8
```
