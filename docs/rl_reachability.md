# RL reachability: auditing an agent over load uncertainty

**Question.** A trained agent operates the grid. Its loads may be anywhere in a box around the current operating
point. Which actions can the agent take, and does each of them keep the grid within its limits one step later, both
in the intact grid (N-0) and after any single line outage (N-1)? Over several steps, which sequences of actions can
occur?

The auditor treats the agent as given. It needs only:

- the trained network;
- what the agent observes;
- what each action does, and which actions are allowed (the mask);
- a model of the grid;
- the grid's limits.

It does **not** use the agent's reward or training data. The limits are properties of the grid, so an agent is
checked against them even if it never saw them during training.

The worked example is a MaskablePPO agent (stable-baselines3) that redispatches generators, changes voltage setpoints
and switches lines on the WSCC 9-bus system (`examples/wscc9_env.py`, `examples/wscc9_actor_reachability.py`). The
verification machinery is generic:

- [`DiscretePolicyVerifier`](../certflow/policy.py#L102), [`MultiStepVerifier`](../certflow/policy.py#L239)
- [`enclose_pf`](../certflow/power/pfcert.py#L43) with an uncertain [`Shunt`](../certflow/power/pf.py#L314)
- [`certflow.power.quantities`](../certflow/power/quantities.py)
- [`n1_analysis`](../certflow/power/contingency.py#L198)

## How this differs from verifying a dispatch controller

| | Dispatch controller ([method.md](method.md)) | RL agent (this document) |
|---|---|---|
| Network input | the loads $x(\varepsilon)$, an affine zonotope | the agent's **observation**, a nonlinear function of the grid state (load power at the actual voltage, slack power, branch loadings) |
| Certified power flow | after the network (setpoints → state) | **before** the network (state → observation) **and after** the action (post-action state) |
| How the uncertainty enters the power flow | specified injections only | load **admittances** $Y(1+\delta)$, which also enter the Jacobian |
| Network output | continuous setpoints → bounds | discrete **argmax** over masked logits → set of reachable actions |
| What is checked | the controlled state is within limits | the **next** state is within limits, for every reachable action |
| Splitting driven by | Krawczyk residual | Krawczyk residual, ambiguity between actions, and unsafe actions |

The key addition is a **certified power flow at the input**. The agent observes quantities that depend on the grid
state, and that state depends on the uncertain loads. All observation entries are therefore computed from one
certified power flow over the load box, sharing its symbols. The agent only sees observations that can actually
occur, with their correlations intact. Three further pieces build on this:

- an extension of the Krawczyk test to uncertain admittances;
- the analysis of a discrete output (`reachable_argmax`);
- a look-ahead check per action, which is a second certified power flow with no network in it.

## The example environment

- **Uncertainty:** the loads at buses 5, 6 and 8 are constant admittances $Y = (1+\delta)\,Y_{nom}$, with
  $Y_{nom} = \overline{S_{nom}}/|V_0|^2$, so the uncertain input is $\delta\in\mathbb R^3$. The audited set is the box
  $\delta \in \delta_c \pm w$, written as $\delta = \delta_c + \mathrm{diag}(w)\,\varepsilon$ with
  $\varepsilon\in[-1,1]^3$.
- **Discrete state:** [`OperatingState`](../examples/wscc9_env.py#L125) holds the line status and the $P_{g2},
  P_{g3}$ and $V_2, V_3$ setpoints. It is fixed within one audit. [`apply(a)`](../examples/wscc9_env.py#L131) gives
  the next state after action $a$, and [`mask()`](../examples/wscc9_env.py#L147) gives the allowed actions.
- **Actions (73):**
  - do nothing;
  - change $P_{g2}$ or $P_{g3}$ by ±5 … ±50 MW;
  - change $V_2$ or $V_3$ by ±0.01 … ±0.05 pu;
  - open or close each of the six lines.
- **Observation (26 entries):**
  - line status (6);
  - $P_{slack}, P_{g2}, P_{g3}$ (3);
  - load P and load Q (3 each);
  - $V_2, V_3$ (2);
  - line loadings % (6) and transformer loadings % (3), each $\max(|S_{from}|, |S_{to}|)$ including the branch shunt.
- **Grid data:** the PSS/E file `WSCC_9_bus.raw` used by the training environment
  ([`read_raw`](../examples/wscc9_env.py#L63) and `tests/test_wscc9_data.py` check the model against it).

### What counts as a violation

The same limits apply in the intact grid and after every line outage:

| Limit | N-0 (state after the action) | N-1 (each line outage on top of it) |
|---|---|---|
| the power flow has a unique solution (no islanding, no collapse) | ✓ | ✓ (islanding is a certain failure) |
| branch loading $\max(\lvert S_{from}\rvert, \lvert S_{to}\rvert)$ ≤ rating (250 MVA lines, 300 MVA transformers) | ✓ | ✓ |
| $\lvert V\rvert \in [0.95, 1.05]$ pu at **every** bus | ✓ | ✓ |
| slack active power $\in [0, 250]$ MW | ✓ | ✓ |

The limits are constants in `wscc9_env.py` (`LINE_RATING`, `XFMR_RATING`, `VMIN_BUS`, `VMAX_BUS`, `SLACK_MIN`,
`SLACK_MAX`), so other limits can be audited by changing them.

## Step by step: the one-step audit (`run_case`)

### Step 1: partition loop

[`DiscretePolicyVerifier.verify`](../certflow/policy.py#L153) keeps a stack of $\varepsilon$-boxes, starting from
$[-1,1]^3$. It [evaluates](../certflow/policy.py#L123) each box (Steps 2–6), stops when the box is finished
([`_done`](../certflow/policy.py#L146)), and otherwise bisects it (Step 7).

> The three callbacks (`observe`, `mask`, `check`) come from [`make_problem(model, st, δc, w, checker)`](../examples/wscc9_env.py#L374).

### Step 2: certified power flow before the action

On a box, the load deviation is a Taylor model $\delta(\varepsilon)$
([`AffineSet.sub_tm`](../certflow/sets.py#L54)). The power flow is certified for **every** $\delta$ in it with the
Krawczyk test ([method.md, Step 5](method.md#5-existence-and-uniqueness-the-krawczyk-test)). Here the uncertainty sits
in the admittance matrix: the shunt at each load bus is $y_{nom}\,\delta(\varepsilon)$, on top of the nominal
$y_{nom}$ already in $Y_{bus}$. This changes two things:

- **Residual:** it gains $g\,\delta\,|V|^2$ and $-b\,\delta\,|V|^2$ terms. These are still quadratic in the state,
  so still exact in a degree-2 Taylor model.
- **Jacobian:** its slope now depends on the input,
  $\partial J/\partial z_j = H_j + H^{sh}_j\,\delta(\varepsilon)$. The slope bound $K_b$ therefore gets a second
  correlated term, $\sum_j |MH^{sh}_j|\,\max|\delta|\, r_j$.

> **In the code**
> - [`WSCC9Model.enclose`](../examples/wscc9_env.py#L214) → [`enclose_pf(..., shunt=Shunt(load_buses, y_nom, δ))`](../certflow/power/pfcert.py#L43).
> - [`Shunt`](../certflow/power/pf.py#L314) holds the admittances and the Taylor model $\delta$.
> - [`RectPFSystem`](../certflow/power/pf.py#L340) includes the shunt in `injections`, `jacobian_tm` (`_jsh` terms) and [`hessian_structure`](../certflow/power/pf.py#L478), which returns `[(base, None), (shunt_structure, scale)]`. [`krawczyk`](../certflow/krawczyk.py#L157) adds one slope term per entry.
> - If the test fails on a box, the observation is *unknown* there and the box is split along the Krawczyk residual score (`Observation.score`).

### Step 3: the observation as a Taylor model

Every observation entry is computed from the certified state **over the same symbols $\varepsilon$**. The entries
therefore stay consistent with each other. For example, a high load at bus 8 and the resulting higher loading of
line 8-9 come from the same $\varepsilon$, not from two independent intervals.

| Entries | Computed as | Function |
|---|---|---|
| line status, $P_{g2/3}$, $V_{2/3}$ | constants | `obs_tm` |
| $P_{slack}$ | certified injection at the slack bus | `enc.P` |
| load P, Q | $\overline{y_{nom}(1+\delta)}\,\lvert V\rvert^2$: the power actually absorbed at the actual voltage | [`admittance_power`](../certflow/power/quantities.py#L29) |
| line / transformer loading | $\max_{ends}\sqrt{P^2+Q^2}$ including the branch shunt, divided by the rating | [`branch_end_flows`](../certflow/power/quantities.py#L39) → [`branch_apparent_power`](../certflow/power/quantities.py#L55) (using [`tm.sqrt`](../certflow/tm.py#L350) and [`tm.maximum`](../certflow/tm.py#L378)) |

> **In the code**: [`WSCC9Model.obs_tm`](../examples/wscc9_env.py#L223) assembles the 26-row Taylor model. Its first 3 symbols are the box directions; the rest come from the power-flow enclosure and the nonlinear operations.

### Step 4: reachable logits

The actor network (26 → 64 → 64 → 73, tanh) is applied to the observation Taylor model with DeepZ
([method.md, Step 3](method.md#3-network-reachability-deepz)). Each relaxed tanh neuron adds one symbol. The
rounding of the deployed float32 network is included.

> [`sb3_actor(W)`](../certflow/nn/network.py#L163) → `Sequential.reach(obs_tm)`; [`_SShaped.reach`](../certflow/nn/layers.py#L194) for tanh.

### Step 5: which actions can win

The mask depends only on the discrete state, so it is constant over the box. An action $a$ can be chosen somewhere
in the box only if, for every allowed competitor $b$, $\sup_\varepsilon (l_a - l_b)(\varepsilon) \ge 0$. The
difference $l_a - l_b$ is formed as a **correlated** Taylor model over the shared symbols. This is much tighter than
comparing separate logit intervals: two logits that move together can have overlapping intervals and still never
swap order.

> [`reachable_argmax(logits, mask)`](../certflow/nn/argmax.py#L51): interval pruning first, then [`pairwise_diff_upper`](../certflow/nn/argmax.py#L19). Ties count as reachable, so the result holds for any tie-breaking rule.

### Step 6: one-step safety of each reachable action

For each reachable action $a$, the next state is $st' = st.\mathrm{apply}(a)$. Its power flow is certified over the
**next** load deviation $\delta' \in [\delta_{lo} - m, \delta_{hi} + m]$, where the margin $m$ is the largest load
change per step. No network is involved. This is a certified power flow with its own splitting
(`enclose_pf_split`), and every limit in [the violation table](#what-counts-as-a-violation) is checked on its result.
The action is:

- **SAFE** if the power flow is proven and every N-0 limit holds for every $\delta'$, plus N-1 when it is enforced;
- **CAN-VIOLATE** if the proof succeeds but a limit can be exceeded, or the new topology islands the grid;
- **UNDECIDED** if the power flow cannot be certified.

**N-1** is reported by default and enforced with `--require-n1`. For every line in service in $st'$:

1. The line is removed. If that islands the grid, it is a certain power-flow failure.
2. Otherwise the power flow is certified over the same $\delta'$-box. Every limited quantity gets a **two-sided**
   bound $[\underline q, \overline q]$: each branch loading, each bus voltage and the slack power.
3. A limit is violated **for certain** if the whole interval lies outside it, and **possibly** if the interval
   crosses it.
4. Sub-boxes where an interval straddles a limit are split further (`refine`), along the direction that moves the
   undecided quantities most.

The result is certified [lower, upper] bounds on four counts:

- overloaded (outage, branch) pairs;
- outages without a power-flow solution;
- (outage, bus) voltage violations;
- outages with a slack violation.

Each action then gets one N-1 status:

- **SECURE:** all upper bounds are 0.
- **INSECURE:** some lower bound is above 0, so the violation happens for every load in the box. The output names
  the outage and the quantity.
- **UNCERTAIN:** otherwise. A sampling search over the box then either finds a concrete violation or reports that
  the bound may be conservative.

> **In the code**
> - [`StepChecker.__call__`](../examples/wscc9_env.py#L352) → [`StepChecker.certify`](../examples/wscc9_env.py#L300) → [`enclose_pf_split`](../certflow/power/pfcert.py#L74) (loading, slack and per-bus voltage bounds).
> - N-1: [`n1_analysis(..., vlim=, slack=)`](../certflow/power/contingency.py#L198) → [`Grid.with_branch_status`](../certflow/power/grid.py#L173) and [`islanded`](../certflow/power/grid.py#L181) → [`_outage_bounds`](../certflow/power/contingency.py#L142) (bounds per leaf, refinement of undecided ones) → [`branch_loading_bounds`](../certflow/power/contingency.py#L126).
> - Counts: [`N1Report.count_bounds(fail_weight, kinds)`](../certflow/power/contingency.py#L80) with `kinds` ⊆ {`"overload"`, `"voltage"`, `"slack"`}. [`N1Report.secure`](../certflow/power/contingency.py#L107) means no possible violation of any screened kind.

### Step 7: when to split, and where

A box is finished when it reaches `max_depth`, or when every reachable action is SAFE. With `refine="single"`, as in
the example, it must also have only one reachable action. Otherwise it is bisected along the direction that
maximises score × width. The score adds two normalised parts:

- **action ambiguity:** for every pair of competing actions, $\sum |C_a - C_b|$ over the box symbols. This is the
  direction that most changes which action wins.
- **unsafe actions:** the check's own split hint, when it gives one.

If the observation could not be certified, the Krawczyk residual score is used instead.

> [`DiscretePolicyVerifier.evaluate`](../certflow/policy.py#L123) (scores), [`ambiguity_split_score`](../certflow/nn/argmax.py#L78), [`verify`](../certflow/policy.py#L153) (bisection). Leaf status: [`PolicyLeaf.status`](../certflow/policy.py#L55).

### Step 8: report and Monte-Carlo cross-check

[`PolicyReport`](../certflow/policy.py#L69) gives:

- `action_sets()`: which set of actions holds on what share of the box;
- `status_shares()`: the share of the box that is SAFE, CAN-VIOLATE, UNDECIDED or UNKNOWN;
- `reachable_actions()`.

The example then samples the box, runs a plain point power flow and the **float32** actor, and checks three things:
the observation lies inside the certified range, the chosen action is in the certified set, and the post-action state
lies inside the certified bounds.

## Multi-step reachability (`run_multistep`)

The load changes from one step to the next by $\delta_{t+1} = \delta_t + G_{step}\,w_t$, with
$w_t \in [-1,1]^r$. The example offers two change models (`--step-change`):

- **independent:** every load moves by up to $\pm m$ on its own ($G_{step} = m I$, $r = 3$);
- **common:** all loads move together by the same amount ($G_{step} = m\mathbf 1$, $r = 1$).

From the root, every reachable action $a$ gets a child node that audits the agent again from
$st.\mathrm{apply}(a)$, over the loads that can occur next. The tree's root-to-leaf paths are all the action
sequences the agent can take. Each node carries its certified one-step safety, and the tree is safe when every node
is.

There are two ways to describe the loads of the next step (`--track`).

**Re-boxing (`rebox`, default).** Take the bounding box in $\delta$ of the leaves where $a$ is reachable, and widen
it by the largest change per step, $|G_{step}|\mathbf 1$. The next step is a fresh 3-dimensional box.

- Cheap: the dimension stays 3 at every step.
- The link between the loads of consecutive steps is lost. With a common-mode change, for example, the widened box
  also contains independent changes that cannot occur.

**History tracking (`history`).** Keep the input as one affine set over all symbols so far,
$\delta = c + G e$. A child keeps the region of the earlier symbols where $a$ is reachable (a box in symbol space)
and appends **new symbols** for the step's change: $G \leftarrow [G,\ G_{step}]$. The one-step safety check of
every node is certified over that same set with the next step's symbols appended, so there is no re-boxing anywhere.

- The correlation between steps is kept. The only over-approximation is the leaf boxes of earlier symbols.
- The dimension grows by $r$ per step: 3, then 3 + r, then 3 + 2r, and so on.

**Which is better depends on the change model.**

- With **independent** changes, the set of loads at the next step is exactly a box: a box plus a box is a box.
  Re-boxing then loses nothing, and history tracking only pays for the extra dimensions. Its splitting budget
  (`--max-depth`) is also spread over more symbols, so with the same depth its leaves are coarser and it can report
  extra (spurious) sequences. That is sound, but looser.
- With **common-mode** or other correlated changes, the next-step set is a thin slab inside the box. History tracking
  gives tighter bounds and can need fewer boxes.

For example, with $w = 0.02$ and $m = 0.03$ at the reset point:

| step-change model | | re-boxing | history tracking |
|---|---|---|---|
| independent | time; boxes; reachable sequences | 39 s; 14; 2 | 66 s; 24; 3 (one spurious) |
| common | time; boxes; reachable sequences | 39 s; 14; 2 | 14 s; 7; 2 |
| common | certified slack after `Pg2-40MW → Pg3-10MW` | 108.6–144.5 MW | 115.4–144.5 MW |

`--track both` runs the two modes and prints a side-by-side comparison of time, tree size, symbols, reachable
sequences and the certified last-step bounds of every sequence:

```bash
python examples/wscc9_actor_reachability.py --study two-step --track both --step-change common
python examples/wscc9_actor_reachability.py --study two-step --track both --step-change common --steps 3 --max-depth 9
```

> - [`MultiStepVerifier(..., track="rebox" | "history", extend=...)`](../certflow/policy.py#L239) with [`MultiStepVerifier.verify`](../certflow/policy.py#L335) and [`_node`](../certflow/policy.py#L291). [`StepNode.input_box()`](../certflow/policy.py#L203) gives the load box of a node; [`StepNode.stats()`](../certflow/policy.py#L231) gives nodes, boxes, symbols and time.
> - Callbacks: [`multistep_problem`](../examples/wscc9_env.py#L394) (re-boxing) and [`multistep_problem_history`](../examples/wscc9_env.py#L441) (history). The latter uses [`make_problem_history`](../examples/wscc9_env.py#L418) and [`StepChecker.check_set`](../examples/wscc9_env.py#L356), which certifies over any affine load set.
> - [`step_generator`](../examples/wscc9_env.py#L407) builds $G_{step}$.

**Soundness:** in both modes the tree contains every possible trajectory, and a Monte-Carlo run of the deployed agent
along random trajectories checks this. Longer horizons (`--steps k`) work the same way; the cost grows with the number
of reachable action sequences and, for history tracking, with the number of symbols.

## Settings used in the example

| | `--quick` | full |
|---|---|---|
| load box half-width $w$ | 0.02 | 0.05 |
| per-step load change $m$ | 0 | 0.06 |
| maximum bisection depth | 5 | 7 |
| Monte-Carlo samples | 100 | 300 |

List the operating points with `python examples/wscc9_actor_reachability.py --list-ops`.

## Reading the output

```
=== reset (delta = 0): lines (1, 1, 1, 1, 1, 1) ... delta = [0. 0. 0.] +- 0.02
  pre-action PF certified (rho 0.019); observation ranges: ...           <- Steps 2-3 on the whole box
  partition: 1 boxes (complete=True); reachable actions: ['Pg2-40MW']    <- Steps 1, 5, 7
  one-step safety ...: SAFE 100.0% ...                                   <- Step 6, share of the box
  N-1 of the agent's actions: ... insecure on 100.0% of the box          <- Step 6, N-1
     after Pg2-40MW:                                                      <- per reachable action
         N-0: OK  max loading 41.2%, |V| 1.002..1.040 pu, slack 106.2..117.6 MW
         N-1: INSECURE  violated for every load: overloads 0, PF failures 0, voltage violations 1, slack violations 0
              outage 5-4: |V| bus 5 0.899..0.906 pu (always out of limits)       <- which outage violates what
  no valid action is certified N-0 safe AND N-1 secure over the whole box
  Monte-Carlo (...): ... 100/100                                         <- Step 8
```

This example shows the kind of finding the audit produces. The agent's action is safe in the intact grid, but losing
line 5-4 leaves bus 5 at about 0.90 pu for every load in the box. That is a property of the grid at this operating
point, and no single allowed action removes it. Reference outputs for all operating points are in
`examples/wscc9_case_studies_output.txt`.

## Auditing another agent or grid

Provide the three callbacks of `DiscretePolicyVerifier` for your problem:

- `observe(lo, hi)` returns the observation as a Taylor model over the box. Build it with `enclose_pf` and
  `certflow.power.quantities` if it depends on the grid state.
- `mask(lo, hi, obs)` returns the allowed actions.
- `check_action(a, lo, hi, obs)` returns whether action `a` is safe. Build it with `enclose_pf_split` and
  `n1_analysis` against your limits.

The network enters through any loader in `certflow.nn` (`from_torch`, `from_state_dict`, `sb3_actor`). Only the
observation layout, the action effects and the mask are specific to an environment.
