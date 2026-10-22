# The verification method

This document explains how certflow proves that a learned dispatch controller keeps the AC grid within its limits,
and where each step lives in the code. The method follows *Verified Safety Layers for AC Surrogates:
Reachability-Based Local Verification and Safety Restoration* (Giraud, Vorwerk, Chatzivasileiadis). Each step ends
with an **In the code** box naming the functions that perform it.

The RL modes reuse Steps 1, 3, 4, 5, 6 and 7 with a different input and output; see
[rl_reachability.md](rl_reachability.md) and [measurement_noise.md](measurement_noise.md).

## 0. The problem

A neural network $f_\theta$ maps the loads $x = [p_d; q_d]$ to generator setpoints $y = [p_g; V_g]$: active power
and voltage magnitude. The grid state that results is not an output of the network. It is the solution of the AC
power flow (PF) with those setpoints. The goal is to prove, for **every** load $x$ in a set $\mathcal X$:

1. the power flow has a solution, and it is **unique** within a known box;
2. every operational quantity of that solution stays within its limits: bus voltage magnitudes, generator active
   and reactive power, slack active and reactive power, branch currents and angle differences.

Standard bound-propagation tools for neural networks cannot do this, because the state comes from an iterative
solver and not from a closed-form computation. certflow replaces the solver by a **validated existence and
uniqueness proof** (Krawczyk) over Taylor models. The models share their symbols with the network's reachable set,
so correlations between inputs, setpoints and state are kept throughout:

```
 load set X ──network reachability──▶ setpoints y(ε) ──▶ specified values S(ε) ──▶ Krawczyk proof: unique PF state?
     ▲                                                                                   │ no             │ yes
     └──────────────────────── split the input set ◀──────────────────────────────────────┘                ▼
                                                                        bounds on |V|, Pg, Qg, slack, |I|, Δθ
                                                                                         │ outside a limit
                                                                                         ▼
                                                       restoration: constant offset d on the setpoints, re-certify
```

`PFVerifier.certify` runs this pipeline over an input set (Steps 1–7), and `Restorer` closes the loop (Step 8).

---

## 1. Validated arithmetic: Taylor models

Every uncertain quantity is a degree-2 Taylor model (TM) over shared symbols $\varepsilon\in[-1,1]^K$:

$$x = c_0 + c^\top \varepsilon + \varepsilon^\top q\, \varepsilon + [-R, R].$$

The polynomial part is exact, and everything of degree three and higher is bounded into the remainder $R$. The
symbols are shared between all quantities, so correlations survive. For example, the network's $p_g$ and the power
flow's voltages depend on the same $\varepsilon$, so their difference does not count the uncertainty twice. Affine
zonotopes, $c_0 + c^\top\varepsilon$, are the special case $q = 0$, $R = 0$.

Every floating-point operation adds its own rounding error to $R$ (forward error analysis with outward rounding). The
enclosures therefore hold for the exact real-valued quantities, not just for the floating-point computation.

> **In the code**
> - [`TM`](../certflow/tm.py#L22) is the `(c0, C, Q, R)` tuple. Operations:
>   - [`add`, `sub`](../certflow/tm.py#L181);
>   - [`mul`](../certflow/tm.py#L238) (exact quadratic part, degree ≥ 3 into `R`);
>   - [`matmul`](../certflow/tm.py#L213);
>   - [`relu`](../certflow/tm.py#L270), [`clip`](../certflow/tm.py#L303);
>   - [`compose`](../certflow/tm.py#L321) (any smooth univariate function), [`sqrt`](../certflow/tm.py#L350), [`maximum`](../certflow/tm.py#L378).
> - [`certflow/fp.py`](../certflow/fp.py) holds the rounding-error bounds (`gam`, `upr`, `odn`, `oup`). Rigorous mode is on by default.

## 2. The input set: a low-rank load zonotope

Grid loads are strongly correlated, so the set of loads to verify is a low-rank zonotope around a nominal operating
point $x_0$:

$$\mathcal R(r_f) = \{\, x_0 \odot (1 + r_f\, G\varepsilon) \;:\; \varepsilon\in[-1,1]^K \,\},$$

where $G$ holds $K$ normalised load-variation modes and $r_f$ is the radius. Verification always runs over
**sub-boxes** $[lo, hi] \subseteq [-1,1]^K$ of $\varepsilon$. A sub-box with centre $e_c$ and half-widths $h$ is
again a zonotope, $x_0 + G_x e_c + (G_x\,\mathrm{diag}\,h)\,\varepsilon'$.

> **In the code**
> - [`low_rank_load_set(x0, Gdirs, frac)`](../certflow/sets.py#L67) builds the set as an [`AffineSet`](../certflow/sets.py#L19)`(c, G)`.
> - [`AffineSet.sub_tm(ec, h)`](../certflow/sets.py#L54) returns a sub-box as a `TM`, including the rounding of its centre and generators, so the sub-boxes provably cover the set.
> - [`box_set`](../certflow/sets.py#L81) gives an axis-aligned box. Any $G$ can be used; how the modes are chosen is up to the user.

## 3. Network reachability (DeepZ)

The input zonotope is pushed through the network layer by layer:

- **Affine layers** act exactly: $Wz + b$ on the centre and the generators.
- **ReLU:** a neuron that is never or always active is exact. A neuron whose input range $[l, u]$ contains 0 is
  over-approximated as $y = \lambda\hat x + \mu + \mu\,\varepsilon_{\text{new}}$, with $\lambda = u/(u-l)$ and
  $\mu = -ul/(2(u-l))$. This is the minimal-area parallelogram, and it **appends one new symbol per relaxed neuron**.
  The output therefore carries the $K$ input symbols plus one relaxation symbol per relaxed neuron. Relaxation errors
  keep their own symbol and can partially cancel later.
- **tanh / sigmoid:** the same construction with slope $\lambda = \min(f'(l), f'(u))$.

The deployed network computes in floating point, e.g. float32. Its own rounding is bounded separately, so the output
enclosure covers what the deployed network actually returns.

> **In the code**
> - [`Linear.reach`](../certflow/nn/layers.py#L119), [`ReLU.reach`](../certflow/nn/layers.py#L151), [`_SShaped.reach`](../certflow/nn/layers.py#L194) (tanh, sigmoid), chained by [`Sequential.reach`](../certflow/nn/network.py#L67). Deployed-arithmetic bound: `ZonoState.dep` and [`exit_identity`](../certflow/nn/layers.py#L81).
> - Loading a trained network: [`from_torch`](../certflow/nn/network.py#L126), [`from_state_dict`](../certflow/nn/network.py#L143), [`sb3_actor`](../certflow/nn/network.py#L163). Any object implementing `reach(TM) -> TM` can be audited ([`NNReachable`](../certflow/nn/network.py#L23)).
> - Dispatch controllers: [`DispatchController.reach`](../certflow/power/controller.py#L41) maps the output to `[Pg; Vg]`, adds a restoration offset if one is given (Step 8), and **clips $V_g$** to its deployable band using two correlated ReLUs ([`tm.clip`](../certflow/tm.py#L303)), exactly as the deployed controller does.

## 4. The parameterised power flow

The state is written in rectangular coordinates, $z = (v_{r,ns}, v_{i,ns})$: real and imaginary voltages of the
non-slack buses. The residual stacks three things:

- active-power mismatch at all non-slack buses;
- reactive-power mismatch at PQ buses;
- $|V|^2 - V_g^2$ at PV buses.

$$h(z, \varepsilon) = h_0(z) - S(\varepsilon) = 0.$$

In rectangular coordinates all three parts of $h_0$ are **quadratic** in $z$, which has two consequences. Degree-2
Taylor models represent them with no truncation error. And the Jacobian $J = \partial h/\partial z$ is **affine** in
$z$, so its derivatives $H_j = \partial J/\partial z_j$ are constant matrices. There are no trigonometric functions
and no angle-wrapping issues.

The uncertainty enters through the specified values $S(\varepsilon)$, all of them Taylor models over the shared
symbols:

- net active injection $p_g(\varepsilon) - p_d(\varepsilon)$ per bus;
- $-q_d(\varepsilon)$ per bus;
- squared voltage setpoints $V_g(\varepsilon)^2$ at PV buses.

> **In the code**
> - [`PFVerifier.enclose`](../certflow/power/verifier.py#L175) builds $S(\varepsilon)$ from the controller's reachable set and the load set (`Pbus`, `Qbus`, `Vset`). [`enclose_pf`](../certflow/power/pfcert.py#L43) stacks `S = [P_ns; Q_pq; Vset²]`.
> - [`RectPFSystem`](../certflow/power/pf.py#L340) is $h_0$:
>   - [`residual`](../certflow/power/pf.py#L420);
>   - [`jacobian`](../certflow/power/pf.py#L426) (point Jacobian via [`jac_rect`](../certflow/power/pf.py#L90));
>   - [`jacobian_tm`](../certflow/power/pf.py#L437) (the Jacobian as a TM: centre $J_0$, linear coefficients, quadratic coefficients);
>   - [`hessian_structure`](../certflow/power/pf.py#L478) (the constant $H_j$, built once by [`h_structure`](../certflow/power/pf.py#L211)).
> - The solver only needs the [`AffineJacobianSystem`](../certflow/krawczyk.py#L28) interface, so any implicit model with an affine Jacobian can be plugged in.

## 5. Existence and uniqueness: the Krawczyk test

The aim is a box around a centre in which, for every $\varepsilon$, exactly one solution of $h(z,\varepsilon)=0$
exists.

1. **Linearisation point:** solve the point power flow at the centre of the input box (Newton), giving $z_0$.
2. **Preconditioner:** $M = J(z_0)^{-1}$. Any $M$ is valid; an inexact inverse only shows up as a larger
   $|I - MJ_0|$.
3. **Centre function:** one Newton step taken *symbolically* over the inputs,
   $x_1(\varepsilon) = z_0 - M\,(h_0(z_0) - S(\varepsilon))$. This centre moves with the inputs, so the residual
   that remains, $corr \ge \sup_\varepsilon |M(h_0(x_1(\varepsilon)) - S(\varepsilon))|$, is small.
4. **Slope bound:** over the candidate box $x_1(\varepsilon) \pm r$, bound $|I - MJ|$ entrywise by
   $$K_b(r) = |I - MJ_0| + \sum_k |MJ^{(k)}_{lin}| + \sum_{a,c}|MJ^{(a,c)}_Q| + \sum_j |MH_j|\, r_j.$$
   - The middle terms are the Jacobian's own dependence on the inputs (linear and quadratic).
   - The last term is the growth of the Jacobian over the box. It is kept **correlated**: $M$ is applied before
     taking absolute values, so sign cancellations are not lost.
5. **Inflation loop:** start at $r = 0$ and iterate $r_{new} = (I - K_b(r))^{-1}\, corr$, keeping $r$ monotone.
   Stop with failure as soon as the spectral radius $\rho(K_b) \ge 1$. Accept when $r_{new} \le r$.

When the test succeeds, $\rho(K_b) < 1$ and $r \ge (I-K_b)^{-1} corr$. The Krawczyk operator then maps the box into
itself and is a contraction there. So **for every input in the box, the power flow has exactly one solution in
$x_1(\varepsilon) \pm r$**.

> **In the code**: [`krawczyk(system, S, z0)`](../certflow/krawczyk.py#L122)
> - Step 1: [`centre_newton`](../certflow/power/pfcert.py#L37), called from `enclose_pf`.
> - Steps 2–3: the start of `krawczyk` (`M`, `x1`, `corr`).
> - Step 4: [`_Kb(r)`](../certflow/krawczyk.py#L166). `c0z` = $|I - MJ_0|$, `base_lin`, `base_quad`, and the correlated term `rem2`, which uses $|MH_j|$ precomputed per column via [`structure_from_triplets`](../certflow/krawczyk.py#L70).
> - Step 5: the loop at the end of `krawczyk`. [`specrad`](../certflow/krawczyk.py#L86) returns a rigorous *upper bound* on $\rho$ (Collatz–Wielandt).
> - Rounding: [`_verify`](../certflow/krawczyk.py#L176) re-checks the accepted radius *a posteriori* with every rounding of $K_b$ and of the Jacobian bounded.
> - Result: [`KrawczykResult.enclosure()`](../certflow/krawczyk.py#L115) gives $x_1(\varepsilon) \pm r^*$ as a TM. `enclose_pf` converts it to full-bus voltages `Vr, Vi` ([`RectPFSystem.state`](../certflow/power/pf.py#L406)) and injections `P, Q` ([`RectPFSystem.injections`](../certflow/power/pf.py#L392)).
> - Inputs that also enter the Jacobian, such as uncertain shunt admittances, are supported: the slope term then has several parts with bounded scales ([`AffineJacobianSystem.hessian_structure`](../certflow/krawczyk.py#L50)). The RL modes use this.

## 6. Splitting the input set

If the test fails ($\rho \ge 1$), the box is bisected and each half is certified on its own, with its own
linearisation point. The bounds for the whole set are the union of the bounds of the leaves, which is sound by set
inclusion.

The direction to bisect is not the widest one. It is the one that contributes most to the residual of the Krawczyk
step: the per-symbol size of the linear and quadratic coefficients of $M(h_0(x_1) - S)$. That residual drives the
spectral radius, and the score comes for free from a quantity already computed.

> **In the code**
> - Score: the `score` field of [`KrawczykResult`](../certflow/krawczyk.py#L106).
> - Direction choice: [`PFVerifier._choose_split`](../certflow/power/verifier.py#L227) → [`split_dims`](../certflow/split.py#L16). This bisects every direction whose score is within `tau = 0.5` of the maximum, up to `m = 4` directions at once, via [`split_children`](../certflow/split.py#L35).
> - Driver: [`adaptive_split`](../certflow/split.py#L104), serial or parallel over an [`ObjectPool`](../certflow/split.py#L79). When the leaf cap `max_leaves` is reached, the result is marked *not formal* (abstain) rather than guessed.
> - Union: [`PFVerifier.certify`](../certflow/power/verifier.py#L232).

## 7. Bounding the operational quantities

The certified state enclosure is pushed through the grid equations. All quantities stay Taylor models over the input
symbols until the final bound is taken.

| Quantity | How it is bounded | In the code |
|---|---|---|
| $\lvert V\rvert$ at PQ buses | the correlated TM of $v_r^2 + v_i^2$ is bounded, then the monotone square root is taken; PV buses use the (clipped) setpoint range | [`VoltageMagnitude.evaluate`](../certflow/power/constraints.py#L142) |
| voltage angles | `atan2` over the four corners of each bus's $(v_r, v_i)$ box; differences between branch ends | [`bus_angle_bands`](../certflow/power/constraints.py#L424), [`AngleDifference`](../certflow/power/constraints.py#L439) |
| $p, q$ injections | the quadratic forms $v^\top H v$ evaluated on the state TM | [`inj_rect_tm`](../certflow/power/pf.py#L139) |
| generator $p_g$ | the controller's reachable output | [`GenActivePower.evaluate`](../certflow/power/constraints.py#L198) |
| generator $q_g$, slack $p/q$ | injection plus bus load, added as **correlated** TMs | [`GenReactivePower`](../certflow/power/constraints.py#L218), [`SlackPower`](../certflow/power/constraints.py#L247) |
| branch current | the branch current $I = Y_\ell v$ is an affine complex zonotope plus a small remainder. Its largest magnitude is bounded by the support function over 512 directions, with a Lipschitz pad for the directions in between. The thermal limit is checked in squared form. | [`zono_imax`](../certflow/power/constraints.py#L292), [`BranchCurrent._end`](../certflow/power/constraints.py#L342) (the smaller of this and the plain quadratic bound) |

A set is **certified safe** when the power flow is proven on every leaf and every bound lies inside its limit
([`Certificate.safe`](../certflow/power/verifier.py#L94)). The default limits are
[`default_constraints`](../certflow/power/constraints.py#L476). A new limit family is a subclass of
[`Constraint`](../certflow/power/constraints.py#L63).

> **Independent check:** [`monte_carlo(ver, cert)`](../certflow/power/validate.py#L39) samples the set, runs the deployed controller and a separate polar Newton power flow, and checks every sample against the certified bounds.

## 8. Safety restoration

If some bound leaves its limit, the controller output gets a **constant offset**
$d = (\Delta p_g, \Delta V_g)$:

$$\tilde f_\theta(x) = f_\theta(x) + d.$$

A constant shifts each bound without widening it, so the same pipeline (Steps 3–7) re-certifies the corrected
controller.

1. **Sensitivities:** one factorisation of the reduced power-flow Jacobian at the box centre gives
   $\partial z/\partial f_\theta = -J^{-1}\partial r/\partial f_\theta$ (implicit function theorem). From it, every
   limit family builds linear rows that say how its bounds move with $d$.
2. **Quadratic program:** find the smallest offset, with voltage changes preferred over active-power redispatch,
   that moves every bound inside its limit. Rows are scaled by each limit's window and columns by each actuator's
   range. The step stays within a trust region. When no offset satisfies all rows, penalised slack variables give
   the best compromise.
3. **Iterate (restore one box):** certify, measure the largest limit exceedance, keep the best offset seen, and stop
   in one of three cases:
   - the box is safe;
   - progress stalls;
   - some bound is wider than its whole limit window, so no constant offset can fit and the box must be split.

   Otherwise take a QP step and adapt the trust region from how well the predicted improvement matched the actual
   one.
4. **Split (piecewise offset):** try the minimum-norm starting offset first, then optional reference starting
   points such as an OPF solution. If no single offset makes the whole box safe, bisect it along the input direction
   that maximises the aggregated violation × sensitivity and restore each half. Each half starts from its parent's
   best offset and never ends up worse than it. Recurse up to a maximum depth.
5. **Deployment:** the result is a piecewise-constant correction. At run time, an operating point gets the offset of
   the sub-box that contains it.

> **In the code**
> - Step 1: [`linearize(grid, Vm, th)`](../certflow/power/sensitivity.py#L29). Each limit family's [`restoration_rows`](../certflow/power/constraints.py#L77) returns `RowBlock`s (sensitivity rows, right-hand side, limit window).
> - Step 2: [`Restorer._step`](../certflow/power/restore.py#L160) assembles the rows and solves the exact minimum-norm problem ([`qp_minnorm`](../certflow/power/restore.py#L41)), or the slack-penalised trust-region problem ([`qp_besteffort`](../certflow/power/restore.py#L72)). `RestoreOptions.eps_vg` weights voltage against active-power changes.
> - Step 3: [`Restorer.restore_box`](../certflow/power/restore.py#L194).
> - Step 4: [`Restorer.restore_split`](../certflow/power/restore.py#L280) and [`split_choice`](../certflow/power/restore.py#L263). Reference starting points: `Restorer(ver, seeds={"opf": fn})`.
> - Step 5: [`PiecewiseOffsetController`](../certflow/power/restore.py#L361).

## 9. Assumptions

- **Arithmetic:** IEEE double precision with round-to-nearest. The library functions `atan2`, `tanh`, `cosh` and
  `exp` are accurate to a few ulp. The tanh and sigmoid relaxations shrink their slope by $(1 - 64u)$ and pad
  their endpoints by 8 ulp to cover this.
- **Deployed network:** it computes in the precision given by `deploy_u` (float32 for stable-baselines3 policies,
  float64 by default).
- **Scope:** the guarantees cover the steady-state AC power flow. They do not cover dynamics or stability.
