# The method: verified safety layers for AC surrogates

This document explains the method of Giraud, Vorwerk & Chatzivasileiadis, *Verified Safety Layers for AC
Surrogates: Reachability-Based Local Verification and Safety Restoration*. Each step is followed by an
**In the code** box naming the function that performs it. Section and equation numbers refer to the paper.

## 0. The problem

An NN surrogate $f_\theta$ maps loads $x = [p_d; q_d]$ to generator setpoints $y = [p_g; V_g]$ (Sec. II-B). The
full grid state is then obtained by solving the AC power flow (PF) with those setpoints. The goal (eq. 10) is to
prove, for **every** load in a set $\mathcal X$:

1. the PF has a solution, and it is **unique** (in a known box);
2. every operational quantity of that solution stays within limits: $|V|$, $p_g$, $q_g$, slack $p/q$, branch
   current, angle difference.

Standard bound propagation cannot do this because the state is the output of an iterative solver. The method
replaces the solver by a validated **Krawczyk** proof over Taylor models that share their symbols with the NN's
reachable set (Fig. 1):

```
 load zonotope X ──DeepZ──▶ setpoints y(ε) ──S(ε)──▶ Krawczyk: K(Z) ⊆ Z ? ──▶ state enclosure ──▶ bounds ──▶ in limits?
        ▲                                                  │ no (ρ ≥ 1)                                   │ no
        └────────────────── bisect the input box ◀─────────┘                          restoration: offset d, re-certify
```

In the code, `PFVerifier.certify` runs this pipeline over a box (Steps 1–7), and `Restorer` closes the loop
(Step 8).

---

## 1. Validated arithmetic: Taylor models (Sec. III-B)

Every uncertain quantity is a degree-2 Taylor model over shared symbols $\varepsilon\in[-1,1]^K$ (eq. 12):

$$x = c_0 + c^\top \varepsilon + \varepsilon^\top q\, \varepsilon + [-R, R].$$

The symbols are shared between all quantities, so correlations survive. For example, $p_g$ from the NN and the
voltages from the PF depend on the same $\varepsilon$, and their difference does not double-count the width.

> **In the code**
> - [`TM`](../certflow/tm.py#L22) is the `(c0, C, Q, R)` tuple. Arithmetic: [`add`/`sub`](../certflow/tm.py#L181), [`mul`](../certflow/tm.py#L238) (exact quadratic part, degree ≥ 3 into `R`), [`matmul`](../certflow/tm.py#L213), [`relu`](../certflow/tm.py#L270), [`clip`](../certflow/tm.py#L303), [`compose`](../certflow/tm.py#L321) (any smooth univariate function), [`sqrt`](../certflow/tm.py#L350), [`maximum`](../certflow/tm.py#L378).
> - **Beyond the paper:** [`certflow/fp.py`](../certflow/fp.py) bounds every floating-point rounding (`gam`, `upr`, `odn`, `oup`) into `R`. The enclosures are therefore sound for the exact-real quantities, not just for the float computation ("rigorous mode", on by default).

## 2. Local input set: the low-rank load zonotope (Sec. III-C, eqs. 13–19)

The set is $\mathcal R(r_f) = \{x_0 \odot (1 + r_f\, G\varepsilon) : \varepsilon\in[-1,1]^K\}$. Here $x_0$ is the
nominal operating load and $G$ holds $K$ RMS-normalised load modes. Verification always runs over **sub-boxes**
$[lo, hi] \subseteq [-1,1]^K$ of $\varepsilon$-space. A sub-box with centre $e_c$ and half-widths $h$ is again a
zonotope, $x_0 + G_x e_c + (G_x\,\mathrm{diag}\,h)\,\varepsilon'$.

> **In the code**
> - [`low_rank_load_set(x0, Gdirs, frac)`](../certflow/sets.py#L67) implements eq. 19 and returns an [`AffineSet`](../certflow/sets.py#L19)`(c, G)`.
> - [`AffineSet.sub_tm(ec, h)`](../certflow/sets.py#L54) gives the sub-box as a `TM` (plus the rounding of the centre/generator computation).
> - The mode matrix itself (eqs. 14–18: geometric spectrum, RMS normalisation) is built by the experiment script and passed in as `Gdirs`. The package takes any $G$.

## 3. NN reachability with DeepZ (Sec. III-D, eqs. 20–22)

The input zonotope is pushed through the network layer by layer:

- **Affine layers:** exact, $Wz + b$ on centre and generators.
- **ReLU:** a neuron that is never active or always active is exact. A crossing neuron ($l<0<u$) is relaxed as
  $y = \lambda\hat x + \mu + \mu\,\varepsilon_{\text{new}}$ with $\lambda = u/(u-l)$ and $\mu = -ul/(2(u-l))$.
  This **appends one new generator per crossing neuron** (eq. 21), so the output (eq. 22) carries $K$ input
  symbols plus $M$ relaxation symbols.
- **tanh / sigmoid:** the same idea with slope $\lambda = \min(f'(l), f'(u))$.

> **In the code**
> - [`Linear.reach`](../certflow/nn/layers.py#L119), [`ReLU.reach`](../certflow/nn/layers.py#L151) (eq. 20–21), [`_SShaped.reach`](../certflow/nn/layers.py#L194) (tanh/sigmoid), chained by [`Sequential.reach`](../certflow/nn/network.py#L67).
> - Translators: [`from_torch`](../certflow/nn/network.py#L126), [`from_state_dict`](../certflow/nn/network.py#L143), [`sb3_actor`](../certflow/nn/network.py#L163).
> - **Beyond the paper:** `ZonoState.dep` bounds the *deployed* network's own rounding (e.g. float32 in torch), so the enclosure covers what the deployed float network outputs ([`exit_identity`](../certflow/nn/layers.py#L81)).
> - The dispatch-specific wrapper [`DispatchController.reach`](../certflow/power/controller.py#L41) maps the NN output to `[Pg; Vg]`, adds the restoration offset $d$ (Step 8), and **clips $V_g$** to the deployable band with two correlated ReLUs ([`tm.clip`](../certflow/tm.py#L303)), as in Sec. II-B.

## 4. The parameterised power flow $h(z,\varepsilon) = h_0(z) - S(\varepsilon)$ (Sec. II-B eq. 9, Sec. III-E eqs. 23–24)

The state is rectangular, $z = (v_{r,ns}, v_{i,ns})$. The residual stacks $P$ at non-slack buses, $Q$ at PQ buses
and $|V|^2$ at PV buses. In rectangular coordinates all three are **quadratic** in $z$, so:

- degree-2 TMs represent them with zero truncation error;
- the Jacobian is **affine** in $z$ (eq. 24), so its state-derivatives $H_j = \partial J/\partial z_j$ are constant.

The uncertainty enters only through the specified values $S(\varepsilon)$ (eq. 23). These are $p_g(\varepsilon) -
p_d(\varepsilon)$ and $-q_d(\varepsilon)$ per bus, and $V_g(\varepsilon)^{\odot 2}$ at PV buses, all TMs over the
shared symbols.

> **In the code**
> - [`PFVerifier.enclose`](../certflow/power/verifier.py#L175) builds $S(\varepsilon)$: `Pbus = Cg·Pg − Cl·Pd`, `Qbus = −Cl·Qd`, `Vset = Cgv·Vg` (lines 194–199). It then calls [`enclose_pf`](../certflow/power/pfcert.py#L43), which stacks `S = [P_ns; Q_pq; Vset²]`.
> - [`RectPFSystem`](../certflow/power/pf.py#L340) is the system $h_0$:
>   - [`residual`](../certflow/power/pf.py#L420) (eq. 9);
>   - [`jacobian`](../certflow/power/pf.py#L426) (point Jacobian, eq. 24, via [`jac_rect`](../certflow/power/pf.py#L90));
>   - [`jacobian_tm`](../certflow/power/pf.py#L437) (the Jacobian as a TM: $J_0$, linear generators, quadratic coefficients; eq. 26);
>   - [`hessian_structure`](../certflow/power/pf.py#L478) (the constant $H_j$, built once by [`h_structure`](../certflow/power/pf.py#L211)).
> - The interface the Krawczyk solver needs is [`AffineJacobianSystem`](../certflow/krawczyk.py#L28). Any implicit model with an affine Jacobian can be plugged in.

## 5. Krawczyk existence and uniqueness (Sec. III-F/G, eqs. 25–29, Algorithm 1)

1. **Linearisation point:** the point PF at the centre of the box, $z_0$ (polar Newton).
2. **Preconditioner:** $M = J(z_0)^{-1}$.
3. **Warm step:** $z_1 = z_0 - M h(z_0)$ (paper). **In the code this is a function of $\varepsilon$:**
   $x_1(\varepsilon) = z_0 - M\,(h_0(z_0) - S(\varepsilon))$ is a TM. Its centre moves with the inputs, which makes
   the residual radius $corr \ge \sup_\varepsilon |M(h_0(x_1(\varepsilon)) - S(\varepsilon))|$ much smaller than
   with a point centre.
4. **Slope majorant (eq. 27):** $K_b(r) = |I - MJ_0| + \sum_k |MJ^{(k)}_{lin}| + \sum_{a,c}|MJ^{(a,c)}_Q| +
   \mathrm{rem}(r)$, with the **correlated remainder** $\mathrm{rem}(r) = \sum_j |MH_j|\, r_j$ (eq. 28). $M$ is
   applied before taking absolute values, so sign cancellations survive.
5. **Monotone inflation (Algorithm 1):** start at $r = 0$ and iterate $r_{new} = (I-K_b(r))^{-1} corr$. Fail as
   soon as $\rho(K_b) \ge 1$. Accept when $r_{new} \le r$. Then (eq. 29) for every $\varepsilon$ there is exactly one
   PF solution in $x_1(\varepsilon) \pm r^*$.

> **In the code**: [`krawczyk(system, S, z0)`](../certflow/krawczyk.py#L122)
> - Centre Newton: [`centre_newton`](../certflow/power/pfcert.py#L37), called from `enclose_pf`.
> - Steps 2–3 (`M`, `x1`, `corr`): [lines 134–141](../certflow/krawczyk.py#L134).
> - Step 4: [`_Kb(r)`](../certflow/krawczyk.py#L166).
>   - `c0z = |I − M J0|`, `base_lin` = $\sum_k|MJ_{lin}^{(k)}|$, `base_quad` = $\sum|MJ_Q|$.
>   - `rem2` = $\sum_j |MH_j| r_j$, precomputed per populated column via [`structure_from_triplets`](../certflow/krawczyk.py#L70).
> - Step 5: [the loop at lines 220–245](../certflow/krawczyk.py#L220). [`specrad`](../certflow/krawczyk.py#L86) is a *rigorous upper bound* on $\rho$ (Collatz–Wielandt), not just an estimate. The Neumann series computes $(I-K_b)^{-1}corr$.
> - Result: [`KrawczykResult.enclosure()`](../certflow/krawczyk.py#L115) $= x_1(\varepsilon) \pm r^*$ as a TM. `enclose_pf` turns it into full-bus `Vr, Vi` TMs ([`RectPFSystem.state`](../certflow/power/pf.py#L406)) and injections `P, Q` ([`RectPFSystem.injections`](../certflow/power/pf.py#L392)).
> - **Beyond the paper:**
>   - [`_verify`](../certflow/krawczyk.py#L176) re-checks the accepted float radius *a posteriori* with every rounding of $K_b$ and of the Jacobian evaluation bounded ($\varepsilon$-inflation).
>   - The Jacobian's state-slope may itself depend on $\varepsilon$ (several Hessian terms with bounded scales, see [`AffineJacobianSystem.hessian_structure`](../certflow/krawczyk.py#L50)). The RL mode needs this for uncertain admittances; see [rl_reachability.md](rl_reachability.md).

## 6. Adaptive domain splitting (Sec. III-H)

When $\rho \ge 1$, the box is bisected and each half is re-certified with its own linearisation. The direction is
not the geometrically widest one. It is the one with the largest **residual-quadratic mass**: the per-symbol size of
$M(h_0(x_1) - S)$, read for free from the residual TM. The certified bounds of the whole box are the union over the
leaves.

> **In the code**
> - Score: [`krawczyk` lines 142–144](../certflow/krawczyk.py#L142) (`score` = $\sum$ |linear + quadratic coefficients| of the residual per symbol).
> - Choice: [`PFVerifier._choose_split`](../certflow/power/verifier.py#L227) → [`split_dims`](../certflow/split.py#L16). This bisects every direction whose score is within `tau = 0.5` of the maximum, up to `m = 4` at once (a refinement of the paper's single-direction bisection), via [`split_children`](../certflow/split.py#L35).
> - Driver: [`adaptive_split`](../certflow/split.py#L104), serial or parallel over an [`ObjectPool`](../certflow/split.py#L79). The leaf cap `max_leaves` → *abstain* (`Certificate.formal = False`).
> - Union: [`PFVerifier.certify`](../certflow/power/verifier.py#L232), `Band.union` over the leaves (line 243).

## 7. Recovering verified bounds (Sec. III-I, eqs. 30–37)

| Quantity | Paper | In the code |
|---|---|---|
| $\lvert V\rvert$ at PQ buses | eq. 30: correlated $\sqrt{v_r^2+v_i^2}$ | [`VoltageMagnitude.evaluate`](../certflow/power/constraints.py#L142): bounds the correlated TM of $\lvert V\rvert^2$, then takes the monotone square root. PV buses use the clipped setpoint range. |
| angles | eq. 31: atan2 on the four corners | [`bus_angle_bands`](../certflow/power/constraints.py#L424); differences in [`AngleDifference`](../certflow/power/constraints.py#L439) |
| $p, q$ injections | eq. 32: $v^\top H v$ | [`inj_rect_tm`](../certflow/power/pf.py#L139) via `RectPFSystem.injections` |
| $p_g$ | eq. 33: the NN output itself | [`GenActivePower.evaluate`](../certflow/power/constraints.py#L198) |
| $q_g$, slack $p/q$ | eq. 33: injection + load **interval** | [`GenReactivePower`](../certflow/power/constraints.py#L218), [`SlackPower`](../certflow/power/constraints.py#L247): injection + load as **correlated TMs** (tighter than adding intervals) |
| branch current | eqs. 34–37: complex-zonotope support function over $\theta$, plus the remainder $\eta_\ell$ | [`zono_imax`](../certflow/power/constraints.py#L292) (512 directions + Lipschitz padding). [`BranchCurrent._end`](../certflow/power/constraints.py#L342) takes the smaller of this and the plain quadratic bound of $\lvert I\rvert^2$. |

A box is **certified safe** when the PF is proven on every leaf and every band is inside its limits:
[`Certificate.safe`](../certflow/power/verifier.py#L94). The default constraint set (eqs. 4–5) is
[`default_constraints`](../certflow/power/constraints.py#L476). New limit families subclass
[`Constraint`](../certflow/power/constraints.py#L63).

> **Monte-Carlo check:** [`monte_carlo(ver, cert)`](../certflow/power/validate.py#L39) samples the box, runs the deployed controller and an independent polar Newton PF, and checks every sample against the certified bands.

## 8. Verified safety restoration (Sec. IV, eqs. 38–39, Algorithms 2–3)

If a band leaves its limits, the dispatch gets a **constant offset** $d = (\Delta p_g, \Delta V_g)$, i.e.
$\tilde f_\theta(x) = f_\theta(x) + d$. A constant shifts each band without widening it, so the same Krawczyk
pipeline re-certifies the shifted controller.

1. **Sensitivities (Sec. IV-B):** one factorisation of the reduced PF Jacobian at the box centre gives
   $\partial z/\partial f_\theta = -J^{-1}\partial r/\partial f_\theta$ (implicit function theorem). Every
   constraint family builds its rows $S_f$ from it.
2. **QP (eq. 39):** minimise $\lVert\Delta p_g \oslash c_P\rVert^2 + \epsilon\lVert\Delta V_g\oslash c_V\rVert^2 +
   w_\xi\sum\xi_f$. Rows are normalised by the limit window $\omega_f$, columns by actuator ranges, inside a trust
   region $\Delta$.
3. **Algorithm 2 (RESTORE):** certify, compute the exceedance $\nu$, keep the best $d$, stop on SAFE / stall /
   "range wider than its window" (needs split). Otherwise take a QP step and adapt $\Delta$ from the reduction ratio.
4. **Algorithm 3 (RESTOREBOXSPLIT):** try the min-norm seed, then the OPF seed. If neither closes, bisect along the
   direction maximising the L2-aggregated (violation × sensitivity). Recurse with keep-best-ancestor up to
   MAXSPLIT.
5. **Deployment (Sec. IV-C):** a piecewise-constant offset controller, where each operating point gets the offset
   of its leaf.

> **In the code**
> - Step 1: [`linearize(grid, Vm, th)`](../certflow/power/sensitivity.py#L29). Each constraint's [`restoration_rows`](../certflow/power/constraints.py#L77) returns `RowBlock`s ($S_f$, the right-hand side, the window $\omega_f$).
> - Step 2: [`Restorer._step`](../certflow/power/restore.py#L160) assembles the rows generically and solves the exact min-norm problem with [`qp_minnorm`](../certflow/power/restore.py#L41) (LDP/NNLS). When that is infeasible, it falls back to the slack-penalised trust-region step [`qp_besteffort`](../certflow/power/restore.py#L72) (eq. 39 with $\xi_f$). `RestoreOptions.eps_vg = 0.1` is the paper's $\epsilon$.
> - Step 3: [`Restorer.restore_box`](../certflow/power/restore.py#L194) (Algorithm 2: keep-best, stall test, `width_excess` → needs split, trust-region update).
> - Step 4: [`Restorer.restore_split`](../certflow/power/restore.py#L280) (Algorithm 3). [`split_choice`](../certflow/power/restore.py#L263) gives the violation × sensitivity direction. `seeds={"opf": fn}` supplies the OPF seed. `_AncestorView` handles keep-best-ancestor.
> - Step 5: [`PiecewiseOffsetController`](../certflow/power/restore.py#L361).

## 9. Where the implementation differs from the paper

Every difference is **still sound** and was checked by Monte-Carlo.

- **Warm step:** $x_1(\varepsilon)$ is a polynomial in the inputs, not a point $z_1$ (Step 5). This gives a smaller residual radius.
- **Rounding:** floating-point rounding is bounded everywhere (`certflow.fp`, `KrawczykResult` a-posteriori check, deployed-network rounding).
- **Correlated sums:** $q_g$ and the slack add injection and load as correlated TMs, not intervals. This was about 10% narrower $q_g$ on PGLib-14/118 in the regression comparison.
- **Multi-direction splitting:** splitting may bisect up to four residual-dominant directions at once.
- **Restoration:** sensitivities are re-linearised at the centre of the box being restored. Slack rows include the $V_g$ sensitivity (losses). The OPF seed is a pluggable callable, so there is no IPOPT dependency.
