# certflow documentation

certflow is an **independent safety auditor for learned grid controllers**. It takes a trained controller or agent
as it is, whatever its training data, loss or reward, and proves whether the power grid stays within its limits for
every operating condition in a given set. If it cannot prove safety, it shows where and why. For continuous
controllers it can also compute a certified correction.

The guarantees are mathematical, not statistical. Every result is computed with validated numerics: Taylor models,
interval bounds and outward rounding. It holds for **all** inputs in the set, not only for sampled ones. Every case
study also runs an independent Monte-Carlo check of the certified claims.

## Documents

| Document | Contents |
|---|---|
| [method.md](method.md) | The verification method, step by step: validated arithmetic, network reachability, the Krawczyk existence and uniqueness proof for the AC power flow, bounding of operational quantities, domain splitting, and safety restoration. Each step names the function that implements it. Read this first. |
| [rl_reachability.md](rl_reachability.md) | Auditing a discrete-action RL agent over load uncertainty: which actions it can take, whether each is safe one step ahead (N-0 and N-1), and which action sequences can occur over two steps. |
| [measurement_noise.md](measurement_noise.md) | Auditing an agent's robustness to measurement noise: the largest noise level for which its decision provably stays the same, or provably stays safe. |

## The three audit modes

| Mode | Question | Uncertain input | Certified power flow | Result | Entry point |
|---|---|---|---|---|---|
| **Controller verification and restoration** | Does a continuous dispatch controller (loads → generator setpoints) keep every operational limit for every load in the set? If not, which constant correction fixes it? | low-rank load set $x = x_0 \odot (1 + r_f G\varepsilon)$ | after the controller: setpoints → grid state | certified bounds per limit; a (piecewise) constant offset that makes them safe | `PFVerifier.certify`, `Restorer.restore_split` |
| **RL reachability** | Which actions can a discrete agent take for every load in a box, and is each action safe one step ahead? Which action sequences are reachable? | load deviation $\delta \in \delta_c \pm w$ | before the agent (grid state → observation) and after the action (post-action state, N-0 and N-1) | certified action sets per sub-box, per-action safety, N-1 bounds, a multi-step tree (uncertainty re-boxed or tracked per step) | `DiscretePolicyVerifier`, `MultiStepVerifier` |
| **Measurement noise** | At a known operating point, how much sensor noise can the agent tolerate and still pick the same action, or a safe one? | independent noise $\lvert\eta_i\rvert \le \varepsilon\lvert o_i\rvert$ on each measurement | only in the safety check of each action the agent might pick | certified and falsified noise radius; action sets per noise level | `robustness_radius` |

All three modes share one core:

- degree-2 Taylor models over shared symbols (`certflow.tm`);
- neural-network reachability (`certflow.nn`);
- the Krawczyk proof that the power flow has a unique solution (`certflow.krawczyk`);
- adaptive splitting of the input set (`certflow.split`).

They differ in where the uncertainty enters and in what is checked.

## What "safe" means

Safety is defined by the grid, not by the controller's training objective. The limits used in the RL case study are
listed in [rl_reachability.md](rl_reachability.md#what-counts-as-a-violation): a unique power-flow solution, branch
ratings, voltage magnitudes at every bus and the slack generator's range, at N-0 and after every line outage. For the
dispatch controller they are the constraint families in [method.md](method.md#7-bounding-the-operational-quantities).
New limit families can be added without changing the core.

## Notation

| Symbol | In the code | Meaning |
|---|---|---|
| $\varepsilon \in [-1,1]^K$ | `e`, the first `K` symbols of a `TM` | input (box) symbols |
| $x = c_0 + c^\top\varepsilon + \varepsilon^\top q\,\varepsilon + [-R,R]$ | `TM(c0, C, Q, R)` | degree-2 Taylor model |
| $\varepsilon_{\text{new},j}$ | symbols appended after the first `K` | relaxation symbols (one per relaxed neuron) |
| $z = (v_{r,ns}, v_{i,ns})$ | `RectPFSystem` state | rectangular voltages of the non-slack buses |
| $h(z,\varepsilon) = h_0(z) - S(\varepsilon)$ | `system.residual(x)` minus `S` | power-flow residual |
| $M \approx J(z_0)^{-1}$ | `M` in `krawczyk()` | Krawczyk preconditioner |
| $K_b$, $\rho(K_b)$ | `_Kb(r)`, `specrad` | slope bound and its spectral radius |
| $r^*$ | `KrawczykResult.r` | radius of the certified state enclosure |
| leaf | `LeafResult`, `PolicyLeaf` | sub-box of the input set |

Links to the code point at the current line numbers. If a line has moved, search for the function name.
