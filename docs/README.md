# certflow documentation

| Document | What it covers |
|---|---|
| [method.md](method.md) | The method of Giraud, Vorwerk & Chatzivasileiadis, *Verified Safety Layers for AC Surrogates*, step by step, with the function that implements each step. Read this first. |
| [rl_reachability.md](rl_reachability.md) | Mode 2: certified reachability of a discrete RL agent (WSCC-9 MaskablePPO) over load uncertainty: one-step, N-1 and two-step. |
| [measurement_noise.md](measurement_noise.md) | Mode 3: certified robustness radius of the agent's decision to measurement noise at an operating point. |

## The three modes at a glance

| Mode | Question | Uncertain input | Where the certified power flow sits | Output | Entry point |
|---|---|---|---|---|---|
| 1. Controller verification + restoration (the paper) | Does the NN dispatch keep every operational limit for every load in the set? If not, which constant offset fixes it? | low-rank load zonotope $x = x_0 \odot (1 + r_f G\varepsilon)$ | **after** the NN: NN → setpoints → PF | certified bands per constraint; (piecewise) offset | `PFVerifier.certify`, `Restorer.restore_split` |
| 2. RL reachability | Which actions can the agent take for every load in a box, and is each one safe one step ahead (N-0, N-1)? Which action sequences are reachable in two steps? | load-admittance deviation $\delta \in \delta_c \pm w$ | **before** the NN (state → observation) **and after** it (post-action state) | certified action sets per sub-box, per-action safety, N-1 count bounds, two-step tree | `DiscretePolicyVerifier`, `MultiStepVerifier` |
| 3. Measurement noise | At a known operating point, how much sensor noise can the agent tolerate and still pick the same action, or a safe one? | independent noise $\lvert\eta_i\rvert \le \varepsilon\lvert o_i\rvert$ on every measurement | none for the observation (the state is exact); only in the safety check of each reachable action | certified and falsified radius $\varepsilon$; action sets per noise level | `robustness_radius` |

All three modes share the same core:

- degree-2 Taylor models over shared noise symbols (`certflow.tm`);
- DeepZ network reachability (`certflow.nn`);
- the Krawczyk existence/uniqueness proof (`certflow.krawczyk`);
- adaptive splitting (`certflow.split`).

The modes differ in where the uncertainty enters and in what is checked.

## Notation used throughout

| Paper | Code | Meaning |
|---|---|---|
| $\varepsilon \in [-1,1]^K$ | `e`, the first `K` symbols of a `TM` | input (box) symbols |
| $x = c_0 + c^\top\varepsilon + \varepsilon^\top q\,\varepsilon + [-R,R]$ | `TM(c0, C, Q, R)` | degree-2 Taylor model |
| $\varepsilon_{\text{new},j}$ | symbols appended after the first `K` | DeepZ relaxation symbols (one per crossing / relaxed neuron) |
| $z = (v_{r,ns}, v_{i,ns})$ | `RectPFSystem` state, `z0`, `x1` | rectangular voltage state of the non-slack buses |
| $h(z,\varepsilon) = h_0(z) - S(\varepsilon)$ | `system.residual(x)` minus `S` | power-flow residual |
| $M \approx J(z_1)^{-1}$ | `M` in `krawczyk()` | Krawczyk preconditioner |
| $K_b$, $\rho(K_b)$ | `_Kb(r)`, `specrad` | slope majorant and its spectral radius |
| $r^*$ | `KrawczykResult.r` | verified enclosure radius |
| leaf | `LeafResult`, `PolicyLeaf` | sub-box of the input box |

Line links point to the current version of the code. If a line has moved, search for the function name.
