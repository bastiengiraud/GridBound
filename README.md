# GridBound

**`certflow`: formal verification of neural-network controllers and RL agents through implicit AC power-flow
models.**

For a controller `y = f_θ(x)` and an input set `x ∈ X` (a low-rank zonotope), certflow proves that the AC power
flow has a **unique** solution for every input. It then encloses every operational quantity (|V|, Pg, Qg, slack,
branch current/loading, angle difference) rigorously, floating-point rounding included. If the result can
violate a limit, it restores safety with a re-certified constant offset on the controller output, piecewise
over a partition when one offset is not enough.

For discrete-action RL agents, certflow certifies which actions the agent can take over a set of observations.
It then checks each reachable action one step ahead (N-0 and N-1), builds multi-step reachability trees, and
computes certified robustness radii to measurement noise.

The method follows *Verified Safety Layers for AC Surrogates: Reachability-Based Local Verification and Safety
Restoration* (Giraud, Vorwerk, Chatzivasileiadis). The RL case study verifies the agent of
[*Multi-Objective Deep Reinforcement Learning for Secure and Stable Power System
Operation*](https://arxiv.org/abs/2608.20914) (Papadopoulos, Tsaousoglou, Vorwerk).

## Contents

```
certflow/                         core: knows nothing about power systems
├── fp.py          rigorous floating-point error bounds (gamma_n, outward rounding)
├── tm.py          degree-2 Taylor models / affine zonotopes  (TM = (c0, C, Q, R))
├── nn/            NN reachability, architecture-agnostic
│   ├── network.py   NNReachable interface: reach(TM) -> TM, forward(x); Sequential; from_torch, from_state_dict, sb3_actor
│   ├── layers.py    abstract transformers: Linear, ReLU, Tanh, Sigmoid, MinMaxNormalize/Denormalize
│   └── argmax.py    discrete outputs: which actions can win over an input set (masked argmax)
├── krawczyk.py    validated Krawczyk existence/uniqueness for F(z, e) = S(e), Jacobian affine in z
├── robustness.py  robustness radius to observation noise: certified (bisection) + falsified (attack)
├── policy.py      DiscretePolicyVerifier (reachable actions -> per-action checks, adaptive partition);
│                  MultiStepVerifier (k-step reachability tree)
├── sets.py        input zonotopes x = c + G e and e-space sub-boxes
└── split.py       adaptive splitting + process pool (Windows-safe)
certflow/power/                   application layer: AC power flow
├── grid.py        Grid (topology-switchable Ybus), from_pandapower, from_matpower
├── pf.py          point Newton PF + rectangular PF as a Krawczyk system (RectPFSystem, uncertain Shunt)
├── pfcert.py      enclose_pf: certified PF for uncertain specs / shunt admittances
├── quantities.py  observables on a certified state: |V|, injections, branch P/Q/|S|/|I|, admittance-load power
├── contingency.py n1_analysis: certified N-1 screening over a box (two-sided loadings, islanding, PF failure)
├── controller.py  DispatchController: NN -> [Pg; Vg] (+ offset, Vg saturation)
├── constraints.py pluggable constraints: V, Pg, Qg, slack, branch |I|², angle difference
├── verifier.py    PFVerifier: certified pipeline + rho-driven splitting -> Certificate
├── sensitivity.py control sensitivities (implicit function theorem)
├── restore.py     restoration: min-norm QP / trust-region SLP / box splitting
├── validate.py    Monte-Carlo validation of a certificate
└── cases.py       MATPOWER case9
examples/          WSCC-9 MaskablePPO case studies (wscc9_env.py, wscc9_actor_reachability.py) + reference outputs
wscc9_maskableppo_policy_extraction/   the trained agent: actor weights (.npz), the SB3 'data' file, the grid (WSCC_9_bus.raw)
tests/             soundness-by-sampling and end-to-end tests
docs/              the method (paper -> code map) and one document per verification mode
shell/scripts/     LSF job scripts (setup, tests, case studies); logs go to shell/outputs/
```

## Documentation

- [docs/method.md](docs/method.md): the method of the paper, step by step, with the function that implements each step
- [docs/rl_reachability.md](docs/rl_reachability.md): reachability of the RL agent over load uncertainty (one-step, N-1, two-step)
- [docs/measurement_noise.md](docs/measurement_noise.md): robustness radius to measurement noise

## Install

Requires Python ≥ 3.9. The core depends only on numpy and scipy.

```bash
git clone https://github.com/bastiengiraud/GridBound.git
cd GridBound
python -m venv .venv && source .venv/bin/activate
pip install -e .              # numpy + scipy
pip install -e ".[all]"       # + pandapower, torch (only for from_pandapower / from_torch), pytest
```

## Quick start

```python
from certflow.nn import from_torch                       # or from_state_dict / sb3_actor / layers by hand
from certflow.power import (from_pandapower, DispatchController, PFVerifier, default_constraints,
                            Restorer, monte_carlo)
from certflow.sets import low_rank_load_set

grid = from_pandapower(net)                              # solved net (or from_matpower("case.m"))
ctrl = DispatchController(from_torch(model), grid.ng, pg_active=..., vg_clip=(vmin, vmax))
U    = low_rank_load_set(x0, Gdirs, frac)                # x = x0 * (1 + frac * Gdirs e), e in [-1,1]^K
ver  = PFVerifier(grid, ctrl, U, default_constraints(grid, vmin, vmax))

cert = ver.certify()                                     # Certificate: .formal, .safe, .bands, .summary()
monte_carlo(ver, cert, n=500)["sound"]                   # independent sampling check
root, leaves = Restorer(ver).restore_split(max_split=8, nproc=4)   # piecewise offsets, each re-certified
```

Parallel: `with ObjectPool(ver, nproc) as pool: ver.certify(pool=pool)` and `Restorer(ver).restore_split(nproc=...)`.
Workers receive the pickled verifier once, so this works with Windows' *spawn*. Guard scripts with
`if __name__ == "__main__":`. Pin BLAS threads (`OMP_NUM_THREADS=1` etc.) for bitwise run-to-run determinism.

## RL case study: WSCC-9 MaskablePPO agent

```bash
python examples/wscc9_actor_reachability.py --list-ops          # the six operating points
python examples/wscc9_actor_reachability.py --quick             # all studies, small boxes: ~1.5 min
python examples/wscc9_actor_reachability.py                     # all studies, full settings: tens of minutes, < 2 GB RAM
python examples/wscc9_actor_reachability.py --study noise --ops 0 4    # one study, some operating points
python examples/wscc9_actor_reachability.py --require-n1        # count possible N-1 violations as unsafe
```

`examples/wscc9_case_studies_output.txt` and `examples/wscc9_noise_robustness_output.txt` are reference outputs
(`--quick` settings).

`examples/wscc9_env.py` models the training environment exactly. It reproduces the observation stored with the
trained model to float32 precision.

**Environment facts the model encodes**
- Loads are fixed admittances (1 + δ)·Y_nom, so the exogenous uncertainty is δ ∈ R³ (buses 5, 6, 8).
- The observed load P/Q, slack MW and line/transformer loadings are power-flow outputs. Loadings are
  max(|S_from|, |S_to|) including the branch shunt, over 250 / 300 MVA.
- Network data is the PSS/E file the agent was trained on (`WSCC_9_bus.raw`; same network as the eRoots
  PublicGridCollection file the paper cites). It differs from textbook WSCC-9 on lines 5-4, 9-6 and 7-8.
  `tests/test_wscc9_data.py` checks the model's data against it.

**Coupled observation:** a certified power flow over the δ-box (admittance loads as an uncertain `Shunt`) gives
the state as Taylor models. All 26 observation entries are built from it over the same symbols, so the actor
sees exactly the observations that can occur.

**Per box**
- `reachable_argmax` gives the certified action set.
- Every reachable action is checked one step ahead: a certified post-action power flow over δ' ∈ δ ± margin
  (existence/uniqueness, no N-0 overload, slack in [0, 250] MW).
- Boxes are bisected until one action remains or the depth limit is reached.
- A Monte-Carlo run with the float32 actor cross-checks every claim.

**N-1:** every post-action state is also screened for all line outages with `n1_analysis`. The result is
certified [lower, upper] bounds on the environment's N-1 metric (overloads + 3 × power-flow failures), with
islanding detected exactly. The output reads each interval as one of three statuses:
- SECURE: upper bound = 0.
- INSECURE: lower bound > 0, i.e. violated at every load in the box.
- UNCERTAIN: lower bound = 0 < upper bound. A sampling search then either finds a real counterexample or
  reports that the bound may be conservative.

N-1 is reported by default; `--require-n1` enforces it. When the agent's actions are not N-1 secure, the
example lists the valid actions that would be N-0 safe and N-1 secure.

**Two-step reachability:** `MultiStepVerifier` re-verifies the agent from every post-action state, over the
load box widened by the per-step change. The result is the tree of every 2-action sequence the agent can take,
each with its certified one-step safety.

**Measurement-noise robustness:** at an operating point the grid state is exact, but the agent sees obs + η with
|η_i| ≤ ε·|obs_i| on every measured entry. For each point the study reports:
- the certified radius ε for "the agent still picks the same action" (a proven lower bound);
- the smallest noise found that flips the action, and to which action (an upper bound);
- the same pair per measurement group, which shows the most sensitive measurements;
- the radius up to which every action the agent could pick is still certified safe.

The generic parts (`enclose_pf`, `Shunt`, `quantities`, `n1_analysis`, `DiscretePolicyVerifier`,
`MultiStepVerifier`, `robustness_radius`) are reusable for other grids and agents. Only the observation layout,
the action list and the mask live in the example.

Not yet covered: damping (small-signal stability). The multi-step tree re-boxes the load at each step, which is
sound but loses correlation across steps.

## Running on an LSF cluster

All scripts are submitted **from the repository root**; logs are written to `shell/outputs/`.

```bash
bash shell/scripts/setup_env.sh                    # once, on a login node: .venv with certflow (editable)
bsub < shell/scripts/run_tests.sh                  # test suite
bsub < shell/scripts/wscc9_rl_case_studies.sh      # all case studies in one serial job
bsub < shell/scripts/wscc9_rl_array.sh             # the same as a 14-element job array, in parallel
ARGS="--require-n1" bsub < shell/scripts/wscc9_rl_array.sh       # extra options for every element
bstat / bjobs                                      # monitor
```

`shell/scripts/common.sh` loads the Python module (`PY_MODULE`, default `python3/3.11.9`; see
`module avail python3`), activates `.venv` and pins BLAS threads to the job's slots. Queue, memory and wall-time
requests are in the `#BSUB` headers of each script.

## Extending

| You want | You write |
|---|---|
| a new activation | a `certflow.nn.layers.Layer` subclass with `reach(ZonoState, ctx)` and `forward` |
| a new architecture | any object implementing `NNReachable.reach(TM) -> TM` (or a `Sequential` of layers) |
| a new limit family (e.g. transformer loading) | a `Constraint` subclass: `evaluate(leaf) -> Band`, plus `restoration_rows` to take part in restoration |
| a different controller | an object with `reach(x_tm, offset) -> (y_raw, y)`, `forward(x, offset)`, `ng`, `vg_clip` |
| another implicit model | an `AffineJacobianSystem` for `certflow.krawczyk.krawczyk` (see `RectPFSystem`) |
| a topology change | `grid.with_branch_status(status)` (Ybus/Yf/Yt rebuilt from branch primitives) |
| a policy whose input is a physical state | `observe(lo, hi) -> Observation(tm)` + `check_action(a, lo, hi, obs)` for `DiscretePolicyVerifier` |
| a nonlinear observable | `tm.compose(x, f, df, d2_range)` (or `tm.sqrt`, `tm.maximum`) on certified TMs |
| N-1 security | `n1_analysis(grid, spec_fn, lo, hi, outages, rating_mva)` -> `N1Report.count_bounds()`, `.secure` |
| multi-step guarantees | `MultiStepVerifier(policy, problem, transition, propagate, steps=k)` -> tree, `.paths()`, `.safe` |
| robustness to measurement noise | `robustness_radius(policy, obs_tm, scale, mask, require="same" or predicate)` -> certified / falsified radius |

## Tests

```bash
python -m tests.run_tests      # no pytest needed
pytest                         # same tests with pytest
```

## Floating-point assumptions

- IEEE double, round-to-nearest.
- `atan2`, `tanh` and `cosh` accurate to a few ulp. The tanh/sigmoid transformers shrink their slope by
  (1 − 64u) and pad endpoints by 8 ulp.
- The deployed actor's float32 forward pass is covered by an explicit error term (`deploy_u = 2^-24`).
