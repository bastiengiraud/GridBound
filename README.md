# GridBound

**`certflow`: an independent safety auditor for learned power-grid controllers.**

Grid operators increasingly receive controllers that were trained rather than designed: neural-network dispatch
surrogates, reinforcement-learning agents and similar. certflow checks such a controller **as it is**, whatever its
training data, loss or reward, against the limits of the grid it will operate. The result is a mathematical
certificate, not a statistical estimate. It holds for every operating condition in a given set, with
floating-point rounding included.

What it can audit:

- **Continuous dispatch controllers** (loads → generator setpoints). certflow proves that the AC power flow has a
  **unique** solution for every load in the set, and bounds every operational quantity: bus voltages, generator
  active and reactive power, slack power, branch currents or loadings, and angle differences. If a limit can be
  violated, it computes a certified constant correction of the setpoints, piecewise over the set if needed.
- **Discrete-action agents (RL).** certflow proves which actions the agent can take for every load in a set. It then
  checks whether each such action keeps the grid within its limits one step later, in the intact grid (N-0) and after
  any single line outage (N-1), and builds multi-step trees of the action sequences that can occur.
- **Robustness to measurement noise.** certflow computes the largest sensor-noise level for which the agent's
  decision provably stays the same, or provably stays safe, together with a concrete noise vector that breaks it.

Every case study also runs an independent Monte-Carlo check of the certified claims.

The method builds on *Verified Safety Layers for AC Surrogates: Reachability-Based Local Verification and Safety
Restoration* (Giraud, Vorwerk, Chatzivasileiadis). The RL case study audits the agent of
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
├── contingency.py n1_analysis: certified N-1 screening over a box (two-sided loadings, |V|, slack; islanding)
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
docs/              how the method works (with links into the code) and one document per audit mode
shell/scripts/     LSF job scripts (setup, tests, case studies); logs go to shell/outputs/
```

## Documentation

- [docs/README.md](docs/README.md): overview of the three audit modes and the notation
- [docs/method.md](docs/method.md): the verification method step by step, with the function that implements each step
- [docs/rl_reachability.md](docs/rl_reachability.md): auditing an RL agent over load uncertainty (one step, N-1, multi-step)
- [docs/measurement_noise.md](docs/measurement_noise.md): auditing an agent's robustness to measurement noise

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

## Case study: auditing an RL agent on the WSCC 9-bus system

The example audits a MaskablePPO agent (stable-baselines3) that redispatches generators, changes voltage setpoints
and switches lines on the WSCC 9-bus system. The auditor uses only:

- the trained weights;
- the definitions of the observation, the actions and the mask;
- the grid data (`WSCC_9_bus.raw`, the training environment's PSS/E file);
- the grid limits.

```bash
python examples/wscc9_actor_reachability.py --list-ops          # the six operating points
python examples/wscc9_actor_reachability.py --quick             # all studies, small boxes: about 2 min
python examples/wscc9_actor_reachability.py                     # all studies, full settings: tens of minutes, < 2 GB RAM
python examples/wscc9_actor_reachability.py --study noise --ops 0 4    # one study, some operating points
python examples/wscc9_actor_reachability.py --require-n1        # also require N-1 security for an action to count as safe
python examples/wscc9_actor_reachability.py --study two-step --track both --step-change common   # re-boxing vs history tracking
```

**Grid limits checked**, the same in the intact grid (N-0) and after every line outage (N-1):

- the power flow has a unique solution (no islanding, no collapse);
- every line and transformer loading is at or below its rating;
- |V| is within [0.95, 1.05] pu at every bus;
- the slack generator's active power is within [0, 250] MW.

**Studies**

- **Reachability over load uncertainty:** for every load in a box around an operating point, the certified set of
  actions the agent takes, and for each action a certified verdict on N-0 safety and N-1 security. When an action is
  not N-1 secure, the output names the outage and the violated quantity.
- **Multi-step reachability:** every sequence of actions the agent can take from the box, each step certified. The
  uncertainty is either re-boxed at every step or tracked with new symbols per step (`--track rebox|history|both`).
- **Measurement noise:** the certified and falsified noise radius for "same action" and for "safe action", overall
  and per measurement group.

`examples/wscc9_case_studies_output.txt` and `examples/wscc9_noise_robustness_output.txt` are reference outputs
(`--quick` settings). `examples/wscc9_env.py` models the environment: the loads are constant admittances, and the
observed quantities are power-flow outputs. The model reproduces the observation stored with the trained agent to
float32 precision. The generic parts (`enclose_pf`, `Shunt`, `quantities`, `n1_analysis`, `DiscretePolicyVerifier`,
`MultiStepVerifier`, `robustness_radius`) apply to other grids and agents. Only the observation layout, the action
effects and the mask are specific to this example. The full step-by-step explanation is in
[docs/rl_reachability.md](docs/rl_reachability.md) and [docs/measurement_noise.md](docs/measurement_noise.md).

Not covered yet: dynamic stability (damping).

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
| N-1 security | `n1_analysis(grid, spec_fn, lo, hi, outages, rating_mva, vlim=, slack=)` -> `N1Report.count_bounds(kinds=)`, `.secure` |
| multi-step guarantees | `MultiStepVerifier(policy, problem, transition, propagate, steps=k)` (re-boxing) or `MultiStepVerifier(..., track="history", extend=...)` (new symbols per step) -> tree, `.paths()`, `.safe`, `.stats()` |
| robustness to measurement noise | `robustness_radius(policy, obs_tm, scale, mask, require="same" or predicate)` -> certified / falsified radius |

## Tests

```bash
python -m tests.run_tests      # no pytest needed
pytest                         # same tests with pytest
```

## Assumptions

The certificates assume IEEE double-precision arithmetic with round-to-nearest, and math-library functions (`atan2`,
`tanh`, `cosh`, `exp`) accurate to a few ulp. The deployed network's own arithmetic (e.g. float32) is covered by an
explicit error term. The guarantees concern the steady-state AC power flow. See
[docs/method.md](docs/method.md#9-assumptions).
