# Changelog

## Unreleased

### New: `certflow.audit`, the user-facing layer

You describe the grid, the controller and the limits in pandapower terms. Every report ends in **SAFE** (proven),
**UNSAFE** (with a concrete counterexample) or **NOT PROVEN**, and includes a Monte-Carlo cross-check. See
[docs/modes/](docs/modes/README.md).

**`GridModel`: a certified model of a pandapower grid as an RL agent sees and changes it**
- Observation blocks (`obs.*`):
  - `bus_vm` and `bus_va`;
  - line and transformer loading and flows;
  - load, sgen and gen values, and the ext_grid exchange;
  - taps and switch states;
  - constants and custom blocks.
- Actions (`act.*`): sgen and gen setpoints, taps, line and transformer switching, combinations and custom actions.
- `ActionSpace`:
  - one output per device (`ActionSpace.multi`, MultiDiscrete);
  - masked agents (`masked=True`).
- `Limits`: |V| (scalar or per bus), loading %, ext_grid P and Q, optional N-1 over line outages.
- `LoadUncertainty`:
  - per-load boxes;
  - low-rank load sets (`directions="common"`, a mode matrix, or `from_samples` for principal components of load
    data);
  - uncertain sgens.
- `check_against_pandapower()` confirms the model reproduces pandapower's results (to about 1e-8 on the examples).

**`RLAuditor`: the three RL modes**
- `next_step` (mode 1): the actions the agent can take for every load in the set, a certified post-action check of
  each (N-0, and N-1 if requested), and a search for a load where the deployed agent really violates a limit.
- `noise_robustness` (mode 2):
  - certified and found noise radii for "same action" and for "safe action", the latter with a concrete noise
    vector that makes the agent pick an action that really violates a limit;
  - the same radii per observation block;
  - the actions the agent can pick at fixed noise levels.
- `multi_step` (mode 3):
  - the certified tree of action sequences over several steps, with loads tracked across steps;
  - per-step certified envelopes of every bus voltage and branch loading;
  - `MultiStepReport.plot()`, a 3D view (element × step × certified band) with sampled trajectories and limit
    planes.
- The policy can be a torch module, a loaded SB3 / sb3-contrib model or a certflow `Sequential`.
  `input_layers=normalizer_layers(mean, var, clip)` adds VecNormalize-style normalisation as exact layers.

**`verify_dispatch`: an NN trained on OPF data (loads → generator setpoints)**
- Accepts a pandapower net, a MATPOWER case or a `Grid`.
- You declare the network's input and output layout and units (`inputs`, `input_units`, `outputs`, `pg_units`,
  `pg_gens`) instead of reshaping the network.
- Optional certified restoration (`restore=True`).

**Examples (each with its output saved next to it)**
- `examples/pandapower_rl_audit.py`: modes 1 and 2.
- `examples/multistep_reachability.py`: mode 3 and the 3D plot.
- `examples/opf_dispatch_audit.py`: the dispatch NN.

### Mode 3 tracking, plots and documentation

- **`RLAuditor.multi_step(..., track="rebox")`** (the default): each step works in the initial set's own
  coordinates, with the box widened by the step change, so the dimension stays constant. The steady-state grid has
  no memory beyond the discrete state, so this is never looser than `track="history"` when the step moves along the
  set's own directions. On the feeder at 4 steps it is about 3.5× faster than history (14 s vs 48 s), and certified
  SAFE where history is NOT PROVEN. See [docs/modes/rl_multistep.md](docs/modes/rl_multistep.md) for the analysis
  and the measured cost.
- **`.plot()` on every report:**
  - `NextStepReport`: a 2D map of action sets and safety per box;
  - `NoiseReport`: guaranteed vs found noise;
  - `DispatchReport`: certified vs sampled ranges vs limits, before and after restoration;
  - `MultiStepReport`: the 3D reachable states.
- **Comparing two multi-step runs:** `compare_3d` (3D side by side), `compare_multistep` (2D per step) and
  `width_table`.
- **[docs/modes/](docs/modes/README.md):** one page per mode (API, how it works, rendered results, figures, limits).
  `python docs/make_figures.py` regenerates every figure and pastes the latest reports into the pages.
- `examples/feeder_demo.py`: the small OLTC feeder and its demo agent, shared by the examples and the docs.

### Fixes

- **Switching off a line behind an open switch** (pandapower's empty auxiliary bus) was treated as the grid splitting:
  the power flow failed and the action or outage was wrongly reported as unsafe. Empty cut-off buses are now dropped,
  as pandapower does; a cut-off part with loads or generation still counts as islanding.
- **`obs.bus_va`** failed for angles beyond ±90°, e.g. behind phase-shifting transformers (YNd5). Each bus is now
  rotated by its own centre angle, which is exact for any angle.
- **Noise radius:** the bisection now checks that the requirement holds without noise, instead of reporting a vacuous
  "guaranteed up to 0 %". The breaking noise vector is reported relative to each entry's scale (entries mix pu, %
  and MW).
- **Plots** no longer switch matplotlib's backend, which disabled inline figures in notebooks.
- `check_against_pandapower` ignores the entries pandapower leaves undefined (out-of-service elements).

- **`from_pandapower` ratings.** pandapower writes real branch ratings into its internal model only when
  `max_loading_percent` exists; otherwise it writes a 100 MVA placeholder, which was used as the rating. Ratings
  are now computed from the element tables, exactly like pandapower's `loading_percent`.
- **`from_pandapower` out-of-service elements.** Out-of-service gens, loads and sgens were counted. They are now
  skipped. The grid also carries lookups from pandapower buses and branches to internal indices.
- **Grids without PV generators** (typical distribution grids) crashed in the Krawczyk setup: an empty index list
  was used with a float dtype.
- **Acceptance tolerances now default to 0.** The previous defaults let a "SAFE" verdict
  exceed a limit by up to 0.1 MW or 1e-4 pu.
- **`n1_analysis` with unsorted `monitored`** put results on the wrong branches. It also accepts per-bus voltage
  limits now.
- **`sb3_actor` always assumed Tanh.** It takes an `activation` argument now, and the new `from_sb3(model)` reads
  the layers from the loaded model.
- **A singular power-flow Jacobian** raised an exception; it now counts as "not certified".
- **Noise robustness:**
  - the attack is stronger (iterative, linearised);
  - falsification works for any concrete "bad action" test, not only "different action";
  - MultiDiscrete agents are supported;
  - when the noise-free action is already unsafe, the report says so.

### Tighter bounds (behaviour change)

- **Krawczyk centre.** `krawczyk(..., centre_iters=2)` takes a second Newton step in Taylor-model arithmetic, so
  the centre function also captures the quadratic load dependence and the remainder shrinks by about 100×.
  - `enclose_pf` (used by the RL audits and N-1 screening) now defaults to 2. The WSCC-9 reference outputs were
    regenerated with it.
  - `PFVerifier` keeps the affine centre (`VerifierOptions.centre_iters=1`), so its results are unchanged.
  - `verify_dispatch` uses 2.
- **Lines ending at an open switch.** pandapower puts an empty auxiliary bus there. The loading of such a line is
  now computed exactly from the voltage at its closed end, instead of multiplying the auxiliary bus's remainder by
  the line's (large) admittance.
- **`DiscretePolicyVerifier(stop_when_decided=True)`** stops splitting a box with one decided action when the check
  does not depend on the box.
- **`reachable_argmax_heads`** handles MultiDiscrete policies (one argmax per head).
