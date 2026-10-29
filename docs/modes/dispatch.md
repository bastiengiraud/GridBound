# Dispatch NN: an OPF proxy (the paper's use case)

**Question:** a network maps loads to generator setpoints (Pg, Vg). Does the resulting AC power flow keep every
operational limit for **every** load in a set? If not, which constant correction of the setpoints makes it
certifiably safe?

## API

```python
from certflow.audit import verify_dispatch

rep = verify_dispatch(net, model, load_width=0.10)                  # pandapower net, MATPOWER case or Grid
rep = verify_dispatch(net, model, load_width=0.10, restore=True)    # + certified piecewise correction
print(rep.summary())
rep.plot("dispatch.png")
```

The network's layout is declared, not assumed. The declarations become exact Linear layers around the network, so
the certified object is exactly the trained network.

| Argument | Values |
|---|---|
| `inputs` | `"pd_qd"` (default), `"pd"`, or indices into `[Pd; Qd]` |
| `input_units` | `"MW"` or `"pu"` |
| `outputs` | `("pg", "vg")`, `("vg", "pg")`, `("pg",)` (Vg held at setpoint) or `("vg",)` |
| `pg_units` | `"MW"` or `"pu"` |
| `pg_gens` | which generators' Pg the network outputs (the others keep their setpoint) |
| `vg_clip` | the deployed Vg saturation, if any |
| `load_set` | any `AffineSet` over `[Pd; Qd]`, e.g. `certflow.sets.low_rank_load_set` (paper eq. 19); default: each load ±`load_width` |
| `vm_pu`, `constraints` | the limits; the default set is \|V\|, Pg, Qg, slack P/Q, branch current and angle difference, with zero tolerance |

## How it works

1. The load set is pushed through the network: the setpoints become a Taylor model in the load symbols.
2. The specified injections (generation minus load) and the PV voltage setpoints become Taylor models.
3. Krawczyk proves that the power flow has a unique solution for every load in the set and encloses the state.
4. Every constraint is bounded on that enclosure:
   - **|V|**: correlated |V|² at PQ buses, the clipped setpoint at PV buses;
   - **generator Q and slack P/Q**: injection plus bus load;
   - **branch current**: a complex-zonotope bound;
   - **angle difference**.
5. **Splitting:** where the proof fails (ρ ≥ 1), the box is split along the directions with the largest residual.
   The certificate is the union over all boxes.
6. **Restoration** (`restore=True): a QP finds a constant offset (dPg, dVg) that moves every certified band inside
   its limits. The offset is re-certified, and the box is split when one offset is not enough.

API used: `PFVerifier.certify` and `Restorer.restore_split` (see [method.md](../method.md) for the mathematics).

## Example: case9, tight reactive limits

The network takes the active loads in pu and outputs [Vg; Pg] with Pg in pu. The loads vary by ±10 % each. The
two PV generators have tight reactive limits (Qmax = 8 and −2 MVAr).

<!-- report:dispatch -->
```text
DISPATCH CONTROLLER AUDIT  ->  UNSAFE
  load set: 3 loads, 3 uncertain directions; power flow certified (unique solution) on 1 box(es), max rho 0.116
  certified range over the whole load set vs limit:
    V            0.97 .. 1.06       pu    limits [0.9, 1.1]   ok
    Pg          76.53 .. 179.3      MW    limits [10, 300]   ok
    Qg          -17.3 .. 16.51      MVAr  limits [-300, 8]   CAN BE VIOLATED   worst excess 8.51 MVAr at element 1
    slack       2.158 .. 80.42      MW/MVAr limits [-300, 300]   ok
    branch loading  max 70.4% (branch 6)   limit 100%   ok
  Monte-Carlo cross-check (300/300 converged samples): every sample inside the certified ranges: True; limit families violated by a sample: 1
  restoration: 1/1 box(es) made certifiably safe with a constant setpoint correction:
    e in [[-1. -1. -1.], [1. 1. 1.]]: SAFE  dPg = [-0.117 -0.031] MW, dVg = [-0.0363 -0.0218] pu
  time 4.7s
```
<!-- /report -->

![dispatch audit](../figures/dispatch.png)

How to read it:

- **Blue:** the certified range over the whole load set.
- **Black:** the range the sampled loads actually reach.
- **Red:** the limits.
- **Green:** the certified range after restoration.

Generator 1's Q can exceed its 8 MVAr limit. A sample confirms it, so the verdict is UNSAFE. One constant correction
(a slight reduction of both Vg) makes the whole load set certifiably safe.

For branch loading, only the upper bound is certified, so the blue bar starts at 0.

## Limits of the guarantee

- **Steady state only:** the AC power flow, with no dynamics.
- **The given load set:** loads outside it are not covered.
- **Restoration** is a certified fix for the boxes it reports SAFE. It is not a retraining of the network.
