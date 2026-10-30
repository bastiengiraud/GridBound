# How the proofs work

This page is for readers who want the mathematics. To use the package, see [the mode pages](modes/README.md).

The method follows *Verified Safety Layers for AC Surrogates: Reachability-Based Local Verification and Safety
Restoration* (Giraud, Vorwerk, Chatzivasileiadis). The function names point into the code.

## The problem

A network maps loads to setpoints, or (for an agent) an observation to an action. The resulting grid state is not an
output of the network: it is the solution of the AC power flow. For **every** load in a set, we want to prove two
things:

1. the power flow has a unique solution;
2. every operational quantity stays within its limits.

```
 load set ──network──▶ setpoints ──▶ specified injections ──▶ Krawczyk: unique power-flow state?
    ▲                                                              │ no            │ yes
    └──────────────────── split the set ◀──────────────────────────┘               ▼
                                                    bounds on |V|, P, Q, loading, angle → limits
```

## 1. Taylor models

Every uncertain quantity is a degree-2 polynomial in shared symbols ε ∈ [−1, 1]ᴷ, plus an interval remainder R:

  x = c₀ + cᵀε + εᵀqε + [−R, R]

The symbols are shared by all quantities (loads, network outputs, voltages), so correlations are kept. Every
floating-point rounding is added to R, so the bounds hold for the exact real values. Code: `certflow.tm`,
`certflow.fp`.

## 2. The input set

The loads form an affine set x = c + Gε, for example x₀ ⊙ (1 + r·Gε) with a few load modes (`low_rank_load_set`,
`LoadUncertainty`). Verification runs over sub-boxes of ε; their union covers the set (`AffineSet.sub_tm`).

## 3. Network reachability

The set is pushed through the network layer by layer (DeepZ):
- **Linear layers** are exact.
- **ReLU:** a neuron that is always on or always off is exact. A neuron that can switch is replaced by the
  minimal-area parallelogram, which adds one new symbol.
- **tanh / sigmoid:** the same construction, with the slope taken from the endpoints.

The deployed float32 rounding is bounded separately. For an agent, the masked argmax over the resulting logits gives
every action it can take: action a is possible only if its logit can reach every other's (`reachable_argmax`).

Code: `certflow.nn` (`Linear`, `ReLU`, `Tanh`, `Sequential.reach`, `from_torch`, `from_sb3`).

## 4. The power flow in rectangular coordinates

The state is the real and imaginary voltage of every non-slack bus. The residual stacks three mismatches:

- active power at non-slack buses;
- reactive power at PQ buses;
- |V|² − V_g² at PV buses.

All three are **quadratic** in the state. So degree-2 Taylor models represent them exactly, and the Jacobian is
affine in the state. The uncertainty enters through the specified injections and setpoints S(ε).

Code: `RectPFSystem`, `enclose_pf`.

## 5. Existence and uniqueness: Krawczyk

1. **Point solution** at the centre of the box (Newton), giving z₀.
2. **Preconditioner** M = J(z₀)⁻¹.
3. **Centre function:** Newton steps taken symbolically over ε, giving a polynomial x₁(ε). The audit layer uses two
   steps, a quadratic centre, which shrinks the remainder about 100×.
4. **Slope bound:** K_b(r) bounds |I − MJ| over x₁(ε) ± r. The growth of the Jacobian is kept correlated (M is
   applied before taking absolute values).
5. **Inflation:** r is grown until r ≥ (I − K_b)⁻¹·corr, failing if the spectral radius ρ(K_b) ≥ 1. The accepted
   radius is re-checked with every rounding bounded.

On success, for every ε the power flow has **exactly one** solution in x₁(ε) ± r.

Code: `krawczyk`, `KrawczykResult.enclosure`.

## 6. Splitting

If the test fails, the box is bisected along the direction that contributes most to the Krawczyk residual. Each half
is certified on its own. The bounds of the set are the union of its leaves. When the leaf budget is exhausted, the
result is "not certified", never a guess.

Code: `adaptive_split`, `enclose_pf_split`, `DiscretePolicyVerifier`.

## 7. Bounding the limited quantities

| Quantity | Bound |
|---|---|
| \|V\| | the correlated Taylor model of vr² + vi², then a square root |
| P, Q injections; generator Q; slack | the quadratic forms on the state, plus bus loads, kept correlated |
| branch current / loading | the complex current as a zonotope. A line ending at an open switch is computed exactly from its closed end. |
| angle difference | `atan2` on the corners of each bus's voltage box |

Where a quantity straddles its limit, the box is bisected further to decide it. Code: `certflow.power.constraints`,
`certflow.power.quantities`, `GridModel.screen`.

## 8. Restoration (dispatch NN)

If a bound crosses a limit, a constant offset d = (ΔPg, ΔVg) is added to the network output. A constant shifts the
bounds without widening them.

1. **Sensitivities** of every bound to d come from one Jacobian factorisation.
2. **A QP** finds the smallest offset that moves every bound inside its limit. Voltage changes are preferred.
3. **Re-certify** with the offset and iterate within a trust region.
4. **Split** the load set when one offset is not enough; the result is a piecewise-constant correction.

Code: `Restorer.restore_split`, `PiecewiseOffsetController`.

## 9. Assumptions

- **Arithmetic:** IEEE double precision with round-to-nearest. The math library (`atan2`, `tanh`, `exp`, …) is
  accurate to a few ulp.
- **Deployed network:** it runs in the precision given by `deploy_u` (float32 for SB3 policies).
- **Scope:** the guarantees cover the steady-state AC power flow, not dynamics or stability.
