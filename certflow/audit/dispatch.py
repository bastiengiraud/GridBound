"""Auditing an NN dispatch controller (loads -> generator setpoints, e.g. an OPF proxy).

    rep = verify_dispatch(net, model, load_width=0.10)          # pandapower net, MATPOWER case or certflow Grid
    print(rep.summary())
    rep = verify_dispatch(net, model, load_width=0.10, restore=True)   # + certified piecewise correction

The network's input / output layout is declared, not assumed:

    inputs   "pd_qd"  [Pd of every load; Qd of every load]  (default)
             "pd"     [Pd of every load]
             an index array into [Pd; Qd]                  (any subset / order)
    outputs  ("pg", "vg") [Pg of the dispatchable gens; Vg of every gen]  (default; any order of the two blocks)
             ("pg",) or ("vg",)  the other block is held at the grid's own setpoints (or pg_fixed / vg_fixed)
    units    input_units / pg_units "MW" or "pu"

Everything is turned into exact Linear layers around the network, so the certified object is exactly the network the
user trained, and the verifier sees the standard layout of :class:`certflow.power.DispatchController`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nn.layers import Linear
from ..nn.network import Sequential, with_input_layers, with_output_layers
from ..power.constraints import default_constraints
from ..power.controller import DispatchController
from ..power.grid import Grid, from_matpower, from_pandapower
from ..power.restore import Restorer
from ..power.validate import monte_carlo
from ..power.verifier import PFVerifier, VerifierOptions
from ..sets import AffineSet
from .rl import as_policy

_UNIT = {"V": ("pu", 1.0), "Pg": ("MW", None), "Qg": ("MVAr", None), "slack": ("MW/MVAr", None), "angle": ("deg", 180 / np.pi)}


def _grid_and_nominal(grid):
    """(Grid, Pd0, Qd0, Pg0, Vg0) from a pandapower net, a MATPOWER case (dict / .m path) or a Grid."""
    if isinstance(grid, Grid):
        g = grid
        Pd0, Qd0 = g.names.get("load_pq", (None, None))
        return g, Pd0, Qd0, g.names.get("gen_pg"), g.names.get("gen_vg")
    if isinstance(grid, str) or (isinstance(grid, dict) and "baseMVA" in grid):     # (a pandapower net is a dict too)
        g = from_matpower(grid)
        Pd0, Qd0 = g.names["load_pq"]
        return g, Pd0, Qd0, g.names["gen_pg"], g.names["gen_vg"]
    import copy
    import warnings

    import pandapower as pp
    net = copy.deepcopy(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp.runpp(net, numba=False)
        g = from_pandapower(net)
    li, gi = g.names["load_index"], g.names["gen_index"]
    sc = net.load.scaling.loc[li].values if "scaling" in net.load else 1.0
    return (g, net.load.p_mw.loc[li].values * sc, net.load.q_mvar.loc[li].values * sc, net.gen.p_mw.loc[gi].values,
            net.gen.vm_pu.loc[gi].values)


def load_box(Pd0, Qd0, rel, power_factor="constant") -> AffineSet:
    """Box of loads around (Pd0, Qd0): each load within +-rel of its value.  "constant" power factor: one symbol per
    load moves P and Q together; "independent": separate symbols for P and Q."""
    x0 = np.r_[Pd0, Qd0].astype(float)
    nl = len(Pd0)
    if power_factor == "constant":
        G = np.zeros((2 * nl, nl))
        G[np.arange(nl), np.arange(nl)] = rel * np.abs(Pd0)
        G[nl + np.arange(nl), np.arange(nl)] = rel * np.asarray(Qd0) * np.sign(np.where(Pd0 == 0, 1.0, Pd0))
    else:
        G = np.diag(rel * np.abs(x0))
    keep = np.abs(G).sum(0) > 0
    return AffineSet(x0, G[:, keep], dict(rel=rel, power_factor=power_factor))


@dataclass
class DispatchReport:
    """Result of :func:`verify_dispatch`."""
    verdict: str                      # SAFE | UNSAFE (a sampled load violates a limit) | NOT PROVEN
    certificate: object               # certflow.power.Certificate (bands per constraint, leaves)
    verifier: PFVerifier
    check: dict                       # Monte-Carlo cross-check
    restored: list = field(default_factory=list)   # RestoreResult per box (restore=True)
    seconds: float = 0.0

    @property
    def safe(self):
        return self.verdict == "SAFE"

    def summary(self) -> str:
        C, g = self.certificate, self.verifier.grid
        L = [f"DISPATCH CONTROLLER AUDIT  ->  {self.verdict}",
             f"  load set: {self.verifier.U.n // 2} loads, {self.verifier.U.K} uncertain directions; "
             f"power flow {'certified (unique solution) on' if C.formal else 'NOT certified on'} "
             f"{C.nleaf} box(es), max rho {C.rho:.3f}"]
        if C.formal:
            L.append("  certified range over the whole load set vs limit:")
            for c in C.constraints:
                b = C.bands[c.name]
                if b is None or not b.lo.size:
                    continue
                ok = c.ok(b)
                if c.name == "branch":
                    ld = np.sqrt(np.maximum(b.hi, 0) / c.lim_hi) * 100
                    k = int(np.argmax(ld))
                    L.append(f"    branch loading  max {ld[k]:.1f}% (branch {int(c.elements[k])})   limit 100%   "
                             f"{'ok' if ok else 'CAN BE EXCEEDED'}")
                    continue
                unit, f = _UNIT.get(c.name, ("", 1.0))
                f = g.sn if f is None else f
                d = c.describe(b, g.sn)
                worst = "" if ok else f"   worst excess {d['worst'] * f:.4g} {unit} at element {d['element']}"
                L.append(f"    {c.name:<6} {b.lo.min() * f:10.4g} .. {b.hi.max() * f:<10.4g} {unit:<5} limits "
                         f"[{np.min(c.lim_lo) * f:.4g}, {np.max(c.lim_hi) * f:.4g}]   {'ok' if ok else 'CAN BE VIOLATED'}{worst}")
        c = self.check
        if c:
            L.append(f"  Monte-Carlo cross-check ({c['n_converged']}/{c['n']} converged samples): every sample inside the "
                     f"certified ranges: {c['sound']}; limit families violated by a sample: {c.get('n_violating', 0)}")
        if self.restored:
            ok = sum(r.restored for r in self.restored)
            L.append(f"  restoration: {ok}/{len(self.restored)} box(es) made certifiably safe with a constant setpoint "
                     f"correction:")
            for r in self.restored:
                L.append(f"    e in [{np.round(r.lo, 2)}, {np.round(r.hi, 2)}]: {'SAFE' if r.restored else 'not restored'}  "
                         f"dPg = {np.round(r.offset[0], 3)} MW, dVg = {np.round(r.offset[1], 4)} pu")
        L.append(f"  time {self.seconds:.1f}s")
        return "\n".join(L)

    __str__ = summary

    def plot(self, file=None):
        """Per element: certified range vs sampled range vs limits (bus |V|, generator Q, branch loading)."""
        from .plots import plot_dispatch
        return plot_dispatch(self, file)


def verify_dispatch(grid, model, load_set=None, *, load_width=0.10, power_factor="constant", inputs="pd_qd",
                    input_units="MW", outputs=("pg", "vg"), pg_units="MW", pg_gens=None, pg_fixed=None,
                    vg_fixed=None, vg_clip=None, vm_pu=None, constraints=None, restore=False, max_split=4,
                    n_check=300, nproc=1) -> DispatchReport:
    """Certify that a dispatch network keeps every operational limit for every load in a set.

    grid         pandapower net, MATPOWER case (dict or .m path) or certflow Grid
    model        the network (torch module, certflow Sequential)
    load_set     AffineSet over [Pd; Qd] of all loads in MW / MVAr; default: every load within +-load_width
    pg_gens      which gens' Pg the network outputs (bool mask / indices into the grid's gens; default: all)
    vg_clip      (vmin, vmax) saturation applied to the Vg outputs before they reach the grid, if the deployed
                 controller does that
    vm_pu        (vmin, vmax) bus voltage limits (default: the grid's own)
    constraints  list of certflow.power constraints (default: |V|, Pg, Qg, slack P/Q, branch current, angle)
    restore      also compute a certified piecewise-constant setpoint correction where limits can be violated"""
    import time
    t0 = time.time()
    g, Pd0, Qd0, Pg0, Vg0 = _grid_and_nominal(grid)
    ng, nl, sn = g.ng, g.nl, g.sn
    if load_set is None:
        if Pd0 is None:
            raise ValueError("pass load_set (the grid carries no nominal loads)")
        load_set = load_box(Pd0, Qd0, load_width, power_factor)
    if load_set.n != 2 * nl:
        raise ValueError(f"load_set must be over [Pd; Qd] of all {nl} loads ({2 * nl} entries), got {load_set.n}")
    net = as_policy(model)
    if not isinstance(net, Sequential):
        raise TypeError("verify_dispatch needs a Sequential network (torch nn.Sequential or certflow Sequential)")
    # input layout: x = [Pd; Qd] (MW) -> network input
    if isinstance(inputs, str):
        sel = {"pd_qd": np.arange(2 * nl), "pd": np.arange(nl)}[inputs]
    else:
        sel = np.asarray(inputs, int)
    S = np.zeros((len(sel), 2 * nl))
    S[np.arange(len(sel)), sel] = 1.0 / (sn if input_units == "pu" else 1.0)
    if net.in_dim is not None and net.in_dim != len(sel):
        raise ValueError(f"the network takes {net.in_dim} inputs; inputs={inputs!r} gives {len(sel)}")
    net = with_input_layers(net, [Linear(S)])
    # output layout: network output -> [Pg of all gens (MW); Vg of all gens (pu)]; gens whose Pg is not an output
    # (or all, if "pg" is not an output block) keep their setpoint through the bias
    pg_set = np.zeros(ng) if Pg0 is None and pg_fixed is None else np.asarray(Pg0 if pg_fixed is None else pg_fixed, float)
    if pg_gens is None:
        out_g = np.arange(ng)
    else:
        pgm = np.asarray(pg_gens)
        out_g = np.where(pgm)[0] if pgm.dtype == bool else pgm.astype(int)
    sizes = {"pg": len(out_g), "vg": ng}
    out_dim = sum(sizes[o] for o in outputs)
    if net.out_dim is not None and net.out_dim != out_dim:
        raise ValueError(f"the network has {net.out_dim} outputs; outputs={outputs!r} expects {out_dim}")
    W = np.zeros((2 * ng, out_dim))
    b = np.r_[pg_set, np.zeros(ng)]
    k = 0
    for o in outputs:
        if o == "pg":
            W[out_g, k + np.arange(len(out_g))] = sn if pg_units == "pu" else 1.0
            b[out_g] = 0.0
        elif o == "vg":
            W[ng + np.arange(ng), k + np.arange(ng)] = 1.0
        else:
            raise ValueError(f"unknown output block {o!r} (use 'pg' and/or 'vg')")
        k += sizes[o]
    if "vg" not in outputs:
        b[ng:] = np.asarray(Vg0 if vg_fixed is None else vg_fixed, float)
    net = with_output_layers(net, [Linear(W, b)])
    ctrl = DispatchController(net, ng, vg_clip=vg_clip or (0.0, 2.0), pg_fallback=float(np.mean(pg_set)))
    vmin, vmax = (None, None) if vm_pu is None else vm_pu
    cons = constraints if constraints is not None else default_constraints(g, vmin, vmax)
    ver = PFVerifier(g, ctrl, load_set, cons, options=VerifierOptions(centre_iters=2))
    if nproc > 1:
        from ..split import ObjectPool
        with ObjectPool(ver, nproc) as pool:
            cert = ver.certify(pool=pool)
    else:
        cert = ver.certify()
    mc = monte_carlo(ver, cert, n=n_check) if n_check else {}
    n_bad = 0
    if mc:
        for c in cons:
            emp = mc["empirical"].get(c.name)
            if emp is not None and c.name in cert.bands and cert.bands[c.name] is not None:
                n_bad += int(not c.ok(type(cert.bands[c.name])(emp[0], emp[1])))
        mc["n_violating"] = n_bad
    verdict = "SAFE" if cert.safe else ("UNSAFE" if n_bad else "NOT PROVEN")
    rep = DispatchReport(verdict, cert, ver, mc)
    if restore and not cert.safe:
        _, leaves = Restorer(ver).restore_split(max_split=max_split, nproc=nproc)
        rep.restored = leaves
    rep.seconds = time.time() - t0
    return rep
