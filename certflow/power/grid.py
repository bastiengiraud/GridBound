"""Grid model: everything the power-flow verifier needs, independent of pandapower.

Branches are stored as their 2x2 admittance primitives (Yff, Yft, Ytf, Ytt) plus an in-service flag, and bus
shunts as Ysh, so the bus admittance matrix is

    Ybus = sum_{l in service} [Yff Yft; Ytf Ytt]_l  (scattered on (f_l, t_l))  + diag(Ysh).

This makes TOPOLOGY CHANGES (line switching, e.g. RL actions) a cheap rebuild: ``grid.with_branch_status(s)``.
Any branch model (lines, transformers with taps/phase shift) is covered because only the primitives are used.

Bus indices are 0..N-1 in the grid's own (pandapower *internal*) order; generators/loads carry the bus index.
Powers in MW/MVAr unless noted; ``sn`` (baseMVA) converts to per unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import scipy.sparse as sp


@dataclass(eq=False)
class Grid:
    sn: float                         # base MVA
    bus_type: np.ndarray              # (N,) 1 = PQ, 2 = PV, 3 = slack
    Ysh: np.ndarray                   # (N,) complex bus shunt admittance (pu)
    f: np.ndarray                     # (L,) from-bus index
    t: np.ndarray                     # (L,) to-bus index
    Yff: np.ndarray                   # (L,) complex branch primitives (pu)
    Yft: np.ndarray
    Ytf: np.ndarray
    Ytt: np.ndarray
    status: np.ndarray                # (L,) bool in service
    rate_a: np.ndarray                # (L,) MVA thermal rating (0 / inf = unlimited)
    angmin: np.ndarray                # (L,) deg
    angmax: np.ndarray                # (L,) deg
    gen_bus: np.ndarray               # (ng,) bus index of each controllable generator (PV)
    gen_pmin: np.ndarray              # (ng,) MW
    gen_pmax: np.ndarray
    gen_qmin: np.ndarray              # (ng,) MVAr
    gen_qmax: np.ndarray
    load_bus: np.ndarray              # (nl,) bus index of each load
    sgen_p: np.ndarray                # (N,) fixed injections (static generators), MW
    sgen_q: np.ndarray                # (N,) MVAr
    slack_lims: tuple = (-1e9, 1e9, -1e9, 1e9)   # slack (pmin, pmax, qmin, qmax), MW/MVAr
    vmin: np.ndarray = None           # (N,) pu
    vmax: np.ndarray = None
    V0: np.ndarray = None             # (N,) complex voltage (warm start / operating point), optional
    gen_cost: np.ndarray = None       # (ng, 3) c2, c1, c0 on MW (optional)
    slack_cost: tuple = None          # (c2, c1, c0) (optional)
    names: dict = field(default_factory=dict)
    _Y_exact: object = None           # optional externally computed Ybus (pandapower) for the base topology

    # ---------------------------------------------------------------- sizes
    @property
    def N(self):
        return len(self.bus_type)

    @property
    def L(self):
        return len(self.f)

    @property
    def ng(self):
        return len(self.gen_bus)

    @property
    def nl(self):
        return len(self.load_bus)

    # ---------------------------------------------------------------- bus classes (internal order)
    @cached_property
    def slk(self) -> int:
        return int(np.where(self.bus_type == 3)[0][0])

    @cached_property
    def pv(self) -> list:
        return list(np.where(self.bus_type == 2)[0])

    @cached_property
    def pq(self) -> list:
        return list(np.where(self.bus_type == 1)[0])

    @cached_property
    def ns(self) -> list:
        return self.pv + self.pq

    # ---------------------------------------------------------------- admittances
    def _Cft(self):
        on = np.where(self.status)[0]
        Cf = sp.csr_matrix((np.ones(len(on)), (np.arange(len(on)), self.f[on])), shape=(len(on), self.N))
        Ct = sp.csr_matrix((np.ones(len(on)), (np.arange(len(on)), self.t[on])), shape=(len(on), self.N))
        return on, Cf, Ct

    @cached_property
    def Yf(self):
        """(L, N) sparse: I_from = Yf V (zero rows for out-of-service branches)."""
        L, N = self.L, self.N
        s = self.status.astype(float)
        r = np.r_[np.arange(L), np.arange(L)]
        c = np.r_[self.f, self.t]
        return sp.csr_matrix((np.r_[self.Yff * s, self.Yft * s], (r, c)), shape=(L, N))

    @cached_property
    def Yt(self):
        L, N = self.L, self.N
        s = self.status.astype(float)
        r = np.r_[np.arange(L), np.arange(L)]
        c = np.r_[self.f, self.t]
        return sp.csr_matrix((np.r_[self.Ytf * s, self.Ytt * s], (r, c)), shape=(L, N))

    @cached_property
    def Ybus(self) -> np.ndarray:
        """Dense complex bus admittance matrix."""
        if self._Y_exact is not None:
            Y = self._Y_exact
            return np.asarray(Y.todense() if hasattr(Y, "todense") else Y).astype(complex)
        N = self.N
        Cf = sp.csr_matrix((np.ones(self.L), (np.arange(self.L), self.f)), shape=(self.L, N))
        Ct = sp.csr_matrix((np.ones(self.L), (np.arange(self.L), self.t)), shape=(self.L, N))
        Y = Cf.T @ self.Yf + Ct.T @ self.Yt + sp.diags(self.Ysh)
        return np.asarray(Y.todense()).astype(complex)

    @cached_property
    def G(self) -> np.ndarray:
        return self.Ybus.real.copy()

    @cached_property
    def B(self) -> np.ndarray:
        return self.Ybus.imag.copy()

    @cached_property
    def branch_blocks(self):
        """Sparse real/imag blocks (Gf, Bf, Gt, Bt) of Yf, Yt (for the branch-current Taylor models)."""
        return (sp.csr_matrix(self.Yf.real), sp.csr_matrix(self.Yf.imag),
                sp.csr_matrix(self.Yt.real), sp.csr_matrix(self.Yt.imag))

    @cached_property
    def rate2(self):
        """Squared current limit (RATE_A / sn)^2 per branch (+inf if unlimited or out of service)."""
        r = np.asarray(self.rate_a, float)
        ok = (r > 1e-9) & np.isfinite(r) & self.status
        return np.where(ok, (np.where(ok, r, 1.0) / self.sn) ** 2, np.inf)

    @cached_property
    def angle_active(self):
        lo, hi = np.asarray(self.angmin, float), np.asarray(self.angmax, float)
        inactive = ((lo <= -359.9) & (hi >= 359.9)) | ((np.abs(lo) < 1e-9) & (np.abs(hi) < 1e-9))
        return ~inactive & self.status

    # ---------------------------------------------------------------- incidences
    @cached_property
    def Cg(self):
        """(N, ng) generator incidence (sum: powers add)."""
        C = np.zeros((self.N, self.ng))
        C[self.gen_bus, np.arange(self.ng)] = 1.0
        return C

    @cached_property
    def Cgv(self):
        """(N, ng) row-normalised generator incidence (voltage setpoint = average of the bus's gens)."""
        return self.Cg / np.maximum(self.Cg.sum(1, keepdims=True), 1)

    @cached_property
    def Cl(self):
        """(N, nl) load incidence."""
        C = np.zeros((self.N, self.nl))
        C[self.load_bus, np.arange(self.nl)] = 1.0
        return C

    # ---------------------------------------------------------------- topology changes
    def with_branch_status(self, status) -> "Grid":
        """New grid with the given in-service flags (bus types/gens unchanged).  Ybus, Yf, Yt are rebuilt."""
        status = np.asarray(status, bool)
        if status.shape != self.status.shape:
            raise ValueError("status must have one flag per branch")
        same = bool((status == self.status).all())
        return replace(self, status=status.copy(), _Y_exact=(self._Y_exact if same else None))

    def islanded(self) -> bool:
        """True if the in-service network is not connected."""
        from scipy.sparse.csgraph import connected_components
        on = np.where(self.status)[0]
        A = sp.csr_matrix((np.ones(len(on)), (self.f[on], self.t[on])), shape=(self.N, self.N))
        return connected_components(A, directed=False)[0] > 1


def from_pandapower(net, use_exact_ybus=True) -> Grid:
    """Build a :class:`Grid` from a SOLVED pandapower net (``pp.runpp`` must have run).

    Uses pandapower's internal ppc order.  Generators are the
    in-service ``net.gen`` (PV units; the ext_grid is the slack), loads the in-service ``net.load``, in-service static
    generators ``net.sgen`` are fixed injections.  Branch primitives are read from ppc's Yf/Yt, shunts are the
    remainder of diag(Ybus).  Branch ratings follow pandapower's ``loading_percent`` (see _pandapower_ratings).
    ``names`` holds the lookups: bus_lookup (pandapower bus -> internal index), branch_element (per branch row:
    ("line" | "trafo", index)), gen_index, load_index (pandapower indices of the gens / loads, in grid order).
    Out-of-service branches are not part of pandapower's internal model; to switch them later, build the grid with
    them in service and use :meth:`Grid.with_branch_status`."""
    ppc = net._ppc["internal"]
    Yb = sp.csr_matrix(ppc["Ybus"])
    bus = np.asarray(ppc["bus"])
    br = np.asarray(ppc["branch"])
    N = bus.shape[0]
    L = br.shape[0]
    try:
        Yf = sp.csr_matrix(ppc["Yf"])
        Yt = sp.csr_matrix(ppc["Yt"])
    except Exception:  # pragma: no cover
        from pandapower.pypower.makeYbus import makeYbus
        _, Yf, Yt = makeYbus(float(ppc.get("baseMVA", net.sn_mva)), bus, br)
        Yf, Yt = Yf.tocsr(), Yt.tocsr()
    f = br[:, 0].real.astype(int)
    t = br[:, 1].real.astype(int)
    ar = np.arange(L)
    Yff = np.asarray(Yf[ar, f]).ravel()
    Yft = np.asarray(Yf[ar, t]).ravel()
    Ytf = np.asarray(Yt[ar, f]).ravel()
    Ytt = np.asarray(Yt[ar, t]).ravel()
    dg = Yb.diagonal()
    Ysh = dg - np.bincount(f, weights=Yff.real, minlength=N) - np.bincount(t, weights=Ytt.real, minlength=N) \
        - 1j * (np.bincount(f, weights=Yff.imag, minlength=N) + np.bincount(t, weights=Ytt.imag, minlength=N))
    status = (br[:, 10].real > 0) if br.shape[1] > 10 else np.ones(L, bool)
    rate = br[:, 5].real.astype(float)
    amn = br[:, 11].real.astype(float) if br.shape[1] > 11 else np.full(L, -360.0)
    amx = br[:, 12].real.astype(float) if br.shape[1] > 12 else np.full(L, 360.0)
    bt = bus[:, 1].real.astype(int)
    V0 = bus[:, 7].real * np.exp(1j * np.deg2rad(bus[:, 8].real))
    # element -> internal bus (as in the reference _grid_consts)
    extnum = net._ppc["bus"][:, 0].astype(int)
    intnum = bus[:, 0].real.astype(int)
    num2int = {int(n): i for i, n in enumerate(intnum)}
    bl = net._pd2ppc_lookups["bus"]
    e2i = lambda b: num2int[int(extnum[bl[b]])]
    in_svc = lambda tab: (tab.in_service.values.astype(bool) if "in_service" in tab else np.ones(len(tab), bool))
    g = net.gen[in_svc(net.gen)]                                  # out-of-service units / loads do not exist
    loads = net.load[in_svc(net.load)]
    gb = np.array([e2i(int(b)) for b in g.bus.values], int)
    lb = np.array([e2i(int(b)) for b in loads.bus.values], int)
    Psg = np.zeros(N)
    Qsg = np.zeros(N)
    if hasattr(net, "sgen") and len(net.sgen):
        sg = net.sgen[in_svc(net.sgen)]
        sc = np.nan_to_num(sg.scaling.values, nan=1.0) if "scaling" in sg else np.ones(len(sg))
        for k, b in enumerate(sg.bus.values):
            i = e2i(int(b))
            Psg[i] += float(np.nan_to_num(sg.p_mw.values[k]) * sc[k])
            Qsg[i] += float(np.nan_to_num(sg.q_mvar.values[k]) * sc[k])
    rate, branch_element = _pandapower_ratings(net, br, rate)
    col = lambda c, d: (np.nan_to_num(g[c].values.astype(float), nan=d) if c in g else np.full(len(g), d))
    eg = net.ext_grid

    def _egl(c, d):
        try:
            v = float(eg[c].iloc[0])
            return v if np.isfinite(v) else d
        except Exception:
            return d

    vmin = np.full(N, 0.0)
    vmax = np.full(N, 10.0)
    if "min_vm_pu" in net.bus and "max_vm_pu" in net.bus:
        for b in net.bus.index:
            i = e2i(int(b))
            lo = net.bus.at[b, "min_vm_pu"]
            hi = net.bus.at[b, "max_vm_pu"]
            if np.isfinite(lo):
                vmin[i] = float(lo)
            if np.isfinite(hi):
                vmax[i] = float(hi)
    cost = None
    scost = None
    if "poly_cost" in net and len(net.poly_cost):
        pc = net.poly_cost

        def _c(et, el):
            r = pc[(pc.et == et) & (pc.element == el)]
            return ((float(r.cp2_eur_per_mw2.iloc[0]), float(r.cp1_eur_per_mw.iloc[0]), float(r.cp0_eur.iloc[0]))
                    if len(r) else (0.0, 1.0, 0.0))
        cost = np.array([_c("gen", k) for k in g.index])
        scost = _c("ext_grid", 0)
    bus_lookup = {int(b): e2i(int(b)) for b in net.bus.index[in_svc(net.bus)]}
    return Grid(sn=float(net.sn_mva), bus_type=bt, Ysh=np.asarray(Ysh, complex), f=f, t=t, Yff=Yff, Yft=Yft,
                Ytf=Ytf, Ytt=Ytt, status=np.asarray(status, bool), rate_a=rate, angmin=amn, angmax=amx,
                gen_bus=gb, gen_pmin=col("min_p_mw", -np.inf), gen_pmax=col("max_p_mw", np.inf),
                gen_qmin=col("min_q_mvar", -np.inf), gen_qmax=col("max_q_mvar", np.inf), load_bus=lb,
                sgen_p=Psg, sgen_q=Qsg,
                slack_lims=(_egl("min_p_mw", -1e9), _egl("max_p_mw", 1e9), _egl("min_q_mvar", -1e9),
                            _egl("max_q_mvar", 1e9)),
                vmin=vmin, vmax=vmax, V0=V0, gen_cost=cost, slack_cost=scost,
                names=dict(intnum=intnum.copy(), bus_lookup=bus_lookup, branch_element=branch_element,
                           gen_index=np.asarray(g.index), load_index=np.asarray(loads.index)),
                _Y_exact=(Yb if use_exact_ybus else None))


def _pandapower_ratings(net, br_int, rate_int):
    """Branch ratings in MVA at nominal voltage, computed like pandapower's ``loading_percent``:
    lines  max_i_ka * df * parallel * sqrt(3) * vn_kv(from bus) * max_loading_percent / 100,
    trafos sn_mva * df * parallel * max_loading_percent / 100.
    pandapower itself only writes real ratings into its internal model when ``max_loading_percent`` exists (an
    OPF-ready net); otherwise RATE_A is a 100 MVA placeholder, which would make every loading check meaningless.
    Also returns, per internal branch row, the element it comes from: ("line", index) / ("trafo", index) / (other, -1).
    Falls back to the internal RATE_A (with a warning) if the internal rows cannot be matched to elements."""
    import warnings
    ext = np.asarray(net._ppc["branch"])
    look = net._pd2ppc_lookups.get("branch", {}) or {}
    rate_ext = ext[:, 5].real.astype(float).copy()
    elem = [("other", -1)] * ext.shape[0]
    for tab, (a, b) in look.items():
        df = net[tab]
        if b - a != len(df):
            continue
        ml = (df.max_loading_percent.values.astype(float) if "max_loading_percent" in df else np.full(len(df), 100.0))
        ml = np.nan_to_num(ml, nan=100.0)
        par = df.parallel.values.astype(float) if "parallel" in df else np.ones(len(df))
        dfac = df.df.values.astype(float) if "df" in df else np.ones(len(df))
        if tab == "line":
            vn = net.bus.loc[df.from_bus.values, "vn_kv"].values.astype(float)
            rate_ext[a:b] = ml / 100.0 * df.max_i_ka.values.astype(float) * dfac * par * vn * np.sqrt(3.0)
        elif tab == "trafo":
            rate_ext[a:b] = ml / 100.0 * df.sn_mva.values.astype(float) * dfac * par
        elif "max_loading_percent" not in df:
            rate_ext[a:b] = 0.0                            # no rating information: unmonitored, not 100 MVA
        elem[a:b] = [(tab, int(i)) for i in df.index]
    keep = ext[:, 10].real > 0                             # pandapower's internal model drops out-of-service branches
    if int(keep.sum()) != br_int.shape[0]:
        warnings.warn("from_pandapower: could not match internal branches to elements; using the internal RATE_A "
                      "(a 100 MVA placeholder unless the net has max_loading_percent columns)")
        return rate_int, [("other", -1)] * br_int.shape[0]
    rows = np.where(keep)[0]
    return rate_ext[rows], [elem[r] for r in rows]


# ==================================================================================================
# MATPOWER (no pandapower needed)
# ==================================================================================================

def _mpc_section(txt, name):
    import re
    m = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", txt, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).split("\n"):
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.replace(",", " ").split()])
    return np.array(rows)


def load_matpower(path):
    """Parse a MATPOWER .m case (bus/gen/branch/gencost/baseMVA) into a dict of arrays."""
    import re
    txt = open(path).read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", txt).group(1))
    return dict(baseMVA=base, bus=_mpc_section(txt, "bus"), gen=_mpc_section(txt, "gen"),
                branch=_mpc_section(txt, "branch"), gencost=_mpc_section(txt, "gencost"))


def from_matpower(mpc) -> Grid:
    """Build a :class:`Grid` from MATPOWER data (dict with baseMVA, bus, gen, branch[, gencost]) or a .m path.

    Generators on PV buses become the controllable units (one setpoint per bus: several gens on one bus share
    the bus's Vg); generators on the slack bus define the slack limits.  Every bus with nonzero Pd/Qd gets one
    load.  Bus order = file order.  Returns the grid plus ``names['load_pq']`` = nominal (Pd, Qd) per load and
    ``names['gen_pg'], names['gen_vg']`` = the case's dispatch."""
    if isinstance(mpc, str):
        mpc = load_matpower(mpc)
    base = float(mpc["baseMVA"])
    bus, gen, br = (np.asarray(mpc[k], float) for k in ("bus", "gen", "branch"))
    N = bus.shape[0]
    num2i = {int(n): i for i, n in enumerate(bus[:, 0])}
    bt = bus[:, 1].astype(int).copy()
    on_g = gen[:, 7] > 0
    gbus = np.array([num2i[int(b)] for b in gen[:, 0]], int)
    slk = int(np.where(bt == 3)[0][0])
    ctrl = on_g & (gbus != slk) & (bt[gbus] == 2)
    for i in np.where(bt == 2)[0]:                       # PV bus without an in-service gen -> PQ
        if not (ctrl & (gbus == i)).any():
            bt[i] = 1
    stat = br[:, 10] > 0
    Ys = 1.0 / (br[:, 2] + 1j * br[:, 3])
    Bc = br[:, 4]
    tap = np.where(br[:, 8] != 0, br[:, 8], 1.0) * np.exp(1j * np.pi / 180 * br[:, 9])
    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap
    Ysh = (bus[:, 4] + 1j * bus[:, 5]) / base
    f = np.array([num2i[int(b)] for b in br[:, 0]], int)
    t = np.array([num2i[int(b)] for b in br[:, 1]], int)
    lb = np.where((bus[:, 2] != 0) | (bus[:, 3] != 0))[0]
    gi = np.where(ctrl)[0]
    sg = on_g & (gbus == slk)
    slim = (float(gen[sg, 9].sum()), float(gen[sg, 8].sum()), float(gen[sg, 4].sum()), float(gen[sg, 3].sum()))
    cost = None
    scost = None
    gc = mpc.get("gencost")
    if gc is not None and gc.shape[0] >= gen.shape[0] and int(gc[0, 0]) == 2:
        nc = int(gc[0, 3])
        co = gc[:, 4:4 + nc]
        co = np.pad(co, ((0, 0), (3 - nc, 0))) if nc < 3 else co[:, -3:]
        cost = co[gi]
        scost = tuple(co[np.where(sg)[0][0]]) if sg.any() else None
    angmin = br[:, 11] if br.shape[1] > 11 else np.full(len(br), -360.0)
    angmax = br[:, 12] if br.shape[1] > 12 else np.full(len(br), 360.0)
    V0 = bus[:, 7] * np.exp(1j * np.deg2rad(bus[:, 8]))
    V0[gbus[ctrl | sg]] = gen[ctrl | sg, 5] * np.exp(1j * np.deg2rad(bus[gbus[ctrl | sg], 8]))
    return Grid(sn=base, bus_type=bt, Ysh=Ysh, f=f, t=t, Yff=Yff, Yft=Yft, Ytf=Ytf, Ytt=Ytt, status=stat,
                rate_a=br[:, 5].copy(), angmin=angmin, angmax=angmax, gen_bus=gbus[gi],
                gen_pmin=gen[gi, 9], gen_pmax=gen[gi, 8], gen_qmin=gen[gi, 4], gen_qmax=gen[gi, 3], load_bus=lb,
                sgen_p=np.zeros(N), sgen_q=np.zeros(N), slack_lims=slim, vmin=bus[:, 12].copy(),
                vmax=bus[:, 11].copy(), V0=V0, gen_cost=cost, slack_cost=scost,
                names=dict(bus_numbers=bus[:, 0].astype(int), load_pq=(bus[lb, 2].copy(), bus[lb, 3].copy()),
                           gen_pg=gen[gi, 1].copy(), gen_vg=gen[gi, 5].copy(), gen_rows=gi))
