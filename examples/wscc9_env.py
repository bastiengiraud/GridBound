"""Certified model of the WSCC-9 RL environment (env_pf_solver_reconciliation.jl + the training script).

Everything environment-specific lives here; the verification machinery is generic certflow:
    certflow.power.enclose_pf / Shunt / quantities   (certified PF with admittance loads, observable TMs)
    certflow.policy.DiscretePolicyVerifier            (reachable actions + per-action checks)

Environment facts reproduced (from the Julia/Python sources):
  * grid: PSS/E "WSCC 9 bus.raw" (eRoots-Analytics/PublicGridCollection, cited by the agent paper), env bus
    numbering; slack bus 1 (|V| = 1.04), PV buses 2, 3
  * loads at buses 5, 6, 8 are FIXED ADMITTANCES  Y = (1 + delta) Y_nom,  Y_nom = conj(S_nom) / |V0|^2 with
    |V0| from the base PF at the raw dispatch -> the exogenous uncertainty is delta in R^3
  * observation (26): [line status 6 | Pg slack, 2, 3 (MW) | load P (3) | load Q (3) | |V| bus 2, 3 |
                       line loading % (6, of 250 MVA) | transformer loading % (3, of 300 MVA)]
    loads observed as conj(Y)|V|^2 (they move with the voltage), loading = max(|S_from|, |S_to|) incl. shunt
  * actions (73): idle | gen 2/3 dP in +-5..50 MW | V setpoint 2/3 in +-0.01..0.05 | open/close each line
  * mask: dP keeps Pg in [0, Pmax], dV keeps V in [0.95, 1.05], open only closed lines / close only open ones
  * a step applies the action and the next load deviation simultaneously; the episode fails if the PF fails;
    N-0 overloads (S > rating) and slack outside [0, 250] MW are penalised.  The verification does not use the
    reward: it checks the grid limits (see StepChecker) at N-0 and after every line outage.

Network data: the PSS/E file of the training environment, wscc9_maskableppo_policy_extraction/WSCC_9_bus.raw (identical
to eRoots-Analytics/PublicGridCollection psse/"WSCC 9 bus.raw" up to the slack's initial dispatch, which the power
flow overwrites).  It differs from the textbook Anderson data on lines 5-4, 9-6, 7-8.  The values used below are
checked against that file by ``read_raw`` (tests/test_wscc9_data.py), and the model is checked against the real
observation stored with the trained model (``infer_state_from_obs`` + ``simulate``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np

from certflow.policy import ActionCheck, Observation
from certflow.power import Shunt, enclose_pf, enclose_pf_split, from_matpower, n1_analysis, point_pf, quantities as qty
from certflow.power.pf import inj_pt
from certflow.sets import AffineSet
from certflow.tm import TM, add, concat, const, pad_symbols, scale

SN = 100.0
LINES = [(5, 4), (6, 4), (7, 5), (9, 6), (7, 8), (8, 9)]
XFMRS = [(4, 1), (2, 7), (9, 3)]
LOAD_BUSES = [5, 6, 8]
S_NOM = np.array([125 + 50j, 90 + 30j, 100 + 35j])          # MVA
LINE_RATING, XFMR_RATING = 250.0, 300.0
PMIN, PMAX = {2: 0.0, 3: 0.0}, {2: 300.0, 3: 270.0}
VMIN_SET, VMAX_SET = 0.95, 1.05                              # PV setpoint band (action mask)
VMIN_BUS, VMAX_BUS = 0.95, 1.05                              # |V| limits checked at EVERY bus, N-0 and N-1
SLACK_MIN, SLACK_MAX = 0.0, 250.0
DP_LEVELS = [-50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0,
             5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
DV_LEVELS = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
_BR = {  # (r, x, b) in pu, from WSCC_9_bus.raw (see read_raw)
    (5, 4): (0.010, 0.068, 0.176), (6, 4): (0.017, 0.092, 0.158), (7, 5): (0.032, 0.161, 0.306),
    (9, 6): (0.039, 0.1738, 0.358), (7, 8): (0.0085, 0.0576, 0.149), (8, 9): (0.0119, 0.1008, 0.209),
    (4, 1): (0.0, 0.0576, 0.0), (2, 7): (0.0, 0.0625, 0.0), (9, 3): (0.0, 0.0586, 0.0)}


RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wscc9_maskableppo_policy_extraction",
                        "WSCC_9_bus.raw")


def read_raw(path=RAW_PATH):
    """Minimal PSS/E v33 reader for what this model uses: branch/transformer (r, x, b) keyed (from, to) as in the
    file, loads (MVA), generator P (MW) and bus |V| setpoints.  Used to check the values hard-coded above."""
    with open(path) as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    sec, out = "bus", dict(branch={}, load={}, pg={}, vm={})
    i = 3
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("0 /") or ln.strip() == "Q":
            low = ln.lower()
            sec = next((k for k in ("load", "fixed shunt", "generator", "branch", "transformer") if f"begin {k}" in low), None)
            i += 1
            continue
        f = [x.strip().strip("'") for x in ln.split(",")]
        if sec == "bus":
            out["vm"][int(f[0])] = float(f[7])
        elif sec == "load":
            out["load"][int(f[0])] = complex(float(f[5]), float(f[6]))
        elif sec == "generator":
            out["pg"][int(f[0])] = float(f[2])
        elif sec == "branch":
            out["branch"][(int(f[0]), int(f[1]))] = (float(f[3]), float(f[4]), float(f[5]))
        elif sec == "transformer":                 # 2-winding: 4 records; r, x on record 2
            r2 = [float(x) for x in lines[i + 1].split(",")[:2]]
            out["branch"][(int(f[0]), int(f[1]))] = (r2[0], r2[1], 0.0)
            i += 3
        i += 1
    return out


def action_list():
    acts = [("idle",)]
    for bus in (2, 3):
        acts += [("gen", bus, dp) for dp in DP_LEVELS]
    for bus in (2, 3):
        acts += [("vset", bus, dv) for dv in DV_LEVELS]
    for (i, j) in LINES:
        acts += [("line", i, j, False), ("line", i, j, True)]
    return acts


ACTIONS = action_list()


def _mpc(load_as_power):
    bus = []
    for b in range(1, 10):
        typ = 3 if b == 1 else (2 if b in (2, 3) else 1)
        Pd = Qd = 0.0
        if load_as_power and b in LOAD_BUSES:
            s = S_NOM[LOAD_BUSES.index(b)]
            Pd, Qd = s.real, s.imag
        bus.append([b, typ, Pd, Qd, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9])
    gen = [[1, 0, 0, 300, -300, 1.04, 100, 1, 1e4, -1e4],
           [2, 163, 0, 300, -300, 1.025, 100, 1, PMAX[2], PMIN[2]],
           [3, 85, 0, 300, -300, 1.025, 100, 1, PMAX[3], PMIN[3]]]
    br = [[i, j, *_BR[(i, j)], 0, 0, 0, 0, 0, 1, -360, 360] for (i, j) in LINES + XFMRS]
    return dict(baseMVA=SN, bus=np.array(bus, float), gen=np.array(gen, float), branch=np.array(br, float))


@dataclass(frozen=True)
class OperatingState:
    """The controllable/discrete part of an environment state (the loads are the uncertain part)."""
    status: tuple = (1, 1, 1, 1, 1, 1)       # line status in LINES order
    pg: tuple = (163.0, 85.0)                # MW at buses 2, 3 (setpoints)
    vg: tuple = (1.025, 1.025)               # pu at buses 2, 3 (setpoints)

    def apply(self, a):
        act = ACTIONS[a]
        if act[0] == "gen":
            pg = list(self.pg)
            pg[act[1] - 2] += act[2]
            return replace(self, pg=tuple(pg))
        if act[0] == "vset":
            vg = list(self.vg)
            vg[act[1] - 2] = round(vg[act[1] - 2] + act[2], 10)
            return replace(self, vg=tuple(vg))
        if act[0] == "line":
            st = list(self.status)
            st[LINES.index((act[1], act[2]))] = int(act[3])
            return replace(self, status=tuple(st))
        return self

    def mask(self):
        m = np.zeros(len(ACTIONS), bool)
        m[0] = True
        k = 1
        for bus in (2, 3):
            for dp in DP_LEVELS:
                m[k] = PMIN[bus] <= self.pg[bus - 2] + dp <= PMAX[bus]
                k += 1
        for bus in (2, 3):
            for dv in DV_LEVELS:
                m[k] = VMIN_SET <= self.vg[bus - 2] + dv <= VMAX_SET
                k += 1
        for li in range(len(LINES)):
            m[k] = self.status[li] == 1
            m[k + 1] = self.status[li] == 0
            k += 2
        return m


class WSCC9Model:
    """Certified observation and transition model of the environment."""

    def __init__(self):
        base = from_matpower(_mpc(load_as_power=True))                   # base PF with constant-power loads
        Vpin = np.zeros(base.N)
        Vpin[base.pv] = base.names["gen_vg"]
        Pd, Qd = base.names["load_pq"]
        Psp = (base.Cg @ base.names["gen_pg"] - base.Cl @ Pd) / SN
        Qsp = -(base.Cl @ Qd) / SN
        th, Vm, mis = point_pf(base, Psp, Qsp, Vpin, (np.angle(base.V0), np.abs(base.V0)))
        assert mis < 1e-10
        self.lb = np.array(LOAD_BUSES) - 1                                # internal index = bus - 1
        self.V0 = Vm[self.lb]
        self.y_nom = np.conj(S_NOM / SN) / self.V0 ** 2                  # nominal load admittance (pu)
        mpc = _mpc(load_as_power=False)
        mpc["bus"][self.lb, 4] = self.y_nom.real * SN                     # nominal admittance as bus shunt
        mpc["bus"][self.lb, 5] = self.y_nom.imag * SN
        self.grid0 = from_matpower(mpc)
        self.slk = self.grid0.slk
        self.line_idx = np.arange(len(LINES))
        self.xfmr_idx = len(LINES) + np.arange(len(XFMRS))
        self._grids = {}

    # ------------------------------------------------------------------ helpers
    def grid(self, status):
        key = tuple(int(s) for s in status)
        if key not in self._grids:
            st = np.r_[np.array(key, bool), np.ones(len(XFMRS), bool)]
            self._grids[key] = self.grid0.with_branch_status(st)
        return self._grids[key]

    def specs(self, g, st, delta: TM):
        """Specification dict for certflow.power.enclose_pf(_split) on grid g (any topology) for load TM delta."""
        _, P, Q, Vs = self._specs(st, delta.C.shape[1], delta.Q.shape[-1], g)
        return dict(P=P, Q=Q, V=Vs, shunt=Shunt(self.lb, self.y_nom, delta), warm=(np.angle(g.V0), np.abs(g.V0)))

    def _specs(self, st, K, kq, g=None):
        g = self.grid(st.status) if g is None else g
        P = np.zeros(g.N)
        P[[1, 2]] = np.array(st.pg) / SN                                  # buses 2, 3
        z = lambda n: TM(np.zeros(n), np.zeros((n, K)), np.zeros((n, kq, kq)), np.zeros(n))
        Pt = z(g.N)
        Pt = TM(P, Pt.C, Pt.Q, Pt.R)
        Vs = z(2)
        Vs = TM(np.array(st.vg, float), Vs.C, Vs.Q, Vs.R)
        return g, Pt, z(g.N), Vs

    def enclose(self, st: OperatingState, delta: TM, warm=None):
        """Certified PF of state ``st`` for every load deviation in the TM ``delta`` (3 rows)."""
        K, kq = delta.C.shape[1], delta.Q.shape[-1]
        g, P, Q, Vs = self._specs(st, K, kq)
        if g.islanded():
            return None
        warm = warm or (np.angle(g.V0), np.abs(g.V0))          # slack |V| = 1.04 comes from V0
        return enclose_pf(g, P, Q, Vs, warm, shunt=Shunt(self.lb, self.y_nom, delta))

    def obs_tm(self, st: OperatingState, delta: TM, enc):
        """The 26-entry observation as a Taylor model (symbols shared with delta)."""
        g = self.grid(st.status)
        K, kq = enc.Vr.C.shape[1], enc.Vr.Q.shape[-1]
        cst = lambda v: TM(np.asarray(v, float), np.zeros((len(v), K)), np.zeros((len(v), kq, kq)), np.zeros(len(v)))
        Ps = scale(np.array([SN]), enc.P.take([self.slk]))
        one_plus = add(const(np.ones(3), delta), delta)
        Pl, Ql = qty.admittance_power(enc, self.lb, self.y_nom, one_plus)
        on = [i for i in self.line_idx if st.status[i]]
        line = cst(np.zeros(len(LINES)))
        parts = []
        if on:
            Sl = scale(np.full(len(on), SN / LINE_RATING * 100), qty.branch_apparent_power(g, enc, on))
            parts.append((on, Sl))
        Sx = scale(np.full(3, SN / XFMR_RATING * 100), qty.branch_apparent_power(g, enc, self.xfmr_idx))
        rows = [cst(np.array(st.status, float)), Ps, cst(st.pg), scale(np.full(3, SN), Pl), scale(np.full(3, SN), Ql),
                cst(st.vg)]
        if on:
            Kmax = max(Sl.C.shape[1], Sx.C.shape[1], K)
            kqm = max(Sl.Q.shape[-1], Sx.Q.shape[-1], kq)
            line = pad_symbols(line, Kmax, kqm)
            Sl = pad_symbols(Sl, Kmax, kqm)
            for c in range(4):
                line[c][on] = Sl[c]
        return concat(rows + [line, Sx])

    # ------------------------------------------------------------------ point model (for validation)
    def simulate(self, st: OperatingState, delta, warm=None):
        """Point PF + observation (float64) for load deviation ``delta`` (3,).  None if the PF fails."""
        g = self.grid(st.status)
        if g.islanded():
            return None
        sh = Shunt(self.lb, self.y_nom, TM(np.zeros(3), np.zeros((3, 1)), np.zeros((3, 1, 1)), np.zeros(3)))
        P = np.zeros(g.N)
        P[[1, 2]] = np.array(st.pg) / SN
        Vpin = np.zeros(g.N)
        Vpin[[1, 2]] = st.vg
        warm = warm or (np.angle(g.V0), np.abs(g.V0))
        th, Vm, mis = point_pf(g, P, np.zeros(g.N), Vpin, warm, shunt_d=np.asarray(delta, float), shunt=sh)
        if mis > 1e-9:
            return None
        V = Vm * np.exp(1j * th)
        Sbr = np.maximum(np.abs(V[g.f] * np.conj(g.Yf @ V)), np.abs(V[g.t] * np.conj(g.Yt @ V))) * SN
        Gd, Bd = sh.effective_GB(g.G, g.B, np.asarray(delta, float))
        Pi, _ = inj_pt(Gd, Bd, th, Vm)
        Sl = np.conj(self.y_nom * (1 + np.asarray(delta))) * Vm[self.lb] ** 2 * SN
        line = np.where(np.array(st.status, bool), Sbr[self.line_idx] / LINE_RATING * 100, 0.0)
        obs = np.r_[st.status, Pi[self.slk] * SN, st.pg, Sl.real, Sl.imag, st.vg, line,
                    Sbr[self.xfmr_idx] / XFMR_RATING * 100]
        return dict(obs=obs, Vm=Vm, th=th, slack=Pi[self.slk] * SN,
                    loading=np.r_[line, Sbr[self.xfmr_idx] / XFMR_RATING * 100])


@dataclass
class StepChecker:
    """Certified one-step safety of an action: post-action PF over the next load deviations.

    Violations (the same limits at N-0 and, with ``n1``, after every line outage):
      * the PF has no (unique) solution, e.g. the outage islands the grid
      * a line / transformer loading max(|S_from|, |S_to|) above its rating (250 / 300 MVA)
      * |V| outside [VMIN_BUS, VMAX_BUS] at ANY bus
      * slack active power outside [SLACK_MIN, SLACK_MAX] MW

    margin     : the next deviation delta' may differ from the observed delta by up to +-margin per load
                 (0 = loads frozen during the step)
    n1         : also screen the post-action state for every line outage (certflow.power.n1_analysis), reported as
                 certified [lower, upper] bounds on the number of overloads, PF failures, (outage, bus) voltage
                 violations and outages with a slack violation (and, for comparison with the training environment,
                 its N-1 penalty count = overloads + 3 x PF failures)
    require_n1 : count a possible N-1 violation of any kind as unsafe"""
    model: WSCC9Model
    margin: float = 0.0
    max_leaves: int = 64
    n1: bool = False
    require_n1: bool = False
    _cache: dict = field(default_factory=dict)

    def certify(self, st: OperatingState, dlo, dhi):
        """Certified post-state bands over the box delta in [dlo, dhi] (with box splitting if needed)."""
        return self.certify_set(st, AffineSet(0.5 * (dlo + dhi), np.diag(0.5 * (dhi - dlo))))

    def certify_set(self, st: OperatingState, U: AffineSet):
        """Certified post-state bands over the affine load set  delta = U.c + U.G e,  e in [-1, 1]^U.K  (any number
        of symbols, e.g. the history-tracked uncertainty of a multi-step tree); split if needed."""
        key = (st, tuple(np.round(U.c, 12)), tuple(np.round(U.G, 12).ravel()), U.G.shape, self.n1)
        if key in self._cache:
            return self._cache[key]
        m = self.model
        g = m.grid(st.status)
        if g.islanded():
            res = dict(formal=True, pf=False, reason="islanded")
            self._cache[key] = res
            return res
        K = U.K
        spec_fn = lambda gg, a, b: m.specs(gg, st, U.sub_tm(0.5 * (a + b), 0.5 * (b - a)))
        on = [i for i in m.line_idx if st.status[i]]

        def reduce(enc, a, b):
            load = np.zeros(len(LINES) + len(XFMRS))
            load[on] = qty.branch_apparent_power(g, enc, on).bounds()[1] * SN / LINE_RATING * 100
            load[m.xfmr_idx] = qty.branch_apparent_power(g, enc, m.xfmr_idx).bounds()[1] * SN / XFMR_RATING * 100
            ps = enc.P.take([m.slk]).bounds()
            vm = qty.vmag(enc, list(range(g.N))).bounds()
            return dict(load=load, slack=(ps[0][0] * SN, ps[1][0] * SN), vlo=vm[0], vhi=vm[1], rho=enc.rho)

        leaves, ok = enclose_pf_split(g, spec_fn, -np.ones(K), np.ones(K), reduce=reduce, max_leaves=self.max_leaves)
        if not ok or not leaves:
            res = dict(formal=False, pf=None, reason="PF not certified")
        else:
            res = dict(formal=True, pf=True, nleaf=len(leaves), load=np.max([L["load"] for L in leaves], 0),
                       slack=(min(L["slack"][0] for L in leaves), max(L["slack"][1] for L in leaves)),
                       vlo=np.min([L["vlo"] for L in leaves], 0), vhi=np.max([L["vhi"] for L in leaves], 0),
                       rho=max(L["rho"] for L in leaves))
            res.update(vmin=float(res["vlo"].min()), vmax=float(res["vhi"].max()))
            if self.n1:
                rating = np.r_[np.full(len(LINES), LINE_RATING), np.full(len(XFMRS), XFMR_RATING)]
                rep = n1_analysis(g, spec_fn, -np.ones(K), np.ones(K), outages=on, rating_mva=rating,
                                  max_leaves=self.max_leaves, vlim=(VMIN_BUS, VMAX_BUS),
                                  slack=(m.slk, SLACK_MIN, SLACK_MAX))
                res.update(n1_over=rep.count_bounds(kinds=("overload",)),            # overloaded (outage, branch) pairs
                           n1_fail=rep.count_bounds(fail_weight=1, kinds=()),        # outages without a PF solution
                           n1=rep.count_bounds(fail_weight=3),                       # training env's metric (overloads + 3 x fails)
                           n1_v=rep.count_bounds(kinds=("voltage",)),                # (outage, bus) |V| violations
                           n1_s=rep.count_bounds(kinds=("slack",)),                  # outages with a slack violation
                           n1_secure=rep.secure, n1_report=rep, n1_worst=rep.worst_loading(),
                           n1_vrange=rep.voltage_range(), n1_srange=rep.slack_range())
        self._cache[key] = res
        return res

    def __call__(self, st, a, dlo, dhi):
        """ActionCheck for taking action a in state st when delta in [dlo, dhi] (next load: the box +- margin)."""
        return self._verdict(self.certify(st.apply(a), dlo - self.margin, dhi + self.margin))

    def check_set(self, st, a, U_next: AffineSet):
        """ActionCheck for taking action a in state st when the NEXT load deviation lies in the affine set U_next."""
        return self._verdict(self.certify_set(st.apply(a), U_next))

    def _verdict(self, res):
        if not res["formal"]:
            return ActionCheck(False, False, res)
        if not res["pf"]:
            return ActionCheck(False, True, res)                       # certain failure (islanding)
        over = res["load"].max() > 100.0
        slack_bad = res["slack"][0] < SLACK_MIN or res["slack"][1] > SLACK_MAX
        volt_bad = res["vmin"] < VMIN_BUS or res["vmax"] > VMAX_BUS
        n1_bad = bool(self.require_n1 and self.n1 and not res["n1_secure"])
        res = dict(res, overload=bool(over), slack_violation=bool(slack_bad), voltage_violation=bool(volt_bad),
                   n1_violation=n1_bad)
        return ActionCheck(not (over or slack_bad or volt_bad or n1_bad), True, res)


def make_problem(model: WSCC9Model, st: OperatingState, delta_c, delta_w, checker: StepChecker = None):
    """Callbacks for certflow.policy.DiscretePolicyVerifier over delta in delta_c +- delta_w (e in [-1,1]^3)."""
    delta_c = np.asarray(delta_c, float) * np.ones(3)
    delta_w = np.asarray(delta_w, float) * np.ones(3)
    U = AffineSet(delta_c, np.diag(delta_w))
    mask = st.mask()

    def observe(lo, hi):
        d = U.sub_tm(0.5 * (lo + hi), 0.5 * (hi - lo))
        enc = model.enclose(st, d)
        if enc is None or not enc.ok:
            return Observation(False, score=(enc.score[:3] if enc is not None and enc.score is not None else None))
        return Observation(True, model.obs_tm(st, d, enc), dict(rho=enc.rho))

    def check(a, lo, hi, obs):
        return checker(st, a, U.point(lo), U.point(hi))

    return U, observe, (lambda lo, hi, obs: mask), (check if checker is not None else None)


def multistep_problem(model: WSCC9Model, checker: StepChecker):
    """Callbacks for certflow.policy.MultiStepVerifier: problem(state, lo, hi), transition, propagate.
    The next load box is the current one widened by the checker's margin (the same assumption the one-step
    safety check uses), so step k+1 starts exactly where step k's safety certificate ends."""
    def problem(st, lo, hi):
        U, observe, mask, check = make_problem(model, st, 0.5 * (lo + hi), 0.5 * (hi - lo), checker)
        return dict(observe=observe, mask=mask, check=check, K=3)

    transition = lambda st, a: st.apply(a)
    propagate = lambda lo, hi: (lo - checker.margin, hi + checker.margin)
    return problem, transition, propagate


def step_generator(margin, kind="independent"):
    """Generator matrix of ONE step's load change  delta_{t+1} = delta_t + G_step w,  w in [-1, 1]^r:
    "independent": every load moves by up to +-margin on its own (G_step = margin * I, r = 3);
    "common":      all loads move together by the same amount (G_step = margin * 1, r = 1)."""
    if kind == "independent":
        return float(margin) * np.eye(3)
    if kind == "common":
        return float(margin) * np.ones((3, 1))
    raise ValueError(f"unknown step-change model {kind!r}")


def make_problem_history(model: WSCC9Model, st: OperatingState, U: AffineSet, checker: StepChecker, step_gen):
    """Callbacks for DiscretePolicyVerifier over the affine load set  delta = U.c + U.G e  (any number of symbols:
    the initial box symbols plus one block of new symbols per earlier step).  The safety check certifies the next
    state over the next load  delta' = delta(e) + step_gen w,  i.e. over the sub-box with the new symbols appended
    (no re-boxing)."""
    mask = st.mask()
    K = U.K

    def observe(lo, hi):
        d = U.sub_tm(0.5 * (lo + hi), 0.5 * (hi - lo))
        enc = model.enclose(st, d)
        if enc is None or not enc.ok:
            return Observation(False, score=(enc.score[:K] if enc is not None and enc.score is not None else None))
        return Observation(True, model.obs_tm(st, d, enc), dict(rho=enc.rho))

    def check(a, lo, hi, obs):
        ec, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        U_next = AffineSet(U.c + U.G @ ec, np.hstack([U.G * h[None, :], step_gen]))
        return checker.check_set(st, a, U_next)

    return dict(observe=observe, mask=(lambda lo, hi, obs: mask), check=check)


def multistep_problem_history(model: WSCC9Model, checker: StepChecker, step_gen):
    """Callbacks for certflow.policy.MultiStepVerifier(track="history"): problem(state, U), transition, extend.
    Each step appends the columns of ``step_gen`` (new symbols) to the load set instead of re-boxing it."""
    problem = lambda st, U: make_problem_history(model, st, U, checker, step_gen)
    transition = lambda st, a: st.apply(a)
    extend = lambda U: AffineSet(U.c, np.hstack([U.G, step_gen]))
    return problem, transition, extend


def infer_state_from_obs(model: WSCC9Model, obs):
    """Recover (OperatingState, delta) from a real observation (loads are conj(Y)|V|^2 -> fixed point)."""
    st = OperatingState(tuple(int(round(v)) for v in obs[:6]), (round(float(obs[7]), 3), round(float(obs[8]), 3)),
                        (round(float(obs[15]), 4), round(float(obs[16]), 4)))      # float32 obs -> setpoint lattice
    P_obs = np.asarray(obs[9:12], float)
    d = np.zeros(3)
    for _ in range(50):
        sim = model.simulate(st, d)
        d_new = P_obs / (S_NOM.real * sim["Vm"][model.lb] ** 2 / model.V0 ** 2) - 1
        if np.abs(d_new - d).max() < 1e-12:
            break
        d = d_new
    return st, d
