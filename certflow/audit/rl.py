"""Auditing a discrete-action RL agent on a pandapower grid.

    auditor = RLAuditor(policy, model)              # policy: torch module, SB3 model or certflow Sequential
    rep = auditor.next_step(net, load_width=0.05)   # mode 1: is every action the agent can take safe?
    print(rep.summary())
    nr = auditor.noise_robustness(net)              # mode 2: how much measurement noise until it acts differently /
    print(nr.summary())                             #         picks an unsafe action?

Both modes return certified results (a proof over the whole set, floating-point rounding included) together with a
search for concrete counterexamples, so every verdict is one of
    SAFE          proven for every load (resp. noise) in the set
    UNSAFE        a concrete load (resp. noise vector) was found where the agent really takes a violating action
    NOT PROVEN    the proof failed somewhere but no concrete counterexample was found (the bounds may be loose)
and a Monte-Carlo cross-check of the certified claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nn.argmax import argmax_heads_point, reachable_argmax, reachable_argmax_heads
from ..nn.network import NNReachable, Sequential, from_sb3, from_torch, with_input_layers
from ..policy import ActionCheck, DiscretePolicyVerifier, Observation
from ..robustness import add_noise, robustness_radius
from ..sets import AffineSet
from ..tm import TM
from .grid_model import GridModel, LoadUncertainty


def as_policy(policy) -> NNReachable:
    """A certflow network from a torch module, a loaded stable-baselines3 model, or a certflow network."""
    if isinstance(policy, NNReachable):
        return policy
    if hasattr(policy, "policy") and hasattr(policy, "predict"):
        return from_sb3(policy)
    if hasattr(policy, "parameters"):
        return from_torch(policy)
    raise TypeError("policy must be a certflow NNReachable, a torch nn.Module or a stable-baselines3 model")


def _fmt(v, unit):
    return f"{v:.4f} {unit}" if unit == "pu" else f"{v:.1f} {unit}" if unit == "%" else f"{v:.3g} {unit}"


@dataclass
class ActionResult:
    """Everything about one action the agent can take."""
    action: int
    label: str
    share: float                       # share of the load box where the agent can take it
    safe: bool                         # certified safe (N-0, and N-1 if requested) over the next-load set
    status: str                        # SAFE | CAN-VIOLATE | VIOLATED | UNDECIDED | ISLANDED
    n0: object = None                  # Screen of the post-action grid (intact)
    n1: dict = field(default_factory=dict)    # outage line -> Screen
    counterexample: dict = None        # concrete loads where the agent takes it AND a limit is violated

    def violations(self):
        out = [(None,) + v for v in (self.n0.violations() if self.n0 is not None and self.n0.lo is not None else [])]
        for o, s in self.n1.items():
            if s.islanded:
                out.append((o, "grid splits", np.nan, np.nan, np.nan, np.nan, "", True))
            elif s.lo is not None:
                out += [(o,) + v for v in s.violations()]
        return out


@dataclass
class NextStepReport:
    """Result of :meth:`RLAuditor.next_step`."""
    verdict: str
    actions: list                      # ActionResult per reachable action, by share
    shares: dict                       # SAFE / CAN-VIOLATE / UNDECIDED / UNKNOWN share of the box
    report: object                     # the underlying certflow.policy.PolicyReport (leaves)
    setup: dict
    check: dict = field(default_factory=dict)        # Monte-Carlo cross-check
    seconds: float = 0.0

    @property
    def safe(self):
        return self.verdict == "SAFE"

    def summary(self, max_violations=4) -> str:
        s = self.setup
        L = [f"NEXT-STEP SAFETY AUDIT  ->  {self.verdict}",
             f"  loads: {s['n_sym']} uncertain directions, {s['width']} around the snapshot; next load: {s['next']}",
             f"  limits: {s['limits']}",
             f"  partition: {len(self.report.leaves)} boxes{'' if self.report.complete else ' (leaf cap hit: incomplete)'}; "
             f"SAFE {100 * self.shares['SAFE']:.1f}%  CAN-VIOLATE {100 * self.shares['CAN-VIOLATE']:.1f}%  "
             f"UNDECIDED {100 * self.shares['UNDECIDED']:.1f}%  PF-NOT-CERTIFIED {100 * self.shares['UNKNOWN']:.1f}%",
             "  actions the agent can take (share of the load box) and their certified effect:"]
        for r in self.actions:
            n0 = r.n0
            rng = ""
            if n0 is not None and n0.lo is not None:
                vlo, vhi = n0.range_of("vm")
                llo, lhi = n0.range_of("loading")
                rng = f"|V| {vlo:.4f}..{vhi:.4f} pu, max loading {lhi:.1f}%"
            L.append(f"    {100 * r.share:6.2f}%  {r.label:<34} {r.status:<11} {rng}")
            for o, name, lo, hi, llo, lhi, unit, cert in r.violations()[:max_violations]:
                where = "" if o is None else f"after outage of line {o}: "
                if name == "grid splits":
                    L.append(f"              {where}the grid splits")
                    continue
                lim = f"<= {_fmt(lhi, unit)}" if np.isfinite(lhi) and hi > lhi else f">= {_fmt(llo, unit)}"
                L.append(f"              {where}{name} in [{_fmt(lo, unit)}, {_fmt(hi, unit)}], limit {lim}"
                         f"{'  (for every load)' if cert else ''}")
            ce = r.counterexample
            if ce:
                L.append(f"              CONFIRMED: {ce['what']}")
        if self.check:
            c = self.check
            L.append(f"  Monte-Carlo cross-check ({c['n']} samples): agent's action in certified set {c['action_ok']}/{c['n']}, "
                     f"post-action state inside certified bounds {c['bounds_ok']}/{c['n_post']}")
            L.append(f"    actions the agent actually took in the samples: {len(c['seen'])} of the {len(self.actions)} "
                     f"certified reachable ({', '.join(self.setup['label'](a) for a in c['seen'][:6])}"
                     f"{', ...' if len(c['seen']) > 6 else ''}); the others are possible but rare, or bound looseness")
        L.append(f"  time {self.seconds:.1f}s")
        return "\n".join(L)

    __str__ = summary

    def plot(self, file=None, dims=(0, 1)):
        """2D map of the load set (two symbol directions): the certified action set and the safety status of every
        box, with the actions the deployed agent took at the Monte-Carlo samples."""
        from .plots import plot_next_step
        return plot_next_step(self, file, dims)


@dataclass
class NoiseReport:
    """Result of :meth:`RLAuditor.noise_robustness` (noise levels are fractions: 0.02 = 2 %)."""
    nominal: str
    same: object                       # RadiusResult for "the agent picks the same action"
    safe: object                       # RadiusResult for "every action the agent can pick is safe" (or None)
    groups: dict                       # group -> RadiusResult (only that group noisy)
    levels: list                       # (eps, [labels], all safe?) at fixed noise levels
    setup: dict
    names: list
    seconds: float = 0.0

    def summary(self) -> str:
        s = self.setup
        pct = lambda v: f"{100 * v:.2f}%"
        L = [f"MEASUREMENT-NOISE AUDIT  (noise on {s['n_noisy']} of {s['n_obs']} observation entries, "
             f"{'relative to each value' if s['relative'] else 'absolute'}; exact entries: {s['exact']})",
             f"  noise-free action: {self.nominal}"]
        r = self.same
        fal = (f"{pct(r.falsified)} (makes it pick {s['label'](r.counterexample['action'])})"
               if np.isfinite(r.falsified) else f"none found up to {pct(r.eps_max)}")
        L.append(f"  same action:  guaranteed up to {pct(r.certified)}{' (whole range)' if r.certified >= r.eps_max else ''}"
                 f"   |  smallest noise found that changes it: {fal}")
        if self.safe is not None:
            r = self.safe
            fal = (f"{pct(r.falsified)} (makes it pick {s['label'](r.counterexample['action'])}, which violates a limit)"
                   if np.isfinite(r.falsified) else f"none found up to {pct(r.eps_max)}")
            if r.falsified == 0.0:
                fal = "0% -- the noise-free action already violates a limit"
            L.append(f"  safe action:  guaranteed up to {pct(r.certified)}{' (whole range)' if r.certified >= r.eps_max else ''}"
                     f"   |  smallest noise found that makes it pick an unsafe action: {fal}")
            if 0.0 < r.falsified < np.inf:
                eta = r.counterexample["eta"]
                top = np.argsort(-np.abs(eta))[:5]
                L.append("     largest entries of that noise: " +
                         ", ".join(f"{self.names[k]} {eta[k]:+.3g}" for k in top if eta[k] != 0))
        if self.groups:
            L.append("  per observation group (only that group noisy), same action:")
            for g, rg in self.groups.items():
                cert = f">= {pct(rg.certified)}" + (" (whole range)" if rg.certified >= rg.eps_max else "")
                fal = (f"breaks at {pct(rg.falsified)}" if np.isfinite(rg.falsified) else "no break found")
                L.append(f"     {g:<16} guaranteed {cert:<24} {fal}")
        if self.levels:
            L.append("  actions the agent can pick at fixed noise levels:")
            for e, labels, ok in self.levels:
                more = f" ... (+{len(labels) - 5})" if len(labels) > 5 else ""
                L.append(f"     +-{pct(e):>7}: {' | '.join(labels[:5])}{more}   {'all safe' if ok else 'NOT all safe'}")
        L.append(f"  time {self.seconds:.1f}s")
        return "\n".join(L)

    __str__ = summary

    def plot(self, file=None):
        """Guaranteed vs found noise level: overall (same action, safe action) and per observation group."""
        from .plots import plot_noise
        return plot_noise(self, file)


class RLAuditor:
    """Audit a discrete-action agent on a :class:`GridModel`.

    policy        the agent's network up to its logits / Q-values: a torch nn.Sequential, a loaded SB3 model
                  (PPO / MaskablePPO / A2C / DQN) or a certflow Sequential
    model         :class:`GridModel` (grid, observation layout, action list, limits)
    input_layers  layers applied to the observation before the network (e.g. ``normalizer_layers(mean, var,
                  clip=10)`` for a VecNormalize wrapper)"""

    def __init__(self, policy, model: GridModel, input_layers=None):
        pol = as_policy(policy)
        if input_layers:
            pol = with_input_layers(pol if isinstance(pol, Sequential) else Sequential(pol.layers), input_layers)
        self.policy = pol
        self.model = model
        self.heads = model.actions.heads
        n_obs = len(model.observation.names(model))
        if pol.in_dim is not None and pol.in_dim != n_obs:
            raise ValueError(f"the network takes {pol.in_dim} inputs but the observation has {n_obs} entries "
                             f"({model.observation.names(model)[:4]} ...)")
        if pol.out_dim is not None and pol.out_dim != len(model.actions):
            raise ValueError(f"the network has {pol.out_dim} outputs but the action space has {len(model.actions)}")

    # ------------------------------------------------------------------ helpers
    def act(self, obs, mask=None):
        """The deployed (deterministic) action at an observation."""
        lg = self.policy.forward(np.asarray(obs, float))
        valid = np.ones(len(lg), bool) if mask is None else mask
        if self.heads is not None:
            return argmax_heads_point(lg, self.heads, valid)
        return int(np.argmax(np.where(valid, lg, -np.inf)))

    def label(self, a):
        return self.model.actions.label(int(a))

    def _post(self, state, a):
        return self.model.actions.joint(int(a)).apply(state, self.model)

    def check_action(self, state, a, U_next, max_leaves=64, refine=24):
        """Certified screening of the grid after action a, over the next-load set (N-0, and N-1 if requested)."""
        m = self.model
        post = self._post(state, a)
        n0 = m.screen(post, U_next, max_leaves=max_leaves, refine=refine)
        n1 = {}
        if m.limits.n1:
            n1 = {o: m.screen(post, U_next, outage=o, max_leaves=max_leaves, refine=refine) for o in m.n1_lines(post)}
        screens = [n0] + list(n1.values())
        safe = all(s.safe for s in screens)
        decided = all(s.formal or s.islanded for s in screens) or any(s.islanded or s.certain.any() for s in screens)
        return safe, decided, n0, n1

    def _falsify_next(self, state, a, n0, n1, U_next, n_random=8, seed=0):
        """A concrete next load in U_next that violates a limit after action a: directed vertices of the leaves where
        a quantity can violate (pushed along its linear coefficients), then random vertices."""
        m = self.model
        post = self._post(state, a)
        rng = np.random.default_rng(seed)
        for outage, scr in [(None, n0)] + list(n1.items()):
            if scr is None:
                continue
            if scr.islanded:
                return dict(outage=outage, u=U_next.c, what=f"line {outage} outage splits the grid")
            if scr.lo is None:
                continue
            cands = []
            for k in np.where(scr.possible)[0]:
                up = scr.hi[k] > scr.lim_hi[k]
                for a_, b_, C, c0 in scr.leaves:
                    ec, h = 0.5 * (a_ + b_), 0.5 * (b_ - a_)
                    cands.append(ec + h * np.sign(C[k] if up else -C[k]))
            cands += [rng.choice([-1.0, 1.0], U_next.K) for _ in range(n_random)]
            for e in cands:
                u = U_next.point(e)
                sim = m.simulate(post, u, outage=outage)
                if sim is None:
                    return dict(outage=outage, u=u, what="the power flow has no solution" +
                                ("" if outage is None else f" after outage of line {outage}"))
                q = sim["quantities"]
                j = int(np.argmax(q["violation"]))
                if q["violation"][j] > 1e-9:
                    lim = q["lim_hi"][j] if q["value"][j] > q["lim_hi"][j] else q["lim_lo"][j]
                    where = "" if outage is None else f" after outage of line {outage}"
                    return dict(outage=outage, u=u, quantity=q["names"][j], value=float(q["value"][j]),
                                limit=float(lim), what=f"{q['names'][j]} = {_fmt(q['value'][j], q['unit'][j])} "
                                                       f"(limit {_fmt(lim, q['unit'][j])}){where}")
        return None

    # ------------------------------------------------------------------ mode 1
    def next_step(self, net, load_width=0.05, uncertainty=None, next_load="same", step_change=0.0, max_depth=6,
                  max_leaves=2000, refine="safe", n_check=200, falsify=True, seed=0) -> NextStepReport:
        """Mode 1: is every action the agent can take safe one step ahead?

        net          snapshot: the grid as it is now (setpoints, taps, switch states, loads)
        load_width   the agent sees any load within +-load_width (relative) of the snapshot; or pass
                     ``uncertainty=LoadUncertainty(...)`` for more control (which loads, absolute widths, ...)
        next_load    "same": the action is applied when the loads are anywhere in the same box (default);
                     "step": the next load is the observed one changed by up to +-step_change (relative), which is
                     tighter because the next load stays correlated with what the agent saw
        max_depth    bisections of the load box (to separate regions where the agent takes different actions)
        n_check      Monte-Carlo samples for the cross-check (0 = skip)"""
        import time
        t0 = time.time()
        m = self.model
        state = m.state_from_net(net)
        unc = uncertainty or LoadUncertainty(rel=load_width)
        U = unc.build(m, net)
        K = U.K
        mask = m.actions.mask(state, m)
        Gs = unc.step(m, U, step_change) if next_load == "step" else None
        if next_load not in ("same", "step"):
            raise ValueError("next_load must be 'same' or 'step'")
        cache = {}

        def next_set(lo, hi):
            if next_load == "same":
                return U
            ec, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            return AffineSet(U.c + U.G @ ec, np.hstack([U.G * h[None, :], Gs]))

        def observe(lo, hi):
            ok, tm, enc = m.observe_set(state, U, lo, hi)
            if not ok:
                sc = enc.score[:K] if enc is not None and enc.score is not None else None
                return Observation(False, score=sc)
            return Observation(True, tm, dict(rho=enc.rho, bounds=tm.bounds()))

        def check(a, lo, hi, obs):
            key = (a,) if next_load == "same" else (a, tuple(lo), tuple(hi))
            if key not in cache:
                Un = next_set(lo, hi)
                safe, decided, n0, n1 = self.check_action(state, a, Un)
                cache[key] = ActionCheck(safe, decided, dict(n0=n0, n1=n1, U=Un))
            return cache[key]

        V = DiscretePolicyVerifier(self.policy, observe, K, (lambda lo, hi, o: mask), check, refine=refine,
                                   max_depth=max_depth if K else 0, max_leaves=max_leaves, heads=self.heads,
                                   stop_when_decided=(next_load == "same"))
        R = V.verify()
        shares = R.status_shares()
        # per action: share, merged status, worst screen
        acts = {}
        for Lf in R.leaves:
            if not Lf.observed:
                continue
            vol = Lf.volume(R.lo, R.hi)
            for a in Lf.actions:
                a = int(a)
                c = Lf.checks[a]
                r = acts.get(a)
                if r is None:
                    n0 = c.info["n0"]
                    st = "SAFE" if c.safe else ("UNDECIDED" if not c.formal else
                                                ("VIOLATED" if any(s.islanded or s.certain.any()
                                                                   for s in [n0] + list(c.info["n1"].values()))
                                                 else "CAN-VIOLATE"))
                    if n0.islanded:
                        st = "ISLANDED"
                    r = acts[a] = ActionResult(a, self.label(a), 0.0, c.safe, st, n0, c.info["n1"])
                    r._leaves = []
                elif not c.safe and r.safe:                       # (step mode) keep the worst leaf's check
                    r.safe, r.status, r.n0, r.n1 = False, "CAN-VIOLATE", c.info["n0"], c.info["n1"]
                r.share += vol
                r._leaves.append(Lf)
        results = sorted(acts.values(), key=lambda r: -r.share)
        rng = np.random.default_rng(seed)
        if falsify:
            for r in results:
                if r.safe:
                    continue
                r.counterexample = self._confirm(state, r, U, mask, next_set, rng)
        check_mc = self._monte_carlo(state, U, R, mask, next_set, n_check, rng) if n_check else {}
        if R.safe:
            verdict = "SAFE"
        elif any(r.counterexample for r in results):
            verdict = "UNSAFE"
        else:
            verdict = "NOT PROVEN"
        d = getattr(unc, "directions", None)
        if uncertainty is None:
            width = f"+-{100 * load_width:g}% per load"
        elif d is not None:
            width = "low-rank: " + ("all loads together" if isinstance(d, str) else f"{np.shape(d)[1]} load modes")
        else:
            width = f"+-{unc.rel} (relative) per load"
        setup = dict(n_sym=K, width=width, limits=m.limits.describe(m), label=self.label,
                     next=("anywhere in the same box" if next_load == "same" else
                           f"the observed load +-{100 * step_change:g}% per load"))
        return NextStepReport(verdict, results, shares, R, setup, check_mc, time.time() - t0)

    def _confirm(self, state, r, U, mask, next_set, rng, n_points=24):
        """Concrete counterexample for a possibly unsafe action: a load where the deployed agent takes it, and a next
        load where the post-action grid violates a limit."""
        m = self.model
        for Lf in sorted(r._leaves, key=lambda L: -L.volume(-np.ones(U.K), np.ones(U.K)))[:6]:
            pts = [0.5 * (Lf.lo + Lf.hi)] + [rng.uniform(Lf.lo, Lf.hi) for _ in range(n_points)]
            for e in pts:
                sim = m.simulate(state, U.point(e))
                if sim is None or self.act(sim["obs"], mask) != r.action:
                    continue
                c = Lf.checks[r.action]
                Un = c.info["U"]
                hit = self._falsify_next(state, r.action, c.info["n0"], c.info["n1"], Un)
                if hit:
                    hit["u_obs"] = U.point(e)
                    hit["what"] = f"agent takes '{r.label}' at a load in the box, then {hit['what']}"
                    return hit
                break                                     # the agent does take it here; the next load cannot violate
        return None

    def _monte_carlo(self, state, U, R, mask, next_set, n, rng):
        m = self.model
        n_ok = a_ok = b_ok = n_post = 0
        seen, pts = set(), []
        for _ in range(n):
            e = rng.uniform(-1, 1, U.K)
            sim = m.simulate(state, U.point(e))
            if sim is None:
                continue
            leaf = next((L for L in R.leaves if L.observed and np.all(e >= L.lo) and np.all(e <= L.hi)), None)
            if leaf is None:
                continue
            n_ok += 1
            a = self.act(sim["obs"], mask)
            seen.add(a)
            pts.append((e, a))
            a_ok += a in set(int(x) for x in leaf.actions)
            c = leaf.checks.get(a)
            if c is None or c.info["n0"].lo is None:
                continue
            Un = c.info["U"]
            post = m.simulate(self._post(state, a), Un.point(rng.uniform(-1, 1, Un.K)))
            if post is None:
                continue
            n_post += 1
            q, s = post["quantities"], c.info["n0"]
            b_ok += bool(np.all(q["value"] >= s.lo - 1e-9) and np.all(q["value"] <= s.hi + 1e-9))
        return dict(n=n_ok, action_ok=a_ok, n_post=n_post, bounds_ok=b_ok, seen=sorted(seen), samples=pts)

    # ------------------------------------------------------------------ mode 3
    def multi_step(self, net, load_width=0.03, uncertainty=None, steps=3, step_change=0.01, max_depth=2,
                   max_leaves=400, n_check=100, seed=0, track="rebox"):
        """Mode 3: which states can the agent drive the grid into over ``steps`` steps?

        The loads start anywhere within +-load_width of the snapshot (or ``uncertainty``) and change by up to
        +-step_change (relative, along the same directions) before every further action.  Returns a
        :class:`~certflow.audit.multistep.MultiStepReport`: the certified tree of action sequences, per-step
        envelopes of every bus voltage and branch loading, a Monte-Carlo check, and ``.plot(file)`` for a 3D view.
        track="rebox" (default): every step works in the initial set's own coordinates with a box widened by the
        step change -- constant dimension.  The grid has no memory beyond the discrete state, so this is never looser
        than "history" when the step change moves along the set's own directions (the LoadUncertainty case).
        track="history": new symbols per step (dimension grows by the step's symbols every step); needed when the
        step change has directions of its own."""
        from .multistep import multi_step
        return multi_step(self, net, load_width, uncertainty, steps, step_change, max_depth, max_leaves, n_check, seed,
                          track)

    # ------------------------------------------------------------------ mode 2
    def noise_robustness(self, net, eps_max=0.2, noisy=None, relative=True, require_safe=True, step_change=0.0,
                         groups=True, levels=(0.01, 0.02, 0.05, 0.10), tol=1e-3) -> NoiseReport:
        """Mode 2: how much measurement noise can the agent take?  The grid state is the snapshot (exact); the agent
        sees  obs + eta  with  |eta_i| <= eps * scale_i.

        noisy        which entries are noisy: None = the measurement blocks of the observation (exact: the agent's
                     own setpoints, taps, switch states), or a bool mask / list of block names / entry indices
        relative     scale_i = |obs_i| (noise as a fraction of each value); False: scale_i = 1 (absolute, in the
                     observation's units), or pass an array of per-entry scales
        require_safe also compute the radius below which every action the agent can pick is certified safe, and a
                     concrete noise vector that makes it pick an action that really violates a limit
        step_change  the action is applied at the snapshot loads changed by up to +-step_change (relative)"""
        import time
        t0 = time.time()
        m = self.model
        state = m.state_from_net(net)
        U0 = LoadUncertainty(rel=0.0).build(m, net)
        sim = m.simulate(state, U0.c)
        if sim is None:
            raise RuntimeError("the power flow of the snapshot does not converge")
        o = sim["obs"]
        n = len(o)
        names = m.observation.names(m)
        sel = self._noisy_mask(noisy, names)
        if isinstance(relative, (bool, np.bool_)):
            sc = np.abs(o) if relative else np.ones(n)
        else:
            sc = np.broadcast_to(np.asarray(relative, float), (n,))
        sc = np.where(sel, sc, 0.0)
        x = TM(o, np.zeros((n, 0)), np.zeros((n, 0, 0)), np.zeros(n))
        mask = m.actions.mask(state, m)
        r_same = robustness_radius(self.policy, x, sc, mask, "same", eps_max, tol, heads=self.heads)
        r_safe = None
        safe_cache, bad_cache = {}, {}
        Un = (U0 if not step_change else
              AffineSet(U0.c, LoadUncertainty(rel=0.0).step(m, U0, step_change)))

        def cert_safe(a):
            if a not in safe_cache:
                safe_cache[a] = self.check_action(state, a, Un)
            return safe_cache[a][0]

        def really_bad(a):
            if a not in bad_cache:
                safe, decided, n0, n1 = safe_cache[a] if a in safe_cache else self.check_action(state, a, Un)
                bad_cache[a] = (not safe) and self._falsify_next(state, a, n0, n1, Un) is not None
            return bad_cache[a]

        if require_safe:
            r_safe = robustness_radius(self.policy, x, sc, mask, lambda acts: all(cert_safe(a) for a in acts),
                                       eps_max, tol, bad_action=really_bad, heads=self.heads)
        grp = {}
        if groups:
            for gname, idx in m.observation.groups(m).items():
                gs = np.zeros(n)
                gs[idx] = sc[idx]
                if gs.any():
                    grp[gname] = robustness_radius(self.policy, x, gs, mask, "same", eps_max, tol, heads=self.heads)
        lv = []
        am = (lambda lg: reachable_argmax(lg, mask)) if self.heads is None else \
             (lambda lg: reachable_argmax_heads(lg, self.heads, mask))
        for e in levels:
            acts = [int(a) for a in am(self.policy.reach(add_noise(x, e * sc))).actions]
            lv.append((e, [self.label(a) for a in acts], all(cert_safe(a) for a in acts)))
        nominal = self.label(r_same.nominal) if r_same.nominal is not None else "ambiguous"
        exact = [nm for nm, s in zip(names, sel) if not s]
        setup = dict(n_noisy=int(sel.sum()), n_obs=n, relative=bool(relative) if isinstance(relative, bool) else False,
                     exact=(", ".join(sorted({e.split('[')[0] for e in exact})) or "none"), label=self.label)
        return NoiseReport(nominal, r_same, r_safe, grp, lv, setup, names, time.time() - t0)

    def _noisy_mask(self, noisy, names):
        m = self.model
        if noisy is None:
            return m.observation.noisy(m)
        arr = np.asarray(noisy)
        if arr.dtype == bool:
            return arr
        sel = np.zeros(len(names), bool)
        groups = m.observation.groups(m)
        for v in noisy:
            if isinstance(v, str):
                if v not in groups:
                    raise KeyError(f"unknown observation block {v!r}; blocks: {list(groups)}")
                sel[groups[v]] = True
            else:
                sel[int(v)] = True
        return sel
