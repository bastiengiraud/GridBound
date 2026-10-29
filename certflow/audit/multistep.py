"""Mode 3: the states an RL agent can drive the grid into over several steps.

    rep = auditor.multi_step(net, load_width=0.03, steps=3, step_change=0.01)
    print(rep.summary())
    rep.plot("reachable.png")            # 3D: element x step x certified [lo, hi] band (voltages, loadings)

Starting from every load in the initial set, the agent acts; the loads then change by up to ``step_change`` per step
and the agent acts again, ``steps`` times.  Every branch of the resulting tree of action sequences is certified:
which actions the agent can take at each step, and certified bounds on every bus voltage and branch loading after
each action.  The loads are tracked with new symbols per step (no re-boxing), so the correlation between steps is
kept.  The per-step envelopes (union over all branches of the tree) are what the plot shows, together with
Monte-Carlo trajectories of the deployed agent, which must lie inside them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..policy import ActionCheck, MultiStepVerifier, Observation
from ..sets import AffineSet


@dataclass
class Envelope:
    """Certified bounds of every limited quantity at one step, over all reachable branches of the tree."""
    lo: dict = field(default_factory=dict)          # name -> lower bound
    hi: dict = field(default_factory=dict)
    lim: dict = field(default_factory=dict)         # name -> (lim_lo, lim_hi)
    unit: dict = field(default_factory=dict)
    complete: bool = True                           # every branch certified (no PF failure / undecided leaf)

    def add(self, scr):
        if scr is None or scr.lo is None:
            self.complete = False
            return
        for k, n in enumerate(scr.names):
            self.lo[n] = min(self.lo.get(n, np.inf), float(scr.lo[k]))
            self.hi[n] = max(self.hi.get(n, -np.inf), float(scr.hi[k]))
            self.lim[n] = (float(scr.lim_lo[k]), float(scr.lim_hi[k]))
            self.unit[n] = scr.unit[k]

    def select(self, prefix):
        """(names, lo, hi, lim_lo, lim_hi) of the quantities whose name starts with ``prefix``."""
        names = sorted((n for n in self.lo if n.startswith(prefix)), key=lambda n: (n.rsplit(" ", 1)[0],
                                                                                   int(n.rsplit(" ", 1)[1])))
        g = lambda d: np.array([d[n] for n in names])
        return (names, g(self.lo), g(self.hi), np.array([self.lim[n][0] for n in names]),
                np.array([self.lim[n][1] for n in names]))

    def violations(self):
        return [n for n in self.lo if self.lo[n] < self.lim[n][0] or self.hi[n] > self.lim[n][1]]


@dataclass
class MultiStepReport:
    """Result of :meth:`RLAuditor.multi_step`."""
    verdict: str                       # SAFE | UNSAFE (a sampled trajectory violates a limit) | NOT PROVEN
    steps: int
    envelopes: list                    # Envelope per step: 0 = before the first action, k = after the k-th action
    paths: list                        # reachable action sequences (tuples of action indices)
    tree: object                       # certflow.policy.StepNode
    setup: dict
    check: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)    # step -> {name: [values of the Monte-Carlo trajectories]}
    seconds: float = 0.0

    @property
    def safe(self):
        return self.verdict == "SAFE"

    def summary(self) -> str:
        s, lab = self.setup, self.setup["label"]
        st = self.tree.stats()
        L = [f"MULTI-STEP REACHABILITY AUDIT ({self.steps} steps)  ->  {self.verdict}",
             f"  loads: {s['width']} at step 0, then +-{100 * s['step_change']:g}% per step "
             f"(tracking: {s.get('track', 'history')}, {st['max_symbols']} symbols per node at most)",
             f"  limits: {s['limits']}",
             f"  tree: {st['nodes']} nodes, {st['leaves']} boxes; {len(self.paths)} reachable action sequence(s):"]
        for p in self.paths[:12]:
            L.append("    " + " -> ".join(lab(a) for a in p))
        if len(self.paths) > 12:
            L.append(f"    ... (+{len(self.paths) - 12})")
        L.append("  certified envelope per step (union over every reachable sequence):")
        for k, E in enumerate(self.envelopes):
            _, vlo, vhi, _, _ = E.select("vm")
            _, llo, lhi, _, _ = E.select("loading")
            viol = E.violations()
            tag = "before any action" if k == 0 else f"after action {k}"
            L.append(f"    step {k} ({tag:<17}): |V| {vlo.min():.4f}..{vhi.max():.4f} pu, max loading {lhi.max():6.1f}%"
                     + ("" if E.complete else "  [incomplete: some branch not certified]")
                     + (f"   can violate: {', '.join(viol[:4])}{' ...' if len(viol) > 4 else ''}" if viol else "   all limits kept"))
        c = self.check
        if c:
            L.append(f"  Monte-Carlo ({c['n']} trajectories of the deployed agent): action sequence in certified set "
                     f"{c['seq_ok']}/{c['n']}, every state inside the certified envelope {c['env_ok']}/{c['n']}, "
                     f"trajectories with a real limit violation {c['n_violating']}")
        L.append(f"  time {self.seconds:.1f}s")
        return "\n".join(L)

    __str__ = summary

    def plot(self, file=None, show=False, quantities=("vm", "loading")):
        """3D view of the certified envelopes: x = element (bus / branch), y = step, z = value.  Each step is a
        translucent band between the certified lower and upper bound; dots are Monte-Carlo trajectories; the
        grey planes are the limits.  Saves to ``file`` if given; returns the matplotlib figure."""
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.5 * len(quantities), 6.5))
        for q_i, q in enumerate(quantities):
            ax = fig.add_subplot(1, len(quantities), q_i + 1, projection="3d")
            draw_3d(ax, self, q)
        fig.suptitle(f"Certified reachable states over {self.steps} agent steps ({self.verdict}); "
                     "bands: certified bounds, dots: sampled trajectories, planes: limits", fontsize=10)
        fig.subplots_adjust(left=0.02, right=0.97, bottom=0.05, top=0.9, wspace=0.08)
        if file:
            fig.savefig(file, dpi=150)
        if show:
            plt.show()
        return fig


TITLES = {"vm": ("bus voltage magnitude", "|V| [pu]"), "loading": ("line / transformer loading", "loading [%]")}


def draw_3d(ax, rep, q, zlim=None, title=None):
    """One 3D panel: x = element, y = step, z = value; a translucent band per step between the certified bounds,
    Monte-Carlo trajectories as dots, limits as grey planes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = plt.get_cmap("viridis")
    names = sorted({n for E in rep.envelopes for n in E.lo if n.startswith(q)},
                   key=lambda n: (n.rsplit(" ", 1)[0], int(n.rsplit(" ", 1)[1])))
    x = np.arange(len(names))
    lims = []
    for k, E in enumerate(rep.envelopes):
        col = cmap(k / max(rep.steps, 1))
        lo = np.array([E.lo.get(n, np.nan) for n in names])
        hi = np.array([E.hi.get(n, np.nan) for n in names])
        ok = np.isfinite(lo) & np.isfinite(hi)
        for seg in np.split(np.arange(len(names)), np.where(~ok)[0]):
            seg = [i for i in seg if ok[i]]
            if len(seg) < 1:
                continue
            xs = np.r_[x[seg] - 0.3, x[seg][::-1] + 0.3] if len(seg) == 1 else np.r_[x[seg], x[seg][::-1]]
            zs = np.r_[lo[seg], hi[seg][::-1]] if len(seg) > 1 else np.r_[lo[seg], hi[seg]]
            verts = [list(zip(xs, np.full(len(xs), k), zs))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=col, edgecolor=col, alpha=0.35, linewidth=0.8))
        for i in np.where(ok)[0]:
            ax.plot([x[i], x[i]], [k, k], [lo[i], hi[i]], color=col, linewidth=2)
        smp = rep.samples.get(k, {})
        pts = [(i, v) for i, n in enumerate(names) for v in smp.get(n, [])]
        if pts:
            pi, pv = np.array(pts).T
            ax.scatter(pi, np.full(len(pi), k), pv, s=3, color="k", alpha=0.25, depthshade=False)
        lims += [E.lim[n] for n in names if n in E.lim]
    for v in sorted({l for lim in lims for l in lim if np.isfinite(l)}):
        X, Y = np.meshgrid([-0.5, len(names) - 0.5], [-0.3, rep.steps + 0.3])
        ax.plot_surface(X, Y, np.full_like(X, v, dtype=float), color="grey", alpha=0.15)
    ax.set_title(title or TITLES.get(q, (q, q))[0], fontsize=10)
    ax.set_xlabel("bus" if q == "vm" else "branch")
    ax.set_xticks(x)
    ax.set_xticklabels([n.rsplit(" ", 1)[1] if q == "vm" else
                        ("T" if "trafo" in n else "L") + n.rsplit(" ", 1)[1] for n in names], fontsize=6)
    ax.set_ylabel("step")
    ax.set_yticks(range(rep.steps + 1))
    ax.set_zlabel(TITLES.get(q, (q, q))[1])
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=22, azim=-60)


def compare_3d(reports, file=None, quantities=("vm", "loading")):
    """The 3D view of several runs side by side (columns: runs, rows: quantities), on the same z-axis per row, e.g.
    ``compare_3d({"history": rep_h, "rebox": rep_r}, "compare.png")``."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    names = list(reports)
    fig = plt.figure(figsize=(7.0 * len(names), 5.6 * len(quantities)))
    for qi, q in enumerate(quantities):
        vals = [v for r in reports.values() for E in r.envelopes for n, v in list(E.lo.items()) + list(E.hi.items())
                if n.startswith(q) and np.isfinite(v)]
        vals += [l for r in reports.values() for E in r.envelopes for n, lim in E.lim.items() if n.startswith(q)
                 for l in lim if np.isfinite(l)]
        pad = 0.03 * (max(vals) - min(vals) + 1e-9)
        for j, nm in enumerate(names):
            r = reports[nm]
            ax = fig.add_subplot(len(quantities), len(names), qi * len(names) + j + 1, projection="3d")
            st = r.tree.stats()
            draw_3d(ax, r, q, zlim=(min(vals) - pad, max(vals) + pad),
                    title=f"{TITLES.get(q, (q, q))[0]}: {nm}\n({r.verdict}, {st['max_symbols']} symbol(s) per node, "
                          f"{r.seconds:.1f}s)")
    fig.suptitle("Certified reachable states per step: " + " vs ".join(names) +
                 "  (bands: certified bounds, dots: sampled trajectories, planes: limits)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.04, top=0.92, wspace=0.05, hspace=0.12)
    if file:
        fig.savefig(file, dpi=140)
    return fig


def width_table(reports, prefixes=(("vm", "max voltage band width [pu]", 1.0), ("loading", "max loading band width [%]", 1.0))):
    """Markdown table: per step, the largest certified band width (hi - lo) of each quantity, for each report."""
    names = list(reports)
    steps = max(r.steps for r in reports.values())
    head = "| step | " + " | ".join(f"{lab}, {nm}" for _, lab, _ in prefixes for nm in names) + " |"
    rows = [head, "|---" * (1 + len(prefixes) * len(names)) + "|"]
    for k in range(steps + 1):
        cells = []
        for p, _, f in prefixes:
            for nm in names:
                E = reports[nm].envelopes[k]
                w = [E.hi[n] - E.lo[n] for n in E.lo if n.startswith(p)]
                cells.append(f"{f * max(w):.4f}" if p == "vm" else f"{f * max(w):.1f}")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def multi_step(auditor, net, load_width=0.03, uncertainty=None, steps=3, step_change=0.01, max_depth=2,
               max_leaves=400, n_check=100, seed=0, track="rebox"):
    """Implementation of :meth:`RLAuditor.multi_step` (see there)."""
    import time

    from .grid_model import LoadUncertainty
    t0 = time.time()
    m = auditor.model
    state = m.state_from_net(net)
    unc = uncertainty or LoadUncertainty(rel=load_width)
    U0 = unc.build(m, net)
    Gs = unc.step(m, U0, step_change) if step_change else np.zeros((m.n_inj, 0))

    def problem(st, U):
        mask = m.actions.mask(st, m)
        K = U.K
        cache = {}

        def observe(lo, hi):
            ok, tm, enc = m.observe_set(st, U, lo, hi)
            if not ok:
                return Observation(False, score=(enc.score[:K] if enc is not None and enc.score is not None else None))
            return Observation(True, tm, dict(rho=enc.rho))

        def check(a, lo, hi, obs):
            key = (a, tuple(lo), tuple(hi))
            if key not in cache:
                ec, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
                Un = AffineSet(U.c + U.G @ ec, np.hstack([U.G * h[None, :], Gs]))
                safe, decided, n0, n1 = auditor.check_action(st, a, Un)
                cache[key] = ActionCheck(safe, decided, dict(n0=n0, n1=n1, U=Un))
            return cache[key]

        return dict(observe=observe, mask=(lambda lo, hi, o: mask), check=check)

    kw = dict(refine="safe", max_depth=max_depth, max_leaves=max_leaves, heads=auditor.heads)
    transition = lambda st, a: auditor._post(st, a)
    if track == "history":
        msv = MultiStepVerifier(auditor.policy, problem, transition, steps=steps, track="history",
                                extend=lambda U: AffineSet(U.c, np.hstack([U.G, Gs])), **kw)
        root = msv.verify(state, U=U0)
    elif track == "rebox":
        # coordinates: the initial set's own symbols, x = c + G e'.  A step adds Gs w = G (S w), so the next box in e'
        # is the current one widened by |S| 1 -- exact when S is diagonal (the step moves along the set's own
        # directions, e.g. LoadUncertainty with or without modes), an interval hull otherwise.
        S, *_ = np.linalg.lstsq(U0.G, Gs, rcond=None) if Gs.shape[1] else (np.zeros((U0.K, 0)),)
        if Gs.shape[1] and np.abs(U0.G @ S - Gs).max() > 1e-9 * max(np.abs(Gs).max(), 1.0):
            raise ValueError("track='rebox' needs the per-step change to lie in the span of the initial set's "
                             "directions; use track='history'")
        grow = np.abs(S).sum(1)

        def problem_box(st, lo, hi):
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            U = AffineSet(U0.c + U0.G @ mid, U0.G * half[None, :])
            return dict(problem(st, U), K=U0.K)

        msv = MultiStepVerifier(auditor.policy, problem_box, transition,
                                propagate=lambda lo, hi: (lo - grow, hi + grow), steps=steps, track="rebox", **kw)
        root = msv.verify(state, -np.ones(U0.K), np.ones(U0.K))
    else:
        raise ValueError("track must be 'history' or 'rebox'")
    # envelopes: step 0 = the grid before any action over the initial set; step k = after the k-th action
    envs = [Envelope() for _ in range(steps + 1)]
    envs[0].add(m.screen(state, U0))
    for node in root.walk():
        for Lf in node.report.leaves:
            if not Lf.observed:
                envs[node.depth + 1].complete = False
                continue
            for c in Lf.checks.values():
                envs[node.depth + 1].add(c.info.get("n0"))
    paths = root.paths()
    # Monte-Carlo trajectories of the deployed agent with the same load model
    rng = np.random.default_rng(seed)
    samples = {k: {} for k in range(steps + 1)}
    n = seq_ok = env_ok = n_bad = 0
    pset = set(paths)
    r = Gs.shape[1]
    for _ in range(n_check):
        u = U0.point(rng.uniform(-1, 1, U0.K))
        st, acts, inside, bad, ok = state, [], True, False, True
        sim = m.simulate(st, u)
        if sim is None:
            continue
        recs = [(0, sim)]
        for k in range(steps):
            a = auditor.act(sim["obs"], m.actions.mask(st, m))
            acts.append(a)
            st = auditor._post(st, a)
            u = u + (Gs @ rng.uniform(-1, 1, r) if r else 0.0)
            sim = m.simulate(st, u)
            if sim is None:
                ok = False
                break
            recs.append((k + 1, sim))
        if not ok:
            continue
        n += 1
        seq_ok += tuple(acts) in pset
        for k, sm in recs:
            q = sm["quantities"]
            E = envs[k]
            for name, v in zip(q["names"], q["value"]):
                samples[k].setdefault(name, []).append(float(v))
                if name in E.lo and not (E.lo[name] - 1e-9 <= v <= E.hi[name] + 1e-9):
                    inside = False
            bad |= bool(np.any(q["violation"] > 1e-9))
        env_ok += inside
        n_bad += bad
    certified = root.safe and all(E.complete for E in envs[1:])
    verdict = "SAFE" if certified else ("UNSAFE" if n_bad else "NOT PROVEN")
    d = getattr(unc, "directions", None)
    width = (f"+-{100 * load_width:g}% per load" if uncertainty is None else
             ("low-rank: " + ("all loads together" if isinstance(d, str) else f"{np.shape(d)[1]} load modes"))
             if d is not None else f"+-{unc.rel} (relative) per load")
    setup = dict(width=width, step_change=step_change, limits=m.limits.describe(m), label=auditor.label, track=track)
    return MultiStepReport(verdict, steps, envs, paths, root, setup,
                           dict(n=n, seq_ok=seq_ok, env_ok=env_ok, n_violating=n_bad), samples, time.time() - t0)
