"""Figures for the audit reports (matplotlib, imported lazily).  Every function returns the figure and saves it if a
file name is given.  The 3D multi-step figure lives in :mod:`certflow.audit.multistep`."""
from __future__ import annotations

import numpy as np

STATUS_COLOR = {"SAFE": "#2e8b57", "CAN-VIOLATE": "#d62728", "UNDECIDED": "#ff9f1c", "UNKNOWN": "#9e9e9e"}


def _plt():
    # never switch the backend here: forcing "Agg" disabled inline figures in notebooks for the whole session.
    # Without a display, matplotlib already falls back to a non-interactive backend, and savefig works with any.
    import matplotlib.pyplot as plt
    return plt


def _save(fig, file):
    if file:
        fig.savefig(file, dpi=150, bbox_inches="tight")
    return fig


# ==================================================================================================
# mode 1: next-step safety
# ==================================================================================================

def plot_next_step(rep, file=None, dims=(0, 1)):
    """Left: certified action set per box of the load set (two symbol directions); dots = action the deployed agent
    took at Monte-Carlo samples.  Right: certified safety status per box."""
    plt = _plt()
    from matplotlib.patches import Patch, Rectangle
    R, lab = rep.report, rep.setup["label"]
    K = len(R.lo)
    if K == 0:
        raise ValueError("the load set has no uncertain direction: nothing to map")
    d0, d1 = (dims[0], dims[1]) if K >= 2 else (0, None)
    keys = sorted({tuple(int(a) for a in L.actions) for L in R.leaves if L.observed}, key=lambda k: (len(k), k))
    singles = sorted({a for k in keys for a in k})
    cmap = plt.get_cmap("tab10")
    acol = {a: cmap(i % 10) for i, a in enumerate(singles)}

    def kcol(k):
        return acol[k[0]] if len(k) == 1 else "#cccccc"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, mode in zip(axes, ("actions", "status")):
        for L in R.leaves:
            x0, x1 = L.lo[d0], L.hi[d0]
            y0, y1 = (L.lo[d1], L.hi[d1]) if d1 is not None else (-1.0, 1.0)
            if mode == "actions":
                k = tuple(int(a) for a in L.actions) if L.observed else None
                fc = kcol(k) if k else "#9e9e9e"
                hatch = "//" if k and len(k) > 1 else None
            else:
                fc, hatch = STATUS_COLOR[L.status], None
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor="white", linewidth=0.8,
                                   alpha=0.55, hatch=hatch))
        pts = rep.check.get("samples", []) if rep.check else []
        if pts and mode == "actions":
            E = np.array([p[0] for p in pts])
            A = np.array([p[1] for p in pts])
            for a in np.unique(A):
                sel = A == a
                y = E[sel, d1] if d1 is not None else np.zeros(sel.sum())
                ax.scatter(E[sel, d0], y, s=10, color=acol.get(int(a), "k"), edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_xlim(R.lo[d0], R.hi[d0])
        ax.set_ylim(*((R.lo[d1], R.hi[d1]) if d1 is not None else (-1, 1)))
        ax.set_xlabel(f"load direction e{d0}")
        ax.set_ylabel(f"load direction e{d1}" if d1 is not None else "")
        ax.set_aspect("equal" if d1 is not None else "auto")
    hand = [Patch(facecolor=acol[a], alpha=0.6, label=lab(a)) for a in singles]
    if any(len(k) > 1 for k in keys):
        hand.append(Patch(facecolor="#cccccc", hatch="//", label="several actions possible"))
    axes[0].legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
    axes[0].set_title("certified action set per box\n(dots: action the deployed agent took)", fontsize=10)
    axes[1].legend(handles=[Patch(facecolor=c, alpha=0.6, label=s) for s, c in STATUS_COLOR.items()],
                   loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8, frameon=False)
    axes[1].set_title("certified one-step safety per box\n(every action possible in the box, every next load)",
                      fontsize=10)
    fig.suptitle(f"Next-step safety audit: {rep.verdict}", fontsize=11)
    return _save(fig, file)


# ==================================================================================================
# mode 2: measurement noise
# ==================================================================================================

def plot_noise(rep, file=None):
    """Guaranteed (certified) noise level vs the smallest breaking noise found, per requirement and group."""
    plt = _plt()
    rows = [("same action (all noisy entries)", rep.same)]
    if rep.safe is not None:
        rows.append(("SAFE action (all noisy entries)", rep.safe))
    rows += [(f"same action, only {g} noisy", r) for g, r in rep.groups.items()]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6))
    eps_max = max(r.eps_max for _, r in rows)
    for i, (name, r) in enumerate(rows[::-1]):
        ax.barh(i, 100 * r.certified, color="#2e8b57", alpha=0.8, label="guaranteed (certified)")
        if np.isfinite(r.falsified):
            ax.barh(i, 100 * (r.falsified - r.certified), left=100 * r.certified, color="#bbbbbb",
                    label="gap (not decided)")
            ax.plot(100 * r.falsified, i, "x", color="#d62728", ms=9, mew=2, label="breaking noise found")
        else:
            ax.annotate("", xy=(100 * r.eps_max, i), xytext=(100 * r.certified, i),
                        arrowprops=dict(arrowstyle="->", color="#2e8b57"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows[::-1]], fontsize=9)
    ax.set_xlabel("noise level  eps  [% of each measured value]")
    ax.set_xlim(0, 100 * eps_max * 1.02)
    ax.grid(axis="x", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="lower right")
    ax.set_title(f"Measurement-noise audit (noise-free action: {rep.nominal})", fontsize=10)
    return _save(fig, file)


# ==================================================================================================
# dispatch controller
# ==================================================================================================

def plot_dispatch(rep, file=None):
    """Per element: certified range (wide bar), sampled range (thin dark bar) and limits, for bus |V|, generator Q and
    branch loading.  With restoration, the certified range after the correction is shown in green."""
    plt = _plt()
    C, g = rep.certificate, rep.verifier.grid
    emp = rep.check.get("empirical", {}) if rep.check else {}
    panels = [("V", "bus |V| [pu]", "bus", lambda v, c: v),
              ("Qg", "generator Q [MVAr]", "generator", lambda v, c: v * g.sn),
              ("branch", "branch loading [%]", "branch", lambda v, c: np.sqrt(np.maximum(v, 0) / c.lim_hi) * 100)]
    panels = [p for p in panels if any(c.name == p[0] for c in C.constraints)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    restored = [r.cert for r in rep.restored if r.restored and r.cert is not None]
    for ax, (name, ylab, xlab, f) in zip(axes, panels):
        c = next(c for c in C.constraints if c.name == name)
        b = C.bands.get(name)
        if b is None or not b.lo.size:
            ax.set_visible(False)
            continue
        x = np.arange(len(b.lo))
        lo, hi = f(b.lo, c), f(b.hi, c)
        if name == "branch":
            lo = np.zeros_like(hi)
        ax.bar(x, hi - lo, bottom=lo, width=0.6, color="#4c72b0", alpha=0.35, label="certified (whole load set)")
        if restored:
            rb = [rc.bands[name] for rc in restored]
            rlo = f(np.min([r.lo for r in rb], 0), c)
            rhi = f(np.max([r.hi for r in rb], 0), c)
            if name == "branch":
                rlo = np.zeros_like(rhi)
            ax.bar(x + 0.2, rhi - rlo, bottom=rlo, width=0.25, color="#2e8b57", alpha=0.7, label="certified after restoration")
        if name in emp:
            elo, ehi = f(emp[name][0], c), f(emp[name][1], c)
            ax.vlines(x, elo, ehi, color="k", linewidth=3, label="sampled")
        if name == "branch":
            ax.axhline(100, color="#d62728", ls="--", lw=1, label="limit")
        else:
            llo, lhi = f(c.lim_lo, c), f(c.lim_hi, c)
            span0 = max(hi.max() - lo.min(), 1e-6)
            wlo, whi = lo.min() - 1.0 * span0, hi.max() + 1.0 * span0       # limits far outside: noted, not drawn
            near = lambda v: (v >= wlo) & (v <= whi)
            ax.hlines(llo[near(llo)], (x - 0.4)[near(llo)], (x + 0.4)[near(llo)], color="#d62728", lw=1.5, label="limit")
            ax.hlines(lhi[near(lhi)], (x - 0.4)[near(lhi)], (x + 0.4)[near(lhi)], color="#d62728", lw=1.5)
            shown = np.r_[lo, hi, llo[near(llo)], lhi[near(lhi)]]
            pad = 0.08 * (shown.max() - shown.min() + 1e-9)
            ax.set_ylim(shown.min() - pad, shown.max() + pad)
            far = [f"{v:.4g}" for v in np.unique(np.r_[llo[~near(llo)], lhi[~near(lhi)]])]
            if far:
                ax.text(0.02, 0.98, "limits off-scale: " + ", ".join(far[:4]), transform=ax.transAxes, va="top",
                        fontsize=7, color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(e)) for e in c.elements], fontsize=7)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Dispatch controller audit: {rep.verdict}", fontsize=11)
    fig.subplots_adjust(bottom=0.2, wspace=0.35)
    return _save(fig, file)


# ==================================================================================================
# mode 3: compare tracking modes
# ==================================================================================================

def compare_multistep(reports, file=None, quantities=("vm", "loading")):
    """Side by side, per step (columns) and quantity (rows): the certified envelope of each report (e.g.
    {"history": rep_h, "rebox": rep_r}) for every element, with the Monte-Carlo trajectories as dots."""
    plt = _plt()
    names = list(reports)
    steps = max(r.steps for r in reports.values())
    cols = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    ylab = {"vm": "|V| [pu]", "loading": "loading [%]"}
    fig, axes = plt.subplots(len(quantities), steps + 1, figsize=(3.3 * (steps + 1), 3.0 * len(quantities)),
                             squeeze=False, sharey="row")
    w = 0.8 / len(names)
    for qi, q in enumerate(quantities):
        for k in range(steps + 1):
            ax = axes[qi, k]
            elems = sorted({n for r in reports.values() for n in r.envelopes[k].lo if n.startswith(q)},
                           key=lambda n: (n.rsplit(" ", 1)[0], int(n.rsplit(" ", 1)[1])))
            x = np.arange(len(elems))
            for j, nm in enumerate(names):
                E = reports[nm].envelopes[k]
                lo = np.array([E.lo.get(n, np.nan) for n in elems])
                hi = np.array([E.hi.get(n, np.nan) for n in elems])
                ax.bar(x - 0.4 + w * (j + 0.5), hi - lo, bottom=lo, width=w, color=cols[j % 4], alpha=0.6,
                       label=f"certified, {nm}")
            smp = next(iter(reports.values())).samples.get(k, {})
            for i, n in enumerate(elems):
                v = smp.get(n, [])
                if v:
                    ax.plot([i] * len(v), v, ".", color="k", ms=2, alpha=0.3)
            lim = [next(iter(reports.values())).envelopes[k].lim.get(n) for n in elems]
            for i, l in enumerate(lim):
                if l is not None:
                    for v in l:
                        if np.isfinite(v):
                            ax.hlines(v, i - 0.45, i + 0.45, color="#d62728", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([n.rsplit(" ", 1)[1] if q == "vm" else ("T" if "trafo" in n else "L") + n.rsplit(" ", 1)[1]
                                for n in elems], fontsize=7)
            if qi == 0:
                ax.set_title("step 0 (before any action)" if k == 0 else f"after action {k}", fontsize=9)
            if k == 0:
                ax.set_ylabel(ylab.get(q, q))
            ax.grid(axis="y", alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="k", marker=".", ls="", label="sampled trajectories"))
    h.append(plt.Line2D([], [], color="#d62728", label="limit"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Multi-step reachability: certified envelopes per step, " + " vs ".join(names), fontsize=11)
    fig.subplots_adjust(bottom=0.14, hspace=0.25, wspace=0.08)
    return _save(fig, file)
