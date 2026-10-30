"""Regenerate every figure and report shown in the documentation (docs/figures/).

    python docs/make_figures.py                 # all
    python docs/make_figures.py dispatch rl1    # some of: dispatch, rl1, rl2, rl3

Each part writes <name>.png and <name>.txt (the printed report) into docs/figures/.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(ROOT, "examples"))
os.makedirs(OUT, exist_ok=True)


def save(name, rep, fig_kw=None):
    rep.plot(os.path.join(OUT, name + ".png"), **(fig_kw or {}))
    txt = rep.summary()
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt, "\n")


def dispatch():
    """Dispatch NN: an NN trained on OPF data (case9), tight reactive limits, restoration."""
    import torch

    from certflow.audit import verify_dispatch
    from certflow.power import case9, from_matpower
    g = from_matpower(case9())
    Pd0 = g.names["load_pq"][0]
    pg0, vg0 = g.names["gen_pg"], g.names["gen_vg"]
    torch.manual_seed(0)
    X = torch.tensor(Pd0 / 100 * np.random.default_rng(0).uniform(0.8, 1.2, (2000, 1)), dtype=torch.float32)
    ratio = X.sum(1, keepdim=True) / float(Pd0.sum() / 100)
    Y = torch.cat([torch.tensor(vg0, dtype=torch.float32) + 0.02 * (ratio - 1),
                   ratio * torch.tensor(pg0 / 100, dtype=torch.float32)], 1)
    model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                torch.nn.Linear(16, 4))
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(1500):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    tight = case9()
    tight["gen"][1:, 3] = [8.0, -2.0]
    rep = verify_dispatch(tight, model, load_width=0.10, inputs="pd", input_units="pu", outputs=("vg", "pg"),
                          pg_units="pu", restore=True)
    save("dispatch", rep)


def _feeder_auditor():
    from feeder_demo import auditor
    return auditor()


def rl1(aud=None):
    """Mode 1 on the feeder: two load directions (loads of buses 2-3 / 4-5) so the partition can be drawn."""
    from certflow.audit import LoadUncertainty
    aud, m, net = aud or _feeder_auditor()
    D = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
    unc = LoadUncertainty(rel=0.35, directions=D)
    rep = aud.next_step(net, uncertainty=unc, max_depth=7, n_check=400, refine="single")
    save("rl_next_step", rep)


def rl2(aud=None):
    """Mode 2 on the feeder snapshot."""
    aud, m, net = aud or _feeder_auditor()
    rep = aud.noise_robustness(net, eps_max=0.10)
    save("rl_noise", rep)


def rl3(aud=None):
    """Mode 3 on the feeder: 3 steps, all loads together +-10 % at step 0, +-3 % per step; both tracking modes."""
    from certflow.audit import LoadUncertainty
    from certflow.audit.multistep import compare_3d, width_table
    from certflow.audit.plots import compare_multistep
    aud, m, net = aud or _feeder_auditor()
    reps = {}
    for track in ("history", "rebox"):
        reps[track] = aud.multi_step(net, uncertainty=LoadUncertainty(rel=0.10, directions="common"), steps=3,
                                     step_change=0.03, max_depth=3, n_check=200, track=track)
        save(f"rl_multistep_{track}", reps[track])
    compare_3d(reps, os.path.join(OUT, "rl_multistep_compare3d.png"))
    compare_multistep(reps, os.path.join(OUT, "rl_multistep_compare.png"))
    with open(os.path.join(OUT, "rl_multistep_widths_timing.txt"), "w", encoding="utf-8") as f:
        f.write(width_table(reps) + "\n")


def rl3_timing(aud=None, horizons=(1, 2, 3, 4)):
    """Wall-clock time of the multi-step audit vs horizon, for both tracking modes (run sequentially, alone)."""
    from certflow.audit import LoadUncertainty
    aud, m, net = aud or _feeder_auditor()
    rows = ["| steps | tracking | time [s] | tree nodes | boxes | symbols per node (max) | reachable sequences | "
            "max loading after last action [%] | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for k in horizons:
        for track in ("history", "rebox"):
            rep = aud.multi_step(net, uncertainty=LoadUncertainty(rel=0.10, directions="common"), steps=k,
                                 step_change=0.03, max_depth=3, n_check=0, track=track)
            st = rep.tree.stats()
            _, _, lhi, _, _ = rep.envelopes[-1].select("loading")
            rows.append(f"| {k} | {track} | {rep.seconds:.1f} | {st['nodes']} | {st['leaves']} | {st['max_symbols']} | "
                        f"{len(rep.paths)} | {lhi.max():.1f} | {rep.verdict} |")
            print(rows[-1])
    import platform
    rows.append("")
    rows.append(f"Measured with `python docs/make_figures.py rl3_timing` on {platform.processor() or platform.machine()}, "
                "single process, nothing else running.")
    with open(os.path.join(OUT, "rl_multistep_timing.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def sync_docs():
    """Paste the latest reports into the mode pages, between <!-- report:NAME --> and <!-- /report --> markers."""
    import glob
    import re
    for page in glob.glob(os.path.join(HERE, "modes", "*.md")):
        s = open(page, encoding="utf-8").read()

        def fill(mt):
            f = os.path.join(OUT, mt.group(1) + ".txt")
            if not os.path.exists(f):
                return mt.group(0)
            body = open(f, encoding="utf-8").read().rstrip()
            if mt.group(1).endswith("_timing"):                       # a markdown table, pasted as is
                return "\n".join([f"<!-- report:{mt.group(1)} -->", body, "<!-- /report -->"])
            return "\n".join([f"<!-- report:{mt.group(1)} -->", "```text", body, "```", "<!-- /report -->"])
        s2 = re.sub(r"<!-- report:([\w-]+) -->.*?<!-- /report -->", fill, s, flags=re.S)
        if s2 != s:
            open(page, "w", encoding="utf-8").write(s2)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["dispatch", "rl1", "rl2", "rl3"]
    t0 = time.time()
    shared = _feeder_auditor() if any(p.startswith("rl") for p in parts) else None
    for p in parts:
        print(f"===== {p}")
        globals()[p]() if p == "dispatch" else globals()[p](shared)
    sync_docs()
    print(f"done in {time.time() - t0:.0f}s; figures in {os.path.relpath(OUT, ROOT)}")
