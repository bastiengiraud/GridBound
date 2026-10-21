"""Case studies: certified verification of the WSCC-9 MaskablePPO agent around operating points.

For each operating point (topology, Pg/V setpoints, load deviation delta_c) and a box delta in delta_c +- w:
  1. the pre-action power flow is certified (Krawczyk, admittance loads) and the agent's 26-entry
     observation is built as a Taylor model in delta -- CONSISTENT with the loads (coupled observation);
  2. the actor (tanh MLP, float32) is pushed through it and the masked argmax gives the certified set of
     actions the agent can take; the box is partitioned until each part has one action (or max depth);
  3. every reachable action is checked one step ahead: certified post-action power flow for the next load
     deviation delta' in [delta - margin, delta + margin]: PF solution exists and is unique, no N-0 overload
     (S <= rating), slack within [0, 250] MW;
     + certified N-1 screening of the post-action state (the env's n1 count: overloads over all line outages
       + 3 x power-flow failures), reported as certified [lower, upper] bounds;
  4. Monte-Carlo cross-check with the plain point model and the float32 actor;
  5. TWO-STEP reachability (certflow.policy.MultiStepVerifier): for every action reachable at step 1, the agent
     is verified again from the post-action state over the propagated load box, and its step-2 actions are
     checked one step ahead -- every 2-action sequence the agent can take, with certified safety;
  6. MEASUREMENT-NOISE ROBUSTNESS around operating points (certflow.robustness): the grid state is exact, the
     agent sees  obs + eta  with |eta_i| <= eps |obs_i| on every measured entry; certified + falsified radius for
     "same action", the most sensitive measurement group, and the radius for "every possible action is safe".

    python examples/wscc9_actor_reachability.py                       # everything (all points, all studies)
    python examples/wscc9_actor_reachability.py --quick               # smaller boxes / depth: a fast smoke run
    python examples/wscc9_actor_reachability.py --study noise         # one study: reach | two-step | noise
    python examples/wscc9_actor_reachability.py --ops 0 3             # only some operating points (--list-ops)
    python examples/wscc9_actor_reachability.py --require-n1          # count possible N-1 violations as unsafe
    python examples/wscc9_actor_reachability.py --weights my.npz      # other actor weights (SB3 state-dict npz)

Each (study, operating point) pair is independent, so on a cluster they can run as separate jobs
(see shell/scripts/wscc9_rl_array.sh).

N-0 (enforced by default): the post-action PF has a unique solution, no branch is overloaded, slack in [0, 250] MW.
N-1 (reported; --require-n1 enforces it): the env's count over all line outages (overloads + 3 x PF failures),
certified as an interval [lower, upper] over the load box:
    SECURE     upper = 0        no outage overloads anything, for every load
    INSECURE   lower > 0        violated for every load in the box (e.g. outages that split the grid)
    UNCERTAIN  lower = 0 < upper  possibly violated somewhere; a sampling search then either finds a real
                                  violation (counterexample) or reports that the bound may be conservative
"""
import argparse
import base64
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace  # noqa: E402

from wscc9_env import (ACTIONS, LINES, XFMRS, OperatingState, StepChecker, WSCC9Model, infer_state_from_obs,  # noqa: E402
                       make_problem, multistep_problem)

from certflow.nn import sb3_actor  # noqa: E402
from certflow.policy import DiscretePolicyVerifier, MultiStepVerifier  # noqa: E402
from certflow.robustness import add_noise, robustness_radius  # noqa: E402
from certflow.nn import reachable_argmax  # noqa: E402
from certflow.tm import TM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "wscc9_maskableppo_policy_extraction")
STUDIES = ("reach", "two-step", "noise")
OP_TITLES = ["reset (delta = 0)", "last training state", "heavy load (delta = +0.3)",
             "light load (delta = -0.3)", "line 8-9 open", "after +0.5 jump at bus 8"]


def _parser():
    p = argparse.ArgumentParser(description="Certified case studies for the WSCC-9 MaskablePPO agent.")
    p.add_argument("weights_pos", nargs="?", default=None, help=argparse.SUPPRESS)   # old positional form
    p.add_argument("--weights", default=os.path.join(ROOT, "extracted", "wscc9_maskableppo_weights.npz"),
                   help="actor weights: SB3 policy state dict saved as .npz")
    p.add_argument("--sb3-data", default=os.path.join(ROOT, "original_files", "data"),
                   help="the 'data' file of the SB3 zip (provides the last training observation)")
    p.add_argument("--study", nargs="+", choices=STUDIES + ("all",), default=["all"],
                   help="which studies to run (default: all)")
    p.add_argument("--ops", nargs="+", type=int, default=None,
                   help="operating-point indices to run (default: all; see --list-ops)")
    p.add_argument("--list-ops", action="store_true", help="print the operating points and exit")
    p.add_argument("--quick", action="store_true", help="smaller boxes / depth / Monte-Carlo (smoke run)")
    p.add_argument("--require-n1", action="store_true",
                   help="count possible N-1 violations as unsafe (default: report only)")
    p.add_argument("--noise-only", action="store_true", help="same as --study noise")
    return p


ARGS = _parser().parse_args(None if __name__ == "__main__" else [])
if ARGS.noise_only:
    ARGS.study = ["noise"]
if "all" in ARGS.study:
    ARGS.study = list(STUDIES)
if ARGS.list_ops:
    for _i, _t in enumerate(OP_TITLES):
        print(f"{_i}: {_t}{'   (also two-step study)' if _i in (0, len(OP_TITLES) - 1) else ''}")
    raise SystemExit(0)
QUICK = ARGS.quick
REQUIRE_N1 = ARGS.require_n1
N_RULE = ("N-0 + N-1 enforced" if REQUIRE_N1 else
          "N-0 enforced: unique PF, no overload, slack in [0, 250] MW; N-1 reported only")
NPZ = ARGS.weights_pos or ARGS.weights
W = dict(np.load(NPZ))
actor = sb3_actor(W)


def name(a):
    act = ACTIONS[a]
    if act[0] == "idle":
        return "idle"
    if act[0] == "gen":
        return f"Pg{act[1]}{act[2]:+.0f}MW"
    if act[0] == "vset":
        return f"V{act[1]}{act[2]:+.2f}"
    return f"{'close' if act[3] else 'open'} {act[1]}-{act[2]}"


def actor_f32(obs, mask):
    """The deployed policy: SB3 float32 forward pass + masked argmax."""
    h = obs.astype(np.float32)
    for k in ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2"):
        h = np.tanh(W[k + ".weight"].astype(np.float32) @ h + W[k + ".bias"].astype(np.float32))
    lg = W["action_net.weight"].astype(np.float32) @ h + W["action_net.bias"].astype(np.float32)
    return int(np.argmax(np.where(mask, lg, np.float32(-1e8))))


def operating_points(model):
    """The case-study operating points, indexed as in OP_TITLES (None if one is unavailable)."""
    ops = [(OP_TITLES[0], OperatingState(), np.zeros(3))]
    try:                                                    # the last state seen in training (stored with the model)
        with open(ARGS.sb3_data) as f:
            d = json.load(f)
        obs = pickle.loads(base64.b64decode(d["_last_obs"][":serialized:"]))[0].astype(float)
        st, dl = infer_state_from_obs(model, obs)
        ops.append((OP_TITLES[1], st, dl))
    except Exception as e:  # pragma: no cover
        print("  (no stored training observation:", e, ")")
        ops.append(None)
    ops += [(OP_TITLES[2], OperatingState(), np.full(3, 0.3)),
            (OP_TITLES[3], OperatingState(), np.full(3, -0.3)),
            (OP_TITLES[4], OperatingState(status=(1, 1, 1, 1, 1, 0)), np.zeros(3)),
            (OP_TITLES[5], OperatingState(), np.array([0.0, 0.0, 0.5]))]
    return ops


def monte_carlo(model, st, U, report, checks_by_leaf, n, seed=0):
    rng = np.random.default_rng(seed)
    mask = st.mask()
    n_obs = n_act = n_post = n_ok = 0
    for e in rng.uniform(-1, 1, (n, 3)):
        sim = model.simulate(st, U.point(e))
        if sim is None:
            continue
        n_ok += 1
        leaf = next(L for L in report.leaves if np.all(e >= L.lo) and np.all(e <= L.hi))
        lo, hi = leaf._obs_bounds
        n_obs += bool(np.all(sim["obs"] >= lo - 1e-9) and np.all(sim["obs"] <= hi + 1e-9))
        a = actor_f32(sim["obs"], mask)
        n_act += a in set(int(x) for x in leaf.actions)
        post = model.simulate(st.apply(a), U.point(e))
        chk = leaf.checks.get(a)
        if post is not None and chk is not None and chk.info.get("pf"):
            n_post += bool(np.all(post["loading"] <= chk.info["load"] + 1e-9)
                           and chk.info["slack"][0] - 1e-9 <= post["slack"] <= chk.info["slack"][1] + 1e-9)
    return n_ok, n_obs, n_act, n_post


def check_summary(report, to_phys=None):
    """Per reachable action: union over the leaves of the certified post-action results (+ the physical load
    boxes of the leaves where the action is reachable, for the N-1 falsification search)."""
    out = {}
    for L in report.leaves:
        for a, c in L.checks.items():
            i = c.info
            o = out.setdefault(a, dict(load=0.0, slack=(np.inf, -np.inf), n1=(np.inf, -np.inf), n1w=0.0, fail=None,
                                       safe=True, n_island=0, n_out=0, boxes=[]))
            o["safe"] &= bool(c.safe)
            if to_phys is not None:
                o["boxes"].append((to_phys(L.lo), to_phys(L.hi)))
            if not i.get("pf"):
                o["fail"] = i.get("reason", "?")
                continue
            o["load"] = max(o["load"], float(np.max(i["load"])))
            o["slack"] = (min(o["slack"][0], i["slack"][0]), max(o["slack"][1], i["slack"][1]))
            if "n1" in i:
                o["n1"] = (min(o["n1"][0], i["n1"][0]), max(o["n1"][1], i["n1"][1]))
                o["n1w"] = max(o["n1w"], float(np.nan_to_num(i["n1_worst"])))
                o["n_island"] = sum(r.islanded for r in i["n1_report"].results)
                o["n_out"] = len(i["n1_report"].results)
    return out


def n1_point(model, st, d):
    """The env's N-1 count at ONE load point (plain power flows): overloads over all line outages + 3 x PF failures.
    Returns (count, worst post-outage loading %, description of the worst case)."""
    count, worst, what = 0, 0.0, ""
    for li, on in enumerate(st.status):
        if not on:
            continue
        so = replace(st, status=tuple(0 if k == li else v for k, v in enumerate(st.status)))
        sim = model.simulate(so, d)
        if sim is None:
            count += 3
            continue
        ld = sim["loading"]
        count += int((ld > 100.0).sum())
        k = int(np.argmax(ld))
        if ld[k] > worst:
            br = (LINES + XFMRS)[k]
            worst, what = float(ld[k]), f"outage {LINES[li][0]}-{LINES[li][1]} -> branch {br[0]}-{br[1]} at {ld[k]:.0f}%"
    return count, worst, what


def falsify_n1(model, st_post, boxes, margin, n=48, seed=0):
    """Search for a REAL N-1 violation after an action: sample next-step loads delta' in (leaf box +- margin) over
    the leaves where the action is reachable (corners included).  Returns (max count, worst loading, where, delta')."""
    rng = np.random.default_rng(seed)
    best = (0, 0.0, "", None)
    for k in range(n):
        lo, hi = boxes[rng.integers(len(boxes))]
        lo, hi = lo - margin, hi + margin
        d = np.where(rng.random(3) < 0.5, lo, hi) if k % 3 == 0 else rng.uniform(lo, hi)
        c, w, what = n1_point(model, st_post, d)
        if (c, w) > best[:2]:
            best = (c, w, what, d)
    return best


def n1_status(o):
    lo, hi = o["n1"]
    if not np.isfinite(lo):
        return "n/a"
    if hi == 0:
        return "SECURE"
    if lo > 0:
        return "INSECURE"
    return "UNCERTAIN"


def print_checks(report, st, model, margin, to_phys, indent="     "):
    """Per reachable action: N-0 status (enforced) and N-1 status (reported), with a falsification search for
    uncertain N-1 bounds."""
    summ = check_summary(report, to_phys)
    for a, o in sorted(summ.items()):
        print(f"{indent}after {name(a)}:")
        if o["fail"]:
            print(f"{indent}    N-0: FAILS       {o['fail']}")
            continue
        n0 = "OK" if (o["load"] <= 100 and o["slack"][0] >= 0 and o["slack"][1] <= 250) else "VIOLATION POSSIBLE"
        print(f"{indent}    N-0: {n0:<11} max loading {o['load']:.1f}%, slack {o['slack'][0]:.1f}..{o['slack'][1]:.1f} MW"
              f"   (certified for every load in the box)")
        stt = n1_status(o)
        lo, hi = o["n1"]
        if stt == "SECURE":
            print(f"{indent}    N-1: SECURE      count 0 for every load: no line outage overloads anything "
                  f"(worst post-outage loading <= {100 * o['n1w']:.0f}%)")
        elif stt == "INSECURE":
            why = (f": {o['n_island']} of {o['n_out']} line outages split the grid (PF fails, 3 points each)"
                   if o["n_island"] else "")
            cnt = f"count {lo}" if lo == hi else f"count between {lo} and {hi}"
            print(f"{indent}    N-1: INSECURE    {cnt} for every load{why}")
        elif stt == "UNCERTAIN":
            print(f"{indent}    N-1: UNCERTAIN   certified count between {lo} and {hi} over the box "
                  f"(worst post-outage loading <= {100 * o['n1w']:.0f}%)")
            c, w, what, d = falsify_n1(model, st.apply(a), o["boxes"], margin)
            if c > 0:
                print(f"{indent}                     -> REAL violation found: count {c} ({what}) at delta' = {np.round(d, 3)}")
            else:
                print(f"{indent}                     -> no violation found in 48 sampled loads (worst {w:.0f}%): "
                      f"the upper bound may be conservative")


def leaf_n1_shares(report):
    """Share of the box where all reachable actions are N-1 secure / some is uncertain / some is insecure."""
    out = dict(secure=0.0, uncertain=0.0, insecure=0.0)
    for L in report.leaves:
        if not L.observed:
            continue
        st = [n1_status(dict(n1=c.info.get("n1", (np.inf, -np.inf)))) for c in L.checks.values()]
        key = "insecure" if "INSECURE" in st else ("uncertain" if "UNCERTAIN" in st else "secure")
        out[key] += L.volume(report.lo, report.hi)
    return out


def secure_alternatives(model, st, U, checker, reachable):
    """Valid actions (mask) the agent does NOT take that are N-0 safe AND N-1 secure over the whole box."""
    good = []
    for a in np.where(st.mask())[0]:
        if a in reachable:
            continue
        c = checker(st, int(a), U.point(-np.ones(3)), U.point(np.ones(3)))
        if c.safe and c.info.get("pf") and c.info["n1"][1] == 0:
            good.append((int(a), float(np.max(c.info["load"]))))
    return good


def run_case(model, title, st, dc, w, margin, depth, nmc):
    t0 = time.time()
    checker = StepChecker(model, margin=margin, n1=True, require_n1=REQUIRE_N1)
    U, observe, mask, check = make_problem(model, st, dc, w, checker)
    o = observe(-np.ones(3), np.ones(3))
    print(f"\n=== {title}:  lines {st.status}  Pg2/3 = {st.pg} MW  V2/3 = {st.vg}  delta = {np.round(dc, 3)} +- {w}"
          f"  (next-step margin +-{margin})")
    if not o.ok:
        print("  pre-action PF NOT certified on the whole box (it will be split)")
    else:
        lo, hi = o.tm.bounds()
        print(f"  pre-action PF certified (rho {o.info['rho']:.3f}); observation ranges: slack {lo[6]:.1f}..{hi[6]:.1f} MW,"
              f" load P5 {lo[9]:.1f}..{hi[9]:.1f} MW, max line loading {hi[17:23].max():.1f}%, max trafo {hi[23:26].max():.1f}%")

    def observe_keep(lo_, hi_):                              # keep the obs bounds of every leaf for the MC check
        ob = observe(lo_, hi_)
        if ob.ok:
            ob.info["bounds"] = ob.tm.bounds()
        return ob

    V = DiscretePolicyVerifier(actor, observe_keep, 3, mask, check, refine="single", max_depth=depth, max_leaves=3000)
    R = V.verify()
    for L in R.leaves:
        L._obs_bounds = L.obs_info.get("bounds", (np.full(26, -np.inf), np.full(26, np.inf)))
    print(f"  partition: {len(R.leaves)} boxes (complete={R.complete}); reachable actions over the box: "
          f"{[name(a) for a in R.reachable_actions()]}")
    for acts, v in list(R.action_sets().items())[:6]:
        print(f"     {100 * v:6.2f}%  {' | '.join(name(a) for a in acts)}")
    sh = R.status_shares()
    n1s = leaf_n1_shares(R)
    print(f"  one-step safety, {N_RULE}: SAFE {100 * sh['SAFE']:.1f}%  CAN-VIOLATE {100 * sh['CAN-VIOLATE']:.1f}%  "
          f"UNDECIDED {100 * sh['UNDECIDED']:.1f}%  PF-UNKNOWN {100 * sh['UNKNOWN']:.1f}%   (share of the load box)")
    print(f"  N-1 of the agent's actions: secure on {100 * n1s['secure']:.1f}%, uncertain on {100 * n1s['uncertain']:.1f}%,"
          f" insecure on {100 * n1s['insecure']:.1f}% of the box")
    print_checks(R, st, model, margin, U.point)
    if n1s["secure"] < 1.0:
        alts = secure_alternatives(model, st, U, checker, set(R.reachable_actions()))
        if alts:
            print(f"  N-1-secure alternatives the agent never takes here (N-0 safe and N-1 count 0 for every load):")
            print("     " + ", ".join(f"{name(a)} (max loading {ld:.0f}%)" for a, ld in alts))
        else:
            print("  no valid action is certified N-0 safe AND N-1 secure over the whole box")
    n_ok, n_obs, n_act, n_post = monte_carlo(model, st, U, R, None, nmc)
    print(f"  Monte-Carlo ({n_ok} samples, float32 actor): observation inside certified range {n_obs}/{n_ok}, "
          f"chosen action in certified set {n_act}/{n_ok}, post-action inside certified band {n_post}/{n_ok}")
    print(f"  time {time.time() - t0:.1f}s")
    return R


def run_two_step(model, title, st, dc, w, margin, depth, nmc, seed=1):
    """Two-step reachability tree from (st, delta in dc +- w), next-step load change +-margin per step."""
    t0 = time.time()
    checker = StepChecker(model, margin=margin, n1=True, require_n1=REQUIRE_N1)
    problem, transition, propagate = multistep_problem(model, checker)
    MV = MultiStepVerifier(actor, problem, transition, propagate, steps=2, refine="single", max_depth=depth,
                           max_leaves=3000)
    lo, hi = np.asarray(dc) - w, np.asarray(dc) + w
    root = MV.verify(st, lo, hi)
    print(f"\n=== TWO-STEP reachability from {title}: delta = {np.round(dc, 3)} +- {w}, load change +-{margin} per step")
    print(f"  step 1: actions {[name(a) for a in root.report.reachable_actions()]}  "
          f"(safety {100 * root.report.status_shares()['SAFE']:.1f}% SAFE)")
    phys = lambda node: (lambda e: node.lo + 0.5 * (np.asarray(e) + 1.0) * (node.hi - node.lo))
    print_checks(root.report, st, model, margin, phys(root), indent="       ")
    for a, child in root.children.items():
        s2 = child.state
        print(f"  after {name(a)} -> Pg2/3 = {s2.pg} MW, V2/3 = {s2.vg}; step-2 load box delta in "
              f"[{np.round(child.lo, 3)}, {np.round(child.hi, 3)}]")
        for acts, v in list(child.report.action_sets().items())[:4]:
            print(f"       {100 * v:6.2f}%  {' | '.join(name(b) for b in acts)}")
        print_checks(child.report, child.state, model, margin, phys(child), indent="       ")
    paths = root.paths()
    print(f"  certified reachable 2-action sequences ({len(paths)}): {[' -> '.join(name(a) for a in p) for p in paths]}")
    print(f"  TWO-STEP SAFE ({N_RULE}): {root.safe}   (every reachable sequence)")
    n1all = [n1_status(o) for nd in root.walk() for o in check_summary(nd.report).values() if not o["fail"]]
    verdict = ("certified N-1 secure along every reachable sequence" if all(x == "SECURE" for x in n1all) else
               f"{n1all.count('INSECURE')} action outcomes certainly N-1 insecure, {n1all.count('UNCERTAIN')} uncertain "
               f"(see above), {n1all.count('SECURE')} secure")
    print(f"  N-1 over the tree (reported only): {verdict}")
    # Monte-Carlo trajectories: delta1 ~ box, delta2 = delta1 + U(-m, m); deployed float32 agent at both steps
    rng = np.random.default_rng(seed)
    seqs = set(paths)
    n = ok_seq = ok_band = 0
    for _ in range(nmc):
        d1 = rng.uniform(lo, hi)
        sim1 = model.simulate(st, d1)
        if sim1 is None:
            continue
        a1 = actor_f32(sim1["obs"], st.mask())
        s2 = st.apply(a1)
        d2 = d1 + rng.uniform(-margin, margin, 3)
        sim2 = model.simulate(s2, d2)
        if sim2 is None:
            continue
        a2 = actor_f32(sim2["obs"], s2.mask())
        d3 = d2 + rng.uniform(-margin, margin, 3)
        sim3 = model.simulate(s2.apply(a2), d3)
        n += 1
        ok_seq += (a1, a2) in seqs
        cs = check_summary(root.children[a1].report).get(a2) if a1 in root.children else None
        ok_band += bool(cs and sim3 is not None and np.all(sim3["loading"] <= cs["load"] + 1e-9)
                        and cs["slack"][0] - 1e-9 <= sim3["slack"] <= cs["slack"][1] + 1e-9)
    print(f"  Monte-Carlo ({n} random 2-step trajectories, float32 agent): sequence in certified set {ok_seq}/{n}, "
          f"step-2 post-action state inside certified band {ok_band}/{n}")
    print(f"  time {time.time() - t0:.1f}s")
    return root


GROUPS = {"generation (Pg slack, 2, 3)": range(6, 9), "loads (P, Q)": range(9, 15), "PV voltages": range(15, 17),
          "line loadings": range(17, 23), "transformer loadings": range(23, 26)}


def run_noise_robustness(model, title, st, delta, margin, eps_max=0.20, tol=5e-4):
    """Measurement-noise robustness of the agent's decision at one operating point (exact grid state)."""
    t0 = time.time()
    delta = np.asarray(delta, float)
    obs = model.simulate(st, delta)["obs"]
    n = len(obs)
    x = TM(obs, np.zeros((n, 0)), np.zeros((n, 0, 0)), np.zeros(n))       # the exact observation (a point)
    mask = st.mask()                                                         # env: mask from the true state
    meas = np.zeros(n)
    meas[6:] = np.abs(obs[6:])                                               # relative noise on every measurement
    checker = StepChecker(model, margin=margin, n1=True, require_n1=REQUIRE_N1)
    print(f"\n=== MEASUREMENT NOISE at {title}: true delta = {np.round(delta, 3)}, lines {st.status}, "
          f"Pg2/3 = {st.pg} MW; noise |eta_i| <= eps*|obs_i| on all 20 measured entries")
    r = robustness_radius(actor, x, meas, mask, "same", eps_max, tol)
    a0 = r.nominal
    print(f"  nominal action: {name(a0)}")
    fal = (f"{100 * r.falsified:.2f}% (noise found that makes it pick {name(r.counterexample['action'])})"
           if np.isfinite(r.falsified) else f"none found up to {100 * eps_max:.0f}%")
    print(f"  same-action radius: certified {100 * r.certified:.2f}%  |  breaking noise: {fal}")
    print("  per measurement group (only that group noisy):")
    for g, idx in GROUPS.items():
        sc = np.zeros(n)
        sc[list(idx)] = meas[list(idx)]
        rg = robustness_radius(actor, x, sc, mask, "same", eps_max, tol)
        fg = (f"breaks at {100 * rg.falsified:.2f}% -> {name(rg.counterexample['action'])}"
              if np.isfinite(rg.falsified) else f"no break found up to {100 * eps_max:.0f}%")
        cert = f">= {100 * rg.certified:.2f}%" if rg.certified < eps_max else f">= {100 * eps_max:.0f}% (whole range)"
        print(f"     {g:<28} certified {cert:<22} {fg}")

    def all_safe(acts):
        return all(checker(st, a, delta, delta).safe for a in acts)
    rs = robustness_radius(actor, x, meas, mask, all_safe, eps_max, tol)
    rule = "N-0 + N-1" if REQUIRE_N1 else "N-0 (unique PF, no overload, slack in [0, 250] MW)"
    print(f"  safe-action radius ({rule}, next-step load change +-{margin}): every action the agent can pick under "
          f"noise up to {100 * rs.certified:.2f}% is certified safe" + (" (whole tested range)" if rs.certified >= eps_max else ""))
    print("  certified action set at fixed noise levels:")
    for e in (0.005, 0.01, 0.02, 0.05, 0.10):
        acts = reachable_argmax(actor.reach(add_noise(x, e * meas)), mask).actions
        safe = all_safe(set(int(a) for a in acts))
        lst = " | ".join(name(int(a)) for a in acts[:6]) + (f" ... (+{len(acts) - 6} more, {len(acts)} in total)" if len(acts) > 6 else "")
        print(f"     +-{100 * e:4.1f}%: {lst:<78} {'all safe' if safe else 'NOT all safe'}")
    print(f"  time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sel =ARGS.ops if ARGS.ops is not None else list(range(len(OP_TITLES)))
    bad = [i for i in sel if not 0 <= i < len(OP_TITLES)]
    if bad:
        raise SystemExit(f"unknown operating point(s) {bad}; valid: 0..{len(OP_TITLES) - 1} (see --list-ops)")
    t_start = time.time()
    model = WSCC9Model()
    print(actor)
    print(f"weights: {os.path.relpath(NPZ)}   studies: {', '.join(ARGS.study)}   operating points: {sel}"
          f"   {'QUICK' if QUICK else 'full'} settings")
    w, margin, depth, nmc = (0.02, 0.0, 5, 100) if QUICK else (0.05, 0.06, 7, 300)
    ops = operating_points(model)
    chosen = [ops[i] for i in sel if ops[i] is not None]
    if "reach" in ARGS.study:
        for title, st, dc in chosen:
            run_case(model, title, st, dc, w, margin, depth, nmc)
    if "two-step" in ARGS.study:                            # two-step studies: reset and after the load jump
        for i in (0, len(OP_TITLES) - 1):
            if i in sel and ops[i] is not None:
                run_two_step(model, *ops[i], w, margin, depth, nmc)
    if "noise" in ARGS.study:                               # measurement-noise robustness at each operating point
        for title, st, dc in chosen:
            run_noise_robustness(model, title, st, dc, margin)
    print(f"\ntotal time {time.time() - t_start:.1f}s")
