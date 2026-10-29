"""Case studies: certified verification of the WSCC-9 MaskablePPO agent around operating points.

For each operating point (topology, Pg/V setpoints, load deviation delta_c) and a box delta in delta_c +- w:
  1. the pre-action power flow is certified (Krawczyk, admittance loads) and the agent's 26-entry
     observation is built as a Taylor model in delta -- CONSISTENT with the loads (coupled observation);
  2. the actor (tanh MLP, float32) is pushed through it and the masked argmax gives the certified set of
     actions the agent can take; the box is partitioned until each part has one action (or max depth);
  3. every reachable action is checked one step ahead: certified post-action power flow for the next load
     deviation delta' in [delta - margin, delta + margin]: PF solution exists and is unique, no overload
     (S <= rating), |V| in [0.95, 1.05] at every bus, slack within [0, 250] MW;
     + certified N-1 screening of the post-action state with the SAME limits after every line outage, reported
       as certified [lower, upper] bounds on the number of overloads, power-flow failures, (outage, bus) voltage
       violations and outages with a slack violation;
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

N-0 (enforced by default): the post-action PF has a unique solution, no branch is overloaded, |V| in [0.95, 1.05]
at every bus, slack in [0, 250] MW.
N-1 (reported; --require-n1 enforces it): the same limits after every line outage; each count (overloads, PF
failures, voltage violations, slack violations) is certified as an interval [lower, upper]:
    SECURE     all upper = 0      no outage violates anything, for every load
    INSECURE   some lower > 0     violated for every load in the box (e.g. outages that split the grid)
    UNCERTAIN  otherwise          possibly violated somewhere; a sampling search then either finds a real
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

from wscc9_env import (ACTIONS, LINES, SLACK_MAX, SLACK_MIN, VMAX_BUS, VMIN_BUS, XFMRS, OperatingState, StepChecker, WSCC9Model, infer_state_from_obs,  # noqa: E402
                       make_problem, multistep_problem, multistep_problem_history, step_generator)
from certflow.sets import AffineSet  # noqa: E402

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
    p.add_argument("--track", choices=("rebox", "history", "both"), default="rebox",
                   help="multi-step study: re-box the load at every step, track its history with new symbols per "
                        "step, or run both and compare speed and tightness (default: rebox)")
    p.add_argument("--step-change", choices=("independent", "common"), default="independent",
                   help="per-step load change: every load by up to +-margin independently, or all loads together")
    p.add_argument("--steps", type=int, default=2, help="horizon of the multi-step study (default: 2)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="maximum number of box bisections per verification (default: 5 with --quick, else 7); "
                        "history tracking has more symbols to split, so it may need a larger value")
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
LIMITS = "unique PF, no overload, |V| in [0.95, 1.05] at all buses, slack in [0, 250] MW"
N_RULE = (f"N-0 + N-1 enforced ({LIMITS})" if REQUIRE_N1 else f"N-0 enforced ({LIMITS}); N-1 (same limits) reported only")
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
                           and chk.info["slack"][0] - 1e-9 <= post["slack"] <= chk.info["slack"][1] + 1e-9
                           and np.all(post["Vm"] >= chk.info["vlo"] - 1e-9) and np.all(post["Vm"] <= chk.info["vhi"] + 1e-9))
    return n_ok, n_obs, n_act, n_post


def box_sampler(to_box, margin):
    """Next-load sampler per leaf for a re-boxed load: delta' in (leaf box of delta) +- margin."""
    def factory(L):
        lo, hi = to_box(L.lo, L.hi)
        lo, hi = lo - margin, hi + margin
        return lambda rng, corner: np.where(rng.random(len(lo)) < 0.5, lo, hi) if corner else rng.uniform(lo, hi)
    return factory


def set_sampler(U, step_gen):
    """Next-load sampler per leaf for a history-tracked load: delta' = U.c + U.G e + step_gen w (e in the leaf box)."""
    r = step_gen.shape[1]

    def factory(L):
        def f(rng, corner):
            e = np.where(rng.random(len(L.lo)) < 0.5, L.lo, L.hi) if corner else rng.uniform(L.lo, L.hi)
            w = rng.choice([-1.0, 1.0], r) if corner else rng.uniform(-1, 1, r)
            return U.c + U.G @ e + step_gen @ w
        return f
    return factory


def check_summary(report, sampler=None):
    """Per reachable action: union over the leaves of the certified post-action results (+ next-load samplers of
    the leaves where the action is reachable, for the N-1 falsification search)."""
    out = {}
    for L in report.leaves:
        for a, c in L.checks.items():
            i = c.info
            inf2 = (np.inf, -np.inf)
            o = out.setdefault(a, dict(load=0.0, slack=inf2, vm=inf2, n1o=inf2, n1f=inf2, n1v=inf2, n1s=inf2, n1w=0.0,
                                       n1vr=inf2, n1sr=inf2, n1det={}, fail=None, safe=True, n_island=0, n_out=0,
                                       samplers=[]))
            o["safe"] &= bool(c.safe)
            if sampler is not None:
                o["samplers"].append(sampler(L))
            if not i.get("pf"):
                o["fail"] = i.get("reason", "?")
                continue
            o["load"] = max(o["load"], float(np.max(i["load"])))
            hull = lambda u, v: (min(u[0], v[0]), max(u[1], v[1]))
            o["slack"] = hull(o["slack"], i["slack"])
            o["vm"] = hull(o["vm"], (i["vmin"], i["vmax"]))
            if "n1" in i:
                for k, ik in (("n1o", "n1_over"), ("n1f", "n1_fail"), ("n1v", "n1_v"), ("n1s", "n1_s"),
                              ("n1vr", "n1_vrange"), ("n1sr", "n1_srange")):
                    o[k] = hull(o[k], i[ik])
                o["n1w"] = max(o["n1w"], float(np.nan_to_num(i["n1_worst"])))
                o["n_island"] = sum(r.islanded for r in i["n1_report"].results)
                o["n_out"] = len(i["n1_report"].results)
                for r in i["n1_report"].results:                     # which outage violates what (union over leaves)
                    if r.islanded or not r.formal:
                        continue
                    out_name = "-".join(map(str, (LINES + XFMRS)[r.outage]))
                    items = [(f"|V| bus {b + 1}", r.v_lo[b], r.v_hi[b], bool(r.v_certain[b]), "pu")
                             for b in np.where(r.v_possible)[0]]
                    items += [(f"loading {'-'.join(map(str, (LINES + XFMRS)[k]))}", 100 * r.loading_lo[j],
                               100 * r.loading_hi[j], bool(r.over_certain[j]), "%")
                              for j, k in enumerate(i["n1_report"].monitored) if r.over_possible[j]]
                    if r.slack_possible:
                        items.append(("slack", r.slack_lo, r.slack_hi, bool(r.slack_certain), "MW"))
                    for what, vlo, vhi, cert, unit in items:
                        key = (out_name, what)
                        p = o["n1det"].get(key)
                        o["n1det"][key] = ((vlo, vhi, cert, unit) if p is None else
                                           (min(p[0], vlo), max(p[1], vhi), p[2] and cert, unit))
    return out


def n1_point(model, st, d):
    """N-1 at ONE load point (plain power flows), same limits as the certified check.  Returns
    (#overloads, #PF failures, #voltage violations, #slack violations, worst loading %, description)."""
    count, nf, nv, ns, worst, what = 0, 0, 0, 0, 0.0, []
    for li, on in enumerate(st.status):
        if not on:
            continue
        so = replace(st, status=tuple(0 if k == li else v for k, v in enumerate(st.status)))
        sim = model.simulate(so, d)
        tag = f"outage {LINES[li][0]}-{LINES[li][1]}"
        if sim is None:
            nf += 1
            what.append(f"{tag}: PF fails")
            continue
        ld = sim["loading"]
        count += int((ld > 100.0).sum())
        if (ld > 100.0).any():
            k = int(np.argmax(ld))
            br = (LINES + XFMRS)[k]
            what.append(f"{tag} -> branch {br[0]}-{br[1]} at {ld[k]:.0f}%")
        worst = max(worst, float(ld.max()))
        bad = np.where((sim["Vm"] < VMIN_BUS) | (sim["Vm"] > VMAX_BUS))[0]
        nv += len(bad)
        if len(bad):
            vs = ", ".join("%d = %.3f" % (b + 1, sim["Vm"][b]) for b in bad)
            what.append(f"{tag} -> |V| bus {vs}")
        if not SLACK_MIN <= sim["slack"] <= SLACK_MAX:
            ns += 1
            what.append(f"{tag} -> slack {sim['slack']:.1f} MW")
    return count, nf, nv, ns, worst, "; ".join(what)


def falsify_n1(model, st_post, samplers, n=48, seed=0):
    """Search for a REAL N-1 violation after an action: sample next-step loads delta' from the uncertainty set of
    the leaves where the action is reachable (every third sample at a corner).  Returns (counts (overloads,
    PF failures, V, slack), worst loading, description, delta') of the worst sample found."""
    rng = np.random.default_rng(seed)
    best = ((0, 0, 0, 0), 0.0, "", None)
    for k in range(n):
        d = samplers[rng.integers(len(samplers))](rng, k % 3 == 0)
        c, nf, nv, ns, w, what = n1_point(model, st_post, d)
        if (c + nf + nv + ns, w) > (sum(best[0]), best[1]):
            best = ((c, nf, nv, ns), w, what, d)
    return best


def n1_status(o):
    """SECURE / INSECURE / UNCERTAIN from the certified count intervals (overloads, PF failures, voltage, slack)."""
    ivs = [o[k] for k in ("n1o", "n1f", "n1v", "n1s") if k in o]
    if not ivs or not np.isfinite(ivs[0][0]):
        return "n/a"
    if all(hi == 0 for lo, hi in ivs):
        return "SECURE"
    if any(lo > 0 for lo, hi in ivs):
        return "INSECURE"
    return "UNCERTAIN"


def _iv(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}..{hi}"


def print_checks(report, st, model, sampler, indent="     "):
    """Per reachable action: N-0 status (enforced) and N-1 status (reported), with a falsification search for
    uncertain N-1 bounds (``sampler``: next-load sampler factory per leaf, see box_sampler / set_sampler)."""
    summ = check_summary(report, sampler)
    for a, o in sorted(summ.items()):
        print(f"{indent}after {name(a)}:")
        if o["fail"]:
            print(f"{indent}    N-0: FAILS       {o['fail']}")
            continue
        n0 = ("OK" if (o["load"] <= 100 and o["slack"][0] >= SLACK_MIN and o["slack"][1] <= SLACK_MAX
                       and o["vm"][0] >= VMIN_BUS and o["vm"][1] <= VMAX_BUS) else "VIOLATION POSSIBLE")
        print(f"{indent}    N-0: {n0:<11} max loading {o['load']:.1f}%, |V| {o['vm'][0]:.3f}..{o['vm'][1]:.3f} pu, "
              f"slack {o['slack'][0]:.1f}..{o['slack'][1]:.1f} MW   (certified for every load in the box)")
        stt = n1_status(o)
        if stt == "n/a":
            continue
        ranges = (f"worst post-outage loading <= {100 * o['n1w']:.0f}%, |V| {o['n1vr'][0]:.3f}..{o['n1vr'][1]:.3f} pu, "
                  f"slack {o['n1sr'][0]:.1f}..{o['n1sr'][1]:.1f} MW")
        counts = (f"overloads {_iv(*o['n1o'])}, PF failures {_iv(*o['n1f'])}, voltage violations {_iv(*o['n1v'])}, "
                  f"slack violations {_iv(*o['n1s'])}")
        det = [f"outage {k[0]}: {k[1]} {v[0]:.3f}..{v[1]:.3f} {v[3]}{' (always out of limits)' if v[2] else ''}"
               for k, v in sorted(o["n1det"].items())]
        if stt == "SECURE":
            print(f"{indent}    N-1: SECURE      no line outage violates any limit, for every load ({ranges})")
        elif stt == "INSECURE":
            why = (f"; {o['n_island']} of {o['n_out']} line outages split the grid" if o["n_island"] else "")
            print(f"{indent}    N-1: INSECURE    violated for every load: {counts}{why}")
            if np.isfinite(o["n1vr"][0]):
                print(f"{indent}                     (non-islanding outages: {ranges})")
            for t in det:
                print(f"{indent}                     {t}")
        else:
            print(f"{indent}    N-1: UNCERTAIN   certified bounds over the box: {counts}")
            print(f"{indent}                     ({ranges})")
            for t in det:
                print(f"{indent}                     {t}")
            (c, nf, nv, ns), w, what, d = falsify_n1(model, st.apply(a), o["samplers"])
            if c + nf + nv + ns > 0:
                print(f"{indent}                     -> REAL violation found at delta' = {np.round(d, 3)}: overloads {c}, "
                      f"PF failures {nf}, voltage {nv}, slack {ns} ({what})")
            else:
                print(f"{indent}                     -> no violation found in 48 sampled loads (worst loading {w:.0f}%): "
                      f"the upper bounds may be conservative")


def leaf_n1_shares(report):
    """Share of the box where all reachable actions are N-1 secure / some is uncertain / some is insecure."""
    out = dict(secure=0.0, uncertain=0.0, insecure=0.0)
    for L in report.leaves:
        if not L.observed:
            continue
        st = [n1_status(dict(n1o=c.info.get("n1_over", (np.inf, -np.inf)), n1f=c.info.get("n1_fail", (0, 0)),
                             n1v=c.info.get("n1_v", (0, 0)), n1s=c.info.get("n1_s", (0, 0)))) for c in L.checks.values()]
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
        if c.safe and c.info.get("pf") and c.info.get("n1_secure"):
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
    print_checks(R, st, model, box_sampler(lambda lo, hi: (U.point(lo), U.point(hi)), margin))
    if n1s["secure"] < 1.0:
        alts = secure_alternatives(model, st, U, checker, set(R.reachable_actions()))
        if alts:
            print(f"  N-1-secure alternatives the agent never takes here (N-0 safe and N-1 secure for every load):")
            print("     " + ", ".join(f"{name(a)} (max loading {ld:.0f}%)" for a, ld in alts))
        else:
            print("  no valid action is certified N-0 safe AND N-1 secure over the whole box")
    n_ok, n_obs, n_act, n_post = monte_carlo(model, st, U, R, None, nmc)
    print(f"  Monte-Carlo ({n_ok} samples, float32 actor): observation inside certified range {n_obs}/{n_ok}, "
          f"chosen action in certified set {n_act}/{n_ok}, post-action inside certified band {n_post}/{n_ok}")
    print(f"  time {time.time() - t0:.1f}s")
    return R


def node_sampler(node, margin, step_gen):
    """Next-load sampler factory for the leaves of a multi-step tree node (either tracking mode)."""
    if node.U is None:
        to_box = lambda a, b: (node.lo + 0.5 * (a + 1.0) * (node.hi - node.lo), node.lo + 0.5 * (b + 1.0) * (node.hi - node.lo))
        return box_sampler(to_box, np.abs(step_gen).sum(1))
    return set_sampler(node.U, step_gen)


def path_bounds(root):
    """Per reachable action sequence: union over the tree's last-step nodes of the certified post-action results."""
    out = {}

    def rec(node, prefix):
        if not node.children:
            for a, o in check_summary(node.report).items():
                key = prefix + (a,)
                p = out.get(key)
                if p is None or o["fail"]:
                    out[key] = dict(o) if p is None else dict(p, fail=o["fail"] or p["fail"])
                    continue
                if p["fail"]:
                    continue
                hull = lambda u, v: (min(u[0], v[0]), max(u[1], v[1]))
                out[key] = dict(p, load=max(p["load"], o["load"]), vm=hull(p["vm"], o["vm"]),
                                slack=hull(p["slack"], o["slack"]), safe=p["safe"] and o["safe"],
                                **{k: hull(p[k], o[k]) for k in ("n1o", "n1f", "n1v", "n1s")})
            return
        for k, c in node.children.items():
            rec(c, prefix + ((k[0] if isinstance(k, tuple) else k),))
    rec(root, ())
    return out


def run_multistep(model, title, st, dc, w, margin, depth, nmc, track="rebox", change="independent", steps=2, seed=1):
    """Multi-step reachability tree from (st, delta in dc +- w), per-step load change delta' = delta + G_step w'
    (``change``: every load by up to +-margin, or all loads together).  ``track``: "rebox" (fresh box per step) or
    "history" (new symbols per step, correlation between steps kept)."""
    t0 = time.time()
    G = step_generator(margin, change)
    checker = StepChecker(model, margin=float(np.abs(G).sum(1).max()), n1=True, require_n1=REQUIRE_N1)
    U0 = AffineSet(np.asarray(dc, float) * np.ones(3), w * np.eye(3))
    kw = dict(steps=steps, refine="single", max_depth=depth, max_leaves=3000)
    if track == "rebox":
        problem, transition, propagate = multistep_problem(model, checker)
        root = MultiStepVerifier(actor, problem, transition, propagate, **kw).verify(st, U0.c - w, U0.c + w)
    else:
        problem, transition, extend = multistep_problem_history(model, checker, G)
        root = MultiStepVerifier(actor, problem, transition, track="history", extend=extend, **kw).verify(st, U=U0)
    stats = root.stats()
    print(f"\n=== {steps}-STEP reachability from {title} [{track.upper()}]: delta = {np.round(dc, 3)} +- {w}, "
          f"load change per step: {change} +-{margin}")

    def show(node, prefix, ind):
        acts = [name(a) for a in node.report.reachable_actions()]
        lo, hi = node.input_box()
        print(f"{ind}step {node.depth + 1}{' after ' + ' -> '.join(name(a) for a in prefix) if prefix else ''}: "
              f"Pg2/3 = {node.state.pg} MW, V2/3 = {node.state.vg}; load box [{np.round(lo, 3)}, {np.round(hi, 3)}]"
              f"{f', {len(node.lo)} symbols' if node.U is not None else ''}; actions {acts} "
              f"({100 * node.report.status_shares()['SAFE']:.1f}% SAFE)")
        if not node.children:
            print_checks(node.report, node.state, model, node_sampler(node, margin, G), indent=ind + "  ")
        for k, c in node.children.items():
            show(c, prefix + ((k[0] if isinstance(k, tuple) else k),), ind + "  ")
    show(root, (), "  ")
    paths = root.paths()
    print(f"  certified reachable {steps}-action sequences ({len(paths)}): {[' -> '.join(name(a) for a in p) for p in paths]}")
    print(f"  {steps}-STEP SAFE ({N_RULE}): {root.safe}   (every reachable sequence)")
    n1all = [n1_status(o) for o in path_bounds(root).values() if not o["fail"]]
    verdict = ("certified N-1 secure along every reachable sequence" if all(x == "SECURE" for x in n1all) else
               f"{n1all.count('INSECURE')} sequences certainly N-1 insecure at the last step, "
               f"{n1all.count('UNCERTAIN')} uncertain, {n1all.count('SECURE')} secure")
    print(f"  N-1 at the last step (reported only): {verdict}")
    # Monte-Carlo trajectories from the same uncertainty model; deployed float32 agent at every step
    rng = np.random.default_rng(seed)
    seqs = set(paths)
    n = ok_seq = ok_band = 0
    r = G.shape[1]
    for _ in range(nmc):
        e = rng.uniform(-1, 1, 3)
        d, s_, acts, node, ok = U0.point(e), st, [], root, True
        for k in range(steps):
            sim = model.simulate(s_, d)
            if sim is None:
                ok = False
                break
            a = actor_f32(sim["obs"], s_.mask())
            acts.append(a)
            wk = rng.uniform(-1, 1, r)
            if k + 1 < steps:                                   # find the child that covers this trajectory
                e = np.r_[e, wk]
                d = d + G @ wk
                kids = [c for kk, c in node.children.items() if (kk[0] if isinstance(kk, tuple) else kk) == a]
                inside = (lambda c: np.all(e >= c.lo - 1e-12) and np.all(e <= c.hi + 1e-12)) if track == "history" else \
                         (lambda c: np.all(d >= c.lo - 1e-12) and np.all(d <= c.hi + 1e-12))
                node = next((c for c in kids if inside(c)), None)
                s_ = s_.apply(a)
                if node is None:
                    break
            else:
                post = model.simulate(s_.apply(a), d + G @ wk)
        if not ok:
            continue
        n += 1
        ok_seq += tuple(acts) in seqs
        if node is not None and len(acts) == steps:
            cs = check_summary(node.report).get(acts[-1])
            ok_band += bool(cs and post is not None and np.all(post["loading"] <= cs["load"] + 1e-9)
                            and cs["slack"][0] - 1e-9 <= post["slack"] <= cs["slack"][1] + 1e-9
                            and cs["vm"][0] - 1e-9 <= post["Vm"].min() and post["Vm"].max() <= cs["vm"][1] + 1e-9)
    print(f"  Monte-Carlo ({n} random {steps}-step trajectories, float32 agent): sequence in certified set {ok_seq}/{n}, "
          f"last post-action state inside certified band {ok_band}/{n}")
    print(f"  tree: {stats['nodes']} nodes, {stats['leaves']} boxes, up to {stats['max_symbols']} symbols; "
          f"verification {stats['seconds']:.1f}s, total {time.time() - t0:.1f}s")
    return root, dict(stats, total=time.time() - t0, paths=paths, safe=root.safe, bounds=path_bounds(root))


def compare_tracking(title, res):
    """Side-by-side cost and tightness of re-boxing vs history tracking for the same multi-step study."""
    rb, hi = res["rebox"], res["history"]
    print(f"\n=== REBOX vs HISTORY from {title}")
    row = lambda lab, a, b: print(f"  {lab:<34} {str(a):>22}   {str(b):>22}")
    row("", "rebox", "history")
    row("total time [s]", f"{rb['total']:.1f}", f"{hi['total']:.1f}")
    row("tree nodes / boxes", f"{rb['nodes']} / {rb['leaves']}", f"{hi['nodes']} / {hi['leaves']}")
    row("max symbols per node", rb["max_symbols"], hi["max_symbols"])
    row("reachable sequences", len(rb["paths"]), len(hi["paths"]))
    row("all sequences N-0 safe", rb["safe"], hi["safe"])
    only = sorted(set(rb["paths"]) - set(hi["paths"]))
    if only:
        print(f"  sequences only in the re-boxed tree (excluded by tracking): {[' -> '.join(name(a) for a in p) for p in only]}")
    print("  certified last-step post-action bounds per sequence (rebox | history):")
    for p in sorted(set(rb["paths"]) | set(hi["paths"])):
        lab = " -> ".join(name(a) for a in p)
        f = lambda o: ("-" if o is None else "PF fails" if o["fail"] else
                       f"load<={o['load']:.1f}% |V| {o['vm'][0]:.3f}..{o['vm'][1]:.3f} slack {o['slack'][0]:.1f}..{o['slack'][1]:.1f}")
        print(f"    {lab}")
        print(f"      rebox:   {f(rb['bounds'].get(p))}")
        print(f"      history: {f(hi['bounds'].get(p))}")


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
    rule = f"N-0 + N-1, {LIMITS}" if REQUIRE_N1 else f"N-0: {LIMITS}"
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
    sel = ARGS.ops if ARGS.ops is not None else list(range(len(OP_TITLES)))
    bad = [i for i in sel if not 0 <= i < len(OP_TITLES)]
    if bad:
        raise SystemExit(f"unknown operating point(s) {bad}; valid: 0..{len(OP_TITLES) - 1} (see --list-ops)")
    t_start = time.time()
    model = WSCC9Model()
    print(actor)
    print(f"weights: {os.path.relpath(NPZ).replace(os.sep, '/')}   studies: {', '.join(ARGS.study)}   operating points: {sel}"
          f"   {'QUICK' if QUICK else 'full'} settings")
    w, margin, depth, nmc = (0.02, 0.0, 5, 100) if QUICK else (0.05, 0.06, 7, 300)
    depth = ARGS.max_depth if ARGS.max_depth is not None else depth
    ops = operating_points(model)
    chosen = [ops[i] for i in sel if ops[i] is not None]
    if "reach" in ARGS.study:
        for title, st, dc in chosen:
            run_case(model, title, st, dc, w, margin, depth, nmc)
    if "two-step" in ARGS.study:                            # multi-step studies: reset and after the load jump
        tracks = ("rebox", "history") if ARGS.track == "both" else (ARGS.track,)
        for i in (0, len(OP_TITLES) - 1):
            if i in sel and ops[i] is not None:
                res = {tr: run_multistep(model, *ops[i], w, margin, depth, nmc, track=tr, change=ARGS.step_change,
                                         steps=ARGS.steps)[1] for tr in tracks}
                if len(tracks) == 2:
                    compare_tracking(ops[i][0], res)
    if "noise" in ARGS.study:                               # measurement-noise robustness at each operating point
        for title, st, dc in chosen:
            run_noise_robustness(model, title, st, dc, margin)
    print(f"\ntotal time {time.time() - t_start:.1f}s")
