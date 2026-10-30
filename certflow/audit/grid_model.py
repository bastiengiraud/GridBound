"""A certified model of a pandapower grid as an RL agent sees and changes it.

You describe four things, in pandapower terms:

    observation  what the agent reads      ObservationSpec([obs.bus_vm(), obs.line_loading(), obs.sgen_p(), ...])
    actions      what the agent can do     ActionSpace([act.idle(), act.shift_tap(0, +1), act.set_sgen(3, q_mvar=0.5)])
    limits       what "safe" means         Limits(vm_pu=(0.95, 1.05), max_loading_percent=100, n1=False)
    uncertainty  which injections vary     LoadUncertainty(rel=0.05)   (given per audit, around a snapshot)

and :class:`GridModel` turns them into what the verifier needs: the observation as a Taylor model over the
uncertain loads (through a certified AC power flow), the effect of each action, and a certified screening of the
post-action grid against the limits (N-0, optionally N-1).  The same model also evaluates single points (plain
Newton power flow), which is used for Monte-Carlo cross-checks and for finding concrete counterexamples, and it can
compare itself against pandapower (:meth:`GridModel.check_against_pandapower`).

Modelling assumptions (checked or warned about at construction):
  * loads are constant-power (ZIP shares are ignored, with a warning); sgens inject their setpoint P, Q;
  * gens are ideal PV units (no reactive limits enforced in the power flow); one ext_grid is the slack;
  * loading_percent follows pandapower (lines: current vs max_i_ka; trafos: current at nominal voltage vs sn_mva);
  * switches keep the state they have in the net; actions switch lines / trafos through ``in_service``.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from ..power import quantities as qty
from ..power.grid import from_pandapower
from ..power.pf import newton
from ..power.pfcert import enclose_pf, enclose_pf_split
from ..sets import AffineSet
from ..split import bisect
from ..tm import TM, add, compose, concat, matmul, mul, scale, sub

DEG = 180.0 / np.pi


def _const(c, like: TM) -> TM:
    """Constant vector c as a TM over the same symbols as ``like`` (any length)."""
    c = np.atleast_1d(np.asarray(c, float))
    K, kq = like.C.shape[-1], like.Q.shape[-1]
    return TM(c, np.zeros((len(c), K)), np.zeros((len(c), kq, kq)), np.zeros(len(c)))


# ==================================================================================================
# state: the controllable / discrete part of the grid
# ==================================================================================================

@dataclass(frozen=True)
class GridState:
    """Everything the agent controls (or that changes only through actions).  The uncertain injections (loads) are
    NOT part of the state.  Tuples are in the order of ``GridModel.sgen_index``, ``gen_index``, ``trafo_index``,
    ``line_index``."""
    sgen_p: tuple = ()        # MW (setpoint, before pandapower's 'scaling')
    sgen_q: tuple = ()        # MVAr
    gen_p: tuple = ()         # MW
    gen_vm: tuple = ()        # pu
    tap: tuple = ()           # tap_pos per trafo
    line_on: tuple = ()
    trafo_on: tuple = ()

    def with_(self, field_name, k, value):
        v = list(getattr(self, field_name))
        v[k] = value
        return replace(self, **{field_name: tuple(v)})


# ==================================================================================================
# actions
# ==================================================================================================

class Action:
    """One discrete action: ``apply(state, model) -> state`` and ``valid(state, model)`` (used for the default
    action mask: e.g. a tap change beyond the tap range is invalid)."""
    label = "action"

    def apply(self, state, model):
        return state

    def valid(self, state, model):
        return True

    def __repr__(self):
        return self.label


class _Idle(Action):
    label = "idle"


class _Sgen(Action):
    def __init__(self, i, p=None, q=None, relative=False, label=None):
        self.i, self.p, self.q, self.rel = i, p, q, relative
        if label is None:
            sgn = lambda v: f"{v:+g}" if relative else f"={v:g}"
            parts = ([f"P{sgn(p)} MW"] if p is not None else []) + ([f"Q{sgn(q)} MVAr"] if q is not None else [])
            label = f"sgen {i} " + ", ".join(parts)
        self.label = label

    def _new(self, state, model):
        k = model.sgen_pos(self.i)
        p, q = state.sgen_p[k], state.sgen_q[k]
        if self.p is not None:
            p = p + self.p if self.rel else self.p
        if self.q is not None:
            q = q + self.q if self.rel else self.q
        return k, p, q

    def apply(self, state, model):
        k, p, q = self._new(state, model)
        lo, hi = model.sgen_p_range[k]
        p = float(np.clip(p, lo, hi))
        qlo, qhi = model.sgen_q_range[k]
        q = float(np.clip(q, qlo, qhi))
        return state.with_("sgen_p", k, round(p, 9)).with_("sgen_q", k, round(q, 9))

    def valid(self, state, model):
        k, p, q = self._new(state, model)
        lo, hi = model.sgen_p_range[k]
        qlo, qhi = model.sgen_q_range[k]
        return bool(lo - 1e-9 <= p <= hi + 1e-9 and qlo - 1e-9 <= q <= qhi + 1e-9)


class _Gen(Action):
    def __init__(self, i, p=None, vm=None, relative=False, label=None):
        self.i, self.p, self.vm, self.rel = i, p, vm, relative
        if label is None:
            sgn = lambda v, u: (f"{v:+g}" if relative else f"={v:g}") + u
            parts = ([f"P{sgn(p, ' MW')}"] if p is not None else []) + ([f"V{sgn(vm, ' pu')}"] if vm is not None else [])
            label = f"gen {i} " + ", ".join(parts)
        self.label = label

    def _new(self, state, model):
        k = model.gen_pos(self.i)
        p, vm = state.gen_p[k], state.gen_vm[k]
        if self.p is not None:
            p = p + self.p if self.rel else self.p
        if self.vm is not None:
            vm = vm + self.vm if self.rel else self.vm
        return k, p, vm

    def apply(self, state, model):
        k, p, vm = self._new(state, model)
        return state.with_("gen_p", k, round(float(p), 9)).with_("gen_vm", k, round(float(vm), 9))

    def valid(self, state, model):
        k, p, vm = self._new(state, model)
        lo, hi = model.gen_p_range[k]
        return bool(lo - 1e-9 <= p <= hi + 1e-9)


class _Tap(Action):
    def __init__(self, t, pos=None, step=None, label=None):
        self.t, self.pos, self.step = t, pos, step
        self.label = label or (f"trafo {t} tap {step:+d}" if step is not None else f"trafo {t} tap ={pos}")

    def _new(self, state, model):
        k = model.trafo_pos(self.t)
        return k, (state.tap[k] + self.step if self.step is not None else self.pos)

    def apply(self, state, model):
        k, v = self._new(state, model)
        lo, hi = model.tap_range[k]
        return state.with_("tap", k, int(np.clip(v, lo, hi)))

    def valid(self, state, model):
        k, v = self._new(state, model)
        lo, hi = model.tap_range[k]
        return bool(lo <= v <= hi)


class _Switch(Action):
    def __init__(self, kind, i, on, label=None):
        self.kind, self.i, self.on = kind, i, on
        self.label = label or f"{'close' if on else 'open'} {kind} {i}"

    def _k(self, model):
        return model.line_pos(self.i) if self.kind == "line" else model.trafo_pos(self.i)

    def apply(self, state, model):
        return state.with_(f"{self.kind}_on", self._k(model), bool(self.on))

    def valid(self, state, model):
        return bool(getattr(state, f"{self.kind}_on")[self._k(model)] != bool(self.on))


class _Combo(Action):
    def __init__(self, actions, label=None):
        self.actions = list(actions)
        self.label = label or " + ".join(a.label for a in self.actions)

    def apply(self, state, model):
        for a in self.actions:
            state = a.apply(state, model)
        return state

    def valid(self, state, model):
        return all(a.valid(state, model) for a in self.actions)


class _Custom(Action):
    def __init__(self, fn, label, valid=None):
        self.fn, self.label, self._valid = fn, label, valid

    def apply(self, state, model):
        return self.fn(state, model)

    def valid(self, state, model):
        return True if self._valid is None else bool(self._valid(state, model))


class act:
    """Action constructors (pandapower element indices).

    act.idle()                                   do nothing
    act.set_sgen(i, p_mw=None, q_mvar=None)      sgen i's setpoint to absolute values
    act.shift_sgen(i, dp_mw=0, dq_mvar=0)        sgen i's setpoint by a step
    act.set_gen(i, p_mw=None, vm_pu=None) / act.shift_gen(i, dp_mw=0, dvm_pu=0)
    act.set_tap(t, pos) / act.shift_tap(t, step) transformer t's tap position
    act.switch_line(l, on) / act.switch_trafo(t, on)
    act.combo(a, b, ...)                          several changes at once
    act.custom(fn, label, valid=None)            anything else: fn(state, model) -> new GridState
    Setpoints / taps outside their range are clipped when applied (and the action counts as invalid for the mask)."""
    idle = staticmethod(lambda: _Idle())
    set_sgen = staticmethod(lambda i, p_mw=None, q_mvar=None, label=None: _Sgen(i, p_mw, q_mvar, False, label))
    shift_sgen = staticmethod(lambda i, dp_mw=None, dq_mvar=None, label=None: _Sgen(i, dp_mw, dq_mvar, True, label))
    set_gen = staticmethod(lambda i, p_mw=None, vm_pu=None, label=None: _Gen(i, p_mw, vm_pu, False, label))
    shift_gen = staticmethod(lambda i, dp_mw=None, dvm_pu=None, label=None: _Gen(i, dp_mw, dvm_pu, True, label))
    set_tap = staticmethod(lambda t, pos, label=None: _Tap(t, pos=int(pos), label=label))
    shift_tap = staticmethod(lambda t, step, label=None: _Tap(t, step=int(step), label=label))
    switch_line = staticmethod(lambda l, on, label=None: _Switch("line", l, on, label))
    switch_trafo = staticmethod(lambda t, on, label=None: _Switch("trafo", t, on, label))
    combo = staticmethod(lambda *a, label=None: _Combo(a, label))
    custom = staticmethod(lambda fn, label, valid=None: _Custom(fn, label, valid))


class ActionSpace:
    """The agent's discrete actions, in the order of its output logits (Q-values).

    heads   : None for one categorical output (gym Discrete).  For a MultiDiscrete agent (one categorical per
              device), pass ``ActionSpace.multi([[actions of head 0], [actions of head 1], ...])``: the network
              outputs the concatenated logits of all heads and every head is an independent argmax.
    masked  : True if the deployed agent masks invalid actions (MaskablePPO).  Only then is the mask applied:
              masking an agent that was not trained with a mask would verify a different policy.
    mask_fn : optional mask_fn(state, model) -> bool array replacing the default (each action's ``valid``)."""

    def __init__(self, actions, masked=False, mask_fn=None, heads=None):
        self.actions = list(actions)
        self.masked = bool(masked)
        self.mask_fn = mask_fn
        self.heads = None if heads is None else [int(h) for h in heads]

    @classmethod
    def multi(cls, heads, masked=False, mask_fn=None):
        flat = [a for h in heads for a in h]
        return cls(flat, masked, mask_fn, heads=[len(h) for h in heads])

    def __len__(self):
        return len(self.actions)

    @property
    def n_joint(self):
        return len(self.actions) if self.heads is None else int(np.prod(self.heads))

    def joint(self, a):
        """The Action for joint action index a (flat over the heads for a MultiDiscrete space)."""
        if self.heads is None:
            return self.actions[a]
        idx = np.unravel_index(int(a), self.heads)
        off = np.r_[0, np.cumsum(self.heads)[:-1]]
        parts = [self.actions[o + i] for o, i in zip(off, idx)]
        parts = [p for p in parts if not isinstance(p, _Idle)] or parts[:1]
        return parts[0] if len(parts) == 1 else _Combo(parts)

    def label(self, a):
        return self.joint(a).label

    def mask(self, state, model):
        """Valid-action mask over the network outputs (all True if the agent is not masked)."""
        if not self.masked:
            return np.ones(len(self.actions), bool)
        if self.mask_fn is not None:
            return np.asarray(self.mask_fn(state, model), bool)
        return np.array([a.valid(state, model) for a in self.actions], bool)


# ==================================================================================================
# observation
# ==================================================================================================

class _Ctx:
    """What an observation item needs, either set-valued (enc: certified PF, u: TM) or at a point (V, u: array)."""

    def __init__(self, model, state, grid, u, enc=None, V=None, P=None, Q=None, Pspec=None, Qspec=None):
        self.model, self.state, self.grid, self.u = model, state, grid, u
        self.enc, self.V, self.P, self.Q, self.Pspec, self.Qspec = enc, V, P, Q, Pspec, Qspec

    @property
    def is_set(self):
        return self.enc is not None


class ObsItem:
    """One block of the observation.  ``values(ctx)`` returns a TM (set) or an array (point); ``names(model)`` the
    entry names; ``noisy`` whether the entries are measurements (subject to measurement noise) rather than values
    the agent knows exactly (its own setpoints, tap positions, switch states)."""

    def __init__(self, kind, elements, fn, unit, noisy, scale=1.0, offset=0.0):
        self.kind, self.elements, self.fn, self.unit, self.noisy = kind, elements, fn, unit, noisy
        self.scale, self.offset = scale, offset

    def ids(self, model):
        return model.default_ids(self.kind) if self.elements is None else list(self.elements)

    def names(self, model):
        return [f"{self.kind}[{i}]" for i in self.ids(model)]

    def values(self, ctx):
        v = self.fn(ctx, self.ids(ctx.model))
        if self.scale == 1.0 and self.offset == 0.0:
            return v
        n = len(self.ids(ctx.model))
        a = np.broadcast_to(np.asarray(self.scale, float), (n,))
        b = np.broadcast_to(np.asarray(self.offset, float), (n,))
        if isinstance(v, TM):
            return add(scale(a, v), _const(b, v))
        return a * v + b


def _cst(ctx, v):
    v = np.asarray(v, float)
    return _const(v, ctx.u) if ctx.is_set else v


def _vm(ctx, ids):
    b = ctx.model.buses(ids)
    return qty.vmag(ctx.enc, b) if ctx.is_set else np.abs(ctx.V[b])


def _atan_d2(lo, hi):
    g = lambda x: -2 * x / (1 + x * x) ** 2
    pts = [lo, hi] + [np.clip(np.full_like(lo, c), lo, hi) for c in (-1 / np.sqrt(3), 1 / np.sqrt(3))]
    vals = np.array([g(p) for p in pts])
    return vals.min(0) - 1e-12, vals.max(0) + 1e-12


def _va(ctx, ids):
    b = ctx.model.buses(ids)
    if not ctx.is_set:
        return np.angle(ctx.V[b]) * DEG
    vr, vi = ctx.enc.Vr.take(b), ctx.enc.Vi.take(b)
    lo, _ = vr.bounds()
    if np.any(lo <= 0):
        raise ValueError("bus angle observation needs Re(V) > 0 (|angle| < 90 deg) over the set")
    inv = compose(vr, lambda x: 1.0 / x, lambda x: -1.0 / x ** 2, lambda a, c: (2.0 / c ** 3, 2.0 / a ** 3))
    t = compose(mul(vi, inv), np.arctan, lambda x: 1.0 / (1.0 + x * x), _atan_d2)
    return scale(np.full(len(b), DEG), t)


def _loading(kind):
    def fn(ctx, ids):
        m, g = ctx.model, ctx.grid
        rows = m.branch_rows(kind, ids)
        fac = g.sn / np.where(g.rate_a[rows] > 0, g.rate_a[rows], np.inf) * 100.0
        if ctx.is_set:
            on = np.where(g.status[rows])[0]                   # out-of-service branches carry no current: 0 %
            if not len(on):
                return _const(np.zeros(len(rows)), ctx.u)
            dang = {k: m.dangling(g, rows[k]) for k in on}
            gen = [k for k in on if dang[k] is None]
            ends = [k for k in on if dang[k] is not None]
            parts, pos = [], []
            if gen:
                parts.append(qty.branch_current_magnitude(g, ctx.enc, [rows[k] for k in gen]))
                pos += gen
            if ends:                                           # |I| = |a| |V_bus| exactly (see GridModel.dangling)
                vb = qty.vmag(ctx.enc, [dang[k][0] for k in ends])
                parts.append(scale(np.array([dang[k][1] for k in ends]), vb))
                pos += ends
            T = concat(parts)
            S = np.zeros((len(rows), len(pos)))
            S[pos, np.arange(len(pos))] = fac[pos]
            return matmul(S, T)
        I = np.maximum(np.abs(g.Yf[rows] @ ctx.V), np.abs(g.Yt[rows] @ ctx.V))
        return fac * I
    return fn


def _flow(kind, end, pq):
    def fn(ctx, ids):
        m, g = ctx.model, ctx.grid
        rows = m.branch_rows(kind, ids)
        if ctx.is_set:
            P, Q = qty.branch_end_flows(g, ctx.enc, rows, end)
            return scale(np.full(len(rows), g.sn), P if pq == "p" else Q)
        Y, bus = (g.Yf, g.f) if end == "from" else (g.Yt, g.t)
        S = ctx.V[bus[rows]] * np.conj(Y[rows] @ ctx.V) * g.sn
        return S.real if pq == "p" else S.imag
    return fn


def _uncertain_or_state(table, attr):
    def fn(ctx, ids):
        m = ctx.model
        rows = [m.inj_row(table, i, attr) for i in ids]
        if table == "sgen":                                    # setpoint (state) unless the sgen is uncertain
            st = ctx.state.sgen_p if attr == "p" else ctx.state.sgen_q
            base = np.array([st[m.sgen_pos(i)] * m.sgen_scaling[m.sgen_pos(i)] for i in ids], float)
        else:
            base = np.zeros(len(ids))
        if all(r is None for r in rows):
            return _cst(ctx, base)
        sel = np.zeros((len(ids), m.n_inj))
        for k, r in enumerate(rows):
            if r is not None:
                sel[k, r] = 1.0
                base[k] = 0.0
        if ctx.is_set:
            return add(_const(base, ctx.u), matmul(sel, ctx.u))
        return base + sel @ ctx.u
    return fn


def _state_value(field_name, pos_fn, mult=None):
    def fn(ctx, ids):
        m = ctx.model
        v = [getattr(ctx.state, field_name)[pos_fn(m, i)] * (1.0 if mult is None else mult(m, i)) for i in ids]
        return _cst(ctx, np.asarray(v, float))
    return fn


def _ext(pq):
    def fn(ctx, ids):
        m, g = ctx.model, ctx.grid
        slk = g.slk
        if ctx.is_set:
            inj = ctx.enc.P if pq == "p" else ctx.enc.Q
            spec = ctx.Pspec if pq == "p" else ctx.Qspec
            return scale(np.array([g.sn]), sub(inj.take([slk]), spec.take([slk])))
        inj = ctx.P if pq == "p" else ctx.Q
        spec = ctx.Pspec if pq == "p" else ctx.Qspec
        return np.array([(inj[slk] - spec[slk]) * g.sn])
    return fn


def _custom(fn_set, fn_point):
    return lambda ctx, ids: fn_set(ctx) if ctx.is_set else fn_point(ctx)


class obs:
    """Observation blocks, in pandapower element indices and pandapower result units.  ``elements=None`` means all
    (in-service) elements of that table, in index order.  Every block takes ``scale``/``offset`` (an affine map
    applied to it, e.g. a fixed normalisation) and ``noisy`` (override the default measurement flag).

    obs.bus_vm()      res_bus.vm_pu               obs.bus_va()        res_bus.va_degree
    obs.line_loading()  res_line.loading_percent  obs.trafo_loading() res_trafo.loading_percent
    obs.line_p_from() / line_q_from() / trafo_p_hv() / trafo_q_hv()   branch flows (MW / MVAr)
    obs.load_p() / load_q()                       res_load (the uncertain loads)
    obs.sgen_p() / sgen_q()                       res_sgen (setpoints; known exactly by default)
    obs.gen_p() / gen_vm()                        gen setpoints (known exactly)
    obs.ext_grid_p() / ext_grid_q()               res_ext_grid (slack exchange)
    obs.trafo_tap() / line_status() / trafo_status()   discrete state (known exactly)
    obs.constant(values, name)                    fixed entries (e.g. time features, padding)
    obs.custom(name, n, fn_set, fn_point, noisy=True)   escape hatch on the certified state (see _Ctx)"""

    @staticmethod
    def _mk(kind, fn, unit, noisy_default):
        def make(elements=None, scale=1.0, offset=0.0, noisy=None):
            return ObsItem(kind, None if elements is None else list(elements), fn, unit,
                           noisy_default if noisy is None else bool(noisy), scale, offset)
        return make

    @staticmethod
    def constant(values, name="const"):
        v = np.atleast_1d(np.asarray(values, float))
        it = ObsItem(name, list(range(len(v))), lambda ctx, ids: _cst(ctx, v), "", False)
        return it

    @staticmethod
    def custom(name, n, fn_set, fn_point, noisy=True, unit=""):
        return ObsItem(name, list(range(int(n))), _custom(fn_set, fn_point), unit, noisy)


obs.bus_vm = obs._mk("bus_vm", _vm, "pu", True)
obs.bus_va = obs._mk("bus_va", _va, "deg", True)
obs.line_loading = obs._mk("line_loading", _loading("line"), "%", True)
obs.trafo_loading = obs._mk("trafo_loading", _loading("trafo"), "%", True)
obs.line_p_from = obs._mk("line_p_from", _flow("line", "from", "p"), "MW", True)
obs.line_q_from = obs._mk("line_q_from", _flow("line", "from", "q"), "MVAr", True)
obs.trafo_p_hv = obs._mk("trafo_p_hv", _flow("trafo", "from", "p"), "MW", True)
obs.trafo_q_hv = obs._mk("trafo_q_hv", _flow("trafo", "from", "q"), "MVAr", True)
obs.load_p = obs._mk("load_p", _uncertain_or_state("load", "p"), "MW", True)
obs.load_q = obs._mk("load_q", _uncertain_or_state("load", "q"), "MVAr", True)
obs.sgen_p = obs._mk("sgen_p", _uncertain_or_state("sgen", "p"), "MW", False)
obs.sgen_q = obs._mk("sgen_q", _uncertain_or_state("sgen", "q"), "MVAr", False)
obs.gen_p = obs._mk("gen_p", _state_value("gen_p", lambda m, i: m.gen_pos(i)), "MW", False)
obs.gen_vm = obs._mk("gen_vm", _state_value("gen_vm", lambda m, i: m.gen_pos(i)), "pu", False)
obs.ext_grid_p = obs._mk("ext_grid_p", _ext("p"), "MW", True)
obs.ext_grid_q = obs._mk("ext_grid_q", _ext("q"), "MVAr", True)
obs.trafo_tap = obs._mk("trafo_tap", _state_value("tap", lambda m, i: m.trafo_pos(i)), "", False)
obs.line_status = obs._mk("line_status", _state_value("line_on", lambda m, i: m.line_pos(i)), "", False)
obs.trafo_status = obs._mk("trafo_status", _state_value("trafo_on", lambda m, i: m.trafo_pos(i)), "", False)


class ObservationSpec:
    """The agent's observation vector: the blocks in order, concatenated."""

    def __init__(self, items):
        self.items = list(items)

    def names(self, model):
        return [n for it in self.items for n in it.names(model)]

    def noisy(self, model):
        """Boolean mask: which entries are measurements (default noise targets)."""
        return np.concatenate([np.full(len(it.ids(model)), it.noisy) for it in self.items])

    def groups(self, model):
        """{block kind: entry indices} (e.g. for per-group noise robustness)."""
        out, k = {}, 0
        for it in self.items:
            n = len(it.ids(model))
            out.setdefault(it.kind, []).extend(range(k, k + n))
            k += n
        return out

    def build(self, ctx):
        vals = [it.values(ctx) for it in self.items]
        if ctx.is_set:
            return concat(vals)
        return np.concatenate([np.atleast_1d(v) for v in vals])


# ==================================================================================================
# limits and uncertainty
# ==================================================================================================

@dataclass
class Limits:
    """What "safe" means after an action (checked with certified bounds; equality with a limit is safe).

    vm_pu               (vmin, vmax) for every bus (scalars or per-bus dicts {bus: value}); None -> the net's
                        min_vm_pu / max_vm_pu columns, else (0.95, 1.05)
    max_loading_percent lines and trafos (pandapower loading_percent); None -> max_loading_percent columns, else 100
    ext_grid_p_mw       (min, max) slack exchange, None = unlimited; likewise ext_grid_q_mvar
    buses               buses whose voltage is checked (default all)
    n1                  also screen every single line outage (``n1_lines``, default all in-service lines) with the
                        same limits; an outage that splits the grid counts as a violation"""
    vm_pu: tuple = None
    max_loading_percent: float = None
    ext_grid_p_mw: tuple = None
    ext_grid_q_mvar: tuple = None
    buses: list = None
    n1: bool = False
    n1_lines: list = None

    def describe(self, model):
        vlo, vhi = model.v_limits()
        v = (f"[{vlo.min():.3g}, {vhi.max():.3g}]" if np.ptp(vlo) == 0 and np.ptp(vhi) == 0
             else f"per bus ({vlo.min():.3g}..{vhi.max():.3g})")
        s = f"|V| in {v} pu, loading <= {model.max_loading():g}%"
        if self.ext_grid_p_mw is not None:
            s += f", ext grid P in [{self.ext_grid_p_mw[0]:g}, {self.ext_grid_p_mw[1]:g}] MW"
        if self.ext_grid_q_mvar is not None:
            s += f", ext grid Q in [{self.ext_grid_q_mvar[0]:g}, {self.ext_grid_q_mvar[1]:g}] MVAr"
        return s + (", N-1 (line outages)" if self.n1 else ", N-0")


@dataclass
class InjectionSet(AffineSet):
    """Affine set over the stacked uncertain injections u = c + G e (MW / MVAr), with what each entry is."""
    rows: list = field(default_factory=list)      # per entry: (table, pandapower index, "p" | "q")
    width: dict = field(default_factory=dict)     # description, for reports


class LoadUncertainty:
    """Which injections are uncertain and by how much, around a snapshot.

    rel           relative half-width per load (0.05 = +-5 % of the snapshot value); scalar or {load: value}
    abs_mw        absolute half-width added on top (MW, applied to P; Q follows the power factor)
    loads         which loads vary (default all in-service loads); the others stay at their snapshot value
    power_factor  "constant": one symbol per load scales P and Q together; "independent": P and Q separately
    sgens         sgens whose output is uncertain too (e.g. uncontrolled PV); ``sgen_rel`` their relative width.
                  Uncertain sgens ignore their setpoint, so they must not be controlled by actions.
    directions    None: one independent symbol per load (two with independent P, Q).  Fewer symbols make the
                  verification much tighter and faster (box splitting works in few dimensions), so for many loads use
                  a low-rank set:  "common" (one symbol: all loads move together by +-rel), or a
                  matrix D (n varying loads x K): load i = snapshot_i * (1 + rel * D[i] e), e in [-1, 1]^K; see
                  :meth:`from_samples` to get D from load data (principal components).
    The symbols are the input directions of the verifier."""

    def __init__(self, rel=0.05, abs_mw=0.0, loads=None, power_factor="constant", sgens=None, sgen_rel=0.0,
                 directions=None):
        if power_factor not in ("constant", "independent"):
            raise ValueError("power_factor must be 'constant' or 'independent'")
        if directions is not None and (power_factor != "constant" or abs_mw):
            raise ValueError("directions need power_factor='constant' and abs_mw=0 (relative changes along modes)")
        self.rel, self.abs_mw, self.loads, self.pf = rel, abs_mw, loads, power_factor
        self.sgens, self.sgen_rel = list(sgens or []), sgen_rel
        self.directions = directions

    @classmethod
    def from_samples(cls, load_p, n_modes=3, coverage=3.0, loads=None, **kw):
        """Low-rank set from load data: ``load_p`` (n_samples x n_loads, MW, columns in the order of ``loads`` /
        all in-service loads).  The relative deviations from the sample mean are decomposed into principal components;
        the set spans +-``coverage`` standard deviations along the first ``n_modes`` of them, around the snapshot."""
        X = np.asarray(load_p, float)
        mu = X.mean(0)
        R = (X - mu) / np.where(np.abs(mu) > 0, np.abs(mu), 1.0)
        _, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        D = coverage * Vt[:n_modes].T * sv[:n_modes] / np.sqrt(len(X))
        return cls(rel=1.0, loads=loads, directions=D, **kw)

    def _dirs(self, n):
        if self.directions is None:
            return None
        if isinstance(self.directions, str):
            if self.directions != "common":
                raise ValueError("directions must be None, 'common' or a matrix")
            return np.ones((n, 1))
        D = np.asarray(self.directions, float)
        if D.shape[0] != n:
            raise ValueError(f"directions has {D.shape[0]} rows, but {n} loads vary")
        return D

    def _relvec(self, ids):
        return np.array([float(self.rel.get(i, 0.0) if isinstance(self.rel, dict) else self.rel) for i in ids])

    def _mode_cols(self, model, c, varying, rel=None):
        vl = [i for i in model.load_index if i in varying]
        D = self._dirs(len(vl))
        if D is None:
            return None
        rp = [model.inj_row("load", i, "p") for i in vl]
        rq = [model.inj_row("load", i, "q") for i in vl]
        r = self._relvec(vl) if rel is None else np.full(len(vl), float(rel))
        Gm = np.zeros((model.n_inj, D.shape[1]))
        Gm[rp] = r[:, None] * c[rp][:, None] * D
        Gm[rq] = r[:, None] * c[rq][:, None] * D
        return Gm, rp + rq

    def build(self, model, net=None) -> InjectionSet:
        net = model.net if net is None else net
        c = np.zeros(model.n_inj)
        cols = []
        varying = set(model.load_index if self.loads is None else self.loads)
        relw = lambda i: float(self.rel.get(i, 0.0) if isinstance(self.rel, dict) else self.rel)
        for i in model.load_index:
            sc = float(np.nan_to_num(net.load.at[i, "scaling"], nan=1.0)) if "scaling" in net.load else 1.0
            p, q = float(net.load.at[i, "p_mw"]) * sc, float(net.load.at[i, "q_mvar"]) * sc
            rp, rq = model.inj_row("load", i, "p"), model.inj_row("load", i, "q")
            c[rp], c[rq] = p, q
            if i not in varying:
                continue
            wp = relw(i) * abs(p) + self.abs_mw
            wq = relw(i) * abs(q) + (self.abs_mw * abs(q) / abs(p) if abs(p) > 0 else 0.0)
            if self.pf == "constant":
                col = np.zeros(model.n_inj)
                col[rp], col[rq] = wp, (wq if p * q >= 0 else -wq)
                if wp or wq:
                    cols.append(col)
            else:
                for r, w in ((rp, wp), (rq, wq)):
                    if w:
                        col = np.zeros(model.n_inj)
                        col[r] = w
                        cols.append(col)
        for i in self.sgens:
            k = model.sgen_pos(i)
            rp, rq = model.inj_row("sgen", i, "p"), model.inj_row("sgen", i, "q")
            if rp is None:
                raise ValueError(f"sgen {i} is not declared uncertain in the GridModel (uncertain_sgens=...)")
            p = float(net.sgen.at[i, "p_mw"]) * model.sgen_scaling[k]
            q = float(net.sgen.at[i, "q_mvar"]) * model.sgen_scaling[k]
            c[rp], c[rq] = p, q
            if self.sgen_rel:
                col = np.zeros(model.n_inj)
                col[rp], col[rq] = self.sgen_rel * abs(p), self.sgen_rel * q
                cols.append(col)
        modes = self._mode_cols(model, c, varying)
        if modes is not None:                                   # low-rank: the modes replace the per-load columns
            Gm, rows = modes
            cols = list(Gm.T) + [col for col in cols if not np.any(col[rows])]
        G = np.array(cols).T if cols else np.zeros((model.n_inj, 0))
        return InjectionSet(c, G, rows=list(model.inj_rows),
                            width=dict(rel=self.rel, abs_mw=self.abs_mw, power_factor=self.pf,
                                       n_varying=len(varying), sgens=self.sgens))

    def step(self, model, base: InjectionSet, rel):
        """Generator columns of a per-step change of the varying loads (relative to the snapshot values)."""
        cols = []
        varying = set(model.load_index if self.loads is None else self.loads)
        modes = self._mode_cols(model, base.c, varying, rel)
        if modes is not None:                                   # same modes as the set itself
            return modes[0]
        for i in model.load_index:
            if i not in varying:
                continue
            rp, rq = model.inj_row("load", i, "p"), model.inj_row("load", i, "q")
            col = np.zeros(model.n_inj)
            col[rp], col[rq] = rel * abs(base.c[rp]), rel * base.c[rq] * np.sign(base.c[rp] or 1.0)
            if col.any():
                cols.append(col)
        return np.array(cols).T if cols else np.zeros((model.n_inj, 0))


# ==================================================================================================
# certified screening of a grid state against the limits
# ==================================================================================================

@dataclass
class Screen:
    """Certified bounds of every limited quantity of one grid state over an injection set (one topology)."""
    formal: bool                       # power flow certified (existence + uniqueness) over the whole set
    islanded: bool = False
    names: list = field(default_factory=list)
    unit: list = field(default_factory=list)
    lo: np.ndarray = None
    hi: np.ndarray = None
    lim_lo: np.ndarray = None
    lim_hi: np.ndarray = None
    nleaf: int = 0
    leaves: list = field(default_factory=list)     # (lo, hi, C over the symbols, c0) per leaf (for falsification)
    outage: object = None

    @property
    def possible(self):
        """Quantities that can leave their limits somewhere in the set."""
        if self.lo is None:
            return np.zeros(0, bool)
        return (self.lo < self.lim_lo) | (self.hi > self.lim_hi)

    @property
    def certain(self):
        """Quantities outside their limits for EVERY point of the set."""
        if self.lo is None:
            return np.zeros(0, bool)
        return (self.hi < self.lim_lo) | (self.lo > self.lim_hi)

    @property
    def safe(self):
        return bool(self.formal and not self.islanded and not self.possible.any())

    @property
    def status(self):
        if self.islanded:
            return "ISLANDED"
        if not self.formal:
            return "UNDECIDED"
        if not self.possible.any():
            return "SAFE"
        return "VIOLATED" if self.certain.any() else "CAN-VIOLATE"

    def violations(self):
        """[(name, lo, hi, lim_lo, lim_hi, unit, certain)] of the quantities that can violate, worst first."""
        out = []
        for k in np.where(self.possible)[0]:
            ex = max(self.lim_lo[k] - self.lo[k], self.hi[k] - self.lim_hi[k])
            out.append((ex, self.names[k], self.lo[k], self.hi[k], self.lim_lo[k], self.lim_hi[k], self.unit[k],
                        bool(self.certain[k])))
        out.sort(key=lambda r: -r[0] / max(abs(r[5] if np.isfinite(r[5]) else r[4]), 1e-9))
        return [r[1:] for r in out]

    def range_of(self, prefix):
        m = [k for k, n in enumerate(self.names) if n.startswith(prefix)]
        return (float(self.lo[m].min()), float(self.hi[m].max())) if m and self.lo is not None else (np.nan, np.nan)


# ==================================================================================================
# the model
# ==================================================================================================

def _in_service(tab):
    return tab.index[tab.in_service.values.astype(bool)] if "in_service" in tab else tab.index


def _quiet_runpp(net):
    import pandapower as pp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pp.runpp(net, numba=False)
            return True
        except Exception:
            return False


class GridModel:
    """Certified model of a pandapower grid for auditing an agent that observes it and acts on it.

    net            the grid (any snapshot: element tables, limits, and the controllable setpoints)
    observation    :class:`ObservationSpec` (or a list of obs blocks) -- the agent's input vector, in order
    actions        :class:`ActionSpace` (or a list of actions) -- the agent's outputs, in order
    limits         :class:`Limits` (default: the net's own limits)
    uncertain_sgens sgens whose output is uncertain (part of the injection set instead of the state)"""

    def __init__(self, net, observation, actions, limits=None, uncertain_sgens=()):
        self.net = copy.deepcopy(net)
        self.observation = observation if isinstance(observation, ObservationSpec) else ObservationSpec(observation)
        self.actions = actions if isinstance(actions, ActionSpace) else ActionSpace(actions)
        self.limits = limits or Limits()
        n = self.net
        self._check_supported(n)
        self.line_index = list(n.line.index)
        self.trafo_index = list(n.trafo.index)
        self.sgen_index = list(_in_service(n.sgen))
        self.gen_index = list(_in_service(n.gen))
        self.load_index = list(_in_service(n.load))
        self.uncertain_sgens = list(uncertain_sgens)
        self.sgen_scaling = np.array([float(np.nan_to_num(n.sgen.at[i, "scaling"], nan=1.0)) if "scaling" in n.sgen
                                      else 1.0 for i in self.sgen_index])
        col = lambda tab, c, i, d: float(np.nan_to_num(tab.at[i, c], nan=d)) if c in tab else d
        self.sgen_p_range = [(col(n.sgen, "min_p_mw", i, 0.0), col(n.sgen, "max_p_mw", i, np.inf)) for i in self.sgen_index]
        self.sgen_q_range = [(col(n.sgen, "min_q_mvar", i, -np.inf), col(n.sgen, "max_q_mvar", i, np.inf))
                             for i in self.sgen_index]
        self.gen_p_range = [(col(n.gen, "min_p_mw", i, -np.inf), col(n.gen, "max_p_mw", i, np.inf)) for i in self.gen_index]
        self.tap_range = [(int(col(n.trafo, "tap_min", t, 0)), int(col(n.trafo, "tap_max", t, 0))) for t in self.trafo_index]
        # injection vector u: [P, Q of every in-service load] + [P, Q of every uncertain sgen]
        self.inj_rows = ([("load", i, "p") for i in self.load_index] + [("load", i, "q") for i in self.load_index]
                         + [("sgen", i, "p") for i in self.uncertain_sgens] + [("sgen", i, "q") for i in self.uncertain_sgens])
        self._inj_pos = {r: k for k, r in enumerate(self.inj_rows)}
        self.n_inj = len(self.inj_rows)
        # compile the base topology (all branches in service, taps as in the net)
        self._grids = {}
        self._base_state = self.state_from_net(n)
        g0 = self._compile(self._base_state.tap)
        self.N, self.slk = g0.N, g0.slk
        self._bus_lookup = g0.names["bus_lookup"]
        self._branch_row = {e: k for k, e in enumerate(g0.names["branch_element"])}
        sn = g0.sn
        sgn = lambda t: -1.0 if t == "load" else 1.0
        self.Mp = np.zeros((self.N, self.n_inj))
        self.Mq = np.zeros((self.N, self.n_inj))
        for k, (t, i, pq) in enumerate(self.inj_rows):
            (self.Mp if pq == "p" else self.Mq)[self._bus_lookup[int(n[t].at[i, "bus"])], k] = sgn(t) / sn
        self.Csg = np.zeros((self.N, len(self.sgen_index)))
        for k, i in enumerate(self.sgen_index):
            if i not in self.uncertain_sgens:
                self.Csg[self._bus_lookup[int(n.sgen.at[i, "bus"])], k] = self.sgen_scaling[k] / sn
        self.Cg = np.zeros((self.N, len(self.gen_index)))
        for k, i in enumerate(self.gen_index):
            self.Cg[self._bus_lookup[int(n.gen.at[i, "bus"])], k] = 1.0 / sn
        self.Cgv = (self.Cg > 0) / np.maximum((self.Cg > 0).sum(1, keepdims=True), 1)
        eg = n.ext_grid[n.ext_grid.in_service.astype(bool)] if "in_service" in n.ext_grid else n.ext_grid
        self.v_slack = float(eg.vm_pu.iloc[0])
        self.va_slack = np.deg2rad(float(eg.va_degree.iloc[0])) if "va_degree" in eg else 0.0
        occ = set(self._bus_lookup.values())                    # internal buses that carry pandapower buses
        self._occupied = occ
        self._dang_cache = {}
        self._warn_inert_taps()

    def _warn_inert_taps(self):
        """pandapower >= 3 ignores tap_pos unless tap_changer_type is set: tap actions would then do nothing."""
        def taps(a):
            if isinstance(a, _Combo):
                return [t for x in a.actions for t in taps(x)]
            return [a.t] if isinstance(a, _Tap) else []
        used = sorted({t for a in self.actions.actions for t in taps(a)})
        tr = self.net.trafo
        if "tap_changer_type" in tr:
            inert = [t for t in used if not isinstance(tr.at[t, "tap_changer_type"], str)]
            if inert:
                warnings.warn(f"GridModel: trafo(s) {inert} have no tap_changer_type, so pandapower (and this model) "
                              "ignore their tap position -- tap actions on them change nothing. Set e.g. "
                              "net.trafo.tap_changer_type = 'Ratio' if the agent's environment applies taps.")

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _check_supported(n):
        eg = n.ext_grid[n.ext_grid.in_service.astype(bool)] if "in_service" in n.ext_grid else n.ext_grid
        if len(eg) != 1:
            raise NotImplementedError(f"exactly one in-service ext_grid (the slack) is supported, found {len(eg)}")
        if "slack" in n.gen and n.gen.slack.astype(bool).any():
            raise NotImplementedError("gens with slack=True are not supported; use an ext_grid as the slack")
        for t in ("ward", "xward", "storage", "dcline", "impedance", "svc", "tcsc", "ssc", "vsc"):
            if t in n and len(n[t]) and ("in_service" not in n[t] or n[t].in_service.astype(bool).any()):
                raise NotImplementedError(f"element table {t!r} is not supported yet")
        zip_cols = [c for c in ("const_z_percent", "const_i_percent", "const_z_p_percent", "const_i_p_percent",
                                "const_z_q_percent", "const_i_q_percent") if c in n.load]
        if zip_cols and np.nan_to_num(n.load[zip_cols].values.astype(float)).any():
            warnings.warn("GridModel: ZIP load shares are ignored (loads are modelled as constant power)")
        if "trafo3w" in n and len(n.trafo3w):
            warnings.warn("GridModel: trafo3w is modelled (as in pandapower) but has no observations / actions")

    def state_from_net(self, net) -> GridState:
        """The controllable state of a pandapower net (setpoints, taps, in_service flags)."""
        f = lambda tab, c, idx, d=0.0: tuple(float(np.nan_to_num(tab.at[i, c], nan=d)) for i in idx)
        return GridState(sgen_p=f(net.sgen, "p_mw", self.sgen_index), sgen_q=f(net.sgen, "q_mvar", self.sgen_index),
                         gen_p=f(net.gen, "p_mw", self.gen_index), gen_vm=f(net.gen, "vm_pu", self.gen_index, 1.0),
                         tap=tuple(int(v) for v in f(net.trafo, "tap_pos", self.trafo_index)),
                         line_on=tuple(bool(net.line.at[i, "in_service"]) for i in self.line_index),
                         trafo_on=tuple(bool(net.trafo.at[i, "in_service"]) for i in self.trafo_index))

    def _compile(self, taps):
        if taps in self._grids:
            return self._grids[taps]
        n = copy.deepcopy(self.net)
        n.line["in_service"] = True                            # compile every branch; status is set per state
        n.trafo["in_service"] = True
        if self.trafo_index:
            n.trafo.loc[self.trafo_index, "tap_pos"] = list(taps)
        conv = _quiet_runpp(n)
        if n._ppc is None or "internal" not in n._ppc:
            raise RuntimeError("pandapower could not build its internal model of the net")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = from_pandapower(n)
        if not conv:
            g = replace(g, V0=None)
        if self._grids and not np.array_equal(g.names["intnum"], next(iter(self._grids.values())).names["intnum"]):
            raise RuntimeError("bus order changed between tap configurations")
        self._grids[taps] = g
        return g

    # ------------------------------------------------------------------ lookups
    def sgen_pos(self, i):
        return self.sgen_index.index(i)

    def gen_pos(self, i):
        return self.gen_index.index(i)

    def trafo_pos(self, i):
        return self.trafo_index.index(i)

    def line_pos(self, i):
        return self.line_index.index(i)

    def inj_row(self, table, i, pq):
        return self._inj_pos.get((table, i, pq))

    def dangling(self, g, row):
        """If branch ``row`` ends at an EMPTY bus (pandapower's auxiliary bus behind an open line switch: no element,
        no other in-service branch), Kirchhoff's current law there (zero injection) makes the branch current an exact
        linear function of the other end's voltage:  V_aux = -Y[aux, b] V_b / Y[aux, aux],  so
        max(|I_from|, |I_to|) = a |V_b|.  Returns (b, a) or None.  This avoids multiplying the enclosure's remainder
        at the auxiliary bus by the (large) admittance of a short line."""
        key = (row, g.status.tobytes(), complex(g.Yff[row]), complex(g.Yft[row]), complex(g.Ytf[row]), complex(g.Ytt[row]))
        if key in self._dang_cache:
            return self._dang_cache[key]
        out = None
        for aux, b, Yab, Yaa_br, Ybb, Yba in ((g.t[row], g.f[row], g.Ytf[row], g.Ytt[row], g.Yff[row], g.Yft[row]),
                                             (g.f[row], g.t[row], g.Yft[row], g.Yff[row], g.Ytt[row], g.Ytf[row])):
            if aux in self._occupied or aux == g.slk:
                continue
            deg = int(((g.f == aux) | (g.t == aux))[g.status].sum())
            if deg != 1:
                continue
            Yaa = g.Ybus[aux, aux]                      # incl. any shunt at the auxiliary bus
            k = -Yab / Yaa                              # V_aux = k V_b
            I_b = abs(Ybb + Yba * k)                    # current entering at the b end, per unit |V_b|
            I_a = abs(Yab + Yaa_br * k)                 # current at the auxiliary end
            out = (int(b), float(max(I_b, I_a)))
            break
        self._dang_cache[key] = out
        return out

    def buses(self, ids):
        return [self._bus_lookup[int(b)] for b in ids]

    def branch_rows(self, kind, ids):
        return [self._branch_row[(kind, int(i))] for i in ids]

    def default_ids(self, kind):
        if kind.startswith("bus"):
            return list(self._bus_lookup)
        if kind.startswith("line"):
            return self.line_index
        if kind.startswith("trafo"):
            return self.trafo_index
        if kind.startswith("load"):
            return self.load_index
        if kind.startswith("sgen"):
            return self.sgen_index
        if kind.startswith("gen"):
            return self.gen_index
        if kind.startswith("ext_grid"):
            return [0]
        raise KeyError(kind)

    def v_limits(self):
        """Per checked bus (Limits.buses order): (vmin, vmax) arrays."""
        L, n = self.limits, self.net
        buses = list(self._bus_lookup) if L.buses is None else list(L.buses)

        def per(v, col, d):
            if v is None:
                return np.array([float(np.nan_to_num(n.bus.at[b, col], nan=d)) if col in n.bus else d for b in buses])
            if isinstance(v, dict):
                return np.array([float(v.get(b, d)) for b in buses])
            return np.full(len(buses), float(v))
        vm = L.vm_pu or (None, None)
        return per(vm[0], "min_vm_pu", 0.95), per(vm[1], "max_vm_pu", 1.05)

    def max_loading(self):
        return 100.0 if self.limits.max_loading_percent is None else float(self.limits.max_loading_percent)

    # ------------------------------------------------------------------ grids and specifications
    def grid(self, state: GridState, outage=None):
        """Grid of ``state`` (taps -> Ybus, in_service flags -> branch status), optionally with a line outaged."""
        g = self._compile(tuple(state.tap))
        st = np.ones(g.L, bool)
        for k, i in enumerate(self.line_index):
            st[self._branch_row[("line", i)]] = state.line_on[k]
        for k, i in enumerate(self.trafo_index):
            st[self._branch_row[("trafo", i)]] = state.trafo_on[k]
        if outage is not None:
            st[self._branch_row[("line", outage)]] = False
        return g if (st == g.status).all() else g.with_branch_status(st)

    def fixed_injections(self, state):
        """Bus injections (pu) that do not depend on the uncertain set: sgens and gens."""
        P = self.Csg @ np.asarray(state.sgen_p, float) + self.Cg @ np.asarray(state.gen_p, float)
        Q = self.Csg @ np.asarray(state.sgen_q, float)
        return P, Q

    def warm(self, g, state):
        th = np.full(self.N, self.va_slack)
        Vm = np.ones(self.N)
        if g.V0 is not None and np.all(np.isfinite(g.V0)):
            th, Vm = np.angle(g.V0).copy(), np.abs(g.V0).copy()
        Vm[g.slk] = self.v_slack
        th[g.slk] = self.va_slack
        Vm[g.pv] = self.vset(g, state)
        return th, Vm

    def vset(self, g, state):
        vb = self.Cgv @ np.asarray(state.gen_vm, float)
        return vb[g.pv]

    def spec(self, g, state, u: TM):
        """(P, Q, Vset) Taylor models of the specified injections for the injection TM ``u``."""
        P0, Q0 = self.fixed_injections(state)
        P = add(_const(P0, u), matmul(self.Mp, u))
        Q = add(_const(Q0, u), matmul(self.Mq, u))
        return P, Q, _const(self.vset(g, state), u)

    def enclose(self, state, U: AffineSet, lo, hi, outage=None):
        """Certified PF of ``state`` over the sub-box [lo, hi] of the injection set U.  Returns (grid, enc, u, P, Q)
        with enc None if the grid is islanded."""
        g = self.grid(state, outage)
        u = U.sub_tm(0.5 * (np.asarray(lo) + hi), 0.5 * (np.asarray(hi) - lo))
        if g.islanded():
            return g, None, u, None, None
        P, Q, V = self.spec(g, state, u)
        enc = enclose_pf(g, P, Q, V, self.warm(g, state))
        return g, enc, u, P, Q

    # ------------------------------------------------------------------ observation
    def observe_set(self, state, U, lo, hi):
        """(ok, TM of the observation over the sub-box, enc) -- ok False if the PF could not be certified."""
        g, enc, u, P, Q = self.enclose(state, U, lo, hi)
        if enc is None or not enc.ok:
            return False, None, enc
        return True, self.observation.build(_Ctx(self, state, g, u, enc=enc, Pspec=P, Qspec=Q)), enc

    def simulate(self, state, u, outage=None):
        """Point power flow (float64 Newton) at injections u (MW / MVAr).  Returns a dict with the observation,
        |V| per bus (pandapower index), loadings (%), ext grid P/Q, and every limited quantity; None if the PF
        fails or the grid is islanded."""
        g = self.grid(state, outage)
        if g.islanded():
            return None
        P0, Q0 = self.fixed_injections(state)
        u = np.asarray(u, float)
        Psp, Qsp = P0 + self.Mp @ u, Q0 + self.Mq @ u
        th0, Vm0 = self.warm(g, state)
        Vpin = np.zeros(self.N)
        Vpin[g.pv] = Vm0[g.pv]
        th, Vm, mis = newton(g.G, g.B, g.ns, g.pq, g.pv, Psp, Qsp, Vpin, th0, Vm0, iters=100)
        if not np.isfinite(mis) or mis > 1e-8:
            return None
        V = Vm * np.exp(1j * th)
        S = V * np.conj(g.Ybus @ V)
        ctx = _Ctx(self, state, g, u, V=V, P=S.real, Q=S.imag, Pspec=Psp, Qspec=Qsp)
        q = self._quantities_point(ctx)
        return dict(obs=self.observation.build(ctx), V=V, quantities=q)

    # ------------------------------------------------------------------ limited quantities
    def _quantity_blocks(self, state, outage=None):
        """[(prefix, kind, ids, lim_lo, lim_hi, unit)] of every limited quantity of a state."""
        L = self.limits
        buses = list(self._bus_lookup) if L.buses is None else list(L.buses)
        vlo, vhi = self.v_limits()
        ml = self.max_loading()
        lines = [i for k, i in enumerate(self.line_index) if state.line_on[k] and i != outage]
        trafos = [i for k, i in enumerate(self.trafo_index) if state.trafo_on[k]]
        g = self._compile(tuple(state.tap))
        rated = lambda kind, ids: [i for i in ids if g.rate_a[self._branch_row[(kind, i)]] > 0]
        out = [("vm bus", "bus_vm", buses, vlo, vhi, "pu"),
               ("loading line", "line_loading", rated("line", lines), -np.inf, ml, "%"),
               ("loading trafo", "trafo_loading", rated("trafo", trafos), -np.inf, ml, "%")]
        if L.ext_grid_p_mw is not None:
            out.append(("ext_grid P", "ext_grid_p", [0], L.ext_grid_p_mw[0], L.ext_grid_p_mw[1], "MW"))
        if L.ext_grid_q_mvar is not None:
            out.append(("ext_grid Q", "ext_grid_q", [0], L.ext_grid_q_mvar[0], L.ext_grid_q_mvar[1], "MVAr"))
        return out

    _QFN = dict(bus_vm=_vm, line_loading=_loading("line"), trafo_loading=_loading("trafo"), ext_grid_p=_ext("p"),
                ext_grid_q=_ext("q"))

    def _quantities(self, ctx, outage=None):
        names, units, lims_lo, lims_hi, vals = [], [], [], [], []
        for prefix, kind, ids, llo, lhi, unit in self._quantity_blocks(ctx.state, outage):
            if not len(ids):
                continue
            vals.append(self._QFN[kind](ctx, ids))
            names += [f"{prefix} {i}" if kind not in ("ext_grid_p", "ext_grid_q") else prefix for i in ids]
            units += [unit] * len(ids)
            lims_lo.append(np.broadcast_to(np.asarray(llo, float), (len(ids),)))
            lims_hi.append(np.broadcast_to(np.asarray(lhi, float), (len(ids),)))
        return names, units, np.concatenate(lims_lo), np.concatenate(lims_hi), vals

    def _quantities_point(self, ctx, outage=None):
        names, units, llo, lhi, vals = self._quantities(ctx, outage)
        v = np.concatenate([np.atleast_1d(x) for x in vals])
        return dict(names=names, unit=units, value=v, lim_lo=llo, lim_hi=lhi,
                    violation=np.maximum(llo - v, v - lhi))

    def screen(self, state, U: AffineSet, outage=None, max_leaves=64, refine=24) -> Screen:
        """Certified bounds of every limited quantity of ``state`` over the whole injection set U.
        Leaves where a quantity straddles its limit are bisected (up to ``refine`` extra splits) along the direction
        that drives the undecided quantities most -- the same bound tightening N-1 screening uses."""
        g = self.grid(state, outage)
        if g.islanded():
            return Screen(False, True, outage=outage)
        K = U.K
        res = Screen(True, outage=outage)

        def spec_fn(gg, a, b):
            u = U.sub_tm(0.5 * (a + b), 0.5 * (b - a))
            P, Q, V = self.spec(gg, state, u)
            return dict(P=P, Q=Q, V=V, warm=self.warm(gg, state), _u=u)

        def red(enc, a, b):
            u = U.sub_tm(0.5 * (a + b), 0.5 * (b - a))
            P, Q, _ = self.spec(g, state, u)
            names, units, llo, lhi, vals = self._quantities(_Ctx(self, state, g, u, enc=enc, Pspec=P, Qspec=Q), outage)
            T = concat(vals)
            lo, hi = T.bounds()
            lo = np.where([nm.startswith(("vm", "loading")) for nm in names], np.maximum(lo, 0.0), lo)   # magnitudes
            win = np.where(np.isfinite(lhi - llo), np.maximum(lhi - llo, 1e-9), np.maximum(np.abs(lhi), 1.0))
            res.names, res.unit, res.lim_lo, res.lim_hi = names, units, llo, lhi
            return dict(lo=lo, hi=hi, a=a, b=b, C=T.C[:, :K], c0=T.c0, sens=np.abs(T.C[:, :K]) / win[:, None])

        stack = [(-np.ones(K), np.ones(K))]
        parts, budget = [], int(refine)
        while stack:
            a, b = stack.pop()
            leaves, ok = enclose_pf_split(g, spec_fn, a, b, reduce=red, max_leaves=max_leaves)
            if not ok or not leaves:
                return Screen(False, outage=outage)
            for L in leaves:
                und = ((L["lo"] < res.lim_lo) & (L["hi"] >= res.lim_lo)) | ((L["hi"] > res.lim_hi) & (L["lo"] <= res.lim_hi))
                if und.any() and budget > 0 and K > 0:
                    budget -= 1
                    j = int(np.argmax(L["sens"][und].sum(0) * (L["b"] - L["a"])))
                    stack += list(bisect(L["a"], L["b"], j))
                else:
                    parts.append(L)
        res.lo = np.min([p["lo"] for p in parts], 0)
        res.hi = np.max([p["hi"] for p in parts], 0)
        res.nleaf = len(parts)
        res.leaves = [(p["a"], p["b"], p["C"], p["c0"]) for p in parts]
        return res

    def n1_lines(self, state):
        L = self.limits
        lines = self.line_index if L.n1_lines is None else list(L.n1_lines)
        return [i for i in lines if state.line_on[self.line_pos(i)]]

    # ------------------------------------------------------------------ trust
    def check_against_pandapower(self, net=None, verbose=True):
        """Point power flow of this model vs pandapower's own runpp at a snapshot: max deviation per observation
        block and per limited quantity.  Run this once to confirm the model reads the grid the way the agent's
        environment does (units, element order, loading definition)."""
        import pandapower as pp
        net = copy.deepcopy(self.net if net is None else net)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp(net, numba=False)
        state = self.state_from_net(net)
        U = LoadUncertainty(rel=0.0).build(self, net)
        sim = self.simulate(state, U.c)
        if sim is None:
            raise RuntimeError("the model's power flow did not converge at this snapshot")
        ref = []
        for it in self.observation.items:
            ids = it.ids(self)
            k = it.kind
            get = {"bus_vm": lambda: net.res_bus.vm_pu.loc[ids].values,
                   "bus_va": lambda: net.res_bus.va_degree.loc[ids].values,
                   "line_loading": lambda: net.res_line.loading_percent.loc[ids].values,
                   "trafo_loading": lambda: net.res_trafo.loading_percent.loc[ids].values,
                   "line_p_from": lambda: net.res_line.p_from_mw.loc[ids].values,
                   "line_q_from": lambda: net.res_line.q_from_mvar.loc[ids].values,
                   "trafo_p_hv": lambda: net.res_trafo.p_hv_mw.loc[ids].values,
                   "trafo_q_hv": lambda: net.res_trafo.q_hv_mvar.loc[ids].values,
                   "load_p": lambda: net.res_load.p_mw.loc[ids].values,
                   "load_q": lambda: net.res_load.q_mvar.loc[ids].values,
                   "sgen_p": lambda: net.res_sgen.p_mw.loc[ids].values,
                   "sgen_q": lambda: net.res_sgen.q_mvar.loc[ids].values,
                   "ext_grid_p": lambda: net.res_ext_grid.p_mw.values[:1],
                   "ext_grid_q": lambda: net.res_ext_grid.q_mvar.values[:1]}.get(k)
            if get is not None:
                ref.append((k, it, get()))
        out, pos = {}, 0
        offsets = {}
        for it in self.observation.items:
            n = len(it.ids(self))
            offsets[id(it)] = (pos, pos + n)
            pos += n
        for k, it, r in ref:
            a, b = offsets[id(it)]
            mine = (sim["obs"][a:b] - np.asarray(it.offset, float)) / np.asarray(it.scale, float)
            out[k] = float(np.max(np.abs(mine - np.asarray(r, float)))) if len(r) else 0.0
        if verbose:
            print("model vs pandapower (max abs deviation per observation block):")
            for k, v in out.items():
                print(f"   {k:<14} {v:.2e}")
        return out
