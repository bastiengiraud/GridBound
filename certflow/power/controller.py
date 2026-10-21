"""Controllers: map the verifier's input set to generator setpoints y = [Pg (ng, MW) ; Vg (ng, pu)].

A controller provides
    reach(x: TM, offset)   -> (y_raw, y)   sound enclosures before / after output saturation
    forward(x, offset)     -> (Pg, Vg)     the deployed point evaluation
where ``offset = (dPg, dVg)`` is the constant restoration correction (or None).
"""
from __future__ import annotations

import numpy as np

from ..fp import CONFIG, U, upr
from ..tm import TM, clip, matmul
from ..nn.network import NNReachable


class DispatchController:
    """NN dispatch policy (P_load, Q_load) -> (Pg, Vg), as in the paper.

    nn          : NNReachable with output [Pg of ACTIVE gens ; Vg of all gens] (compact layout)
    ng          : number of generators
    pg_active   : bool (ng,) gens whose Pg the NN outputs (others, e.g. synchronous condensers, are pinned to 0)
    vg_clip     : (vmin, vmax) saturation of the Vg setpoints before they reach the grid (deployed behaviour)
    pg_fallback : value substituted for a NaN Pg output in deployment (reference: mean nominal Pg)"""

    def __init__(self, nn: NNReachable, ng, pg_active=None, vg_clip=(0.94, 1.06), pg_fallback=0.0):
        self.nn = nn
        self.ng = int(ng)
        self.pg_active = np.ones(ng, bool) if pg_active is None else np.asarray(pg_active, bool)
        self.vg_clip = (float(vg_clip[0]), float(vg_clip[1]))
        self.pg_fallback = float(pg_fallback)
        na = int(self.pg_active.sum())
        self.S = None
        if na < self.ng:                                         # compact [Pg_active, Vg] -> full [Pg, Vg]
            S = np.zeros((2 * ng, na + ng))
            S[np.where(self.pg_active)[0], np.arange(na)] = 1.0
            S[ng:, na:] = np.eye(ng)
            self.S = S

    # ---------------------------------------------------------------- set-valued
    def reach(self, x: TM, offset=None):
        ng = self.ng
        y_raw = self.nn.reach(x)
        if self.S is not None:
            y_raw = matmul(self.S, y_raw)
        if offset is not None and (offset[0] is not None or offset[1] is not None):
            dpg, dvg = offset
            y_raw = TM(y_raw[0].copy(), y_raw[1].copy(), y_raw[2], y_raw[3].copy())
            if dpg is not None:
                y_raw[0][:ng] += dpg
            if dvg is not None:
                y_raw[0][ng:] += dvg
            if CONFIG.rigorous:                                  # rounding of the centre add AND of the deployed fl(y + d)
                y_raw[3][:] = upr(y_raw[3] + U * (2.0 * np.abs(y_raw[0]) + np.abs(y_raw[1]).sum(-1) + y_raw[3]),
                                  y_raw[1].shape[1] + 6)
        lo = np.concatenate([np.full(ng, -1e6), np.full(ng, self.vg_clip[0])])
        hi = np.concatenate([np.full(ng, 1e6), np.full(ng, self.vg_clip[1])])
        return y_raw, clip(y_raw, lo, hi)

    # ---------------------------------------------------------------- point
    def raw(self, x):
        y = self.nn.forward(np.asarray(x, float))
        if self.S is not None:
            y = self.S @ y
        return y

    def forward(self, x, offset=None):
        ng = self.ng
        y = self.raw(x)
        Pg = np.nan_to_num(y[:ng], nan=self.pg_fallback)
        Vg = np.nan_to_num(y[ng:], nan=1.0)
        if offset is not None:
            if offset[0] is not None:
                Pg = Pg + offset[0]
            if offset[1] is not None:
                Vg = Vg + offset[1]
        return Pg, np.clip(Vg, *self.vg_clip)
