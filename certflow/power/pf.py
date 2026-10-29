"""AC power flow: point Newton solver (polar) and the rectangular-coordinate Taylor-model system for Krawczyk.

Rectangular coordinates make P, Q bilinear in (vr, vi), so the reduced Jacobian is AFFINE in the state and its
state-derivatives H_j are constant -- exactly the structure :func:`certflow.krawczyk.krawczyk` exploits.
Residual rows: [P_ns ; Q_pq ; |V|^2_pv], state: [vr_ns ; vi_ns].
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..fp import CONFIG, U, upr
from ..krawczyk import mdotsp, structure_from_triplets
from ..tm import TM, add, mag, matmul, mul, nt, sub


def spsolve(A, b):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve as _sps
    return _sps(csr_matrix(A), b)


# ==================================================================================================
# point power flow (polar)
# ==================================================================================================

def inj_pt(G, B, th, Vm):
    """Bus injections (P, Q) in pu at a point."""
    c, s = np.cos(th), np.sin(th)
    Vc, Vs = Vm * c, Vm * s
    a = G @ Vc - B @ Vs
    b = G @ Vs + B @ Vc
    return Vm * (c * a + s * b), Vm * (s * a - c * b)


def jac_blocks(G, B, th, Vm):
    """Full N x N polar Jacobian blocks (dP/dth, dP/dV, dQ/dth, dQ/dV)."""
    N = G.shape[0]
    P, Q = inj_pt(G, B, th, Vm)
    D = th[:, None] - th[None, :]
    c, s = np.cos(D), np.sin(D)
    VV = Vm[:, None] * Vm[None, :]
    H = VV * (G * s - B * c)
    Nn = Vm[:, None] * (G * c + B * s)
    M = -VV * (G * c + B * s)
    L = Vm[:, None] * (G * s - B * c)
    bii, gii = np.diag(B), np.diag(G)
    di = np.diag_indices(N)
    H[di] = -Q - bii * Vm ** 2
    Nn[di] = P / Vm + gii * Vm
    M[di] = P - gii * Vm ** 2
    L[di] = Q / Vm - bii * Vm
    return H, Nn, M, L


def jac(G, B, ns, pq, th, Vm):
    """Reduced polar Jacobian of the mismatch h = [P_spec - P ; Q_spec - Q] (negated, as in the reference)."""
    H, Nn, M, L = jac_blocks(G, B, th, Vm)
    return -np.block([[H[np.ix_(ns, ns)], Nn[np.ix_(ns, pq)]], [M[np.ix_(pq, ns)], L[np.ix_(pq, pq)]]])


def newton(G, B, ns, pq, pv, Psp, Qsp, Vpin, th0, Vm0, iters=60, tol=1e-12):
    """Damped Newton PF from (th0, Vm0) with PV magnitudes pinned to Vpin[pv].  Returns (th, Vm, max mismatch)."""
    na = len(ns)
    th = th0.copy()
    Vm = Vm0.copy()
    Vm[pv] = Vpin[pv]
    d = np.ones(na + len(pq))
    for _ in range(iters):
        P, Q = inj_pt(G, B, th, Vm)
        d = np.concatenate([(Psp - P)[ns], (Qsp - Q)[pq]])
        if np.abs(d).max() < tol:
            break
        dz = spsolve(jac(G, B, ns, pq, th, Vm), d)
        a = min(1.0, 0.5 / max(np.abs(dz).max(), 1e-9))
        th[ns] -= a * dz[:na]
        Vm[pq] -= a * dz[na:]
    return th, Vm, float(np.abs(d).max())


# ==================================================================================================
# rectangular formulation
# ==================================================================================================

def rect_ab(G, B, vr, vi):
    return G @ vr - B @ vi, G @ vi + B @ vr


def jac_rect(G, B, ns, pq, pv, vr, vi):
    """Reduced rectangular Jacobian: rows [P_ns; Q_pq; V^2_pv], cols [vr_ns; vi_ns].  Linear in (vr, vi)."""
    N = G.shape[0]
    di = np.diag_indices(N)
    a, b = rect_ab(G, B, vr, vi)
    dP_dvr = G * vr[:, None] + B * vi[:, None]
    dP_dvr[di] += a
    dP_dvi = -B * vr[:, None] + G * vi[:, None]
    dP_dvi[di] += b
    dQ_dvr = G * vi[:, None] - B * vr[:, None]
    dQ_dvr[di] -= b
    dQ_dvi = -B * vi[:, None] - G * vr[:, None]
    dQ_dvi[di] += a
    dV_dvr = np.zeros((N, N))
    dV_dvr[di] = 2 * vr
    dV_dvi = np.zeros((N, N))
    dV_dvi[di] = 2 * vi
    top = np.concatenate([dP_dvr[np.ix_(ns, ns)], dP_dvi[np.ix_(ns, ns)]], 1)
    mid = np.concatenate([dQ_dvr[np.ix_(pq, ns)], dQ_dvi[np.ix_(pq, ns)]], 1)
    bot = np.concatenate([dV_dvr[np.ix_(pv, ns)], dV_dvi[np.ix_(pv, ns)]], 1)
    return np.concatenate([top, mid, bot], 0)


def jac_rect_absrad(G, B, ns, pq, pv, rvr, rvi):
    """Entrywise |Jacobian| applied to nonnegative state radii (rvr, rvi) -> (m, m) majorant."""
    N = G.shape[0]
    di = np.diag_indices(N)
    aG = np.abs(G)
    aB = np.abs(B)
    a = aG @ rvr + aB @ rvi
    b = aG @ rvi + aB @ rvr
    dP_dvr = aG * rvr[:, None] + aB * rvi[:, None]
    dP_dvr[di] += a
    dP_dvi = aB * rvr[:, None] + aG * rvi[:, None]
    dP_dvi[di] += b
    dQ_dvr = aG * rvi[:, None] + aB * rvr[:, None]
    dQ_dvr[di] += b
    dQ_dvi = aB * rvi[:, None] + aG * rvr[:, None]
    dQ_dvi[di] += a
    dV_dvr = np.zeros((N, N))
    dV_dvr[di] = 2 * rvr
    dV_dvi = np.zeros((N, N))
    dV_dvi[di] = 2 * rvi
    top = np.concatenate([dP_dvr[np.ix_(ns, ns)], dP_dvi[np.ix_(ns, ns)]], 1)
    mid = np.concatenate([dQ_dvr[np.ix_(pq, ns)], dQ_dvi[np.ix_(pq, ns)]], 1)
    bot = np.concatenate([dV_dvr[np.ix_(pv, ns)], dV_dvi[np.ix_(pv, ns)]], 1)
    return np.concatenate([top, mid, bot], 0)


def inj_rect_tm(G, B, Vr: TM, Vi: TM):
    """Full-bus rectangular voltage TMs -> (P, Q) injection TMs (pu)."""
    a = sub(matmul(G, Vr), matmul(B, Vi))
    b = add(matmul(G, Vi), matmul(B, Vr))
    return add(mul(Vr, a), mul(Vi, b)), sub(mul(Vi, a), mul(Vr, b))


def vmag2_tm(Vr: TM, Vi: TM):
    """|V|^2 = vr^2 + vi^2 as a CORRELATED TM product (no vr/vi decorrelation)."""
    return add(mul(Vr, Vr), mul(Vi, Vi))


def _djac_rect(G, B, ns, pq, pv, Vr, Vi):
    m = 2 * len(ns)
    Ks = Vr[1].shape[1]
    J0 = jac_rect(G, B, ns, pq, pv, Vr[0], Vi[0])
    JlinC = (np.stack([jac_rect(G, B, ns, pq, pv, Vr[1][:, k], Vi[1][:, k]) for k in range(Ks)], 2)
             if Ks else np.zeros((m, m, 0)))
    R_rad = jac_rect_absrad(G, B, ns, pq, pv, Vr[3], Vi[3])
    return J0, JlinC, (Vr[2], Vi[2]), R_rad


def quad_w(Qs):
    """Per-state-row weight of the quadratic generators: sum_a |Q_aa| + sum_{a<c} |Q_ac + Q_ca|."""
    Qsum = Qs + np.swapaxes(Qs, -1, -2)
    K = Qs.shape[-1]
    dg = np.abs(np.diagonal(Qs, axis1=-2, axis2=-1)).sum(-1)
    return dg + (np.abs(Qsum) * np.triu(np.ones((K, K)), 1)).sum((-1, -2))


def quad_sum_rect(M, G, B, ns, pq, pv, VrQ, ViQ):
    """sum_{a<=c} |M J_quad,ac| streamed (Jacobian linear -> its (a,c) coefficient is Jac of the state's)."""
    m = 2 * len(ns)
    Ks = VrQ.shape[1]
    q = np.zeros((m, m))
    for a in range(Ks):
        for c in range(a, Ks):
            vr = VrQ[:, a, c] if c == a else VrQ[:, a, c] + VrQ[:, c, a]
            vi = ViQ[:, a, c] if c == a else ViQ[:, a, c] + ViQ[:, c, a]
            if not (vr.any() or vi.any()):
                continue
            q = q + np.abs(mdotsp(M, jac_rect(G, B, ns, pq, pv, vr, vi)))
    return q


# ---------------------------------------------------------------- caches (content-keyed, bounded)
class _LRU(OrderedDict):
    def __init__(self, maxsize=64):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return v


_H_CACHE = _LRU(64)
_LINPAT = _LRU(64)
_LIN_CHK = [0]
LIN_NCHECK = 12


def _gb_key(G, B):
    return (G.shape[0], hash(G.tobytes()), hash(B.tobytes()))


def h_structure(G, B, ns, pq, pv):
    """Nonzeros of the CONSTANT tensor H[:,:,j] = d(reduced rect Jacobian)/d(state_j), as sparse selectors:
    (rows, inv, vals, cols_u, j_u, m, ShT, ScT) so that |M H_j| and sum_j |M H_j| r_j are O(m nnz)."""
    ns = np.asarray(ns, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pv = np.asarray(pv, dtype=int)
    na = len(ns)
    npq = len(pq)
    m = 2 * na
    N = G.shape[0]
    nspos = -np.ones(N, int)
    nspos[ns] = np.arange(na)
    pqpos = -np.ones(N, int)
    pqpos[pq] = np.arange(npq)
    pvpos = -np.ones(N, int)
    pvpos[pv] = np.arange(len(pv))
    arm = np.arange(m)
    arna = np.arange(na)
    RR, CC, JJ, VV = [], [], [], []
    for j in range(m):
        k = int(ns[j]) if j < na else int(ns[j - na])
        T = 0 if j < na else 1
        Gkn = G[k, ns]
        Bkn = B[k, ns]
        Gnk = G[ns, k]
        Bnk = B[ns, k]
        rows = [np.full(m, nspos[k])]
        cols = [arm]
        vals = [np.concatenate([Gkn, -Bkn]) if T == 0 else np.concatenate([Bkn, Gkn])]
        if pqpos[k] >= 0:
            rows.append(np.full(m, na + pqpos[k]))
            cols.append(arm)
            vals.append(np.concatenate([-Bkn, -Gkn]) if T == 0 else np.concatenate([Gkn, -Bkn]))
        if pvpos[k] >= 0:
            rows.append(np.array([na + npq + pvpos[k]]))
            cols.append(np.array([nspos[k] if T == 0 else na + nspos[k]]))
            vals.append(np.array([2.0]))
        rows += [arna, arna]
        cols += [arna, na + arna]
        vals += [Gnk if T == 0 else -Bnk, Bnk if T == 0 else Gnk]
        if npq:
            qr = na + np.arange(npq)
            rows += [qr, qr]
            cols += [nspos[pq], na + nspos[pq]]
            vals += [-B[pq, k] if T == 0 else -G[pq, k], G[pq, k] if T == 0 else -B[pq, k]]
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        key = r * m + c
        uk, inv0 = np.unique(key, return_inverse=True)
        s = np.zeros(len(uk))
        np.add.at(s, inv0, v)
        keep = np.abs(s) > 1e-12
        RR.append(uk[keep] // m)
        CC.append(uk[keep] % m)
        JJ.append(np.full(int(keep.sum()), j))
        VV.append(s[keep])
    return structure_from_triplets(np.concatenate(RR), np.concatenate(CC), np.concatenate(JJ),
                                   np.concatenate(VV), m)


def lin_pattern(G, B, ns, pq, pv, m):
    """Fixed (topology-only) sparsity pattern of the reduced rect Jacobian (superset, from two random evals)."""
    def build():
        from scipy.sparse import csr_matrix
        rng = np.random.default_rng(12345)
        N = G.shape[0]
        p1 = jac_rect(G, B, ns, pq, pv, rng.uniform(0.85, 1.15, N), rng.uniform(0.10, 0.60, N))
        p2 = jac_rect(G, B, ns, pq, pv, rng.uniform(0.85, 1.15, N), rng.uniform(0.10, 0.60, N))
        rows, cols = np.nonzero((np.abs(p1) + np.abs(p2)) > 0.0)
        indptr = np.zeros(m + 1, np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr).astype(np.int32)
        return rows, cols, cols.astype(np.int32), indptr, csr_matrix
    return _LINPAT.get_or((m, len(ns), len(pq)) + _gb_key(G, B), build)


def lin_sum(M, JlinC, G, B, ns, pq, pv):
    """sum_k |M dJ/de_k| on the cached sparsity pattern (self-checked against the dense reference at first)."""
    Ks = JlinC.shape[2]
    m = M.shape[0]
    if Ks == 0:
        return 0
    rows, cols, indices, indptr, csr = lin_pattern(G, B, ns, pq, pv, m)
    Jnz = JlinC[rows, cols, :]
    MT = M.T
    acc = 0
    for k in range(Ks):
        Sk = csr((Jnz[:, k], indices, indptr), shape=(m, m))
        acc = acc + np.abs(np.asarray(Sk.T @ MT).T)
    if _LIN_CHK[0] < LIN_NCHECK:
        _LIN_CHK[0] += 1
        ref = sum(np.abs(mdotsp(M, JlinC[:, :, k])) for k in range(Ks))
        d = float(np.abs(acc - ref).max())
        scale = float(np.abs(ref).max()) + 1e-12
        assert d <= 1e-9 * scale + 1e-12, f"[lin fast-path BUG] max|fast-ref|={d:.3e} (scale {scale:.3e})"
    return acc


# ==================================================================================================
# the Krawczyk system
# ==================================================================================================

class Shunt:
    """Uncertain bus shunt admittances:  y_total(bus_k) = y_k * d_k(e)  added to the network (on top of
    Ybus), with d a Taylor model over the input symbols (e.g. the admittance-load deviation delta)."""

    def __init__(self, bus, y, d: TM):
        self.bus = np.asarray(bus, int)
        self.y = np.asarray(y, complex)
        self.d = d

    def centre(self):
        return self.d[0]

    def dmax(self):
        from ..tm import bounds
        lo, hi = bounds(self.d)
        return np.maximum(np.abs(lo), np.abs(hi))

    def effective_GB(self, G, B, d=None):
        """(G, B) with the shunts at value d (default: centre) folded into the diagonal."""
        d = self.centre() if d is None else d
        G2, B2 = G.copy(), B.copy()
        np.add.at(G2, (self.bus, self.bus), self.y.real * d)
        np.add.at(B2, (self.bus, self.bus), self.y.imag * d)
        return G2, B2


class RectPFSystem:
    """AC power flow  F(z) = [P_ns; Q_pq; |V|^2_pv](z)  in rectangular state z = [vr_ns; vi_ns], with the
    non-ns (slack) bus fixed at (vr0, vi0).  Implements certflow.krawczyk.AffineJacobianSystem.

    shunt : optional :class:`Shunt` -- parameter-dependent bus admittances.  Their injection terms
            g d |V|^2, -b d |V|^2 enter F, and their Jacobian slope d(e) is bounded via a second Hessian term."""

    def __init__(self, G, B, ns, pq, pv, vr0, vi0, shunt: Shunt = None):
        self.G, self.B = G, B
        self.ns, self.pq, self.pv = list(ns), list(pq), list(pv)
        self.vr0, self.vi0 = vr0, vi0
        self.N = G.shape[0]
        self.na = len(self.ns)
        self.m = 2 * self.na
        nzY = int(np.count_nonzero((G != 0) | (B != 0), axis=1).max())
        self.jac_entry_terms = 2 * nzY + 12
        self.shunt = None
        if shunt is not None:
            nspos = -np.ones(self.N, int)
            nspos[self.ns] = np.arange(self.na)
            pqpos = -np.ones(self.N, int)
            pqpos[self.pq] = np.arange(len(self.pq))
            keep = nspos[shunt.bus] >= 0                          # a shunt on the slack bus does not enter F
            self.shunt = shunt
            self._sb = shunt.bus
            self._si = nspos[shunt.bus]
            self._sq = np.where(pqpos[shunt.bus] >= 0, self.na + pqpos[shunt.bus], -1)
            self._skeep = keep
            self.jac_entry_terms += 4

    # ---------------------------------------------------------------- shunt helpers
    def _jsh(self, ur, ui, absval=False):
        """Shunt Jacobian contribution, linear in u = d * v at the shunt buses (m x m)."""
        J = np.zeros((self.m, self.m))
        g = self.shunt.y.real
        b = self.shunt.y.imag
        if absval:
            g, b = np.abs(g), np.abs(b)
        for s in np.where(self._skeep)[0]:
            i = self._si[s]
            J[i, i] += 2 * g[s] * ur[s]
            J[i, self.na + i] += 2 * g[s] * ui[s]
            if self._sq[s] >= 0:
                q = self._sq[s]
                J[q, i] += (2 * b[s] if absval else -2 * b[s]) * ur[s]
                J[q, self.na + i] += (2 * b[s] if absval else -2 * b[s]) * ui[s]
        return J

    def _u(self, Vr, Vi):
        sb = self._sb
        return mul(self.shunt.d, Vr.take(sb)), mul(self.shunt.d, Vi.take(sb))

    def injections(self, Vr: TM, Vi: TM):
        """Full-bus injection TMs (P, Q) including the uncertain shunts."""
        P, Q = inj_rect_tm(self.G, self.B, Vr, Vi)
        if self.shunt is None:
            return P, Q
        from ..tm import scale
        sb = self._sb
        dV2 = mul(self.shunt.d, vmag2_tm(Vr.take(sb), Vi.take(sb)))
        return _add_rows(P, sb, scale(self.shunt.y.real, dV2)), _add_rows(Q, sb, scale(-self.shunt.y.imag, dV2))

    @property
    def z0(self):
        return np.concatenate([self.vr0[self.ns], self.vi0[self.ns]])

    def state(self, x: TM):
        """Reduced state TM -> full-bus (Vr, Vi) TMs (fixed buses carry a few-ulp rounding remainder)."""
        N, ns, na = self.N, self.ns, self.na
        Kc = x[1].shape[-1]
        Kq = x[2].shape[-1]
        padR = upr(8 * U * np.abs(self.vr0), 2) if CONFIG.rigorous else np.zeros(N)
        padI = upr(8 * U * np.abs(self.vi0), 2) if CONFIG.rigorous else np.zeros(N)
        Vr = (self.vr0.copy(), np.zeros((N, Kc)), np.zeros((N, Kq, Kq)), padR)
        Vi = (self.vi0.copy(), np.zeros((N, Kc)), np.zeros((N, Kq, Kq)), padI)
        for c in range(4):
            Vr[c][ns] = x[c][:na]
            Vi[c][ns] = x[c][na:]
        return TM(*Vr), TM(*Vi)

    def residual(self, x: TM) -> TM:
        Vr, Vi = self.state(x)
        P, Q = self.injections(Vr, Vi)
        V2 = vmag2_tm(Vr, Vi)
        return TM(*(np.concatenate([P[c][self.ns], Q[c][self.pq], V2[c][self.pv]]) for c in range(4)))

    def jacobian(self, z):
        vr = self.vr0.copy()
        vi = self.vi0.copy()
        vr[self.ns] = z[: self.na]
        vi[self.ns] = z[self.na:]
        J = jac_rect(self.G, self.B, self.ns, self.pq, self.pv, vr, vi)
        if self.shunt is not None:
            dc = self.shunt.centre()
            J = J + self._jsh(dc * vr[self._sb], dc * vi[self._sb])
        return J

    def jacobian_tm(self, x: TM):
        Vr, Vi = self.state(x)
        J0, JlinC, quad, R_rad = _djac_rect(self.G, self.B, self.ns, self.pq, self.pv, Vr, Vi)
        if self.shunt is None:
            return J0, JlinC, quad, R_rad
        ur, ui = self._u(Vr, Vi)
        Ks = JlinC.shape[2]
        J0 = J0 + self._jsh(ur[0], ui[0])
        JlinC = JlinC.copy()
        for k in range(Ks):
            JlinC[:, :, k] += self._jsh(ur[1][:, k], ui[1][:, k])
        R_rad = R_rad + self._jsh(ur[3], ui[3], absval=True)
        return J0, JlinC, quad + (ur[2], ui[2]), R_rad

    def lin_sum(self, M, JlinC):
        return lin_sum(M, JlinC, self.G, self.B, self.ns, self.pq, self.pv)

    def quad_sum(self, M, quad_src):
        q = quad_sum_rect(M, self.G, self.B, self.ns, self.pq, self.pv, quad_src[0], quad_src[1])
        if self.shunt is not None:
            UrQ, UiQ = quad_src[2], quad_src[3]
            Ks = UrQ.shape[1]
            for a in range(Ks):
                for c in range(a, Ks):
                    ur = UrQ[:, a, c] if c == a else UrQ[:, a, c] + UrQ[:, c, a]
                    ui = UiQ[:, a, c] if c == a else UiQ[:, a, c] + UiQ[:, c, a]
                    if ur.any() or ui.any():
                        q = q + np.abs(M @ self._jsh(ur, ui))
        return q

    def _sh_structure(self):
        rr, cc, jj, vv = [], [], [], []
        g, b = self.shunt.y.real, self.shunt.y.imag
        for s in np.where(self._skeep)[0]:
            i = self._si[s]
            for j in (i, self.na + i):
                rr.append(i); cc.append(j); jj.append(j); vv.append(2 * g[s])
                if self._sq[s] >= 0:
                    rr.append(self._sq[s]); cc.append(j); jj.append(j); vv.append(-2 * b[s])
        return structure_from_triplets(rr, cc, jj, vv, self.m)

    def hessian_structure(self):
        key = (self.N, len(self.ns), len(self.pq)) + _gb_key(self.G, self.B)
        base = _H_CACHE.get_or(key, lambda: h_structure(self.G, self.B, self.ns, self.pq, self.pv))
        if self.shunt is None or not self._skeep.any():
            return base
        scale = np.zeros(self.m)
        dm = self.shunt.dmax()
        for s in np.where(self._skeep)[0]:
            scale[self._si[s]] = scale[self.na + self._si[s]] = upr(dm[s], 2)
        return [(base, None), (self._sh_structure(), scale)]

    def quad_weights(self, quad_src):
        VrQ, ViQ = quad_src[0], quad_src[1]
        w = np.concatenate([quad_w(VrQ[self.ns]), quad_w(ViQ[self.ns])])
        if self.shunt is None or not self._skeep.any():
            return w
        wu = np.zeros(self.m)
        qr, qi = quad_w(quad_src[2]), quad_w(quad_src[3])
        for s in np.where(self._skeep)[0]:
            wu[self._si[s]] += qr[s]
            wu[self.na + self._si[s]] += qi[s]
        return [w, wu]

    def jac_majorant(self, x: TM, r):
        Vr, Vi = self.state(x)
        rR = upr(mag(Vr) + Vr[3], nt(Vr) + 2)
        rI = upr(mag(Vi) + Vi[3], nt(Vi) + 2)
        rR[self.ns] = upr(rR[self.ns] + r[: self.na], 2)
        rI[self.ns] = upr(rI[self.ns] + r[self.na:], 2)
        J = jac_rect_absrad(self.G, self.B, self.ns, self.pq, self.pv, rR, rI)
        if self.shunt is not None:
            dm = self.shunt.dmax()
            J = J + self._jsh(upr(dm * rR[self._sb], 2), upr(dm * rI[self._sb], 2), absval=True)
        return J


def _add_rows(A: TM, rows, B: TM) -> TM:
    """A + B where B lives on the subset ``rows`` of A's rows."""
    from ..tm import pad_symbols
    K = max(A[1].shape[-1], B[1].shape[-1])
    kq = max(A[2].shape[-1], B[2].shape[-1])
    A = pad_symbols(A, K, kq)
    B = pad_symbols(B, K, kq)
    full = TM(np.zeros_like(A[0]), np.zeros_like(A[1]), np.zeros_like(A[2]), np.zeros_like(A[3]))
    for c in range(4):
        np.add.at(full[c], rows, B[c])
    return add(A, full)
