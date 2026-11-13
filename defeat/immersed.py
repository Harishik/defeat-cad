"""Stress check on a fixed grid, for parts no tetrahedral mesh can be built for.

The part is not meshed at all. A regular grid of hexahedra is laid over its bounding box, and every
integration point of every cell is classified inside or outside the part by its generalized winding
number (Jacobson et al. 2013; fast evaluation by Barill et al. 2018). The winding number is computed
from the triangulated boundary and stays meaningful when that boundary is non-manifold, has gaps, or
overlaps itself, which is exactly when a conforming mesher gives up.

Cells with no material are dropped. The rest are quadratic hexahedra whose stiffness is integrated
only over the material (the finite cell method, Parvizian, Duester and Rank 2007; the same idea as the
immersed solvers shown at CDFAM 2026). Empty integration points keep a stiffness of ALPHA so cut cells
never make the system singular.

The load cases are those of fea.solve: the last 5% of the material clamped at one end of the long
axis, the last 5% at the other end carrying bending both ways, tension and torsion. The original and
the defeatured part are solved on the SAME grid with the same end regions, so the error of the grid
itself is shared by both and largely cancels in the percent change the guard judges.
"""
from __future__ import annotations

import time

import numpy as np

from .fea import END_SHARE, FORCE_N, MATERIAL

ALPHA = 1e-6        # stiffness left in empty integration points, relative to the material
INTORDER = 4        # 3x3x3 Gauss points per cell: the material boundary is resolved inside each cell
TARGET_CELLS = 2500  # active quadratic cells: about 80k unknowns, one LU factorization, about a minute per part


def boundary_triangles(shape, defl=None):
    """Oriented triangle soup of the part's boundary. No watertightness is assumed."""
    from .tessellate import tessellate
    V, T, _ = tessellate(shape, lin_defl=defl)
    return np.asarray(V, float), np.asarray(T, np.int64)


def inside(V, T, Q):
    """Generalized winding number > 1/2 means inside."""
    import igl
    if not len(Q):
        return np.zeros(0, bool)
    w = igl.fast_winding_number(np.ascontiguousarray(V), np.ascontiguousarray(T), np.ascontiguousarray(Q))
    return np.abs(w) > 0.5


def grid_for(shape, V=None, T=None, target=TARGET_CELLS):
    """Cell size so the part fills about `target` cells: volume / g^3 = target."""
    from . import occ
    x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
    vol = abs(occ.volume(shape))
    ext = np.array([x1 - x0, y1 - y0, z1 - z0])
    g = (max(vol, 1e-9) / target) ** (1 / 3)
    # thin parts: the boundary cells dominate, so cap the count of cells over the whole box as well
    while np.prod(np.ceil(ext / g) + 2) > 40 * target:
        g *= 1.15
    lo = np.array([x0, y0, z0]) - 0.5 * g
    return {"origin": [float(v) for v in lo], "g": float(g), "n": [int(v) for v in np.ceil((ext + g) / g)]}


def _hex_mesh(grid):
    from skfem import MeshHex
    o, g, n = np.asarray(grid["origin"]), grid["g"], grid["n"]
    xs = [o[i] + g * np.arange(n[i] + 1) for i in range(3)]
    return MeshHex.init_tensor(*xs)


def _composite(sub=4, gauss=2):
    """Fine rule used only to measure each cut cell's moments: the cell split into sub^3 parts, each
    with a gauss^3 Gauss rule, on the reference cube [0, 1]^3."""
    g, w = np.polynomial.legendre.leggauss(gauss)
    g, w = (g + 1) / 2, w / 2
    pts1 = np.concatenate([(i + g) / sub for i in range(sub)])
    w1 = np.tile(w, sub) / sub
    P = np.array(np.meshgrid(pts1, pts1, pts1, indexing="ij")).reshape(3, -1)
    W = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    return P, W


def _legendre3(P, deg=2):
    """Tensor Legendre polynomials up to `deg` per direction at reference points P (3, n) in [0, 1]^3."""
    L = [np.polynomial.legendre.legvander(2 * P[i] - 1, deg) for i in range(3)]   # (n, deg+1) each
    return np.einsum("na,nb,nc->abcn", *L).reshape((deg + 1) ** 3, -1)


def moment_fitted_weights(chi_fine, fine, gauss):
    """Weights on the standard Gauss points of each cut cell that integrate every polynomial of degree
    <= 2 per direction exactly over the cell's material (moment fitting, as in Intact's immersed method).
    chi_fine: (cells, fine points) inside indicator. Returns (cells, gauss points), in reference units."""
    Pf, Wf = fine
    Pg, _ = gauss
    moments = (_legendre3(Pf) * Wf) @ chi_fine.T              # (27, cells)
    w = np.linalg.solve(_legendre3(Pg), moments).T          # (cells, 27)
    # A cell that only grazes the part can get negative weights, which would make its stiffness
    # indefinite. Those cells are fitted with non-negative weights to the moments of degree <= 1 per
    # direction (volume, centroid and their products), which a non-negative rule can still match.
    from scipy.optimize import nnls
    bad = np.nonzero((w < 0).any(axis=1))[0]
    if len(bad):
        A1 = _legendre3(Pg, 1)
        m1 = (_legendre3(Pf, 1) * Wf) @ chi_fine[bad].T
        for j, i in enumerate(bad):
            w[i] = nnls(A1, m1[:, j])[0]
    return w


def _element_matrices(g, e, quad, lam, mu):
    """Stiffness of one grid cube split by integration point: K_cell = sum_q phi_q * Kq[q]. Every cell of
    the grid is the same cube, so these 27 matrices serve them all."""
    from skfem import Basis, BilinearForm, MeshHex, asm
    from skfem.helpers import ddot, sym_grad, trace
    m1 = MeshHex.init_tensor(np.array([0.0, g]), np.array([0.0, g]), np.array([0.0, g]))
    b1 = Basis(m1, e, quadrature=quad)

    @BilinearForm
    def stiff(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return w.phi * (2 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev))

    ed = b1.element_dofs[:, 0]
    nq = len(quad[1])
    Kq = np.empty((nq, len(ed), len(ed)))
    for q in range(nq):
        phi = np.zeros((1, nq))
        phi[0, q] = 1.0
        Kq[q] = asm(stiff, b1, phi=phi).toarray()[np.ix_(ed, ed)]
    return Kq


def _assemble(Kq, phi, edofs, n):
    """Global stiffness from per-cell integration weights phi (cells, nq): one matrix product, then a
    sparse scatter, in chunks to bound memory."""
    from scipy.sparse import coo_matrix
    nl = edofs.shape[0]
    flat = Kq.reshape(Kq.shape[0], -1)
    K = None
    for s in range(0, phi.shape[0], 2000):
        ph, ed = phi[s:s + 2000], edofs[:, s:s + 2000].astype(np.int32)
        data = (ph @ flat).ravel()
        rows = np.repeat(ed.T, nl, axis=1).ravel()
        cols = np.tile(ed.T, (1, nl)).ravel()
        part = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
        K = part if K is None else K + part
    return K


def solve(V, T, bc):
    """Same outputs as fea.solve for the static check (cases, tip_disp, ends, load_center)."""
    from skfem import Basis, ElementHex2, ElementVector, LinearForm, asm
    from skfem.models.elasticity import lame_parameters
    from skfem.quadrature import get_quadrature
    from skfem.refdom import RefHex

    from .fea import _Solver

    t0 = time.time()
    times = {}
    m = _hex_mesh(bc["grid"])
    g = bc["grid"]["g"]
    # A cell whose centre is farther from the boundary than half its diagonal lies entirely on one side.
    centers = m.p[:, m.t].mean(axis=1).T
    ins_c = inside(V, T, centers)
    near = _near_surface(V, T, centers, 0.9 * g)
    full = ins_c & ~near
    cand = np.nonzero(near)[0]
    fine = _composite()
    gauss = get_quadrature(RefHex, INTORDER)
    probe = Basis(m, ElementHex2(), quadrature=fine, elements=cand)
    X = probe.global_coordinates().value
    chi = inside(V, T, X.reshape(3, -1).T).reshape(len(cand), -1).astype(float)
    frac = chi.mean(axis=1)
    full[cand[frac >= 1]] = True
    sel = (frac > 0) & (frac < 1)
    cut, chi = cand[sel], chi[sel]
    wfit = moment_fitted_weights(chi, fine, gauss)
    times["classify"] = time.time() - t0
    active = np.concatenate([np.nonzero(full)[0], cut])
    if not len(active):
        return {"ok": False, "error": "the grid found no material"}
    nf = int(full.sum())
    # phi: integration weight per Gauss point relative to the full cell's
    phi = np.ones((len(active), len(gauss[1])))
    phi[nf:] = wfit / gauss[1][None, :]
    order = np.argsort(active)
    m = m.restrict(active[order])
    phi = phi[order]
    is_cut = np.zeros(len(active), bool)
    is_cut[nf:] = True
    is_cut = is_cut[order]
    e = ElementVector(ElementHex2())
    ib = Basis(m, e, quadrature=gauss)
    lam, mu = lame_parameters(MATERIAL["E"], MATERIAL["nu"])
    Kq = _element_matrices(g, e, gauss, lam, mu)
    K = _assemble(Kq, phi + ALPHA * is_cut[:, None], ib.element_dofs, ib.N)
    times["assemble"] = time.time() - t0 - times["classify"]
    ax = bc["axis"]
    X = ib.global_coordinates().value
    dx = np.asarray(ib.dx)
    dxm = dx * phi                                       # material volume per integration point
    if bc.get("ends"):
        lo, hi = bc["ends"]
    else:
        xa = X[ax].ravel()
        o = np.argsort(xa)
        cum = np.cumsum(dxm.ravel()[o]) / dxm.sum()
        lo = float(xa[o][np.searchsorted(cum, END_SHARE)])
        hi = float(xa[o][min(np.searchsorted(cum, 1 - END_SHARE), cum.size - 1)])
    D = np.nonzero(ib.doflocs[ax] <= lo)[0]
    load = (X[ax] >= hi) * phi
    size = float((load * dx).sum())
    if not len(D) or size <= 0:
        return {"ok": False, "error": "could not find support / load regions"}
    F = FORCE_N
    c = np.asarray(bc.get("load_center") or (X * load * dx).sum(axis=(1, 2)) / size)
    r = (X - c[:, None, None]) * load
    r[ax] = 0
    polar = float((np.sum(r ** 2, axis=0) * dx).sum()) or 1.0
    k = F * bc["length"] / 10 / polar
    a1, a2 = (ax + 1) % 3, (ax + 2) % 3

    def directed(i, sign):
        return LinearForm(lambda v, w: sign * F / size * w.load * v.value[i])

    tors = LinearForm(lambda v, w: k * (v.value[a2] * w.r1 - v.value[a1] * w.r2))
    forms = {"bend_thin": asm(directed(bc["load_dir"], -1), ib, load=load),
             "bend_wide": asm(directed(bc["wide_dir"], -1), ib, load=load),
             "tension": asm(directed(ax, 1), ib, load=load), "torsion": asm(tors, ib, r1=r[a1], r2=r[a2])}
    I = np.setdiff1d(np.arange(ib.N), D)
    t1 = time.time()
    lu = _Solver(K[I][:, I], direct_limit=10 ** 7)
    cases, U = {}, {}
    for name, f in forms.items():
        u = np.zeros(ib.N)
        u[I] = lu.solve(f[I])
        U[name] = u
        cases[name] = float(f @ u) / F
    times["solve"] = time.time() - t1
    if not all(np.isfinite(v) and v > 0 for v in cases.values()):
        return {"ok": False, "error": "singular stiffness (part not connected to the supports)"}
    tip = cases["bend_thin"]
    return {"ok": True, "method": "grid", "cells": int(len(active)), "cut_cells": int(len(cut)), "dofs": int(ib.N), "grid": bc["grid"],
            "cases": cases, "tip_disp": tip, "stiffness": F / tip, "max_disp": float(np.abs(U["bend_thin"]).max()),
            "volume": float(dxm.sum()), "ends": [lo, hi], "load_center": [float(v) for v in c],
            "seconds": time.time() - t0, "times": {k: round(v, 1) for k, v in times.items()}}


def _near_surface(V, T, Q, dist):
    """Points within `dist` of the boundary, through a KD-tree over points sampled on the triangles."""
    from scipy.spatial import cKDTree
    a, b, cc = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    edge = max(dist / 2, 1e-9)
    pts = [a, b, cc, (a + b + cc) / 3]
    # subdivide long triangles so no boundary point is farther than dist/2 from a sample
    longest = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(cc - b, axis=1), np.linalg.norm(a - cc, axis=1)]), axis=0)
    k = np.clip(np.ceil(longest / edge).astype(int), 1, 64)
    for n in np.unique(k[k > 1]):
        sel = k == n
        s = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
        A, B, C = a[sel], b[sel], cc[sel]
        P = A[:, None, :] + s[None, :, 0, None] * (B - A)[:, None, :] + s[None, :, 1, None] * (C - A)[:, None, :]
        pts.append(P.reshape(-1, 3))
    tree = cKDTree(np.concatenate(pts))
    d, _ = tree.query(Q, distance_upper_bound=dist + edge)
    return np.isfinite(d)


def compare(orig, new, bc):
    """Solve both shapes on the original's grid with its end regions. bc comes from fea.load_case."""
    Va, Ta = boundary_triangles(orig)
    bc = dict(bc)
    bc.setdefault("grid", grid_for(orig))
    a = solve(Va, Ta, bc)
    if not a.get("ok"):
        return a, None
    bc["ends"], bc["load_center"] = a["ends"], a["load_center"]
    Vb, Tb = boundary_triangles(new)
    b = solve(Vb, Tb, bc)
    return a, b


def solve_pair(orig_brep: bytes, new_brep: bytes, bc: dict):
    """Worker entry point: both parts as BREP strings."""
    from . import occ
    from .fea import compare as pct
    try:
        a, b = compare(occ.from_brep_string(orig_brep), occ.from_brep_string(new_brep), bc)
    except Exception as e:  # noqa
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not (a.get("ok") and b and b.get("ok")):
        return {"ok": False, "error": (b or a).get("error"), "original": a, "defeatured": b}
    return {"ok": True, "original": a, "defeatured": b, "delta": pct(a, b)}
