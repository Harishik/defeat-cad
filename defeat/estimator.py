"""How much did defeaturing change the part's stiffness? Bounds from ONE solve, on the simplified part.

Setting. The part is Omega, the simplified part Omega0. A load case is a linear functional l (our loads
sit in the end regions, away from every feature) and its compliance is J = l(u), with u the
displacement on that domain. For a NEGATIVE feature F, material the part lacks and defeaturing filled
in (a hole, pocket, slot, convex round), Omega0 = Omega u F. With e = u - u0 the two weak problems give,
exactly,

    J(u) - J(u0) = a_Omega(e, e) + a_F(u0, u0),        a_D(v, w) = int_D sigma(v) : eps(w) dx

  a_F(u0, u0) : the strain energy the simplified solution stores in the filled-in material. Exact,
                computable without meshing the original, and a guaranteed LOWER bound on the change.
  a_Omega(e,e): the energy-norm defeaturing error. Antolin and Chanon (IJNME 2023, arXiv 2212.03141)
                bound it by an estimator computed from u0 on the feature's own faces gamma:
                    E_gamma^2 = (1/mu) ( |gamma|^(1/2) ||d - mean d||^2 + |gamma|^(3/2) |mean d|^2 )
                    d = g - sigma(u0) n   (g = 0 on a free surface: the traction u0 carries there)
                with |||e||| <~ E, a constant independent of the number and size of features. That
                constant is not known in closed form: it is measured on validation parts, never assumed.

POSITIVE features (bosses, ribs, concave rounds: material the simplified part drops) need a local
solve on the feature and are reported, not bounded, here.
"""
from __future__ import annotations

import numpy as np

from . import occ
from .fea import MATERIAL


def _wall_samples(shape, face_ids, defl=None):
    """Points, outward normals (of the part) and areas on the given B-Rep faces, from its triangulation."""
    from .tessellate import tessellate
    V, T, fid = tessellate(shape, lin_defl=defl)
    V, T = np.asarray(V, float), np.asarray(T, np.int64)
    sel = np.isin(fid, list(face_ids))
    a, b, c = V[T[sel, 0]], V[T[sel, 1]], V[T[sel, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    keep = area > 1e-12
    n = cr[keep] / (2 * area[keep, None])
    # 3-point rule per triangle (edge midpoints): exact for quadratics, so smooth tractions integrate well
    a, b, c, area = a[keep], b[keep], c[keep], area[keep]
    pts = np.concatenate([(a + b) / 2, (b + c) / 2, (c + a) / 2])
    nrm = np.concatenate([n, n, n])
    w = np.concatenate([area, area, area]) / 3
    return pts, nrm, w, fid[sel][keep]


def classify(orig, simple, features, removed, defl=None):
    """For each removed feature: 'negative' (defeaturing added material at it), 'positive' (it took
    material away) or 'mixed', by testing just across the feature's own faces against both solids."""
    from .immersed import boundary_triangles, inside
    Vo, To = boundary_triangles(orig, defl)
    Vs, Ts = boundary_triangles(simple, defl)
    box = occ.bbox(orig)
    eps = 1e-3 * occ.norm(occ.sub(box[3:], box[:3]))
    out = {}
    for f in features:
        if f.id not in removed:
            continue
        p, n, w, _ = _wall_samples(orig, f.faces, defl)
        if not len(p):
            continue
        # triangle centroids: the edge midpoints used for integration sit on rims, where inside is ambiguous
        k = len(p) // 3
        p, n, w = (p[:k] + p[k:2 * k] + p[2 * k:]) / 3, n[:k], 3 * w[:k]
        tot = float(w.sum()) or 1.0
        # probe at a distance scaled to the feature: a 1 mm chamfer leaves a wedge thinner than a part-sized step
        for step in sorted({min(eps, 0.05 * f.size) if f.size > 0 else eps, 0.2 * eps, 0.02 * eps}, reverse=True):
            across = inside(Vs, Ts, p + step * n)      # just outside the part: is it material now?
            under = ~inside(Vs, Ts, p - step * n)      # just inside the part: was material removed?
            neg, pos = float(w[across].sum()) / tot, float(w[under].sum()) / tot
            if neg + pos > 0.5:
                break
        out[f.id] = ("negative" if neg >= 0.6 and pos < 0.1 else "positive" if pos >= 0.6 and neg < 0.1
                     else "negligible" if neg + pos < 0.2 else "mixed")
    return out, (Vo, To)


def _locate(m, x):
    """For each point (3, n): a tet of the straight-sided mesh m that contains it and the point itself,
    or, for a point outside the mesh, the nearest tet among close candidates and the point clamped into
    it. A feature smaller than an element (a filled 0.1 mm chamfer on a 1 mm mesh) lies partly outside
    the straight-sided tets, and continuing a polynomial there is unbounded."""
    from scipy.spatial import cKDTree
    n = x.shape[1]
    P = m.p[:, m.t]                                   # (3, 4, nt)
    A = np.moveaxis(P[:, 1:, :] - P[:, :1, :], 2, 0)  # (nt, coord, j): x - p0 = A l
    Ainv = np.linalg.inv(A)
    k = min(24, m.t.shape[1])
    _, cand = cKDTree(P.mean(axis=1).T).query(x.T, k=k)
    cand = cand.reshape(n, k)
    best, bestd, bestx = cand[:, 0].copy(), np.full(n, np.inf), x.copy()
    for j in range(k):
        c = cand[:, j]
        l = np.einsum("pij,pj->pi", Ainv[c], (x - P[:, 0, c]).T)
        lc = np.clip(np.column_stack([1 - l.sum(axis=1), l]), 0, None)
        lc /= lc.sum(axis=1, keepdims=True)
        y = np.einsum("ikp,pk->ip", P[:, :, c], lc)
        d = np.linalg.norm(y - x, axis=0)
        take = d < bestd - 1e-15
        best[take], bestd[take], bestx[:, take] = c[take], d[take], y[:, take]
        if (bestd == 0).all():
            break
    out = bestd > 0                                   # nudge clamped points a hair into their tet
    cen = P[:, :, best].mean(axis=1)
    bestx[:, out] += 1e-9 * (cen[:, out] - bestx[:, out])
    return best, bestx


def _probes(sb, x, grad=False):
    """Like skfem's basis.probes for a scalar basis, robust to points just outside the mesh (see
    _locate). With grad=True, returns the three matrices giving the field's raw derivatives d/dx, d/dy,
    d/dz there instead. Raw, not recovered: an L2-projected gradient rings on coarse thin-walled meshes
    and overstated the strain energy in a small filled feature 40-fold."""
    from scipy.sparse import coo_matrix
    x = np.asarray(x, float)
    if x.shape[0] != 3:
        x = x.T
    n = x.shape[1]
    best, y = _locate(sb.mesh, x)
    pts = sb.mapping.invF(y[:, :, None], tind=best)
    fields = [sb.elem.gbasis(sb.mapping, pts, i, tind=best)[0] for i in range(sb.Nbfun)]
    rows = np.tile(np.arange(n, dtype=np.int32), sb.Nbfun)
    cols = sb.element_dofs[:, best].flatten()
    mk = lambda vals: coo_matrix((vals, (rows, cols)), shape=(n, sb.N)).tocsr()
    if not grad:
        return mk(np.array([f.value for f in fields]).flatten())
    return [mk(np.array([f.grad[d] for f in fields]).flatten()) for d in range(3)]


def _load_shift(m, e, cases, U, J0, bc, xq, w):
    """2 (l_orig(u0) - l_simple(u0)) per load case, for loads spread over the material at the loaded end
    (fea.solve): directed loads F/V_load over the loaded volume, torsion k (u_a2 r1 - u_a1 r2) with
    k = F L / 10 / polar. The original's loaded material is the simplified one's minus the filled-in part
    of it (points xq, weights w), so both its volume or polar moment and its load integral follow exactly."""
    from skfem import Basis
    ax, (lo, hi), F = bc["axis"], bc["ends"], bc["force"]
    a1, a2 = (ax + 1) % 3, (ax + 2) % 3
    c = np.asarray(bc["load_center"], float)
    b4 = Basis(m, e, intorder=4)
    X = b4.global_coordinates().value.reshape(3, -1)
    dx = np.asarray(b4.dx).ravel()
    inL = X[ax] >= hi
    size0 = float(dx[inL].sum())
    r0 = X[:, inL] - c[:, None]
    polar0 = float(((r0[a1] ** 2 + r0[a2] ** 2) * dx[inL]).sum())
    sel = xq[:, ax] >= hi
    if not sel.any() or size0 <= 0:
        return {k: 0.0 for k in cases}
    xf, wf = xq[sel], w[sel]
    ib = Basis(m, e, intorder=2)
    pr = _probes(ib.split_bases()[0], xf.T)
    comp = _component_dofs(ib)
    rf = xf.T - c[:, None]
    size = size0 - float(wf.sum())
    polar = polar0 - float(((rf[a1] ** 2 + rf[a2] ** 2) * wf).sum())
    if size <= 0 or polar <= 0:
        return {k: 0.0 for k in cases}
    directed = {"bend_thin": (bc["load_dir"], -1.0), "bend_wide": (bc["wide_dir"], -1.0), "tension": (ax, 1.0)}
    out = {}
    for name in cases:
        u = U[name]
        uf = [pr @ u[comp[i]] for i in range(3)]
        if name in directed:
            i, sgn = directed[name]
            # l0(u0) = sgn F / size0 * int_{L0} u_i  ->  int_{L0} u_i = sgn J0 size0 / F
            integral = sgn * J0[name] * size0 / F - float((uf[i] * wf).sum())
            l_orig = sgn * F / size * integral
        elif name == "torsion":
            k0 = F * bc["length"] / 10 / polar0
            k = F * bc["length"] / 10 / polar
            integral = J0[name] / k0 - float(((uf[a2] * rf[a1] - uf[a1] * rf[a2]) * wf).sum())
            l_orig = k * integral
        else:
            out[name] = 0.0
            continue
        out[name] = 2.0 * (l_orig - J0[name])
    return out


def _gradients(ib, U, x):
    """Raw displacement gradients of every case at points x (3, n): {case: (3, 3, n)}, g[i, j] = du_i/dx_j."""
    sb = ib.split_bases()[0]
    D = _probes(sb, x, grad=True)
    out = {}
    for name, u in U.items():
        uc = [u[sb_dofs] for sb_dofs in _component_dofs(ib)]
        out[name] = np.stack([[D[j] @ uc[i] for j in range(3)] for i in range(3)])
    return out


def _component_dofs(ib):
    """Global dof indices of each displacement component, ordered like the scalar split basis."""
    return [ib.split_indices()[c] for c in range(3)]


def _added_material(orig, simple, h, min_tets=400):
    """Quadrature points and weights filling exactly the material defeaturing added (simple minus orig),
    from a tet mesh of that solid alone, refined until it has enough elements to follow it. Returns
    (points (n, 3), weights (n,), exact volume) or None if the boolean or the mesh fails."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from skfem import Basis, ElementTetP1, MeshTet

    from .validate import mesh_stats, step_bytes
    def cut(a, b):
        c = BRepAlgoAPI_Cut(a, b)
        c.Build()
        if not c.IsDone():
            return None, 0.0
        s = c.Shape()
        return s, (abs(occ.volume(s)) if occ.solids_of(s) else 0.0)
    piece, vol = cut(simple, orig)
    if piece is None:
        return None
    # booleans on coincident faces can fail silently (returning an argument whole); added minus removed
    # volume must equal the change in volume, or the result is not trusted
    _, rem = cut(orig, simple)
    vs, vo = abs(occ.volume(simple)), abs(occ.volume(orig))
    if abs((vol - rem) - (vs - vo)) > max(0.05 * (vol + rem), 1e-5 * vs):
        return None
    if vol < 1e-9 * vs:
        return np.zeros((0, 3)), np.zeros(0), 0.0
    data, hf = step_bytes(piece), h
    for _ in range(6):
        ms = mesh_stats(data, hf, keep_mesh=True)
        if not ms.get("ok"):
            return None
        if ms["tets"] >= min_tets or hf < 1e-3 * h:
            break
        hf /= 2
    pm = MeshTet(np.asarray(ms["mesh"]["p"], float).T, np.asarray(ms["mesh"]["t"], int).T)
    b = Basis(pm, ElementTetP1(), intorder=4)
    w = np.asarray(b.dx).ravel()
    if abs(w.sum() - vol) > 0.05 * vol:              # the mesh does not fill the solid: not trusted
        return None
    return b.global_coordinates().value.reshape(3, -1).T, w, vol


def _dense_outside(m, Vo, To, near_pts, max_points=3_000_000):
    """Fallback when the boolean or its mesh fails: quadrature points of the simplified mesh m that lie
    outside the original part, from a composite rule (each tet split into 8^L sub-tets) on the tets near
    the removed features, so a feature smaller than an element is still resolved. Returns (points, weights)."""
    from scipy.spatial import cKDTree
    from skfem import Basis, ElementTetP1, MeshTet

    from .immersed import inside
    P = m.p[:, m.t]
    cen = P.mean(axis=1).T
    edge = np.linalg.norm(P[:, 0] - P[:, 1], axis=0)
    d, _ = cKDTree(near_pts).query(cen)
    sel = np.where(d < 1.5 * edge + 1e-12)[0]
    if not len(sel):
        return np.zeros((0, 3)), np.zeros(0)
    L = 4
    while L > 1 and len(sel) * 4 * 8 ** L > max_points:
        L -= 1
    ref = MeshTet.init_refdom().refined(L)
    rb = Basis(ref, ElementTetP1(), intorder=2)
    lp = rb.global_coordinates().value.reshape(3, -1)          # reference points
    lw = np.asarray(rb.dx).ravel()                            # reference weights (sum 1/6)
    A = P[:, 1:, sel] - P[:, :1, sel]                         # (3, 3, ns)
    det = np.abs(np.linalg.det(np.moveaxis(A, 2, 0)))
    X = (P[:, 0, sel][:, :, None] + np.einsum("ijs,jq->isq", A, lp)).reshape(3, -1).T
    W = (det[:, None] * lw[None, :]).ravel()
    out = ~inside(Vo, To, X)
    return X[out], W[out]


def _stress(G, lam, mu):
    eps = 0.5 * (G + np.swapaxes(G, 0, 1))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    return 2 * mu * eps + lam * tr * np.eye(3)[:, :, None]


def bounds(p, t, U, forms, orig, simple, features, removed, defl=None, bc=None):
    """Per load case and per removed feature: the lower bound a_F(u0, u0) and the estimator E^2, both
    in N mm, plus the simplified compliance J0 = l(u0). p, t: the simplified part's tet mesh; U and
    forms: its displacements and load vectors per case (from fea.solve with want_system).
    bc: the load cases as fea.solve used them (axis, ends, load_dir, wide_dir, force, length, load_center).
    The loads are spread over the material in the loaded end region, so material filled in there changes
    the load itself. By minimum potential energy, u0 restricted to the original part is still admissible
    (it is zero wherever the original is clamped), so for every case
        J(u) >= 2 l_orig(u0) - J0 + a_F(u0, u0),
    and l_orig(u0), the original's load applied to u0, is computed exactly from the loaded material minus
    the filled-in part of it. res["load_shift"] holds 2 (l_orig(u0) - J0) per case; it is zero when no
    filled material reaches the loaded end."""
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet
    from skfem.models.elasticity import lame_parameters
    from scipy.spatial import cKDTree

    from .immersed import inside

    lam, mu = lame_parameters(MATERIAL["E"], MATERIAL["nu"])
    kinds, (Vo, To) = classify(orig, simple, features, removed, defl)
    neg = [f for f in features if kinds.get(f.id) == "negative"]
    m = MeshTet(np.asarray(p, float).T, np.asarray(t, int).T)
    e = ElementVector(ElementTetP2())
    ib = Basis(m, e, intorder=2)
    cases = {k: v for k, v in U.items() if not k.startswith("_")}
    J0 = {k: float(forms[k] @ U[k]) for k in cases}
    res = {"kinds": kinds, "J0": J0, "features": {}}
    if not any(f.id in removed for f in features):
        return res

    # 1. lower bound: strain energy of u0 in the filled-in material Omega0 minus Omega, integrated on a
    # mesh of that material alone (classifying the simplified mesh's own quadrature points against the
    # original over-counts any feature smaller than an element), with u0's raw gradient there
    samples = [(f.id, _wall_samples(orig, f.faces, defl)) for f in neg]
    h = float(np.median(np.linalg.norm(m.p[:, m.t[0]] - m.p[:, m.t[1]], axis=0)))
    add = _added_material(orig, simple, h)
    lower = {f.id: {k: 0.0 for k in cases} for f in neg}
    res["lower_method"] = "added-material mesh"
    if add is None:                                   # boolean or mesh failed: dense sampling near the features
        res["lower_method"] = "dense sampling near features"
        near = np.concatenate([_wall_samples(orig, f.faces, defl)[0] for f in features if f.id in removed] or [np.zeros((0, 3))])
        xq, w = _dense_outside(m, Vo, To, near) if len(near) else (np.zeros((0, 3)), np.zeros(0))
    else:
        xq, w, res["added_volume"] = add
        res["added_volume_meshed"] = float(w.sum())
    res["load_shift"] = {k: 0.0 for k in cases}
    if bc is not None and len(xq) and not bc.get("user"):
        res["load_shift"] = _load_shift(m, e, cases, U, J0, bc, xq, w)
    if len(xq):
        # attribute each point to the nearest face of any removed feature; only negative ones are reported
        allf = [(f.id, _wall_samples(orig, f.faces, defl)[0]) for f in features if f.id in removed]
        allf = [(i, s) for i, s in allf if len(s)]
        pts = np.concatenate([s for _, s in allf])
        owner = np.concatenate([np.full(len(s), i) for i, s in allf])
        who = owner[cKDTree(pts).query(xq)[1]]
        Gq = _gradients(ib, {k: U[k] for k in cases}, xq.T)
        res["lower_other"] = {}
        for name in cases:
            g = Gq[name]
            eps = 0.5 * (g + np.swapaxes(g, 0, 1))
            tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
            dens = lam * tr ** 2 + 2 * mu * np.einsum("ijp,ijp->p", eps, eps)
            for f in neg:
                lower[f.id][name] = float((dens * w)[who == f.id].sum())
            # added material next to positive or mixed features still counts toward the total lower bound
            res["lower_other"][name] = float((dens * w)[~np.isin(who, [f.id for f in neg])].sum())

    # 2. estimator: traction residual of u0 on each feature's faces of the ORIGINAL part
    for fid, (P, N, W, _) in samples:
        Gs = _gradients(ib, {k: U[k] for k in cases}, P.T)
        area = float(W.sum())
        row = {"area": area, "samples": int(len(P)), "lower": {}, "est2": {}}
        for name in cases:
            g = Gs[name]                                       # (3, 3, pts)
            s = _stress(g, lam, mu)
            d = -np.einsum("ijp,pj->pi", s, N)                 # d = g - sigma n, with g = 0 on a free surface
            dbar = (d * W[:, None]).sum(axis=0) / area
            osc = float(((d - dbar) ** 2).sum(axis=1) @ W)
            row["est2"][name] = (area ** 0.5 * osc + area ** 1.5 * float(dbar @ dbar)) / mu
            row["lower"][name] = lower[fid][name]
        res["features"][fid] = row
    return res


def removed_material_energy(orig, simple, p, t, U, h):
    """Upper bound on how much stiffer the removed material made the part: by minimum potential energy,
    any displacement equal to u0 on the simplified part and continued into the removed material Omega
    minus Omega0 gives  J(u) - J(u0) >= a_neg(u0, u0) - a_pos(u_P, u_P). The smallest a_pos comes from
    solving elasticity on the removed material alone with u0 held where it meets the simplified part.
    Returns {case: a_pos} in N mm, or None if there is no removed material or it can't be meshed."""
    import igl
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from scipy.sparse.linalg import splu
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet, asm
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    from .immersed import boundary_triangles
    from .validate import mesh_stats, step_bytes

    cut = BRepAlgoAPI_Cut(orig, simple)
    cut.Build()
    if not cut.IsDone():
        return None
    piece = cut.Shape()
    if not occ.solids_of(piece) or abs(occ.volume(piece)) < 1e-9 * abs(occ.volume(orig)):
        return {k: 0.0 for k in U}
    ms = mesh_stats(step_bytes(piece), h, keep_mesh=True)
    if not ms.get("ok"):
        return None
    pm = MeshTet(np.asarray(ms["mesh"]["p"], float).T, np.asarray(ms["mesh"]["t"], int).T)
    e = ElementVector(ElementTetP2())
    bp = Basis(pm, e, intorder=2)
    lam, mu = lame_parameters(MATERIAL["E"], MATERIAL["nu"])
    K = asm(linear_elasticity(lam, mu), bp).tocsc()
    # interface: boundary facets of the removed material that lie on the simplified part's surface
    Vs, Ts = boundary_triangles(simple)
    bf = pm.boundary_facets()
    mid = pm.p[:, pm.facets[:, bf]].mean(axis=1).T
    box = occ.bbox(orig)
    tol = 1e-3 * occ.norm(occ.sub(box[3:], box[:3]))
    on = np.sqrt(igl.point_mesh_squared_distance(mid, Vs, Ts)[0]) < tol
    if not on.any():
        return None
    D = bp.get_dofs(facets=bf[on]).all()
    # u0 at those dofs, from the simplified solution
    ms0 = MeshTet(np.asarray(p, float).T, np.asarray(t, int).T)
    b0 = Basis(ms0, e, intorder=2)
    loc = bp.doflocs[:, D]
    out = {}
    I = np.setdiff1d(np.arange(bp.N), D)
    lu = splu(K[I][:, I].tocsc())
    # each Dirichlet dof belongs to one displacement component: dofs are interleaved x, y, z per node
    which = np.empty(bp.N, int)
    for c in range(3):
        which[bp.nodal_dofs[c]] = c
        if bp.edge_dofs.size:
            which[bp.edge_dofs[c]] = c
    for name, u in U.items():
        parts = [_probes(b, loc) @ uc for uc, b in b0.split(u)]     # (3, len(D)) values of u0 there
        x = np.zeros(bp.N)
        x[D] = np.array(parts)[which[D], np.arange(len(D))]
        x[I] = lu.solve(-(K[I][:, D] @ x[D]))
        out[name] = float(x @ (K @ x))
    return out


def summarize(res):
    """Totals per load case as percent of the simplified compliance J0: the change the guard reports
    is at least `lower_pct`; `est_pct` is the estimator's share, before its calibrated constant."""
    out = {}
    for name, j0 in res["J0"].items():
        lo = (sum(r["lower"][name] for r in res["features"].values()) + res.get("lower_other", {}).get(name, 0.0)
              + res.get("load_shift", {}).get(name, 0.0))
        e2 = sum(r["est2"][name] for r in res["features"].values())
        out[name] = {"lower_pct": 100 * lo / j0, "est_pct": 100 * e2 / j0}
    return out
