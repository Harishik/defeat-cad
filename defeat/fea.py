"""The physics check: same cases on the original and the defeatured model, chosen by analysis type.
Static: four load cases. Modal: first three natural frequencies. Thermal: conduction along each axis.
CFD: frontal area along each axis and wetted area.

Both models are meshed with gmsh at the same target size h and solved with scikit-fem
(quadratic tetrahedra, isotropic material). The material in the last 5% of the part's volume at
one end of its long axis is clamped; the same share at the other end carries the load, in bending
about both axes, tension and torsion. Those two regions are fixed on the ORIGINAL model and reused
as they are for the defeatured one, so rounding or removing something right at an end cannot change
the test itself.
"""
from __future__ import annotations

import time

import numpy as np

# default: PC/ABS, typical injection-molded housing material (mm, N, MPa)
MATERIAL = {"name": "PC/ABS", "E": 2300.0, "nu": 0.36, "rho": 1.1e-9}  # rho in t/mm3
FORCE_N = 100.0
MAX_TETS = 17000         # about 80k quadratic dofs, what a laptop factorizes in well under a minute
COARSE_CURVATURE = 8
END_SHARE = 0.05         # share of the part's volume clamped at one end and loaded at the other


def user_case(shape_faces, boundary, bc):
    """Put your picked supports and loads (face ids, force vector) into a load case."""
    from . import occ
    def pack(ids):
        return occ.to_brep_string(occ.compound_of([shape_faces[i] for i in ids])) if ids else None
    bc["user"] = {"fixed": pack(boundary["fixed"]), "loaded": pack(boundary.get("loaded") or []), "force": boundary.get("force")}
    f = boundary.get("force") or [0, 0, -FORCE_N]
    def count(n):
        return f"{n} face" + ("" if n == 1 else "s")
    load = f"force ({f[0]:g}, {f[1]:g}, {f[2]:g}) N on {count(len(boundary['loaded']))} you chose" if boundary.get("loaded") else "no load"
    bc["text"] = f"Your supports on {count(len(boundary['fixed']))}, {load} ({MATERIAL['name']}, E={MATERIAL['E']:g} MPa, ν={MATERIAL['nu']})"
    return bc


CHECK_NAMES = {"static": "load case", "modal": "natural frequency", "thermal": "heat path", "cfd": "flow area"}


def load_case(bbox, analysis="static"):
    x0, y0, z0, x1, y1, z1 = bbox
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    ext = hi - lo
    axis = int(np.argmax(ext))
    others = [i for i in range(3) if i != axis]
    load_dir = others[int(np.argmin(ext[others]))]
    wide_dir = [i for i in others if i != load_dir][0]
    names = "XYZ"
    return {"analysis": analysis, "axis": axis, "load_dir": load_dir, "wide_dir": wide_dir, "force": FORCE_N,
            "center": [float(v) for v in (lo + hi) / 2], "length": float(ext[axis]),
            "text": f"Last {END_SHARE:.0%} of the material clamped at the {names[axis]}min end, {FORCE_N:g} N along −{names[load_dir]} "
                    f"through the last {END_SHARE:.0%} at the {names[axis]}max end, plus bending the other way, tension and torsion "
                    f"({MATERIAL['name']}, E={MATERIAL['E']:g} MPa, ν={MATERIAL['nu']})"}


def lock(bc, result):
    """Reuse the mesh size and end regions of the original's solve for every later model of the part."""
    if result and result.get("ok"):
        bc["h_fea"], bc["ends"], bc["load_center"] = result.get("h_fea"), result.get("ends"), result.get("load_center")
        bc["ends_all"] = result.get("ends_all")
    return bc


def _end_regions(p, t, ax, share=END_SHARE):
    """Axis positions below and above which `share` of the volume lies."""
    a, b, c = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]], p[t[:, 3]] - p[t[:, 0]]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6
    x = p[t, ax].mean(axis=1)
    o = np.argsort(x)
    cum = np.cumsum(vol[o]) / vol.sum()
    return float(x[o][np.searchsorted(cum, share)]), float(x[o][min(np.searchsorted(cum, 1 - share), len(o) - 1)])


def solve(p, t, bc, order=2):
    """Quadratic tetrahedra by default. The load is spread through the loaded material by volume and
    deflection is its work-conjugate (volume-weighted) displacement, so neither depends on where the
    mesh happens to put its nodes. Four load cases share one factorization."""
    from scipy.sparse.linalg import splu
    from skfem import Basis, ElementTetP1, ElementTetP2, ElementVector, FacetBasis, LinearForm, MeshTet, asm
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    p, t = np.asarray(p, float), np.asarray(t, int)
    used = np.unique(t)
    remap = -np.ones(len(p), int)
    remap[used] = np.arange(len(used))
    p, t = p[used], remap[t]
    # positive orientation
    d = np.einsum("ij,ij->i", np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), p[t[:, 3]] - p[t[:, 0]])
    t[d < 0] = t[d < 0][:, [0, 2, 1, 3]]
    m = MeshTet(p.T, t.T)
    e = ElementVector(ElementTetP2() if order == 2 else ElementTetP1())
    ib = Basis(m, e, intorder=2)
    lam, mu = lame_parameters(MATERIAL["E"], MATERIAL["nu"])
    K = asm(linear_elasticity(lam, mu), ib).tocsc()
    ax = bc["axis"]
    lo, hi = bc.get("ends") or _end_regions(p, t, ax)
    user = bc.get("user")
    F = bc["force"]
    if user:      # your own supports and loads, picked on faces of the part
        fix = _facets_on(m, p, user["fixed"])
        if not len(fix):
            return {"ok": False, "error": "could not find the faces you chose as supports"}
        D = ib.get_dofs(facets=fix).all()
        touching, c = np.zeros(0, int), np.asarray(bc["center"])
        if user.get("loaded"):
            lf = _facets_on(m, p, user["loaded"])
            if not len(lf):
                return {"ok": False, "error": "could not find the faces you chose for the load"}
            fb = FacetBasis(m, e, facets=lf, intorder=3)
            area = float(np.asarray(fb.dx).sum())
            vec = np.asarray(user.get("force") or [0, 0, -FORCE_N], float)
            F = float(np.linalg.norm(vec)) or 1.0
            forms = {"your_load": asm(LinearForm(lambda v, w: (vec[0] * v.value[0] + vec[1] * v.value[1] + vec[2] * v.value[2]) / area), fb)}
        else:      # no load picked (modal needs only supports): a unit body force keeps the rest of the report meaningful
            forms = {"your_load": asm(LinearForm(lambda v, w: F * v.value[bc["load_dir"]] / np.asarray(ib.dx).sum()), ib)}
    else:
        D = np.nonzero(ib.doflocs[ax] <= lo)[0]
        # The loaded region is cut at quadrature points, not whole elements, so its size barely moves with the mesh
        touching = np.nonzero(p[t, ax].max(axis=1) >= hi)[0]
        if not len(D) or not len(touching):
            return {"ok": False, "error": "could not find support / load regions"}
        lb = Basis(m, e, elements=touching, intorder=4)
        x = lb.global_coordinates().value
        inside = (x[ax] >= hi).astype(float)
        weight = inside * lb.dx
        size = float(weight.sum())
        if size <= 0:
            return {"ok": False, "error": "could not find support / load regions"}
        F = bc["force"]
        # Torque F*L/10 about the loaded region's own centroid, so it carries no net force and always has a
        # real lever arm. The centroid comes from the original and is reused for the defeatured model.
        c = np.asarray(bc.get("load_center") or (x * weight).sum(axis=(1, 2)) / size)
        r = (x - c[:, None, None]) * inside
        r[ax] = 0
        polar = float((np.sum(r ** 2, axis=0) * lb.dx).sum()) or 1.0
        k = F * bc["length"] / 10 / polar
        a1, a2 = (ax + 1) % 3, (ax + 2) % 3

        def directed(i, sign):
            return LinearForm(lambda v, w: sign * F / size * w.inside * v.value[i])

        tors = LinearForm(lambda v, w: k * (v.value[a2] * w.r1 - v.value[a1] * w.r2))
        forms = {"bend_thin": asm(directed(bc["load_dir"], -1), lb, inside=inside),
                 "bend_wide": asm(directed(bc["wide_dir"], -1), lb, inside=inside),
                 "tension": asm(directed(ax, 1), lb, inside=inside), "torsion": asm(tors, lb, r1=r[a1], r2=r[a2])}
    primary = next(iter(forms))
    I = np.setdiff1d(np.arange(ib.N), D)
    lu = splu(K[I][:, I].tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}, diag_pivot_thresh=0.0)
    U, cases = {}, {}
    for name, f in forms.items():
        u = np.zeros(ib.N)
        u[I] = lu.solve(f[I])
        U[name] = u
        # work-conjugate displacement: mean deflection for forces, rotation-like measure for torsion
        cases[name] = float(f @ u) / F
    if not all(np.isfinite(v) and v > 0 for v in cases.values()):
        return {"ok": False, "error": "singular stiffness (part not connected to the supports)"}
    Ut = None
    analysis = bc.get("analysis", "static")
    extra = {}
    if analysis == "modal":
        cases, U = _modes(K, I, lu, ib, U)
    elif analysis == "thermal":
        cases, extra, Ut = _conduction(m, p, t, bc)
    elif analysis == "cfd":
        cases = _flow_areas(m, p)
    if not all(np.isfinite(v) and v > 0 for v in cases.values()):
        return {"ok": False, "error": f"the {analysis} check did not solve"}
    u = U[primary] if primary in U else next(iter(U.values()))
    du = ib.interpolate(u).grad
    eps = 0.5 * (du + np.swapaxes(du, 0, 1))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = {(i, j): 2 * mu * eps[i, j] + (lam * tr if i == j else 0) for i in range(3) for j in range(i, 3)}
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                 + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2))
    w = np.asarray(ib.dx).ravel()
    vm = np.asarray(vm).ravel()
    o = np.argsort(vm)
    cum = np.cumsum(w[o]) / w.sum()
    tip = float(forms[primary] @ U["_static"]) / F if "_static" in U else cases.get(primary, float(forms[primary] @ u) / F)
    # the stress peak away from the supports and load, where the applied boundary conditions don't set it
    if user:
        mid = np.ones(w.size, bool)
    else:
        xe = p[t, ax].mean(axis=1)
        mid = np.tile((xe > lo) & (xe < hi), w.size // t.shape[0])
        if not mid.any():
            mid = np.ones(w.size, bool)
    o_mid = np.argsort(vm[mid])
    cum_mid = np.cumsum(w[mid][o_mid]) / w[mid].sum()
    vm_peak = float(vm[mid][o_mid][min(int(np.searchsorted(cum_mid, 0.999)), o_mid.size - 1)])
    field = None
    if bc.get("want_field"):   # where each load case's strain energy sits, for the feature sensitivity
        dx = np.asarray(ib.dx)
        field = {"c": p[t].mean(axis=1).astype("float32"), "v": dx.sum(axis=1).astype("float32"), "w": {}}
        for name, ui in U.items():
            if name.startswith("_"):
                continue
            g = ib.interpolate(ui).grad
            ei = 0.5 * (g + np.swapaxes(g, 0, 1))
            tri = ei[0, 0] + ei[1, 1] + ei[2, 2]
            dens = lam * tri ** 2 + 2 * mu * sum(ei[i, j] ** 2 for i in range(3) for j in range(3))
            w_e = np.asarray(dens * dx).sum(axis=1)
            field["w"][name] = (w_e / max(float(w_e.sum()), 1e-30)).astype("float32")
        if analysis == "thermal":
            field["w"] = _thermal_field(Ut)
        elif analysis == "cfd":
            field["w"] = {"volume": field["v"] / max(float(field["v"].sum()), 1e-30)}
    export = None
    if bc.get("want_system"):   # the discrete problem, so an outside solver can be handed exactly the same one
        export = {"p": p, "t": t, "edges": m.t2e.T, "edge_nodes": m.edges.T, "nodal": ib.nodal_dofs, "edge": ib.edge_dofs,
                  "fixed": D, "forms": forms}
    return {"ok": True, "field": field, "system": export, "dofs": int(ib.N), "order": order, "max_disp": float(np.abs(u).max()), "tip_disp": tip,
            "stiffness": F / tip, "cases": cases,
            "vm_p99": float(vm[o][np.searchsorted(cum, 0.99)]), "vm_p95": float(vm[o][np.searchsorted(cum, 0.95)]),
            "vm_max": float(vm.max()), "vm_peak": vm_peak, "vm_mean": float((vm * w).sum() / w.sum()),
            "analysis": analysis, "ends": [lo, hi], "ends_all": extra.get("ends_all"), "load_center": [float(v) for v in c], "n_loaded": int(len(touching))}


def _facets_on(m, p, brep):
    """Boundary facets of the mesh lying on the given faces (a B-Rep string), by true distance to them."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    from . import occ
    shape = occ.from_brep_string(brep)
    bf = m.boundary_facets()
    tri = m.facets[:, bf].T
    cen = p[tri].mean(axis=1)
    size = float(np.median(np.linalg.norm(p[tri[:, 0]] - p[tri[:, 1]], axis=1)))
    tol = 0.1 * size
    x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
    near = np.nonzero((cen[:, 0] > x0 - tol) & (cen[:, 0] < x1 + tol) & (cen[:, 1] > y0 - tol) & (cen[:, 1] < y1 + tol)
                      & (cen[:, 2] > z0 - tol) & (cen[:, 2] < z1 + tol))[0]
    keep = []
    for i in near:
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(*[float(a) for a in cen[i]])).Vertex()
        d = BRepExtrema_DistShapeShape(v, shape)
        if d.IsDone() and d.Value() <= tol:
            keep.append(bf[i])
    return np.asarray(keep, int)


def _modes(K, I, lu, ib, U, n=3):
    """First natural frequencies with the same supports, by shift-invert on the same factorization."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    from skfem import BilinearForm, asm
    rho = MATERIAL["rho"]
    M = asm(BilinearForm(lambda u, v, w: rho * (u.value[0] * v.value[0] + u.value[1] * v.value[1] + u.value[2] * v.value[2])), ib).tocsc()
    Kii, Mii = K[I][:, I], M[I][:, I]
    op = LinearOperator(Kii.shape, matvec=lu.solve, dtype=float)
    vals, vecs = eigsh(Kii, k=n, M=Mii, sigma=0.0, which="LM", OPinv=op, tol=1e-6, v0=np.ones(len(I)))
    o = np.argsort(vals)
    out, modes = {}, {"_static": next(iter(U.values()))}
    for k, j in enumerate(o, 1):
        out[f"mode_{k}"] = float(np.sqrt(max(vals[j], 0.0)) / (2 * np.pi))
        x = np.zeros(ib.N)
        x[I] = vecs[:, j]
        modes[f"mode_{k}"] = x
    return out, modes


def _conduction(m, p, t, bc):
    """Steady conduction along each axis: a volume source at one end, the other held cold. The result is
    the mean temperature rise in the heated region, which is the thermal resistance up to a constant."""
    from scipy.sparse.linalg import splu
    from skfem import Basis, ElementTetP2, LinearForm, asm
    from skfem.models.poisson import laplace
    sb = Basis(m, ElementTetP2(), intorder=2)
    K = asm(laplace, sb).tocsc()
    ends = bc.get("ends_all") or [_end_regions(p, t, a) for a in range(3)]
    cases, temps = {}, {}
    user = bc.get("user")
    if user and user.get("loaded"):   # your cold faces and heated faces
        from skfem import FacetBasis
        D = sb.get_dofs(facets=_facets_on(m, p, user["fixed"])).all()
        fb = FacetBasis(m, ElementTetP2(), facets=_facets_on(m, p, user["loaded"]), intorder=3)
        area = float(np.asarray(fb.dx).sum()) or 1.0
        f = asm(LinearForm(lambda v, w: v.value / area), fb)
        I = np.setdiff1d(np.arange(sb.N), D)
        x = np.zeros(sb.N)
        x[I] = splu(K[I][:, I].tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}, diag_pivot_thresh=0.0).solve(f[I])
        return {"your_heat_path": float(f @ x)}, {"ends_all": None}, {"your_heat_path": (sb, x)}
    for a, (lo, hi) in enumerate(ends):
        D = np.nonzero(sb.doflocs[a] <= lo)[0]
        touching = np.nonzero(p[t, a].max(axis=1) >= hi)[0]
        if not len(D) or not len(touching):
            continue
        hb = Basis(m, ElementTetP2(), elements=touching, intorder=4)
        inside = (hb.global_coordinates().value[a] >= hi).astype(float)
        size = float((inside * hb.dx).sum()) or 1.0
        f = asm(LinearForm(lambda v, w: w.inside * v.value / size), hb, inside=inside)
        I = np.setdiff1d(np.arange(sb.N), D)
        x = np.zeros(sb.N)
        x[I] = splu(K[I][:, I].tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}, diag_pivot_thresh=0.0).solve(f[I])
        name = f"conduction_{'xyz'[a]}"
        cases[name] = float(f @ x)
        temps[name] = (sb, x)
    return cases, {"ends_all": [[float(v) for v in e] for e in ends]}, temps


def _thermal_field(temps):
    out = {}
    for name, (sb, x) in temps.items():
        g = sb.interpolate(x).grad
        w_e = np.asarray(sum(g[i] ** 2 for i in range(3)) * np.asarray(sb.dx)).sum(axis=1)
        out[name] = (w_e / max(float(w_e.sum()), 1e-30)).astype("float32")
    return out


def _flow_areas(m, p):
    """What the air sees: the projected (frontal) area along each axis and the wetted area."""
    tri = m.facets[:, m.boundary_facets()].T
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    n = 0.5 * np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    return {"frontal_x": float(0.5 * np.abs(n[:, 0]).sum()), "frontal_y": float(0.5 * np.abs(n[:, 1]).sum()),
            "frontal_z": float(0.5 * np.abs(n[:, 2]).sum()), "wetted": float(area.sum())}


def solve_mesh(step_bytes: bytes, st: dict, h: float, bc: dict):
    """Solve on the mesh in st, or on a coarser one if the quadratic system would not fit the direct
    solver. The size chosen for the original (bc['h_fea']) is reused for the defeatured model."""
    from .validate import mesh_stats
    hf = bc.get("h_fea") or h
    mesh = st
    coarse = COARSE_CURVATURE if bc.get("stat_curvature", 12) else 0   # a round with it off stays off: it hangs
    if hf != h:
        mesh = mesh_stats(step_bytes, hf, keep_mesh=True, curvature=coarse)
    elif not bc.get("h_fea"):
        while mesh.get("ok") and mesh["tets"] > MAX_TETS:
            hf *= max(1.2, min(2.5, 1.05 * (mesh["tets"] / MAX_TETS) ** (1 / 3)))   # elements go as 1/h^3
            mesh = mesh_stats(step_bytes, hf, keep_mesh=True, curvature=coarse)
    if not mesh.get("ok"):
        return {"ok": False, "error": "mesh: " + str(mesh.get("error"))[:100]}
    r = solve(mesh["mesh"]["p"], mesh["mesh"]["t"], bc)
    r["h_fea"] = hf
    r["tets_fea"] = mesh["tets"]
    return r


def mesh_and_solve(step_bytes: bytes, h: float, bc: dict):
    """Worker entry point: gmsh mesh + mesh stats + static solve (mesh data never leaves the worker)."""
    from .validate import mesh_stats
    st = mesh_stats(step_bytes, h, keep_mesh=True, curvature=bc.get("stat_curvature", 12))
    if not st.get("ok"):
        return st
    if bc.get("max_tets") and st["tets"] > bc["max_tets"]:
        return {"ok": False, "error": "too large for a finer check"}
    t0 = time.time()
    try:
        r = solve_mesh(step_bytes, st, h, bc)
    except Exception as e:  # noqa
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    st.pop("mesh")
    r["seconds"] = time.time() - t0
    st["fea"] = r
    return st


PEAK_TOL = 10.0     # stress peak, percent. Twice the stiffness tolerance: a peak depends far more on the mesh


def judged(delta, tol=5.0):
    """One number for the guard: the worst load case, or the move in the stress peak put on the same
    scale, so that a part passes only if its stiffness and its stress peak both stayed put."""
    d = worst_case(delta)
    if d is None:
        return None
    peak = (delta or {}).get("vm_peak_delta_pct")
    return d if peak is None else max(d, abs(peak) * tol / PEAK_TOL)


def worst_case(delta):
    """Largest change across the load cases, in percent."""
    cases = (delta or {}).get("cases_delta_pct")
    return max(abs(v) for v in cases.values()) if cases else None


def guard_message(guard, analysis="static"):
    """Plain-English account of what the stress guard did, or None if it had nothing to do."""
    if not (guard and guard.get("triggered")):
        return None
    if guard.get("peak_before_pct") is not None:
        before = f"Removing the features moved the stress peak by {guard['peak_before_pct']:.1f}%"
    else:
        before = f"Removing the features moved a {CHECK_NAMES.get(analysis, 'load case')} by {guard['before_pct']:.1f}%"
    if not guard.get("accepted"):
        return f"{before}, and no re-check could improve on that. Review this part before you rely on it."
    if guard.get("undone"):
        return f"{before}, and only the full part stayed within tolerance, so all {guard['of']} removals were undone."
    now = guard["peak_after_pct"] if guard.get("peak_after_pct") is not None else guard["after_pct"]
    after = f"{guard['restored']} of {guard['of']} features were put back, which brought it to {now:.1f}%"
    if guard.get("within_tol"):
        return f"{before}. {after}."
    return f"{before}. {after}, still outside tolerance. Review this part before you rely on it."


def compare(a, b):
    """Percent changes defeatured vs original for the headline quantities."""
    if not (a.get("ok") and b.get("ok")):
        return None

    def pct(k):
        return 100 * (b[k] - a[k]) / a[k] if a.get(k) else None

    out = {"tip_disp_delta_pct": pct("tip_disp"), "max_disp_delta_pct": pct("max_disp"),
           "vm_p99_delta_pct": pct("vm_p99"), "vm_peak_delta_pct": pct("vm_peak"), "vm_mean_delta_pct": pct("vm_mean"),
           "stiffness_delta_pct": pct("stiffness") if a.get("stiffness") and b.get("stiffness") else None}
    if a.get("cases") and b.get("cases"):
        out["cases_delta_pct"] = {k: 100 * (b["cases"][k] - a["cases"][k]) / a["cases"][k] for k in a["cases"]}
    return out
