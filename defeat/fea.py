"""Linear static check: same load cases on the original and the defeatured model.

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
MATERIAL = {"name": "PC/ABS", "E": 2300.0, "nu": 0.36}
FORCE_N = 100.0
MAX_TETS = 17000         # about 80k quadratic dofs, what a laptop factorizes in well under a minute
COARSE_CURVATURE = 8
END_SHARE = 0.05         # share of the part's volume clamped at one end and loaded at the other


def load_case(bbox):
    x0, y0, z0, x1, y1, z1 = bbox
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    ext = hi - lo
    axis = int(np.argmax(ext))
    others = [i for i in range(3) if i != axis]
    load_dir = others[int(np.argmin(ext[others]))]
    wide_dir = [i for i in others if i != load_dir][0]
    names = "XYZ"
    return {"axis": axis, "load_dir": load_dir, "wide_dir": wide_dir, "force": FORCE_N,
            "center": [float(v) for v in (lo + hi) / 2], "length": float(ext[axis]),
            "text": f"Last {END_SHARE:.0%} of the material clamped at the {names[axis]}min end, {FORCE_N:g} N along −{names[load_dir]} "
                    f"through the last {END_SHARE:.0%} at the {names[axis]}max end, plus bending the other way, tension and torsion "
                    f"({MATERIAL['name']}, E={MATERIAL['E']:g} MPa, ν={MATERIAL['nu']})"}


def lock(bc, result):
    """Reuse the mesh size and end regions of the original's solve for every later model of the part."""
    if result and result.get("ok"):
        bc["h_fea"], bc["ends"], bc["load_center"] = result.get("h_fea"), result.get("ends"), result.get("load_center")
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
    from skfem import Basis, ElementTetP1, ElementTetP2, ElementVector, LinearForm, MeshTet, asm
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
    u = U["bend_thin"]
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
    tip = cases["bend_thin"]
    return {"ok": True, "dofs": int(ib.N), "order": order, "max_disp": float(np.abs(u).max()), "tip_disp": tip,
            "stiffness": F / tip, "cases": cases,
            "vm_p99": float(vm[o][np.searchsorted(cum, 0.99)]), "vm_p95": float(vm[o][np.searchsorted(cum, 0.95)]),
            "vm_max": float(vm.max()), "vm_mean": float((vm * w).sum() / w.sum()),
            "ends": [lo, hi], "load_center": [float(v) for v in c], "n_loaded": int(len(touching))}


def solve_mesh(step_bytes: bytes, st: dict, h: float, bc: dict):
    """Solve on the mesh in st, or on a coarser one if the quadratic system would not fit the direct
    solver. The size chosen for the original (bc['h_fea']) is reused for the defeatured model."""
    from .validate import mesh_stats
    hf = bc.get("h_fea") or h
    mesh = st
    if hf != h:
        mesh = mesh_stats(step_bytes, hf, keep_mesh=True, curvature=COARSE_CURVATURE)
    elif not bc.get("h_fea"):
        while mesh.get("ok") and mesh["tets"] > MAX_TETS:
            hf *= 1.35
            mesh = mesh_stats(step_bytes, hf, keep_mesh=True, curvature=COARSE_CURVATURE)
    if not mesh.get("ok"):
        return {"ok": False, "error": "mesh: " + str(mesh.get("error"))[:100]}
    r = solve(mesh["mesh"]["p"], mesh["mesh"]["t"], bc)
    r["h_fea"] = hf
    r["tets_fea"] = mesh["tets"]
    return r


def mesh_and_solve(step_bytes: bytes, h: float, bc: dict):
    """Worker entry point: gmsh mesh + mesh stats + static solve (mesh data never leaves the worker)."""
    from .validate import mesh_stats
    st = mesh_stats(step_bytes, h, keep_mesh=True)
    if not st.get("ok"):
        return st
    t0 = time.time()
    try:
        r = solve_mesh(step_bytes, st, h, bc)
    except Exception as e:  # noqa
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    st.pop("mesh")
    r["seconds"] = time.time() - t0
    st["fea"] = r
    return st


def worst_case(delta):
    """Largest change across the load cases, in percent."""
    cases = (delta or {}).get("cases_delta_pct")
    return max(abs(v) for v in cases.values()) if cases else None


def guard_message(guard):
    """Plain-English account of what the stress guard did, or None if it had nothing to do."""
    if not (guard and guard.get("triggered")):
        return None
    before = f"Removing the features moved a load case by {guard['before_pct']:.1f}%"
    if not guard.get("accepted"):
        return f"{before}, and no re-check could improve on that. Review this part before you rely on it."
    if guard.get("undone"):
        return f"{before}, and only the full part stayed within tolerance, so all {guard['of']} removals were undone."
    after = f"{guard['restored']} of {guard['of']} features were put back, which brought it to {guard['after_pct']:.1f}%"
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
           "vm_p99_delta_pct": pct("vm_p99"), "vm_mean_delta_pct": pct("vm_mean"),
           "stiffness_delta_pct": pct("stiffness") if a.get("stiffness") and b.get("stiffness") else None}
    if a.get("cases") and b.get("cases"):
        out["cases_delta_pct"] = {k: 100 * (b["cases"][k] - a["cases"][k]) / a["cases"][k] for k in a["cases"]}
    return out
