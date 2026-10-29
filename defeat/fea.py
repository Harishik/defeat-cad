"""Linear static check: same load case on the original and the defeatured model.

Both models are meshed with gmsh at the same target size h and solved with scikit-fem
(linear tetrahedra, isotropic material). Boundary conditions are picked from the ORIGINAL
part's bounding box so both solves see identical supports and loads: a cantilever along the
longest axis (one end clamped, the other loaded across the thinnest direction).
"""
from __future__ import annotations

import time

import numpy as np

# default: PC/ABS, typical injection-molded housing material (mm, N, MPa)
MATERIAL = {"name": "PC/ABS", "E": 2300.0, "nu": 0.36}
FORCE_N = 100.0


def load_case(bbox):
    x0, y0, z0, x1, y1, z1 = bbox
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    ext = hi - lo
    axis = int(np.argmax(ext))
    others = [i for i in range(3) if i != axis]
    load_dir = others[int(np.argmin(ext[others]))]
    names = "XYZ"
    return {"axis": axis, "lo": float(lo[axis]), "hi": float(hi[axis]), "tol": float(0.015 * ext[axis]),
            "load_dir": load_dir, "force": FORCE_N,
            "text": f"Clamped at {names[axis]}min face, {FORCE_N:g} N along −{names[load_dir]} on {names[axis]}max end "
                    f"({MATERIAL['name']}, E={MATERIAL['E']:g} MPa, ν={MATERIAL['nu']})"}


def solve(p, t, bc):
    from skfem import Basis, ElementTetP1, ElementVector, MeshTet, asm, condense, solve as fsolve
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
    ib = Basis(m, ElementVector(ElementTetP1()))
    lam, mu = lame_parameters(MATERIAL["E"], MATERIAL["nu"])
    K = asm(linear_elasticity(lam, mu), ib)
    ax = bc["axis"]
    fixed = np.where(p[:, ax] <= bc["lo"] + bc["tol"])[0]
    loaded = np.where(p[:, ax] >= bc["hi"] - bc["tol"])[0]
    if len(fixed) < 3 or len(loaded) < 1:
        return {"ok": False, "error": "could not find support / load regions"}
    f = np.zeros(ib.N)
    f[ib.nodal_dofs[bc["load_dir"], loaded]] = -bc["force"] / len(loaded)
    D = ib.nodal_dofs[:, fixed].ravel()
    u = fsolve(*condense(K, f, D=D))
    U = u[ib.nodal_dofs].T                     # (n_nodes, 3)
    disp = np.linalg.norm(U, axis=1)
    # element stresses (constant per P1 tet)
    X, Ue = p[t], U[t]
    Dm = X[:, 1:] - X[:, 0:1]                  # (M,3,3) rows = edge vectors
    dU = Ue[:, 1:] - Ue[:, 0:1]
    G = np.linalg.solve(Dm, dU).transpose(0, 2, 1)   # du_i/dx_j
    eps = 0.5 * (G + G.transpose(0, 2, 1))
    tr = np.trace(eps, axis1=1, axis2=2)
    sig = lam * tr[:, None, None] * np.eye(3) + 2 * mu * eps
    s = sig
    vm = np.sqrt(0.5 * ((s[:, 0, 0] - s[:, 1, 1]) ** 2 + (s[:, 1, 1] - s[:, 2, 2]) ** 2 + (s[:, 2, 2] - s[:, 0, 0]) ** 2)
                 + 3 * (s[:, 0, 1] ** 2 + s[:, 1, 2] ** 2 + s[:, 0, 2] ** 2))
    vol = np.abs(np.einsum("ij,ij->i", np.cross(Dm[:, 0], Dm[:, 1]), Dm[:, 2])) / 6
    # stiffness = load / tip deflection (mean over loaded nodes, in load direction)
    tip = float(np.mean(-U[loaded, bc["load_dir"]]))
    return {"ok": True, "dofs": int(ib.N), "max_disp": float(disp.max()), "tip_disp": tip,
            "stiffness": bc["force"] / tip if tip > 0 else None,
            "vm_p99": float(np.percentile(vm, 99)), "vm_p95": float(np.percentile(vm, 95)),
            "vm_max": float(vm.max()), "vm_mean": float(np.average(vm, weights=vol)),
            "n_fixed": int(len(fixed)), "n_loaded": int(len(loaded))}


def mesh_and_solve(step_bytes: bytes, h: float, bc: dict):
    """Worker entry point: gmsh mesh + mesh stats + static solve (mesh data never leaves the worker)."""
    from .validate import mesh_stats
    st = mesh_stats(step_bytes, h, keep_mesh=True)
    if not st.get("ok"):
        return st
    mesh = st.pop("mesh")
    t0 = time.time()
    try:
        r = solve(mesh["p"], mesh["t"], bc)
    except Exception as e:  # noqa
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    r["seconds"] = time.time() - t0
    st["fea"] = r
    return st


def compare(a, b):
    """Percent changes defeatured vs original for the headline quantities."""
    if not (a.get("ok") and b.get("ok")):
        return None

    def pct(k):
        return 100 * (b[k] - a[k]) / a[k] if a.get(k) else None

    return {"tip_disp_delta_pct": pct("tip_disp"), "max_disp_delta_pct": pct("max_disp"),
            "vm_p99_delta_pct": pct("vm_p99"), "vm_mean_delta_pct": pct("vm_mean"),
            "stiffness_delta_pct": pct("stiffness") if a.get("stiffness") and b.get("stiffness") else None}
