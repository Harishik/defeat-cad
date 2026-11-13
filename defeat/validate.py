"""Validation: geometric deltas, surface deviation, and real gmsh meshing of both models."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from . import occ
from .tessellate import tessellate


def _samples(shape, n=20000):
    V, T, _ = tessellate(shape)
    tri = V[T]
    a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    rng = np.random.default_rng(0)
    idx = rng.choice(len(T), size=n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]
    return np.vstack([pts, V])


def deviation(a, b):
    """Symmetric surface deviation (max = Hausdorff, plus mean and 95th pct)."""
    from scipy.spatial import cKDTree
    pa, pb = _samples(a), _samples(b)
    dab = cKDTree(pb).query(pa)[0]
    dba = cKDTree(pa).query(pb)[0]
    d = np.concatenate([dab, dba])
    return {"max": float(d.max()), "mean": float(d.mean()), "p95": float(np.percentile(d, 95))}


def geometry_report(orig, new):
    bo, bn = occ.bbox(orig), occ.bbox(new)
    vo, vn = occ.volume(orig), occ.volume(new)
    ao, an = occ.area(orig), occ.area(new)
    dev = deviation(orig, new)
    return {
        "faces": [len(occ.faces_of(orig)), len(occ.faces_of(new))],
        "edges": [len(occ.indexed(orig, occ.TopAbs_EDGE) and occ.sub_shapes(orig, occ.TopAbs_EDGE)) // 2,
                  len(occ.sub_shapes(new, occ.TopAbs_EDGE)) // 2],
        "volume": [vo, vn], "volume_delta_pct": 100 * (vn - vo) / vo if vo else 0.0,
        "area": [ao, an], "area_delta_pct": 100 * (an - ao) / ao if ao else 0.0,
        "bbox": [bo, bn],
        "bbox_changed": any(abs(x - y) > 5e-3 * max(bo[3] - bo[0], bo[4] - bo[1], bo[5] - bo[2]) for x, y in zip(bo, bn)),
        "valid": occ.is_valid(new),
        "deviation": dev,
    }


# ---------------------------------------------------------------- gmsh (runs in the worker)
MESHERS = (1, 10, 4)   # Delaunay first (deterministic); HXT and frontal when it fails on a surface mesh


def _stl_surface(step_path, h, stl_path):
    """OpenCASCADE's own watertight triangulation of the solid, written as STL. Its edges are shared by
    the faces on both sides, so the surface is closed even where the CAD surfaces are hard to parametrize."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    shape = occ.heal(occ.read_cad(step_path))
    box = occ.bbox(shape)
    diag = occ.norm(occ.sub(box[3:], box[:3]))
    BRepMesh_IncrementalMesh(shape, min(h / 8, diag / 400), False, 0.3, True)
    w = StlAPI_Writer()
    w.ASCIIMode = False
    if not w.Write(shape, stl_path):
        raise RuntimeError("could not triangulate the part")


def _volume_from_stl(gmsh, stl_path, h, remesh):
    """Volume mesh inside a closed triangulation. remesh: rebuild smooth patches from the triangles and
    remesh them at size h, so the elements match the CAD route's; otherwise keep the triangles as they are."""
    import math
    gmsh.model.add("stl")
    gmsh.merge(stl_path)
    gmsh.model.mesh.removeDuplicateNodes()
    if remesh:
        gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, True, True, math.pi)
        gmsh.model.mesh.createGeometry()
    surfaces = [e[1] for e in gmsh.model.getEntities(2)]
    loop = gmsh.model.geo.addSurfaceLoop(surfaces)
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    for algorithm in MESHERS:
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
        try:
            gmsh.model.mesh.generate(3)
            if len(gmsh.model.mesh.getElementsByType(4)[0]):
                return
        except Exception:
            pass
        gmsh.model.mesh.clear()
    raise RuntimeError("the mesher produced no volume elements")


def mesh_stats(step_bytes: bytes, h: float, order: int = 1, keep_mesh: bool = False, curvature: int = 12, route: str = "cad"):
    """route 'cad': gmsh meshes the CAD surfaces directly (best quality, follows fillets by curvature).
    route 'stl': gmsh works from OpenCASCADE's watertight triangulation instead, remeshed at size h, or kept
    as it is if remeshing fails. It needs no surface parametrization, so it gets through the periodic
    surfaces, slivers and gaps that make the CAD route give up or hang."""
    import gmsh
    fd, path = tempfile.mkstemp(suffix=".step")
    os.write(fd, step_bytes)
    os.close(fd)
    stl = path[:-5] + ".stl"
    out = {"ok": False}
    try:
        gmsh.initialize(["-noenv"])
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)   # deterministic meshes, run to run
        gmsh.option.setNumber("Mesh.Optimize", 1)         # remove sliver tetrahedra
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        t = time.time()
        if route == "stl":
            _stl_surface(path, h, stl)
            try:
                _volume_from_stl(gmsh, stl, h, remesh=True)
            except Exception:
                gmsh.model.remove()
                _volume_from_stl(gmsh, stl, h, remesh=False)
        else:
            gmsh.model.add("m")
            gmsh.model.occ.importShapes(path)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.05)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
            for algorithm in MESHERS:
                gmsh.option.setNumber("Mesh.Algorithm3D", algorithm)
                try:
                    gmsh.model.mesh.generate(3)
                    if not len(gmsh.model.mesh.getElementsByType(4)[0]):
                        raise RuntimeError("the mesher produced no volume elements")
                    break
                except Exception:
                    if algorithm == MESHERS[-1]:
                        raise
                    gmsh.model.mesh.clear()
        dt = time.time() - t
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        tets = np.concatenate([tg for ty, tg in zip(types, tags) if ty in (4, 11)]) if len(types) else np.array([])
        q = np.asarray(gmsh.model.mesh.getElementQualities(tets.tolist(), "minSICN")) if len(tets) else np.array([0.0])
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        out = {"ok": True, "tets": int(len(tets)), "nodes": int(len(ntags)), "seconds": dt, "route": route,
               "q_min": float(q.min()), "q_mean": float(q.mean()),
               "q_poor_pct": float(100 * (q < 0.2).mean()),
               "q_hist": np.histogram(q, bins=10, range=(0, 1))[0].tolist()}
        if keep_mesh:
            # nodes + tets (1-based gmsh tags remapped to 0-based)
            idx = {int(t_): i for i, t_ in enumerate(ntags)}
            et, en = gmsh.model.mesh.getElementsByType(4)
            conn = np.vectorize(idx.get)(np.asarray(en).reshape(-1, 4))
            out["mesh"] = {"p": coords.reshape(-1, 3).tolist(), "t": conn.tolist()}
    except Exception as e:  # noqa
        out = {"ok": False, "error": str(e), "route": route}
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        os.unlink(path)
        if os.path.exists(stl):
            os.unlink(stl)
    return out


def step_bytes(shape) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    occ.write_step(shape, path)
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def _mesh_rounds(h, diag):
    """(element size, curvature sizing, share of the time left). A part too heavy at h gets a coarser mesh
    that still follows its rounds. Curvature sizing is what hangs: on parts that timed out in every round
    it was the cause, and with it off they mesh in about a second. So the last rounds switch it off and
    size elements from the part itself, in case h is tiny for this part. They come last because a mesh
    that ignores curvature does not refine around fillets, so it understates what removing them saves.
    The 'stl' rounds mesh from OpenCASCADE's own watertight triangulation instead of the CAD surfaces, which
    gets through the periodic surfaces and gaps that make the CAD route give up."""
    return ((h, 12, "cad", 0.45), (h, 0, "stl", 0.35), (2 * h, 6, "cad", 0.4), (max(h, diag / 20), 0, "stl", 0.6),
            (max(2 * h, diag / 10), 0, "cad", 1.0))


def grid_compare(orig, new, analysis="static", boundary=None, timeout=300.0):
    """The stress check on a fixed grid (defeat.immersed), for parts no tetrahedral mesh can be built
    for. Only the default static load cases; None when it does not apply or does not solve."""
    from . import fea
    from .execute import worker
    if analysis != "static" or (boundary and boundary.get("fixed")):
        return None
    bc = fea.load_case(occ.bbox(orig), analysis)
    r = worker().call("defeat.immersed:solve_pair", occ.to_brep_string(orig), occ.to_brep_string(new), bc, timeout=timeout)
    if not r.get("ok"):
        return None
    return {"method": "grid", "load_case": bc["text"] + ". Solved on a fixed grid, without a mesh",
            "material": fea.MATERIAL, "original": r["original"], "defeatured": r["defeatured"], "delta": r["delta"]}


# ---------------------------------------------------------------- the original's solve, computed once
# The original part's mesh and stress solve depend only on its geometry, the mesh size and the load
# settings, never on which features are removed. So it is solved once (in the background as soon as a
# part is opened) and reused by every run and every stress-guard re-check on that part.
_ORIG = {}             # key -> {"done": threading.Event, "result": dict | None}
_ORIG_LOCK = None
_ORIG_MAX = 24


def _orig_lock():
    global _ORIG_LOCK
    if _ORIG_LOCK is None:
        import threading
        _ORIG_LOCK = threading.Lock()
    return _ORIG_LOCK


def _orig_key(shape, size: float, bc: dict):
    """By the exact geometry (the BREP text; STEP files carry a time stamp), mesh size and load settings."""
    import hashlib
    import json
    return (hashlib.sha1(occ.to_brep_string(shape)).hexdigest(), round(float(size), 9), json.dumps(bc, sort_keys=True, default=str))


def _orig_bc(orig, analysis, boundary, want_field, curve, rt):
    from . import fea
    bc = fea.load_case(occ.bbox(orig), analysis)
    if boundary and boundary.get("fixed"):
        fea.user_case(occ.faces_of(orig), boundary, bc)
    bc["want_field"] = want_field
    bc["stat_curvature"] = curve
    bc["mesh_route"] = rt
    return bc


def solve_original(w, shape, data: bytes, size: float, bc: dict, timeout: float):
    """mesh_and_solve for the original part, through the cache: a finished result is reused, one being
    computed in the background is waited for, and only a missing one is computed here."""
    import copy
    import threading
    key = _orig_key(shape, size, bc)
    with _orig_lock():
        ent = _ORIG.get(key)
        mine = ent is None
        if mine:
            ent = _ORIG[key] = {"done": threading.Event(), "result": None}
    if not mine:
        if ent["done"].wait(timeout) and ent["result"] is not None:
            return copy.deepcopy(ent["result"])
        if not ent["done"].is_set():
            return {"ok": False, "error": f"kernel timed out after {timeout:.0f}s"}
        # the background solve failed: fall through and try here, in the main worker
        with _orig_lock():
            ent = _ORIG[key] = {"done": threading.Event(), "result": None}
    try:
        r = w.call("defeat.fea:mesh_and_solve", data, size, bc, timeout=timeout)
    except Exception as e:  # noqa
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with _orig_lock():
        if (r.get("fea") or {}).get("ok"):
            ent["result"] = copy.deepcopy(r)
            while len(_ORIG) > _ORIG_MAX:
                _ORIG.pop(next(iter(_ORIG)))
        else:
            _ORIG.pop(key, None)
        ent["done"].set()
    return r


def prewarm_solve(shape, size, bc, timeout=240.0):
    """solve_original for this shape, size and load settings in a thread with its own worker process.
    A no-op when that solve is already cached or running."""
    import threading
    from .execute import Worker
    with _orig_lock():
        if _orig_key(shape, size, bc) in _ORIG:
            return
    data = step_bytes(shape)

    def job():
        w = Worker()
        try:
            solve_original(w, shape, data, size, bc, timeout)
        finally:
            w.close()
    threading.Thread(target=job, daemon=True, name="prewarm-solve").start()


def prewarm_original(orig, h, analysis="static", boundary=None, want_field=False, timeout=240.0):
    """Start solving the original at the first mesh round mesh_compare will try, in its own worker
    process and thread, so it runs while the user reviews features and while the kernel defeatures."""
    import threading
    from .execute import Worker
    if analysis not in ("static", "modal", "thermal", "cfd"):
        return
    box = occ.bbox(orig)
    size, curve, rt, _ = _mesh_rounds(h, occ.norm(occ.sub(box[3:], box[:3])))[0]
    prewarm_solve(orig, size, _orig_bc(orig, analysis, boundary, want_field, curve, rt), timeout)


def mesh_compare(orig, new, h, timeout=300.0, with_fea=True, want_field=False, analysis="static", boundary=None, budget=300.0):
    """budget: seconds for the whole comparison, shared out so a coarser mesh always gets its turn."""
    from . import fea
    from .execute import worker
    w = worker()
    if with_fea:
        import time
        b = {"ok": False, "error": "the original could not be meshed"}
        curvature, route = 12, "cad"
        end = time.time() + budget
        box = occ.bbox(orig)
        rounds = _mesh_rounds(h, occ.norm(occ.sub(box[3:], box[:3])))
        timed_out_at = 0.0
        for size, curve, rt, share in rounds:
            left = end - time.time()
            if left < 5:
                break
            if size <= timed_out_at:
                continue   # a round at the same size would time out again: go straight to a coarser mesh
            timeout = left * share / 2
            bc = _orig_bc(orig, analysis, boundary, want_field, curve, rt)
            a = solve_original(w, orig, step_bytes(orig), size, bc, timeout)
            if "timed out" in str(a.get("error", "")):
                timed_out_at = size
            if not (a.get("fea") or {}).get("ok"):
                continue
            bc["want_field"] = False
            fea.lock(bc, a.get("fea"))   # same mesh size and same end regions for both models
            b = w.call("defeat.fea:mesh_and_solve", step_bytes(new), size, bc, timeout=max(timeout, 5.0))
            if "timed out" in str(b.get("error", "")):
                timed_out_at = size
            if (b.get("fea") or {}).get("ok"):
                h = size
                curvature, route = curve, rt
                break
    else:
        a = w.call("defeat.validate:mesh_stats", step_bytes(orig), h, timeout=timeout)
        b = w.call("defeat.validate:mesh_stats", step_bytes(new), h, timeout=timeout)
    out = {"h": h, "original": a, "defeatured": b, "stat_curvature": curvature if with_fea else 12, "mesh_route": route if with_fea else "cad"}
    if with_fea and a.get("fea") and b.get("fea"):
        out["fea"] = {"load_case": bc["text"], "material": fea.MATERIAL, "original": a.pop("fea"),
                      "defeatured": b.pop("fea")}
        out["fea"]["delta"] = fea.compare(out["fea"]["original"], out["fea"]["defeatured"])
    if a.get("ok") and b.get("ok") and a["tets"]:
        out["tet_reduction_pct"] = 100 * (a["tets"] - b["tets"]) / a["tets"]
        out["speedup"] = a["seconds"] / max(b["seconds"], 1e-3)
    return out
