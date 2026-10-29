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
def mesh_stats(step_bytes: bytes, h: float, order: int = 1, keep_mesh: bool = False):
    import gmsh
    fd, path = tempfile.mkstemp(suffix=".step")
    os.write(fd, step_bytes)
    os.close(fd)
    out = {"ok": False}
    try:
        gmsh.initialize(["-noenv"])
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)   # deterministic meshes, run to run
        gmsh.model.add("m")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.05)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)    # Delaunay (deterministic)
        gmsh.option.setNumber("Mesh.Optimize", 1)         # remove sliver tetrahedra
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        t = time.time()
        gmsh.model.mesh.generate(3)
        dt = time.time() - t
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        tets = np.concatenate([tg for ty, tg in zip(types, tags) if ty in (4, 11)]) if len(types) else np.array([])
        q = np.asarray(gmsh.model.mesh.getElementQualities(tets.tolist(), "minSICN")) if len(tets) else np.array([0.0])
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        out = {"ok": True, "tets": int(len(tets)), "nodes": int(len(ntags)), "seconds": dt,
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
        out = {"ok": False, "error": str(e)}
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        os.unlink(path)
    return out


def step_bytes(shape) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    occ.write_step(shape, path)
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def mesh_compare(orig, new, h, timeout=300.0, with_fea=True):
    from . import fea
    from .execute import worker
    w = worker()
    if with_fea:
        bc = fea.load_case(occ.bbox(orig))
        a = w.call("defeat.fea:mesh_and_solve", step_bytes(orig), h, bc, timeout=timeout)
        b = w.call("defeat.fea:mesh_and_solve", step_bytes(new), h, bc, timeout=timeout)
    else:
        a = w.call("defeat.validate:mesh_stats", step_bytes(orig), h, timeout=timeout)
        b = w.call("defeat.validate:mesh_stats", step_bytes(new), h, timeout=timeout)
    out = {"h": h, "original": a, "defeatured": b}
    if with_fea and a.get("fea") and b.get("fea"):
        out["fea"] = {"load_case": bc["text"], "material": fea.MATERIAL, "original": a.pop("fea"),
                      "defeatured": b.pop("fea")}
        out["fea"]["delta"] = fea.compare(out["fea"]["original"], out["fea"]["defeatured"])
    if a.get("ok") and b.get("ok") and a["tets"]:
        out["tet_reduction_pct"] = 100 * (a["tets"] - b["tets"]) / a["tets"]
        out["speedup"] = a["seconds"] / max(b["seconds"], 1e-3)
    return out
