"""Triangulate a B-Rep for display, keeping a per-triangle face id."""
from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

from . import occ


def tessellate(shape, faces=None, lin_defl=None, ang_defl=0.35):
    """Returns (vertices[N,3] float32, triangles[M,3] int32, face_id[M] int32)."""
    if faces is None:
        faces = occ.faces_of(shape)
    if lin_defl is None:
        x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
        lin_defl = max(1e-3, 0.0015 * max(x1 - x0, y1 - y0, z1 - z0))
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    verts, tris, fids = [], [], []
    offset = 0
    for fid, f in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        n = tri.NbNodes()
        for i in range(1, n + 1):
            p = tri.Node(i).Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        rev = f.Orientation() == TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            if rev:
                b, c = c, b
            tris.append((a - 1 + offset, b - 1 + offset, c - 1 + offset))
            fids.append(fid)
        offset += n
    return (np.asarray(verts, np.float32), np.asarray(tris, np.int32), np.asarray(fids, np.int32))


def render_png(shape, path, face_colors=None, faces=None, elev=28, azim=-60, title=None):
    """Quick static render (matplotlib) for reports and debugging."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    V, T, F = tessellate(shape, faces)
    fig = plt.figure(figsize=(6, 5), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    tri = V[T]
    base = np.array([0.80, 0.83, 0.88])
    cols = np.tile(base, (len(T), 1))
    if face_colors:
        for fid, c in face_colors.items():
            cols[F == fid] = c
    # simple lambert shading
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
    pc = Poly3DCollection(tri, facecolors=np.clip(cols * shade[:, None], 0, 1), edgecolors="none")
    ax.add_collection3d(pc)
    mn, mx = V.min(0), V.max(0)
    c, r = (mn + mx) / 2, (mx - mn).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
