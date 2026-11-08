"""Rich B-Rep tensors for the recognition network, on top of the face graph.

Three ideas from the literature, put together:
  * UV-Net (Jayaraman et al., CVPR 2021): a grid of surface samples for every face, with a trim mask,
    and a grid of curve samples for every edge, so the network sees real geometry.
  * BRepNet (Lambourne et al., CVPR 2021): oriented coedges with next / previous / mate pointers, so
    the network can walk the boundary, which a plain face-adjacency graph can't express.
  * The graph's hand-made features and rule hints stay as extra inputs (node_features).
"""
from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
from OCP.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

from . import occ

UV = 10      # surface samples per direction
CURVE = 10   # samples along an edge
EDGE_TYPES = (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_BSplineCurve)


def _face_grid(face, c, s):
    """UV × UV × 7: position, outward normal, inside-the-trim mask."""
    grid = np.zeros((7, UV, UV), "float32")
    a = BRepAdaptor_Surface(face, True)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    if not (np.isfinite([u0, u1, v0, v1]).all() and u1 > u0 and v1 > v0):
        return grid
    cls = BRepTopAdaptor_FClass2d(face, 1e-6)
    flip = face.Orientation() == TopAbs_REVERSED
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    for i, u in enumerate(np.linspace(u0, u1, UV)):
        for j, v in enumerate(np.linspace(v0, v1, UV)):
            a.D1(float(u), float(v), p, du, dv)
            n = du.Crossed(dv)
            m = n.Magnitude()
            nv = (n.X() / m, n.Y() / m, n.Z() / m) if m > 1e-12 else (0.0, 0.0, 0.0)
            if flip:
                nv = (-nv[0], -nv[1], -nv[2])
            st = cls.Perform(gp_Pnt2d(float(u), float(v)))
            grid[:, i, j] = ((p.X() - c[0]) * s, (p.Y() - c[1]) * s, (p.Z() - c[2]) * s, *nv,
                             1.0 if st in (TopAbs_IN, TopAbs_ON) else 0.0)
    return grid


def _edge_grid(edge, c, s):
    """CURVE × 6 samples (position, unit tangent), plus line/circle/ellipse/spline/closed flags."""
    grid = np.zeros((6, CURVE), "float32")
    kind = np.zeros(5, "float32")
    if BRep_Tool.Degenerated_s(edge):
        return grid, kind
    a = BRepAdaptor_Curve(edge)
    t0, t1 = a.FirstParameter(), a.LastParameter()
    p, d = gp_Pnt(), gp_Vec()
    for i, t in enumerate(np.linspace(t0, t1, CURVE)):
        a.D1(float(t), p, d)
        m = d.Magnitude() or 1.0
        grid[:, i] = ((p.X() - c[0]) * s, (p.Y() - c[1]) * s, (p.Z() - c[2]) * s, d.X() / m, d.Y() / m, d.Z() / m)
    ty = a.GetType()
    for k, g in enumerate(EDGE_TYPES):
        kind[k] = 1.0 if ty == g else 0.0
    kind[4] = 1.0 if a.IsClosed() else 0.0
    return grid, kind


def _wire_edges(wire, face):
    """Edges of a wire in loop order; falls back to file order if the wire explorer drops any."""
    raw = []
    ex = TopExp_Explorer(wire, TopAbs_EDGE)
    while ex.More():
        raw.append(TopoDS.Edge_s(ex.Current()))
        ex.Next()
    ordered = []
    we = BRepTools_WireExplorer(TopoDS.Wire_s(wire), face)
    while we.More():
        ordered.append(we.Current())
        we.Next()
    return ordered if len(ordered) == len(raw) else raw


def brep_tensors(g):
    """Surface grids, curve grids and coedge pointers for a FaceGraph."""
    x0, y0, z0, x1, y1, z1 = occ.bbox(g.shape)
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    s = 2.0 / max(x1 - x0, y1 - y0, z1 - z0, 1e-9)
    uv = np.stack([_face_grid(f, c, s) for f in g.faces]) if g.faces else np.zeros((0, 7, UV, UV), "float32")
    emap = g.emap
    ne = emap.Extent()
    cg, ck, conv = np.zeros((ne, 6, CURVE), "float32"), np.zeros((ne, 5), "float32"), np.zeros((ne, 5), "float32")
    for k in range(1, ne + 1):
        cg[k - 1], ck[k - 1] = _edge_grid(TopoDS.Edge_s(emap.FindKey(k)), c, s)
        lid = g.edge_index.get(k)
        if lid is not None:
            e = g.edges[lid]
            conv[k - 1] = (e.convexity == -1, e.convexity == 0, e.convexity == 1, e.angle / np.pi,
                           np.log10(max(e.length / g.char_len, 1e-6)) / 6 + 1)
    co_face, co_edge, co_rev, nxt, prv = [], [], [], [], []
    for fi, face in enumerate(g.faces):
        ex = TopExp_Explorer(face, TopAbs_WIRE)
        while ex.More():
            edges = _wire_edges(ex.Current(), TopoDS.Face_s(face))
            start = len(co_face)
            for n, e in enumerate(edges):
                co_face.append(fi)
                co_edge.append(emap.FindIndex(e) - 1)
                co_rev.append(1.0 if e.Orientation() == TopAbs_REVERSED else 0.0)
                nxt.append(start + (n + 1) % len(edges))
                prv.append(start + (n - 1) % len(edges))
            ex.Next()
    co_edge = np.asarray(co_edge, "int32")
    mate = np.arange(len(co_edge), dtype="int32")
    seen = {}
    for i, e in enumerate(co_edge.tolist()):
        if e in seen:
            mate[i], mate[seen[e]] = seen[e], i
        else:
            seen[e] = i
    return {"uv": uv.astype("float16"), "cg": cg.astype("float16"), "ek": np.concatenate([conv, ck], 1).astype("float32"),
            "cf": np.asarray(co_face, "int32"), "ce": co_edge, "cr": np.asarray(co_rev, "float32"),
            "cn": np.asarray(nxt, "int32"), "cp": np.asarray(prv, "int32"), "cm": mate}
