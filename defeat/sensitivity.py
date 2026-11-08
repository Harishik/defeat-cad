"""What each removed feature costs and what it buys, from the original part's own stress solution.

Cost: the strain energy around a feature times the material it changes. This is the first-order
estimate behind feature sensitivity analysis (Gopalakrishnan & Suresh; Turevsky et al.) and the same
question analysis-aware defeaturing asks (Buffa, Chanon & Vázquez): a large hole in a quiet corner can
go, while a small fillet in the load path may have to stay.

Benefit: how many mesh elements crowd around the feature, since those are what removing it saves.
Together they give the stress guard an order to work in: put back whatever costs most per element saved.
"""
from __future__ import annotations

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from OCP.gp import gp_Pnt2d
from scipy.spatial import cKDTree

from .decide import est_volume

N = 6


def face_points(face):
    """Points on the trimmed face, on an N × N parameter grid; always at least the kept samples' centre."""
    a = BRepAdaptor_Surface(face, True)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    cls = BRepTopAdaptor_FClass2d(face, 1e-6)
    pts, fallback = [], None
    for u in np.linspace(u0, u1, N):
        for v in np.linspace(v0, v1, N):
            p = a.Value(float(u), float(v))
            fallback = (p.X(), p.Y(), p.Z())
            if cls.Perform(gp_Pnt2d(float(u), float(v))) in (TopAbs_IN, TopAbs_ON):
                pts.append(fallback)
    return pts or [fallback]


def feature_scores(features, graph, field, h):
    """{feature id: {"cost": worst share of compliance per load case, "benefit": elements nearby}}."""
    c = np.asarray(field["c"], float)
    v = np.asarray(field["v"], float)
    tree = cKDTree(c)
    out = {}
    for f in features:
        pts = np.array([p for x in f.faces for p in face_points(graph.faces[x])])
        r = max(float(f.size or 0), h) * 1.5
        near = np.unique(np.concatenate([np.asarray(q, int) for q in tree.query_ball_point(pts, r)] + [np.zeros(0, int)]))
        if not len(near):
            near = np.unique(tree.query(pts)[1])
        vol = max(float(v[near].sum()), 1e-12)
        dv = est_volume(f, graph)
        cost = max(float(np.asarray(w)[near].sum()) / vol * dv for w in field["w"].values())
        out[f.id] = {"cost": cost, "benefit": int(len(near))}
    return out


def hot_features(features, graph, field, h, axis=None, ends=None, ratio=1.0):
    """Features sitting on a stress peak of the original: the densest 1% of strain energy, away from the
    supports and load. Removing one could pass every global check and still hide the part's real hot spot."""
    c = np.asarray(field["c"], float)
    v = np.maximum(np.asarray(field["v"], float), 1e-30)
    keep = np.ones(len(c), bool) if axis is None or not ends else (c[:, axis] > ends[0]) & (c[:, axis] < ends[1])
    if not keep.any():
        return set()
    tree = cKDTree(c)
    limits = {}
    for name, w in field["w"].items():
        d = np.asarray(w, float) / v
        o = np.argsort(d[keep])
        cum = np.cumsum(v[keep][o]) / v[keep].sum()
        limits[name] = (d, d[keep][o][min(int(np.searchsorted(cum, 0.99)), o.size - 1)] * ratio)
    hot = set()
    for f in features:
        pts = np.array([p for x in f.faces for p in face_points(graph.faces[x])])
        r = max(float(f.size or 0), h) * 1.0
        near = np.unique(np.concatenate([np.asarray(q, int) for q in tree.query_ball_point(pts, r)] + [np.zeros(0, int)]))
        near = near[keep[near]]
        if len(near) and any(d[near].max() >= lim for d, lim in limits.values()):
            hot.add(f.id)
    return hot


def restore_order(features, scores):
    """Features in the order the guard should put them back: most stiffness risk per element saved first."""
    return sorted(features, key=lambda f: -scores[f.id]["cost"] / (scores[f.id]["benefit"] + 1.0))
