"""Procedural mold-part generator with face labels taken from how each part was built.

Labels come from the generator's own construction parameters (hole radii, boss axes, rib
boxes, text region, fillet radius...), NOT from the rule recognizer, so the GNN learns an
independent opinion that we can compare against the rules.
"""
from __future__ import annotations

import math
import random

import cadquery as cq

from . import occ
from .graph import build_graph

LABELS = ["base", "hole", "boss", "rib", "emboss", "fillet", "chamfer", "pocket", "step"]


def make_part(seed: int):
    R = random.Random(seed)
    L, W, T = R.uniform(70, 180), R.uniform(50, 120), R.uniform(2.5, 6)
    part = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    spec = {"holes": [], "bosses": [], "ribs": [], "text": None, "fillet_r": None, "chamfer": False,
            "pockets": [], "T": T, "L": L, "W": W}
    # optional perimeter wall
    wall_h = 0
    wt = 0
    if R.random() < 0.5:
        wall_h = R.uniform(8, 20)
        wt = R.uniform(2.0, 5.0)
        part = part.union(cq.Workplane("XY").rect(L, W).rect(L - 2 * wt, W - 2 * wt).extrude(wall_h + T))
    # triangular gussets against the wall
    if wall_h:
        for k in range(R.randint(0, 3)):
            t = R.uniform(1.5, 4.5)
            gh, gl = R.uniform(0.5, 0.9) * wall_h, R.uniform(8, 22)
            x0 = -L / 2 + wt
            y = R.uniform(-W / 2 + 12, W / 2 - 12 - t)
            gus = (cq.Workplane("XZ", origin=(0, y + t, 0))
                   .polyline([(x0, T), (x0 + gl, T), (x0, T + gh)]).close().extrude(t))
            part = part.union(gus)
            spec["ribs"].append((x0, y, T, x0 + gl, y + t, T + gh))
    # ribs
    nrib = R.randint(0, 4)
    for i in range(nrib):
        t = R.uniform(1.0, 3.0)
        h = R.uniform(4, 12)
        if R.random() < 0.5:
            x = R.uniform(-L / 2 + 10, L / 2 - 10)
            ln = R.uniform(0.3, 0.8) * W
            box = (x - t / 2, -ln / 2, T, x + t / 2, ln / 2, T + h)
        else:
            y = R.uniform(-W / 2 + 10, W / 2 - 10)
            ln = R.uniform(0.3, 0.8) * L
            box = (-ln / 2, y - t / 2, T, ln / 2, y + t / 2, T + h)
        part = part.union(cq.Workplane("XY").box(box[3] - box[0], box[4] - box[1], h, centered=False)
                          .translate((box[0], box[1], T)))
        spec["ribs"].append(box)
    # bosses (some with pilot holes)
    for i in range(R.randint(0, 4)):
        r = R.uniform(2.5, 6)
        h = R.uniform(4, 14)
        x, y = R.uniform(-L / 2 + 12, L / 2 - 12), R.uniform(-W / 2 + 12, W / 2 - 12)
        if any(b[0] - 3 < x < b[3] + 3 and b[1] - 3 < y < b[4] + 3 for b in spec["ribs"]):
            continue
        part = part.union(cq.Workplane("XY", origin=(x, y, T)).circle(r).extrude(h))
        spec["bosses"].append((x, y, r, T + h))
        if R.random() < 0.6:
            pr = r * R.uniform(0.3, 0.5)
            part = part.cut(cq.Workplane("XY", origin=(x, y, T + 1)).circle(pr).extrude(h + 1))
            spec["holes"].append((x, y, pr))
    # through holes
    for i in range(R.randint(1, 6)):
        r = R.uniform(0.8, 6)
        x, y = R.uniform(-L / 2 + 8, L / 2 - 8), R.uniform(-W / 2 + 8, W / 2 - 8)
        if any(b[0] - r - 2 < x < b[3] + r + 2 and b[1] - r - 2 < y < b[4] + r + 2 for b in spec["ribs"]):
            continue
        if any(math.hypot(x - bx, y - by) < br + r + 2 for bx, by, br, _ in spec["bosses"]):
            continue
        part = part.cut(cq.Workplane("XY", origin=(x, y, -1)).circle(r).extrude(T + 2 + wall_h))
        spec["holes"].append((x, y, r))
    # rectangular cut-outs / vents
    for i in range(R.randint(0, 3)):
        w, l = R.uniform(1.2, 10), R.uniform(6, 25)
        x, y = R.uniform(-L / 2 + 15, L / 2 - 15), R.uniform(-W / 2 + 15, W / 2 - 15)
        box = (x - w / 2, y - l / 2, x + w / 2, y + l / 2)
        if any(b[0] - 2 < box[2] and box[0] < b[3] + 2 and b[1] - 2 < box[3] and box[1] < b[4] + 2 for b in spec["ribs"]):
            continue
        if any(box[0] - br - 2 < bx < box[2] + br + 2 and box[1] - br - 2 < by < box[3] + br + 2 for bx, by, br, _ in spec["bosses"]):
            continue
        part = part.cut(cq.Workplane("XY", origin=(x, y, -1)).rect(w, l).extrude(T + 2))
        spec["pockets"].append(box)
    # embossed text on the bottom
    if R.random() < 0.7:
        h = R.uniform(0.4, 1.2)
        s = R.uniform(6, 12)
        txt = R.choice(["AB", "CAE", "PC-ABS", "MADE IN KR", "R1", "A3", "LOGO"])
        x, y = R.uniform(-L / 4, L / 4), R.uniform(-W / 4, W / 4)
        try:
            tw = cq.Workplane("XY", origin=(x, y, 0)).transformed(rotate=(180, 0, 0)).text(txt, s, h, halign="center", valign="center")
            part = part.union(tw)
            spec["text"] = (z := -h, 0.0)
        except Exception:
            pass
    shape = part.val().clean()
    # edge blends on the outer vertical edges, sometimes chamfers on the bottom outline
    try:
        fr = R.uniform(1.0, 4.0)
        shape2 = cq.Workplane().add(shape).edges("|Z").edges(
            cq.selectors.BoxSelector((-L, -W, T - 0.5), (L, W, T + wall_h + 0.5))).edges(">X or <X").fillet(fr).val()
        if shape2.isValid():
            shape, spec["fillet_r"] = shape2, fr
    except Exception:
        pass
    for _ in range(R.randint(0, 3)):
        try:
            ch = R.uniform(0.6, min(2.0, T * 0.4))
            z = 0.0 if (wall_h or R.random() < 0.6) else T
            if R.random() < 0.5:
                yy = R.choice([-W / 2, W / 2])
                sel = cq.selectors.BoxSelector((-L, yy - 0.2, z - 0.2), (L, yy + 0.2, z + 0.2))
                shape2 = cq.Workplane().add(shape).edges("|X").edges(sel).chamfer(ch).val()
            else:
                xx = R.choice([-L / 2, L / 2])
                sel = cq.selectors.BoxSelector((xx - 0.2, -W, z - 0.2), (xx + 0.2, W, z + 0.2))
                shape2 = cq.Workplane().add(shape).edges("|Y").edges(sel).chamfer(ch).val()
            if shape2.isValid():
                shape, spec["chamfer"] = shape2, True
        except Exception:
            pass
    return shape.wrapped, spec


def _close(a, b, tol=1e-3):
    return abs(a - b) < tol


def label_faces(shape, spec, g=None):
    g = g or build_graph(shape)
    T = spec["T"]
    labels = []
    for n in g.nodes:
        f = g.faces[n.id]
        lab = "base"
        c = n.centroid
        if n.kind in ("cylinder", "cone"):
            r = n.info.radius
            ax = n.info.axis or (0, 0, 1)
            vertical = abs(abs(ax[2]) - 1) < 1e-3
            if vertical and any(_close(r, hr) and math.hypot(n.info.loc[0] - hx, n.info.loc[1] - hy) < 1e-2
                                for hx, hy, hr in spec["holes"]):
                lab = "hole"
            elif vertical and any(_close(r, br) and math.hypot(n.info.loc[0] - bx, n.info.loc[1] - by) < 1e-2
                                  for bx, by, br, _ in spec["bosses"]):
                lab = "boss"
            elif spec["fillet_r"] and _close(r, spec["fillet_r"], 1e-2):
                lab = "fillet"
        if lab == "base" and n.kind in ("torus", "sphere") and spec["fillet_r"]:
            lab = "fillet"
        if lab == "base" and n.kind == "plane":
            for bx, by, br, top in spec["bosses"]:
                if _close(c[2], top, 1e-3) and math.hypot(c[0] - bx, c[1] - by) < br:
                    lab = "boss"
            for b in spec["ribs"]:
                if (b[0] - 1e-3 <= c[0] <= b[3] + 1e-3 and b[1] - 1e-3 <= c[1] <= b[4] + 1e-3
                        and c[2] > T + 1e-3):
                    nz = abs(n.info.normal[2])
                    thin = min(b[3] - b[0], b[4] - b[1])
                    # side walls, top cap, end faces of the rib
                    lab = "rib"
            for p in spec["pockets"]:
                if p[0] - 1e-3 <= c[0] <= p[2] + 1e-3 and p[1] - 1e-3 <= c[1] <= p[3] + 1e-3 and abs(n.info.normal[2]) < 1e-3 \
                        and 0 < c[2] < T:
                    lab = "pocket"
        if lab == "base" and spec["chamfer"] and n.kind == "plane" and 0.3 < abs(n.info.normal[2]) < 0.95 \
                and (abs(n.info.normal[0]) < 1e-3 or abs(n.info.normal[1]) < 1e-3) and c[2] < T + 1e-3:
            lab = "chamfer"
        if spec["text"] and c[2] < -1e-3:
            lab = "emboss"
        labels.append(LABELS.index(lab))
    return g, labels


# ---------------------------------------------------------------- features for the GNN
SURF = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "revolution", "extrusion", "other"]


def node_features(g):
    import numpy as np
    Lc = g.char_len
    X = []
    for n in g.nodes:
        oh = [1.0 if n.kind == s else 0.0 for s in SURF]
        conv = [0, 0, 0]
        deg = 0
        for m, eids in n.neighbors.items():
            for e in eids:
                conv[g.edges[e].convexity + 1] += 1
                deg += 1
        deg = max(deg, 1)
        r = n.info.radius / Lc if n.info.radius else 0.0
        X.append(oh + [
            math.log10(max(n.area / g.total_area, 1e-9)) / 9.0,
            math.tanh(n.mean_curv * Lc / 50), math.tanh(n.gauss_curv * Lc * Lc / 2500), math.tanh(n.max_curv * Lc / 50),
            1.0 if n.concave_axis else 0.0, 1.0 if n.concave_axis is False else 0.0,
            min(r * 20, 1.0), min(n.n_inner_loops, 5) / 5.0,
            conv[0] / deg, conv[1] / deg, conv[2] / deg, math.log1p(deg) / 4,
        ])
    E, EA = [], []
    for e in g.edges:
        a = [1.0 if e.convexity == k else 0.0 for k in (-1, 0, 1)] + [e.angle / math.pi, math.log10(max(e.length / Lc, 1e-6)) / 6 + 1]
        E += [(e.f1, e.f2), (e.f2, e.f1)]
        EA += [a, a]
    return np.asarray(X, "float32"), np.asarray(E, "int64").T if E else np.zeros((2, 0), "int64"), np.asarray(EA, "float32")


def sample(seed):
    """Build one labelled training graph (runs in a worker process)."""
    try:
        shape, spec = make_part(seed)
        g, y = label_faces(shape, spec)
        X, E, EA = node_features(g)
        return {"x": X, "e": E, "ea": EA, "y": y, "seed": seed}
    except Exception as ex:  # noqa
        return None
