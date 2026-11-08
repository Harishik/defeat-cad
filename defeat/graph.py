"""Attributed face-adjacency graph of a B-Rep solid.

Nodes are faces, links are shared edges labelled convex / concave / smooth.
Everything is pose-invariant so the same graph feeds the rule recognizer and the network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from OCP.BRepAdaptor import BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt2d, gp_Vec2d

from . import occ

CONVEX, CONCAVE, SMOOTH = 1, -1, 0
SMOOTH_ANGLE = math.radians(12)


@dataclass
class FaceNode:
    id: int
    kind: str
    area: float
    centroid: tuple
    info: occ.SurfInfo
    n_inner_loops: int
    mean_curv: float = 0.0
    gauss_curv: float = 0.0
    max_curv: float = 0.0
    concave_axis: bool | None = None   # cylinder/cone: normal points toward axis (hole-like)
    neighbors: dict = field(default_factory=dict)  # nid -> list of edge ids


@dataclass
class EdgeLink:
    id: int
    f1: int
    f2: int
    convexity: int
    angle: float      # angle between face normals (rad)
    length: float
    tangent: tuple | None = None  # unit direction at the edge midpoint


@dataclass
class FaceGraph:
    shape: object
    faces: list
    nodes: list
    edges: list
    char_len: float
    total_area: float
    inner_wires: dict  # fid -> list[list[edge ids]]
    emap: object = None         # indexed edge -> faces map of the shape
    edge_index: dict = field(default_factory=dict)  # emap key (1-based) -> EdgeLink id

    def link(self, a, b):
        return [self.edges[e] for e in self.nodes[a].neighbors.get(b, [])]


def _inward_point(face, edge, t, eps_frac=2e-3):
    """3D point slightly inside `face` from parameter t on `edge` (material-side rule)."""
    fF = TopoDS.Face_s(face.Oriented(TopAbs_FORWARD))
    ori = None
    ex = TopExp_Explorer(fF, TopAbs_EDGE)
    while ex.More():
        if ex.Current().IsSame(edge):
            ori = ex.Current().Orientation()
            break
        ex.Next()
    c2 = BRepAdaptor_Curve2d(TopoDS.Edge_s(edge), fF)
    tt = min(max(t, c2.FirstParameter()), c2.LastParameter())
    p, v = gp_Pnt2d(), gp_Vec2d()
    c2.D1(tt, p, v)
    vx, vy = v.X(), v.Y()
    if ori == TopAbs_REVERSED:
        vx, vy = -vx, -vy
    nx, ny = -vy, vx
    nn = math.hypot(nx, ny) or 1.0
    a = BRepAdaptor_Surface(fF, True)
    du = (a.LastUParameter() - a.FirstUParameter())
    dv = (a.LastVParameter() - a.FirstVParameter())
    step = eps_frac * max(min(du, dv), 1e-9)
    u = p.X() + step * nx / nn
    w = p.Y() + step * ny / nn
    q = a.Value(u, w)
    return (q.X(), q.Y(), q.Z())


def classify_edge(f1, f2, edge):
    """Convexity of the dihedral between f1 and f2 along edge."""
    try:
        t, p, tan = occ.edge_mid(edge)
        n1 = occ.normal_at_point(f1, p)
        n2 = occ.normal_at_point(f2, p)
        ang = math.acos(max(-1.0, min(1.0, occ.dot(occ.unit(n1), occ.unit(n2)))))
        if ang < SMOOTH_ANGLE:
            return SMOOTH, ang
        q = _inward_point(f2, edge, t)
        d = occ.dot(occ.unit(occ.sub(q, p)), occ.unit(n1))
        return (CONVEX if d < 0 else CONCAVE), ang
    except Exception:
        return CONVEX, math.pi / 2


def _cyl_concave(face, info):
    """True if the cylinder/cone material is outside (hole-like)."""
    p, n = occ.face_point_normal(face)
    if info.axis is None or info.loc is None:
        return None
    ax, lo = occ.unit(info.axis), info.loc
    v = occ.sub(p, lo)
    radial = occ.sub(v, tuple(occ.dot(v, ax) * c for c in ax))
    return occ.dot(n, radial) < 0


def build_graph(shape, curvature=True) -> FaceGraph:
    faces = occ.faces_of(shape)
    fmap = occ.indexed(shape, occ.TopAbs_FACE)
    x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
    char_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    nodes = []
    for i, f in enumerate(faces):
        info = occ.surf_info(f)
        ow = BRepTools.OuterWire_s(f)
        wires = occ.sub_shapes(f, TopAbs_WIRE)
        n_inner = sum(1 for w in wires if not w.IsSame(ow))
        node = FaceNode(i, info.kind, occ.area(f), occ.centroid(f), info, n_inner)
        if curvature:
            node.mean_curv, node.gauss_curv, node.max_curv = occ.face_curvatures(f)
        if info.kind in ("cylinder", "cone"):
            node.concave_axis = _cyl_concave(f, info)
        nodes.append(node)

    emap = occ.edge_face_map(shape)
    edges = []
    edge_index = {}
    for k in range(1, emap.Extent() + 1):
        e = emap.FindKey(k)
        lst = emap.FindFromIndex(k)
        fs = []
        it = lst.cbegin() if hasattr(lst, "cbegin") else None
        for fshape in lst:
            fi = fmap.FindIndex(fshape) - 1
            if fi >= 0 and fi not in fs:
                fs.append(fi)
        if len(fs) != 2:
            continue  # seam or free edge
        if BRep_Tool.Degenerated_s(TopoDS.Edge_s(e)):
            continue
        a, b = fs
        cvx, ang = classify_edge(faces[a], faces[b], e)
        try:
            tan = occ.unit(occ.edge_mid(e)[2])
        except Exception:
            tan = None
        link = EdgeLink(len(edges), a, b, cvx, ang, occ.edge_length(e), tan)
        edge_index[k] = link.id
        edges.append(link)
        nodes[a].neighbors.setdefault(b, []).append(link.id)
        nodes[b].neighbors.setdefault(a, []).append(link.id)

    inner_wires = {}
    for i, f in enumerate(faces):
        if nodes[i].n_inner_loops == 0:
            continue
        ow = BRepTools.OuterWire_s(f)
        loops = []
        for w in occ.sub_shapes(f, TopAbs_WIRE):
            if w.IsSame(ow):
                continue
            ids = []
            for e in occ.sub_shapes(w, TopAbs_EDGE):
                k = emap.FindIndex(e)
                if k in edge_index:
                    ids.append(edge_index[k])
            loops.append(ids)
        inner_wires[i] = loops

    total_area = sum(n.area for n in nodes)
    return FaceGraph(shape, faces, nodes, edges, char_len, total_area, inner_wires, emap, edge_index)
