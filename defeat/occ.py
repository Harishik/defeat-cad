"""Thin helpers over OpenCASCADE (OCP) used across the engine."""
from __future__ import annotations

import math
from dataclasses import dataclass

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone,
                         GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                         GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution,
                         GeomAbs_Torus)
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Reader
from OCP.Interface import Interface_Static
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                          TopTools_IndexedMapOfShape)
from OCP.gp import gp_Pnt, gp_Vec

SURF_TYPES = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bspline", GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
}
SURF_NAMES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "revolution", "extrusion", "other"]


# ---------------------------------------------------------------- I/O
def read_cad(path: str) -> TopoDS_Shape:
    p = path.lower()
    if p.endswith((".step", ".stp")):
        r = STEPControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            raise ValueError(f"Could not read STEP file: {path}")
        r.TransferRoots()
        shape = r.OneShape()
    elif p.endswith((".iges", ".igs")):
        r = IGESControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            raise ValueError(f"Could not read IGES file: {path}")
        r.TransferRoots()
        shape = r.OneShape()
    elif p.endswith(".brep"):
        shape = TopoDS_Shape()
        BRepTools.Read_s(shape, path, BRep_Builder())
    else:
        raise ValueError("Supported formats: .step .stp .iges .igs .brep")
    return shape


def _quiet():
    from OCP.Message import Message, Message_Gravity
    try:
        for p in Message.DefaultMessenger_s().Printers():
            p.SetTraceLevel(Message_Gravity.Message_Fail)
    except Exception:
        pass


def write_step(shape: TopoDS_Shape, path: str) -> None:
    _quiet()
    Interface_Static.SetCVal_s("write.step.schema", "AP214")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(path)


def to_brep_string(shape: TopoDS_Shape) -> bytes:
    import io, os, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".brep")
    os.close(fd)
    BRepTools.Write_s(shape, tmp)
    data = open(tmp, "rb").read()
    os.unlink(tmp)
    return data


def from_brep_string(data: bytes) -> TopoDS_Shape:
    import os, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".brep")
    os.write(fd, data)
    os.close(fd)
    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, tmp, BRep_Builder())
    os.unlink(tmp)
    return shape


def heal(shape: TopoDS_Shape) -> TopoDS_Shape:
    if is_valid(shape):
        return shape
    fix = ShapeFix_Shape(shape)
    fix.Perform()
    return fix.Shape()


def is_valid(shape: TopoDS_Shape) -> bool:
    return BRepCheck_Analyzer(shape).IsValid()


# ---------------------------------------------------------------- topology
def sub_shapes(shape, kind):
    out, ex = [], TopExp_Explorer(shape, kind)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def indexed(shape, kind) -> TopTools_IndexedMapOfShape:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def faces_of(shape):
    """Unique faces in stable (indexed-map) order. Face id = position in this list."""
    m = indexed(shape, TopAbs_FACE)
    return [TopoDS.Face_s(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def edge_face_map(shape) -> TopTools_IndexedDataMapOfShapeListOfShape:
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return m


def solids_of(shape):
    return [TopoDS.Solid_s(s) for s in sub_shapes(shape, TopAbs_SOLID)]


def make_compound(shapes) -> TopoDS_Compound:
    c, b = TopoDS_Compound(), BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


# ---------------------------------------------------------------- measures
def volume(shape) -> float:
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def area(shape) -> float:
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return p.Mass()


def centroid(shape):
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    c = p.CentreOfMass()
    return (c.X(), c.Y(), c.Z())


def bbox(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b, True)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return (x0, y0, z0, x1, y1, z1)


def edge_length(edge) -> float:
    edge = TopoDS.Edge_s(edge)
    p = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, p)
    return p.Mass()


@dataclass
class SurfInfo:
    kind: str
    radius: float = 0.0          # cylinder / sphere / cone ref radius / torus minor
    major: float = 0.0           # torus major radius
    axis: tuple | None = None    # cylinder/cone/torus axis direction
    loc: tuple | None = None     # axis location
    normal: tuple | None = None  # plane normal (geometric, orientation-corrected)


def surf_info(face) -> SurfInfo:
    a = BRepAdaptor_Surface(face, True)
    t = a.GetType()
    kind = SURF_TYPES.get(t, "other")
    s = SurfInfo(kind)
    rev = face.Orientation() == TopAbs_REVERSED
    if t == GeomAbs_Plane:
        pl = a.Plane()
        n = pl.Axis().Direction()
        s.normal = (-n.X(), -n.Y(), -n.Z()) if rev else (n.X(), n.Y(), n.Z())
        if not pl.Direct():
            s.normal = tuple(-v for v in s.normal)
    elif t == GeomAbs_Cylinder:
        c = a.Cylinder()
        s.radius = c.Radius()
        d, l = c.Axis().Direction(), c.Axis().Location()
        s.axis, s.loc = (d.X(), d.Y(), d.Z()), (l.X(), l.Y(), l.Z())
    elif t == GeomAbs_Cone:
        c = a.Cone()
        s.radius = c.RefRadius()
        d, l = c.Axis().Direction(), c.Axis().Location()
        s.axis, s.loc = (d.X(), d.Y(), d.Z()), (l.X(), l.Y(), l.Z())
    elif t == GeomAbs_Sphere:
        s.radius = a.Sphere().Radius()
    elif t == GeomAbs_Torus:
        tr = a.Torus()
        s.radius, s.major = tr.MinorRadius(), tr.MajorRadius()
        d = tr.Axis().Direction()
        s.axis = (d.X(), d.Y(), d.Z())
    return s


def face_point_normal(face, u=None, v=None):
    """Point and outward normal at (u,v) (default: mid-parameter)."""
    a = BRepAdaptor_Surface(face, True)
    if u is None:
        u = 0.5 * (a.FirstUParameter() + a.LastUParameter())
    if v is None:
        v = 0.5 * (a.FirstVParameter() + a.LastVParameter())
    props = BRepLProp_SLProps(a, u, v, 2, 1e-6)
    p = props.Value()
    if not props.IsNormalDefined():
        return (p.X(), p.Y(), p.Z()), (0.0, 0.0, 0.0)
    n = props.Normal()
    sgn = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    return (p.X(), p.Y(), p.Z()), (sgn * n.X(), sgn * n.Y(), sgn * n.Z())


def face_curvatures(face, n=3):
    """Mean and gaussian curvature sampled on an n x n UV grid (orientation-corrected)."""
    a = BRepAdaptor_Surface(face, True)
    u0, u1, v0, v1 = a.FirstUParameter(), a.LastUParameter(), a.FirstVParameter(), a.LastVParameter()
    sgn = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    Hs, Ks = [], []
    for i in range(n):
        for j in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            v = v0 + (v1 - v0) * (j + 0.5) / n
            try:
                pr = BRepLProp_SLProps(a, u, v, 2, 1e-6)
                if pr.IsCurvatureDefined():
                    Hs.append(sgn * pr.MeanCurvature())
                    Ks.append(pr.GaussianCurvature())
            except Exception:
                pass
    if not Hs:
        return 0.0, 0.0, 0.0
    return sum(Hs) / len(Hs), sum(Ks) / len(Ks), max(abs(h) for h in Hs)


def edge_mid(edge):
    c = BRepAdaptor_Curve(TopoDS.Edge_s(edge))
    t = 0.5 * (c.FirstParameter() + c.LastParameter())
    p, d = gp_Pnt(), gp_Vec()
    c.D1(t, p, d)
    return t, (p.X(), p.Y(), p.Z()), (d.X(), d.Y(), d.Z())


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def normal_at_point(face, pnt):
    """Outward normal of face at the projection of a 3D point."""
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    surf = BRep_Tool.Surface_s(face)
    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*pnt), surf)
    if proj.NbPoints() == 0:
        return face_point_normal(face)[1]
    u, v = proj.LowerDistanceParameters()
    a = BRepAdaptor_Surface(face, False)
    pr = BRepLProp_SLProps(a, u, v, 1, 1e-6)
    if not pr.IsNormalDefined():
        return (0.0, 0.0, 0.0)
    n = pr.Normal()
    sgn = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    return (sgn * n.X(), sgn * n.Y(), sgn * n.Z())
