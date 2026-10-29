"""Original showcase parts for the intro film, modelled in CadQuery (ours to use anywhere).

    python video/parts.py          # writes video/parts/<name>.step
"""
from __future__ import annotations

import math
import os

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))


def safe(p, op):
    """Apply a finishing op (fillet/chamfer); keep the part as-is if the kernel refuses that edge set."""
    try:
        q = op(p)
        return q if q.val().isValid() else p
    except Exception as e:
        print("   skipped a finishing op:", type(e).__name__)
        return p
OUT = os.path.join(HERE, "parts")


def mold_plate():
    """Injection-mold cavity plate: filleted cavity, guide-pin bores, ejector holes, cooling lines, engraving."""
    p = cq.Workplane("XY").box(140, 100, 32).edges("|Z").fillet(6).faces(">Z").edges().chamfer(1.2)
    p = p.faces(">Z").workplane().rect(78, 50).cutBlind(-14)
    p = safe(p, lambda q: q.edges(cq.selectors.BoxSelector((-40, -26, 1), (40, 26, 15.5))).fillet(4))
    p = p.faces(">Z").workplane().rect(118, 78, forConstruction=True).vertices().cboreHole(10, 15, 4)
    p = p.faces(">Z").workplane().pushPoints([(x, y) for x in (-28, -14, 0, 14, 28) for y in (-16, 16)]).hole(3, 30)
    for y in (-30, 30):
        p = p.cut(cq.Workplane("YZ").workplane(offset=-80).center(y, 4).circle(4).extrude(160))
    p = p.faces(">Z").workplane().center(0, -42).text("CAVITY A", 6, -0.8, kind="bold")
    return p


def gearbox_cover():
    """Gearbox cover: bearing boss, bolt flange, stiffening ribs, fillets."""
    base = cq.Workplane("XY").slot2D(150, 80).extrude(8)
    boss = cq.Workplane("XY").workplane(offset=8).center(-30, 0).circle(26).extrude(22)
    boss2 = cq.Workplane("XY").workplane(offset=8).center(35, 0).circle(18).extrude(16)
    p = base.union(boss).union(boss2)
    for a in (0, 60, 120, 180, 240, 300):
        r = cq.Workplane("XY").workplane(offset=8).center(-30, 0).transformed(rotate=(0, 0, a)).center(36, 0).rect(22, 3).extrude(10)
        p = p.union(r)
    p = p.faces(">Z").workplane().pushPoints([(-30, 0)]).hole(28, 40)
    p = p.faces(">Z").workplane().pushPoints([(35, 0)]).hole(16, 40)
    pts = [(-68, 0), (68, 0), (-40, 34), (-10, 36), (25, 34), (50, 30), (-40, -34), (-10, -36), (25, -34), (50, -30)]
    p = p.faces("<Z").workplane().pushPoints(pts).hole(5, 10)
    p = safe(p, lambda q: q.faces(">Z").edges().chamfer(1.2))
    p = safe(p, lambda q: q.edges(cq.selectors.BoxSelector((-200, -200, 7.5), (200, 200, 8.5))).fillet(1.5))
    return p


def flanged_hub():
    """Flanged hub: bolt circle, keyway, chamfers, fillet at the flange root."""
    p = cq.Workplane("XY").circle(55).extrude(12).faces(">Z").workplane().circle(26).extrude(40)
    p = safe(p, lambda q: q.edges(cq.selectors.NearestToPointSelector((26, 0, 12))).fillet(5))
    p = p.faces(">Z").workplane().hole(24)
    p = p.faces(">Z").workplane().center(0, 12).rect(8, 6).cutThruAll()
    p = p.faces("<Z").workplane().polarArray(42, 0, 360, 8).hole(7, 12)
    p = safe(p, lambda q: q.faces(">Z").edges("%CIRCLE").chamfer(1))
    return p


def heat_sink():
    """Heat sink: base plate with fins, mounting holes, rounded corners, logo."""
    p = cq.Workplane("XY").box(110, 80, 8).edges("|Z").fillet(5)
    for i in range(11):
        x = -45 + i * 9
        p = p.union(cq.Workplane("XY").workplane(offset=4).center(x, 0).rect(2.4, 64).extrude(30))
    p = p.faces("<Z").workplane().rect(96, 66, forConstruction=True).vertices().hole(4.2, 8)
    p = p.faces("<Z").workplane().center(0, 0).text("THERMAL", 7, -0.6, kind="bold")
    return p


def robot_link():
    """Robot arm link: two bearing bores, lightening pockets, fillets and chamfers."""
    p = cq.Workplane("XY").slot2D(200, 60).extrude(18)
    p = p.faces(">Z").workplane().pushPoints([(-70, 0), (70, 0)]).hole(30)
    p = p.faces(">Z").workplane().pushPoints([(-28, 0), (28, 0)]).slot2D(36, 26).cutBlind(-10)
    p = p.faces(">Z").workplane().pushPoints([(-70, 22), (-70, -22), (70, 22), (70, -22), (-92, 0), (92, 0)]).hole(4, 18)
    p = safe(p, lambda q: q.faces(">Z").edges("%CIRCLE").chamfer(0.8))
    p = safe(p, lambda q: q.faces(">Z").edges("not %CIRCLE").fillet(1.5))
    return p


PARTS = {"mold_plate": mold_plate, "gearbox_cover": gearbox_cover, "flanged_hub": flanged_hub, "heat_sink": heat_sink, "robot_link": robot_link}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in PARTS.items():
        try:
            shape = fn()
            path = os.path.join(OUT, f"{name}.step")
            cq.exporters.export(shape, path)
            s = shape.val()
            print(f"{name}: valid={s.isValid()} faces={len(s.Faces())} -> {path}")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
