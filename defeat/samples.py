"""Sample parts: an L-bracket and a printer-chassis style injection-molded part."""
from __future__ import annotations

import cadquery as cq


def l_bracket() -> cq.Shape:
    """L-bracket test part: big hole + slot (keep),
    corner holes, fillets, chamfers and raised 'CAE' text (remove)."""
    L, W, T, H = 80.0, 60.0, 6.0, 70.0
    base = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    wall = cq.Workplane("XY").box(T, W, H, centered=(False, True, False)).translate((-L / 2, 0, 0))
    part = base.union(wall)
    # gussets (ribs) between base and wall
    for y in (-W / 2 + 4, W / 2 - 4 - 4):
        g = (cq.Workplane("XZ", origin=(0, y + 4, 0))
             .polyline([(-L / 2 + T, T), (-L / 2 + T + 30, T), (-L / 2 + T, T + 42)]).close()
             .extrude(4))
        part = part.union(g)
    # functional features (keep): large hole in wall, slot in base
    part = part.cut(cq.Workplane("YZ", origin=(-L / 2 - 1, 0, 0)).center(0, 42).circle(14).extrude(T + 2))
    part = part.cut(cq.Workplane("XY", origin=(0, 0, -1)).center(15, 0).slot2D(36, 10, 90).extrude(T + 2))
    # small corner holes (remove)
    for x, y in [(30, -24), (30, 24), (6, -24), (6, 24)]:
        part = part.cut(cq.Workplane("XY", origin=(x, y, -1)).circle(2.0).extrude(T + 2))
    for y, z in [(-22, 62), (22, 62)]:
        part = part.cut(cq.Workplane("YZ", origin=(-L / 2 - 1, 0, 0)).center(y, z).circle(2.0).extrude(T + 2))
    # cosmetic edge rounds and chamfers (remove)
    part = part.edges("|Z and >X").fillet(4.0)
    part = part.edges("|X and >Z").fillet(2.0)
    part = part.faces(">X").edges("|Y").chamfer(1.0)
    # raised "CAE" logo on the base (remove)
    txt = cq.Workplane("XY", origin=(-12, 0, T)).text("CAE", 9, 0.8, kind="bold", halign="center", valign="center")
    part = part.union(txt.rotate((-12, 0, 0), (-12, 0, 1), 90))
    return part.val().clean()


def printer_chassis(seed: int = 7) -> cq.Shape:
    """Printer-chassis style injection-molded plate: rib grid, screw bosses,
    snap windows, small vents, fillets, mounting holes and a logo."""
    import random
    rnd = random.Random(seed)
    L, W, T = 180.0, 110.0, 3.0
    part = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
    # perimeter wall
    wall = (cq.Workplane("XY").rect(L, W).rect(L - 2 * T, W - 2 * T).extrude(18))
    part = part.union(wall)
    # rib grid (thin 1.6mm ribs)
    for x in (-60, -20, 20, 60):
        part = part.union(cq.Workplane("XY", origin=(x, 0, T)).box(1.6, W - 2 * T, 10, centered=(True, True, False)))
    for y in (-25, 25):
        part = part.union(cq.Workplane("XY", origin=(0, y, T)).box(L - 2 * T, 1.6, 10, centered=(True, True, False)))
    # screw bosses with pilot holes
    for x, y in [(-80, -45), (80, -45), (-80, 45), (80, 45), (-40, 0), (40, 0)]:
        part = part.union(cq.Workplane("XY", origin=(x, y, T)).circle(4.5).extrude(12))
        part = part.cut(cq.Workplane("XY", origin=(x, y, 4)).circle(1.4).extrude(20))
    # large functional opening (keep) and mounting holes (keep)
    part = part.cut(cq.Workplane("XY", origin=(0, 0, -1)).rect(30, 30).extrude(T + 2))
    for x, y in [(-70, -15), (70, 15)]:
        part = part.cut(cq.Workplane("XY", origin=(x, y, -1)).circle(3.5).extrude(T + 2))
    # vent slots (small)
    for i in range(6):
        part = part.cut(cq.Workplane("XY", origin=(-5 + 0, 0, -1)).center(-50 + 3.5 * i, 40).rect(1.5, 12).extrude(T + 2))
    # tiny cosmetic holes
    for _ in range(5):
        x, y = rnd.uniform(-85, 85), rnd.uniform(-50, -30)
        part = part.cut(cq.Workplane("XY", origin=(x, y, -1)).circle(0.8).extrude(T + 2))
    # outer corner rounds and rounded boss tops
    part = part.edges("|Z").edges(cq.selectors.BoxSelector((-L, -W, 5), (L, W, 17))).edges(">X or <X").fillet(1.5)
    # embossed logo on the bottom face
    logo = cq.Workplane("XY", origin=(55, -38, 0)).transformed(rotate=(180, 0, 0)).text("PC-ABS", 7, 0.6, kind="bold", halign="center", valign="center")
    part = part.union(logo)
    return part.val().clean()


SAMPLES = {
    "l_bracket": l_bracket,
    "printer_chassis": printer_chassis,
}
