"""End-to-end checks on the bundled sample parts (run: python -m pytest -q)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRACKET = os.path.join(ROOT, "samples", "l_bracket.step")
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def bracket():
    from defeat import occ
    from defeat.recognize import recognize
    shape = occ.read_cad(BRACKET)
    g, feats = recognize(shape)
    return shape, g, feats


def test_recognizes_l_bracket(bracket):
    _, g, feats = bracket
    kinds = [f.kind for f in feats]
    assert len(g.nodes) == 84
    assert kinds.count("hole") == 7
    assert kinds.count("emboss") == 3          # C, A, E
    assert kinds.count("rib") == 2             # gussets
    assert "slot" in kinds
    assert any(f.kind == "hole" and abs(f.size - 28) < 1e-6 for f in feats)


def test_wall_between_two_rounds_is_not_a_fillet():
    """A flange wall whose top and bottom rims are both rounded touches two blends smoothly,
    but only the rounds are fillets; the wall itself is body and must never be removed."""
    import cadquery as cq
    from defeat.recognize import recognize
    p = cq.Workplane("XY").circle(55).extrude(12).faces(">Z").workplane().circle(26).extrude(40)
    p = p.newObject([e for e in p.edges("%CIRCLE").vals() if abs(e.radius() - 55) < 1e-6]).fillet(2)
    g, feats = recognize(p.val().wrapped, use_gnn=False)
    fillets = [f for f in feats if f.kind == "fillet"]
    assert len(fillets) == 2
    assert all(g.nodes[x].kind == "torus" for f in fillets for x in f.faces)


def test_decisions_keep_functional_features(bracket):
    from defeat.decide import Policy, decide
    _, g, feats = bracket
    d = decide(feats, Policy("static", 4.0))
    big = next(f for f in feats if f.kind == "hole" and f.size > 20)
    assert d[big.id].action == "keep"
    assert all(d[f.id].action == "remove" for f in feats if f.kind in ("emboss", "fillet", "chamfer"))
    assert all(d[f.id].action == "keep" for f in feats if f.kind in ("rib", "slot"))


def test_defeature_and_stress_check(bracket):
    from defeat import occ
    from defeat.decide import Policy, decide
    from defeat.execute import defeature, worker
    from defeat.validate import geometry_report, mesh_compare
    shape, g, feats = bracket
    d = decide(feats, Policy("static", 4.0))
    r = defeature(shape, feats, d)
    try:
        assert occ.is_valid(r.shape)
        assert len(occ.faces_of(r.shape)) == 31
        geo = geometry_report(shape, r.shape)
        assert abs(geo["volume_delta_pct"]) < 2.0
        m = mesh_compare(shape, r.shape, 4.0)
        assert m["tet_reduction_pct"] > 40
        assert abs(m["fea"]["delta"]["tip_disp_delta_pct"]) < 10
    finally:
        worker().close()


def test_exported_python_reproduces_result(tmp_path):
    from defeat.execute import worker
    from defeat.session import new_session
    s = new_session(BRACKET)
    try:
        s.run(with_mesh=False)
        v = s.verify_python_export()
        assert v["ok"], v
        assert v["volume_diff_pct"] < 1e-6
    finally:
        worker().close()


def test_offline_intent_parser():
    from defeat.agent import parse_offline
    calls = parse_offline("Remove the logo and fillets under 2 mm but keep the holes, then run it")
    names = [c["name"] for c in calls]
    assert "run_defeaturing" in names
    rules = {c["arguments"]["feature_class"]: c["arguments"] for c in calls if c["name"] == "set_class_rule"}
    assert rules["emboss"]["action"] == "remove" and rules["hole"]["action"] == "keep"
    assert rules["fillet"]["max_size_mm"] == 2.0


def test_native_cad_formats_get_a_clear_message(tmp_path):
    from defeat import occ
    p = tmp_path / "part.prt"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="nx/export_step.py"):
        occ.read_cad(str(p))


def test_stress_check_uses_the_same_regions_for_both_models():
    import numpy as np
    from defeat import fea
    # a 10 x 10 x 100 bar: deflection must match beam theory, and locking the end regions must hold
    pts = []
    for z in range(0, 101, 10):
        for y in (0, 5, 10):
            for x in (0, 5, 10):
                pts.append((x, y, z))
    p = np.array(pts, float)
    idx = lambda i, j, k: k * 9 + j * 3 + i
    t = []
    for k in range(10):
        for j in range(2):
            for i in range(2):
                c = [idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                     idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
                t += [[c[0], c[1], c[3], c[4]], [c[1], c[2], c[3], c[6]], [c[1], c[3], c[4], c[6]],
                      [c[1], c[4], c[5], c[6]], [c[3], c[4], c[6], c[7]]]
    bc = fea.load_case((0, 0, 0, 10, 10, 100))
    a = fea.solve(p, t, bc)
    assert a["ok"] and len(a["ends"]) == 2
    a["h_fea"] = 5.0
    fea.lock(bc, a)
    b = fea.solve(p, t, bc)
    assert b["ends"] == a["ends"]
    assert b["tip_disp"] == pytest.approx(a["tip_disp"])
    # free length about 90 of 100: F L^3 / 3 E I, within a factor that covers the end regions
    L, I = 90.0, 10 ** 4 / 12
    beam = fea.FORCE_N * L ** 3 / (3 * fea.MATERIAL["E"] * I)
    assert 0.5 * beam < a["tip_disp"] < 1.5 * beam


def test_volume_budget_keeps_the_largest_features(bracket):
    from defeat.decide import Policy, decide, est_volume
    from defeat import occ
    shape, g, feats = bracket
    free = decide(feats, Policy("static", 4.0))
    vol = occ.volume(shape)
    d = decide(feats, Policy("static", 4.0), g, vol)
    moved = sum(est_volume(f, g) for f in feats if d[f.id].action == "remove")
    assert moved <= 0.01 * vol
    removed = [f for f in feats if d[f.id].action == "remove"]
    kept_by_budget = [f for f in feats if free[f.id].action == "remove" and d[f.id].action == "keep"]
    if removed and kept_by_budget:
        assert max(est_volume(f, g) for f in removed) <= min(est_volume(f, g) for f in kept_by_budget)


def test_removal_that_rebuilds_too_much_material_is_rejected(bracket, monkeypatch):
    from defeat import execute, occ
    from defeat.execute import _remove_faces
    monkeypatch.setattr(execute, "TINY_VOLUME", 0.0)
    monkeypatch.setattr(execute, "TINY_GROWTH", 0.0)
    shape, g, feats = bracket
    f = next(x for x in feats if x.kind == "fillet")
    brep = occ.to_brep_string(shape)
    assert _remove_faces(brep, f.faces)["ok"]
    r = _remove_faces(brep, f.faces, 0.0, (0.0, 0.0))
    assert not r["ok"] and "material" in r["error"]
