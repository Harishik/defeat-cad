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
