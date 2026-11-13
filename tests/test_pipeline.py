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
    g, feats = recognize(p.val().wrapped, use_net=False)
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
        # 25, not the 31 of earlier releases: those left 6 blend faces in place while reporting them removed
        assert len(occ.faces_of(r.shape)) == 25
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
    d = decide(feats, Policy("static", 4.0, verified=False), g, vol)
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
    # over the estimate but provably local: accepted
    assert _remove_faces(brep, f.faces, 0.0, (0.0, 0.0))["ok"]
    # over the estimate and not local: rejected
    monkeypatch.setattr(execute, "_change_is_local", lambda *a: False)
    r = _remove_faces(brep, f.faces, 0.0, (0.0, 0.0))
    assert not r["ok"] and "material" in r["error"]


def test_change_is_local_sees_material_far_from_the_removed_faces(bracket):
    from defeat import execute, occ
    from defeat.execute import _change_is_local, _remove_faces
    shape, g, feats = bracket
    f = next(x for x in feats if x.kind == "fillet")
    new = occ.from_brep_string(_remove_faces(occ.to_brep_string(shape), f.faces)["brep"])
    diag = g.char_len
    assert _change_is_local(shape, new, [g.faces[i] for i in f.faces], 0.01 * diag, diag)
    fc = occ.bbox(g.faces[f.faces[0]])
    far = max((x for x in g.faces if occ.area(x) > 0), key=lambda x: sum((a - b) ** 2 for a, b in zip(occ.bbox(x), fc)))
    assert not _change_is_local(shape, new, [far], 0.0, diag)


def test_every_text_to_cad_example_builds_what_it_says(tmp_path):
    from defeat import agent, occ
    for keys, desc, code in agent.EXAMPLES:
        out = str(tmp_path / "x.step")
        ok, err = agent.run_cadquery("import cadquery as cq\n" + code, out)
        assert ok, f"{desc}: {err}"
        assert occ.is_valid(occ.read_cad(out)), desc


def test_nx_journal_calls_the_expected_api_against_a_stub(tmp_path, monkeypatch):
    """No NX here, so this checks the journal's own logic only, against a stand-in for NXOpen."""
    import runpy
    import sys
    import types
    from unittest import mock
    calls = []

    class Creator:
        ObjectTypes = types.SimpleNamespace(Solids=False)

        def Commit(self):
            calls.append(("commit", self.ExportAs, self.InputFile, self.ObjectTypes.Solids, self.ProcessHoldFlag, self.LayerMask))
            if "broken" not in self.OutputFile:
                open(self.OutputFile, "w").write("ISO-10303-21;")

        def Destroy(self):
            calls.append(("destroy",))

    nx = mock.MagicMock()
    nx.StepCreator.ExportAsOption.Ap214 = "ap214"
    session = nx.Session.GetSession.return_value
    session.DexManager.CreateStepCreator.side_effect = Creator

    def opened(path):
        return types.SimpleNamespace(FullPath=path, Close=lambda *a: calls.append(("close", path))), mock.MagicMock()
    session.Parts.OpenBaseDisplay.side_effect = opened
    monkeypatch.setitem(sys.modules, "NXOpen", nx)
    good, broken = str(tmp_path / "a.prt"), str(tmp_path / "broken.prt")
    journal = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nx", "export_step.py")

    monkeypatch.setattr(sys, "argv", ["export_step.py", good])
    runpy.run_path(journal, run_name="__main__")
    assert os.path.isfile(str(tmp_path / "a.step"))
    assert calls == [("commit", "ap214", good, True, True, "1-256"), ("destroy",), ("close", good)]

    monkeypatch.setattr(sys, "argv", ["export_step.py", broken])
    with pytest.raises(RuntimeError, match="did not write"):
        runpy.run_path(journal, run_name="__main__")


def test_guard_judges_the_stress_peak_as_well_as_stiffness():
    from defeat import fea
    stiff = {"cases_delta_pct": {"bend_thin": 2.0, "tension": -1.0}}
    assert fea.judged({**stiff, "vm_peak_delta_pct": 4.0}) == 2.0            # both fine: the stiffness number leads
    assert fea.judged({**stiff, "vm_peak_delta_pct": -12.0}) == 6.0          # a peak past 10% fails the 5% check
    assert fea.judged(stiff) == 2.0
    msg = fea.guard_message({"triggered": True, "accepted": True, "before_pct": 6.0, "peak_before_pct": 12.0, "after_pct": 2.0,
                             "peak_after_pct": 4.0, "restored": 1, "of": 3, "within_tol": True})
    assert "stress peak by 12.0%" in msg and "brought it to 4.0%" in msg


def test_a_group_the_kernel_refuses_is_split_so_good_features_still_go(monkeypatch):
    """Bisection: one feature the kernel won't touch must not take the others down with it."""
    from defeat import execute, occ
    from defeat.session import Session
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = Session(os.path.join(root, "samples", "l_bracket.step"))
    todo = [f for f in s.features if s.decisions[f.id].action == "remove"]
    bad = todo[0]
    bad_faces = list(bad.faces)
    real = execute._remove_faces
    faces0 = occ.faces_of(s.shape)
    bad_sig = [occ.centroid(faces0[i]) for i in bad_faces]

    def refuse(brep, ids, *a, **k):
        faces = occ.faces_of(occ.from_brep_string(brep))
        if any(occ.norm(occ.sub(occ.centroid(faces[i]), c)) < 1e-6 for i in ids for c in bad_sig):
            return {"ok": False, "error": "kernel returned the part unchanged"}
        return real(brep, ids, *a, **k)

    class Local:
        def call(self, fn, *a, timeout=30.0):
            return refuse(*a) if fn == "_remove_faces" else getattr(execute, fn)(*a)
    monkeypatch.setattr(execute, "worker", lambda: Local())
    er = execute.defeature(s.shape, s.features, s.decisions, graph=s.graph)
    assert bad.id in er.failed
    assert len(er.removed) >= len(todo) - 3   # blend neighbours of the refused feature may stay with it
    assert occ.is_valid(er.shape)


def test_a_part_too_heavy_to_check_keeps_only_small_removals(monkeypatch):
    from defeat import session as sm
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = sm.Session(os.path.join(root, "samples", "l_bracket.step"))
    monkeypatch.setattr(sm, "mesh_compare", lambda *a, **k: {"h": 4, "original": {"ok": False, "error": "mesh timed out after 60s"}, "defeatured": {"ok": False}})
    monkeypatch.setattr(sm, "grid_compare", lambda *a, **k: None)   # the fixed-grid fallback isn't available either
    s.run(with_mesh=True)
    assert s.metrics["unchecked"] is True
    assert "too heavy" in s.metrics["guard_message"]
    assert abs(s.metrics["geometry"]["volume_delta_pct"]) <= 1.0 + 1e-6
    # the fallback was never meshed: no mesh change may be reported from the abandoned removal
    assert s.metrics["mesh"]["tet_reduction_pct"] is None


def test_an_empty_fallback_reports_the_original_mesh_not_the_abandoned_one(monkeypatch):
    from defeat import session as sm
    from defeat.execute import ExecResult
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = sm.Session(os.path.join(root, "samples", "l_bracket.step"))
    orig = {"ok": True, "tets": 1000, "seconds": 1.0}
    abandoned = {"ok": True, "tets": 400, "seconds": 0.4}
    monkeypatch.setattr(sm, "mesh_compare", lambda *a, **k: {"h": 4, "original": orig, "defeatured": abandoned, "tet_reduction_pct": 60.0,
                                                           "speedup": 2.5, "fea": {"original": {"ok": False, "error": "solver failed"}}})
    real = sm.defeature
    calls = []
    def second_removes_nothing(shape, *a, **k):
        calls.append(1)
        return real(shape, *a, **k) if len(calls) == 1 else ExecResult(shape)
    monkeypatch.setattr(sm, "defeature", second_removes_nothing)
    s.run(with_mesh=True)
    m = s.metrics["mesh"]
    assert len(calls) == 2 and not s.metrics["removed"]
    assert m["defeatured"] is orig and m["tet_reduction_pct"] == 0.0 and m["speedup"] == 1.0
    assert s.metrics["unchecked"] is False


def test_parser_handles_lists_pronouns_and_numbered_features():
    from defeat.agent import parse_offline
    def rules(t):
        return {(c["arguments"].get("feature_class"), c["arguments"].get("action")) for c in parse_offline(t) if c["name"] == "set_class_rule"}
    assert rules("Delete the fillets, the chamfers and the logo") == {("fillet", "remove"), ("chamfer", "remove"), ("emboss", "remove")}
    assert rules("Ribs carry load, keep them") == {("rib", "keep")}
    assert rules("No fillets please") == {("fillet", "remove")}
    ids = [c for c in parse_offline("Remove #2, #5 and #8") if c["name"] == "set_features"]
    assert ids[0]["arguments"]["feature_ids"] == [2, 5, 8]
    assert [c["name"] for c in parse_offline("Extract the flow volume")] == ["make_fluid_domain"]


def test_planner_falls_back_to_the_parser_when_the_model_gives_nothing(monkeypatch):
    from defeat import agent
    def down(*a, **k):
        raise TimeoutError
    monkeypatch.setattr(agent, "llm_chat", down)
    calls, _, via = agent.plan_calls([], "Remove the logo")
    assert via == "keyword parser" and calls[0]["arguments"]["feature_class"] == "emboss"


def test_a_normal_part_is_stress_checked_not_called_too_heavy():
    from defeat import session as sm
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = sm.Session(os.path.join(root, "samples", "l_bracket.step"))
    s.run(with_mesh=True)
    assert s.metrics["unchecked"] is False
    assert "too heavy" not in (s.metrics["guard_message"] or "")
    assert s.metrics["fea"]["delta"]["cases_delta_pct"]


def test_features_touching_a_chosen_support_face_are_kept():
    from defeat import session as sm
    from defeat.decide import Policy, decide
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = sm.Session(os.path.join(root, "samples", "l_bracket.step"))
    hole = next(f for f in s.features if f.kind == "hole" and s.decisions[f.id].action == "remove")
    face = next(iter(s.graph.nodes[hole.faces[0]].neighbors))
    assert face not in hole.faces
    d = decide(s.features, Policy("static", s.policy.h, boundary={"fixed": [face]}), s.graph, s.part_volume)
    assert d[hole.id].action == "keep"


def _sleep_forever():
    import time
    time.sleep(600)


def test_a_hung_kernel_call_is_killed_and_the_worker_recovers():
    """The promise behind "any part": a call that never returns costs its timeout and nothing more."""
    import time
    from defeat.execute import Worker
    w = Worker()
    try:
        t = time.time()
        r = w.call("tests.test_pipeline:_sleep_forever", timeout=3.0)
        assert not r["ok"] and "timed out" in r["error"]
        assert time.time() - t < 30
        assert w.call("_unify", _box_brep(), timeout=60.0)["ok"]   # the next call gets a fresh worker
    finally:
        w.close()


def _box_brep():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from defeat import occ
    return occ.to_brep_string(BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape())


def test_growth_far_from_the_feature_is_rejected_without_booleans():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from defeat import occ
    from defeat.execute import _change_is_local
    a = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    b = BRepPrimAPI_MakeBox(10.0, 10.0, 30.0).Shape()   # grew 20 upward
    corner = [f for f in occ.faces_of(a) if occ.bbox(f)[5] < 1e-6]   # the bottom face, far from the growth
    assert _change_is_local(a, b, corner, 1.0, 17.0) is False
    top = [f for f in occ.faces_of(a) if occ.bbox(f)[2] > 9.9]
    assert _change_is_local(a, b, top, 25.0, 17.0) is True   # within reach: still judged by the real check


def _sample_session():
    from defeat import session as sm
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return sm, sm.Session(os.path.join(root, "samples", "l_bracket.step"))


def test_no_comparison_is_never_reported_as_checked(monkeypatch):
    """The simplified model could not be solved: the part must say so, not pass silently."""
    sm, s = _sample_session()
    no_delta = {"h": 1, "original": {"ok": True}, "defeatured": {"ok": True}, "stat_curvature": 12,
                "fea": {"original": {"ok": True}, "defeatured": {"ok": False, "error": "could not find support / load regions"}, "delta": None}}
    monkeypatch.setattr(sm, "mesh_compare", lambda *a, **k: no_delta)
    monkeypatch.setattr(sm, "grid_compare", lambda *a, **k: None)
    s.run(with_mesh=True)
    assert s.metrics["unchecked"] is True
    assert "supports" in s.metrics["guard_message"] and "too heavy" not in s.metrics["guard_message"]


def test_the_smaller_removal_is_checked_when_the_first_comparison_fails(monkeypatch):
    sm, s = _sample_session()
    real = sm.mesh_compare
    calls = []

    def first_fails(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return {"h": 1, "original": {"ok": True}, "defeatured": {"ok": True}, "stat_curvature": 12,
                    "fea": {"original": {"ok": True}, "defeatured": {"ok": False}, "delta": None}}
        return real(*a, **k)
    monkeypatch.setattr(sm, "mesh_compare", first_fails)
    monkeypatch.setattr(sm, "grid_compare", lambda *a, **k: None)
    s.run(with_mesh=True)
    assert len(calls) >= 2 and s.metrics["unchecked"] is False
    assert s.metrics["fea"]["delta"]["cases_delta_pct"]


def test_a_part_that_cannot_be_meshed_says_that_not_too_heavy(monkeypatch):
    sm, s = _sample_session()
    bad = {"h": 1, "original": {"ok": False, "error": "the mesher produced no volume elements"},
           "defeatured": {"ok": False, "error": "the original could not be meshed"}, "stat_curvature": 12}
    monkeypatch.setattr(sm, "mesh_compare", lambda *a, **k: bad)
    monkeypatch.setattr(sm, "grid_compare", lambda *a, **k: None)
    s.run(with_mesh=True)
    assert s.metrics["unchecked"] is True and "mesher" in s.metrics["guard_message"]


def test_when_curvature_sizing_hangs_the_mesh_retries_without_it(monkeypatch):
    from defeat import execute, occ, validate
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shape = occ.read_cad(os.path.join(root, "samples", "l_bracket.step"))
    real = execute.Worker.call
    seen = []

    def call(self, fn, *a, timeout=30.0):
        if fn == "defeat.fea:mesh_and_solve":
            seen.append(a[2]["stat_curvature"])
            if a[2]["stat_curvature"]:
                return {"ok": False, "error": "kernel timed out after 1s"}
        return real(self, fn, *a, timeout=timeout)
    monkeypatch.setattr(execute.Worker, "call", call)
    out = validate.mesh_compare(shape, shape, 3.0, budget=200.0)
    assert seen[0] == 12 and 0 in seen
    assert out["stat_curvature"] == 0 and out["fea"]["delta"]["cases_delta_pct"]


def test_multigrid_and_direct_solvers_agree():
    """Above DIRECT_DOFS the stress check switches to multigrid-preconditioned CG: same answers, linear memory."""
    from defeat import fea, occ
    from defeat.validate import mesh_stats, step_bytes
    shape = occ.read_cad(BRACKET)
    st = mesh_stats(step_bytes(shape), 6.0, keep_mesh=True)
    bc = fea.load_case(occ.bbox(shape))
    old = fea.DIRECT_DOFS
    try:
        fea.DIRECT_DOFS = 10 ** 9
        a = fea.solve(st["mesh"]["p"], st["mesh"]["t"], bc)
        fea.DIRECT_DOFS = 0
        b = fea.solve(st["mesh"]["p"], st["mesh"]["t"], bc)
    finally:
        fea.DIRECT_DOFS = old
    assert a["solver"] == "direct" and b["solver"] == "multigrid"
    for k in a["cases"]:
        assert abs(a["cases"][k] - b["cases"][k]) <= 1e-6 * abs(a["cases"][k])


def test_the_triangulation_route_meshes_a_part():
    """The fallback mesher works from OpenCASCADE's watertight triangulation, not the CAD surfaces."""
    from defeat import occ
    from defeat.validate import mesh_stats, step_bytes
    r = mesh_stats(step_bytes(occ.read_cad(BRACKET)), 4.0, route="stl")
    assert r["ok"] and r["route"] == "stl" and r["tets"] > 1000


def test_mesh_rounds_try_the_triangulation_route_after_the_cad_route():
    from defeat.validate import _mesh_rounds
    routes = [r[2] for r in _mesh_rounds(2.0, 100.0)]
    assert routes[0] == "cad" and routes[1] == "stl"


def test_negation_and_fill_words_are_read_right():
    from defeat import agent
    assert agent.polarity("fill the pockets") == "remove"
    assert agent.polarity("don't fill the pockets") == "keep"
    assert agent.polarity("no need to keep the fillets") == "remove"
    assert agent.polarity("never remove the bosses") == "keep"
    assert agent.stated_actions("Fill the pockets and keep the holes") == {"pocket": "remove", "hole": "keep"}
    assert agent.is_question("How many holes are there?") and not agent.is_question("Can you remove the logo?")


def test_verification_overrides_a_flipped_action_and_drops_unmentioned_classes():
    from defeat import agent
    llm = [{"name": "set_class_rule", "arguments": {"feature_class": "pocket", "action": "keep"}},
           {"name": "set_class_rule", "arguments": {"feature_class": "fillet", "action": "remove"}}]
    out = agent.verify_calls(llm, "Fill the pockets")
    assert out == [{"name": "set_class_rule", "arguments": {"feature_class": "pocket", "action": "remove"}}]
    assert agent.verify_calls(llm, "What features does this part have?") == []


def test_consensus_picks_the_candidate_the_pool_agrees_with():
    from defeat import agent
    c = [{"m": {"dims": [10, 20, 30], "volume": 6000}}, {"m": {"dims": [10, 20, 31], "volume": 6100}},
         {"m": {"dims": [40, 80, 90], "volume": 90000}}]
    assert agent._consensus(c) in (0, 1)


def test_upload_limit_and_session_cap(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from defeat import server, session
    c = TestClient(server.app)
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0.001)
    r = c.post("/api/upload", files={"file": ("big.step", b"x" * 5000)})
    assert r.status_code == 413
    monkeypatch.setattr(session, "MAX_SESSIONS", 2)
    for _ in range(3):
        session.new_session(BRACKET)
    assert len(session.SESSIONS) <= 2


def test_grid_assembly_matches_scikit_fem():
    import numpy as np
    from skfem import Basis, BilinearForm, ElementHex2, ElementVector, MeshHex, asm
    from skfem.helpers import ddot, sym_grad, trace
    from skfem.quadrature import get_quadrature
    from skfem.refdom import RefHex
    from defeat import immersed
    g = 2.0
    xs = np.arange(4) * g
    m = MeshHex.init_tensor(xs, xs[:3], xs[:3])
    e = ElementVector(ElementHex2())
    q = get_quadrature(RefHex, immersed.INTORDER)
    ib = Basis(m, e, quadrature=q)
    phi = np.random.default_rng(1).uniform(0.1, 1, (m.t.shape[1], len(q[1])))

    @BilinearForm
    def stiff(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return w.phi * (2 * 1.0 * ddot(eu, ev) + 0.5 * trace(eu) * trace(ev))
    K1 = asm(stiff, ib, phi=phi)
    K2 = immersed._assemble(immersed._element_matrices(g, e, q, 0.5, 1.0), phi, ib.element_dofs, ib.N)
    assert abs(K1 - K2).max() < 1e-10 * abs(K1).max()


def test_moment_fitted_weights_integrate_the_cut_exactly():
    import numpy as np
    from skfem.quadrature import get_quadrature
    from skfem.refdom import RefHex
    from defeat import immersed
    q, fine = get_quadrature(RefHex, immersed.INTORDER), immersed._composite()
    chi = (fine[0][0] + 0.3 * fine[0][1] < 0.52).astype(float)[None]
    w = immersed.moment_fitted_weights(chi, fine, q)
    assert abs(w.sum() - (chi * fine[1]).sum()) < 1e-9
    assert abs((w * q[0][0]).sum() - (chi * fine[1] * fine[0][0]).sum()) < 1e-9
    assert (w >= 0).all()


def test_grid_check_matches_tetrahedra_on_a_bar_with_a_hole():
    """The grid and the tetrahedral check agree on a plain bar and on what drilling a hole costs it."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from defeat import fea, immersed, occ
    from defeat.validate import mesh_stats, step_bytes
    bar = BRepPrimAPI_MakeBox(120.0, 20.0, 10.0).Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(60, 10, -1), gp_Dir(0, 0, 1)), 5.0, 12.0).Shape()
    holed = BRepAlgoAPI_Cut(bar, hole).Shape()
    bc = fea.load_case(occ.bbox(bar))
    bc["grid"] = immersed.grid_for(bar, target=500)
    ga, gb = immersed.compare(holed, bar, bc)          # filling the hole: the "defeatured" part is the plain bar
    ma = mesh_stats(step_bytes(holed), 4.0, keep_mesh=True)
    ta = fea.solve(ma["mesh"]["p"], ma["mesh"]["t"], dict(fea.load_case(occ.bbox(bar)), ends=ga["ends"], load_center=ga["load_center"]))
    mb = mesh_stats(step_bytes(bar), 4.0, keep_mesh=True)
    tb = fea.solve(mb["mesh"]["p"], mb["mesh"]["t"], dict(fea.load_case(occ.bbox(bar)), ends=ga["ends"], load_center=ga["load_center"]))
    assert abs(ga["volume"] / occ.volume(holed) - 1) < 0.02
    for k in ("bend_thin", "tension"):
        assert abs(ga["cases"][k] / ta["cases"][k] - 1) < 0.08
        grid_change = 100 * (gb["cases"][k] - ga["cases"][k]) / ga["cases"][k]
        tet_change = 100 * (tb["cases"][k] - ta["cases"][k]) / ta["cases"][k]
        assert grid_change < 0 and abs(grid_change - tet_change) < 1.5


def test_an_unmeshable_part_is_checked_on_the_grid(monkeypatch):
    sm, s = _sample_session()
    bad = {"h": 1, "original": {"ok": False, "error": "the mesher produced no volume elements"},
           "defeatured": {"ok": False, "error": "the original could not be meshed"}, "stat_curvature": 12}
    grid = {"method": "grid", "load_case": "grid", "material": {}, "original": {"ok": True, "cases": {"a": 1.0}, "tip_disp": 1.0},
            "defeatured": {"ok": True, "cases": {"a": 0.99}, "tip_disp": 0.99},
            "delta": {"cases_delta_pct": {"a": -1.0}, "tip_disp_delta_pct": -1.0}}
    monkeypatch.setattr(sm, "mesh_compare", lambda *a, **k: bad)
    monkeypatch.setattr(sm, "grid_compare", lambda *a, **k: grid)
    s.run(with_mesh=True)
    assert s.metrics["unchecked"] is False and s.metrics["fea"]["method"] == "grid"
    assert "fixed grid" in s.metrics["guard_message"] and s.metrics["removed"]


def test_one_solve_bounds_bracket_the_change_from_filling_a_hole():
    """Plate in tension with a small hole (plane stress): filling the hole changes compliance by
    3 pi a^2 / (W L); the strain energy in the filled material is exactly a third of that and is a lower
    bound; the Antolin-Chanon estimator covers the rest. All from one solve of the simplified plate."""
    import numpy as np
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from defeat import estimator, fea, occ
    from defeat.recognize import Feature
    from defeat.validate import mesh_stats, step_bytes
    L, W, T, a, h = 300.0, 100.0, 2.0, 8.0, 3.5
    plate = BRepPrimAPI_MakeBox(L, W, T).Shape()
    holed = BRepAlgoAPI_Cut(plate, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(L / 2, W / 2, -1), gp_Dir(0, 0, 1)), a, T + 2).Shape()).Shape()
    walls = [i for i, f in enumerate(occ.faces_of(holed)) if occ.surf_info(f).kind == "cylinder"]
    bc = fea.load_case(occ.bbox(plate))

    def solve(shape, bc):
        m = mesh_stats(step_bytes(shape), h, keep_mesh=True)
        return fea.solve(m["mesh"]["p"], m["mesh"]["t"], dict(bc, want_system=True))
    rs = solve(plate, bc)
    rh = solve(holed, dict(bc, ends=rs["ends"], load_center=rs["load_center"]))
    s = rs["system"]
    lbc = dict(bc, ends=rs["ends"], load_center=rs["load_center"])
    res = estimator.bounds(s["p"], s["t"], s["U"], s["forms"], holed, plate, [Feature(0, "hole", walls)], {0}, bc=lbc)
    assert res["kinds"] == {0: "negative"} and all(abs(v) < 1e-12 for v in res["load_shift"].values())
    # a hole inside the loaded end changes the load itself (it is spread over less material); the
    # load-shift term keeps the bound valid there too
    end_hole = BRepAlgoAPI_Cut(plate, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(L - 6, W / 2, -1), gp_Dir(0, 0, 1)), 4.0, T + 2).Shape()).Shape()
    ew = [i for i, f in enumerate(occ.faces_of(end_hole)) if occ.surf_info(f).kind == "cylinder"]
    re_ = estimator.bounds(s["p"], s["t"], s["U"], s["forms"], end_hole, plate, [Feature(0, "hole", ew)], {0}, bc=lbc)
    # check the term against the original's own load vector applied to u0 carried onto its mesh
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet
    so = solve(end_hole, lbc)["system"]
    mo = MeshTet(np.asarray(so["p"], float).T, np.asarray(so["t"], int).T)
    bo = Basis(mo, ElementVector(ElementTetP2()))
    ms_ = MeshTet(np.asarray(s["p"], float).T, np.asarray(s["t"], int).T)
    bs = Basis(ms_, ElementVector(ElementTetP2()))
    sb, comp = bs.split_bases()[0], estimator._component_dofs(bs)
    P = estimator._probes(sb, bo.doflocs.T)
    which = np.empty(bo.N, int)
    for c in range(3):
        which[bo.nodal_dofs[c]] = c
        which[bo.edge_dofs[c]] = c
    for k in ("bend_thin", "bend_wide", "tension", "torsion"):
        vals = np.stack([P @ s["U"][k][comp[c]] for c in range(3)])
        u0_on_orig = vals[which, np.arange(bo.N)]
        direct = 2 * (float(so["forms"][k] @ u0_on_orig) - res["J0"][k])
        assert abs(re_["load_shift"][k] - direct) <= 0.15 * abs(direct) + 5e-4 * res["J0"][k], k   # 0.05% of J0: the meshes cut the loaded region differently
    summ = estimator.summarize(res)
    for k in rs["cases"]:
        measured = 100 * (rh["cases"][k] - rs["cases"][k]) / rs["cases"][k]
        # The bound is exact for exact solutions; the two-mesh referee itself moves by tenths of a point
        # between meshes (gmsh meshes this plate differently on Apple Silicon), so allow for that here and
        # keep the tight check for tension, which has a closed-form answer below
        noise = 0.02 if k == "tension" else max(0.02, 0.25 * abs(measured))
        assert summ[k]["lower_pct"] <= measured + noise, k
        assert summ[k]["lower_pct"] + summ[k]["est_pct"] >= measured
    theory = 100 * np.pi * a * a / (W * L)
    assert abs(summ["tension"]["lower_pct"] / theory - 1) < 0.2
    measured_t = 100 * (rh["cases"]["tension"] - rs["cases"]["tension"]) / rs["cases"]["tension"]
    assert 2.4 < measured_t / summ["tension"]["lower_pct"] < 3.6


def test_every_feature_reported_removed_is_gone_from_the_part(bracket):
    """A removal elsewhere can drop the face count while a blend stays put. Only features whose faces
    the kernel really deleted may be reported as removed."""
    import igl
    import numpy as np
    from defeat import estimator, occ
    from defeat.decide import Policy, decide
    from defeat.execute import defeature
    from defeat.immersed import boundary_triangles
    shape, g, feats = bracket
    dec = decide(feats, Policy("static", 2.0), g, occ.volume(shape))
    for d in dec.values():
        d.action = "remove"
    er = defeature(shape, feats, dec, graph=g)
    V, T = boundary_triangles(er.shape)
    for f in feats:
        if f.id in er.removed:
            p, _, _, _ = estimator._wall_samples(shape, f.faces)
            assert np.median(np.sqrt(igl.point_mesh_squared_distance(p, V, T)[0])) > 1e-3, f"{f.kind} #{f.id} is still there"
    assert len(er.removed) + len(er.failed) == len(feats)


def test_energy_bound_covers_removed_ribs_and_filled_holes_together():
    """By minimum potential energy, J(u) - J(u0) >= a_neg(u0, u0) - a_pos(u_P, u_P) for any mix of
    filled holes and removed ribs, all from the simplified solve plus a small solve on the removed rib."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from defeat import estimator, fea, occ
    from defeat.recognize import Feature
    from defeat.validate import mesh_stats, step_bytes
    L, W, T, h = 300.0, 100.0, 4.0, 5.0
    plate = BRepPrimAPI_MakeBox(L, W, T).Shape()
    ribbed = BRepAlgoAPI_Fuse(plate, BRepPrimAPI_MakeBox(gp_Pnt(100, 40, T), 100.0, 6.0, 8.0).Shape()).Shape()
    part = BRepAlgoAPI_Cut(ribbed, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(150, 20, -1), gp_Dir(0, 0, 1)), 8.0, T + 2).Shape()).Shape()
    bc = fea.load_case(occ.bbox(plate))

    def solve(shape, bc):
        m = mesh_stats(step_bytes(shape), h, keep_mesh=True)
        return fea.solve(m["mesh"]["p"], m["mesh"]["t"], dict(bc, want_system=True))
    rs = solve(plate, bc)
    ro = solve(part, dict(bc, ends=rs["ends"], load_center=rs["load_center"]))
    s = rs["system"]
    walls = [i for i, f in enumerate(occ.faces_of(part)) if occ.surf_info(f).kind == "cylinder"]
    res = estimator.bounds(s["p"], s["t"], s["U"], s["forms"], part, plate, [Feature(0, "hole", walls)], {0})
    summ = estimator.summarize(res)
    apos = estimator.removed_material_energy(part, plate, s["p"], s["t"], {k: s["U"][k] for k in rs["cases"]}, h / 2)
    for k in rs["cases"]:
        measured = 100 * (ro["cases"][k] - rs["cases"][k]) / rs["cases"][k]
        assert summ[k]["lower_pct"] - 100 * apos[k] / res["J0"][k] <= measured + 0.05
        assert apos[k] > 0


def test_field_sampling_is_exact_inside_and_bounded_just_outside_the_mesh():
    """Values and raw gradients of a quadratic field are reproduced exactly at interior points; a point
    off the mesh takes the field's value at its nearest point on the mesh instead of an extrapolation."""
    import numpy as np
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet
    from defeat.estimator import _gradients, _probes
    m = MeshTet().refined(2)
    ib = Basis(m, ElementVector(ElementTetP2()))
    u = ib.project(lambda x: np.array([x[0] ** 2, 2 * x[1] * x[2], x[2] + x[0] * x[1]]))
    x = np.random.default_rng(0).random((3, 200)) * 0.98 + 0.01
    g = _gradients(ib, {"a": u}, x)["a"]
    ex = np.zeros_like(g)
    ex[0, 0], ex[1, 1], ex[1, 2], ex[2, 2], ex[2, 0], ex[2, 1] = 2 * x[0], 2 * x[2], 2 * x[1], 1, x[1], x[0]
    assert np.abs(g - ex).max() < 1e-8
    sb = Basis(m, ElementTetP2())
    v = sb.project(lambda x: x[0] ** 2 + x[1])
    assert np.abs(_probes(sb, x) @ v - (sb.probes(x) @ v)).max() < 1e-8
    off = np.array([[1.3, 0.5, 0.5]]).T                   # 0.3 past the face x = 1: value there is 1 + 0.5
    assert abs((_probes(sb, off) @ v)[0] - 1.5) < 1e-6


def test_lower_bound_resolves_a_feature_smaller_than_an_element():
    """The dense fallback integrates exactly the material outside the original near the features, even
    when the feature is a fraction of an element: here a 0.2 mm notch filled on a 2 mm mesh."""
    import numpy as np
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from skfem import MeshTet
    from defeat import estimator, occ
    from defeat.immersed import boundary_triangles
    from defeat.validate import mesh_stats, step_bytes
    bar = BRepPrimAPI_MakeBox(40.0, 6.0, 6.0).Shape()
    notch = BRepPrimAPI_MakeBox(gp_Pnt(19.9, -1, 5.8), 0.2, 8.0, 1.0).Shape()   # 0.2 x 6 x 0.2 slot on top
    part = BRepAlgoAPI_Cut(bar, notch).Shape()
    exact = occ.volume(bar) - occ.volume(part)
    ms = mesh_stats(step_bytes(bar), 2.0, keep_mesh=True)
    m = MeshTet(np.asarray(ms["mesh"]["p"], float).T, np.asarray(ms["mesh"]["t"], int).T)
    Vo, To = boundary_triangles(part)
    near = np.array([[20.0, y, 5.9] for y in np.linspace(0, 6, 13)])
    x, w = estimator._dense_outside(m, Vo, To, near)
    assert abs(w.sum() / exact - 1) < 0.1
