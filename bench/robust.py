"""Robustness benchmark: run the FULL pipeline (read, heal, recognize, decide, defeature, validate,
optionally mesh + stress check) on random real-world STEP models and report how often each stage
succeeds, how long it takes, and how much the models are simplified.

    python -m bench.robust --n 500 --mesh 500

Every part that has something to remove is meshed and solved before and after (parts with nothing to
remove have nothing to compare). Rates carry 95% Wilson intervals, medians a part-level bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
from collections import Counter

from bench.common import DATA, bootstrap, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def run_one(args):
    """Full shipped pipeline via Session (rules + network, static preset, stress guard when meshing)."""
    name, with_mesh = args
    import shutil
    from defeat import fea, occ
    from defeat.session import Session
    out = {"stage": "read"}
    s = Session(os.path.join(BASE, "breps", "step", name + ".stp"))
    try:
        out["input_valid"] = occ.is_valid(s.shape)
        out["solids"] = len(occ.solids_of(s.shape))
        if out["solids"] == 0:
            return {"ok": False, "error": "no solid in file", **out}
        out["faces"] = len(s.graph.nodes)
        out["features"] = len(s.features)
        n_rm = sum(1 for d in s.decisions.values() if d.action == "remove")
        out["to_remove"] = n_rm
        out["stage"] = "defeature"
        s.run(with_mesh=bool(with_mesh and n_rm))
        r = s.result
        out["removed"] = len(r.removed)
        out["kept_by_kernel"] = len(r.failed)
        out["output_valid"] = occ.is_valid(r.shape)
        g = s.metrics["geometry"]
        if n_rm:
            out["faces_after"] = g["faces"][1]
            out["volume_delta_pct"] = g["volume_delta_pct"]
        m = s.metrics.get("mesh") or {}
        if m.get("tet_reduction_pct") is not None:
            out["tet_reduction_pct"] = m["tet_reduction_pct"]
        fd = (m.get("fea") or {}).get("delta") or {}
        if fd.get("tip_disp_delta_pct") is not None:
            out["tip_disp_delta_pct"] = fd["tip_disp_delta_pct"]
            out["stress_p99_delta_pct"] = fd["vm_p99_delta_pct"]
            out["worst_case_pct"] = fea.worst_case(fd)
            out["stress_peak_delta_pct"] = fd.get("vm_peak_delta_pct")
            out["cases_pct"] = fd.get("cases_delta_pct")
        if s.metrics.get("unchecked"):
            out["unchecked"] = s.metrics.get("unchecked_why")
        out["mesh_curvature"] = m.get("stat_curvature")
        gd = s.metrics.get("guard") or {}
        if gd.get("triggered"):
            out["guard_before_pct"] = gd.get("before_pct")
            out["guard_stiffness_before_pct"] = gd.get("stiffness_before_pct")
            out["guard_accepted"] = bool(gd.get("accepted"))
            out["guard_restored"] = gd.get("restored", 0)
            out["guard_undone"] = bool(gd.get("undone"))
        out["stage"] = "done"
        return out
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mesh", type=int, default=500, help="how many of the parts also get mesh + stress check")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=240, help="seconds per part")
    ap.add_argument("--val", action="store_true", help="validation parts, for tuning; saved apart from the test results")
    ap.add_argument("--skip", type=int, default=0, help="start this far into the shuffled test split (for parts not scored before)")
    ap.add_argument("--retry", type=int, default=0,
                    help="re-run only the parts that timed out in the saved rows, with this per-part timeout")
    a = ap.parse_args()
    split = json.load(open(os.path.join(BASE, "train_test.json")))
    if a.val:
        tr = split["train"]
        random.Random(0).shuffle(tr)
        names = tr[4000: 4000 + a.n]
    else:
        test = split["test"]
        random.Random(1).shuffle(test)
        names = test[a.skip: a.skip + a.n]
    tag = "robust_val" if a.val else "robustness"
    items = [(n, i < a.mesh) for i, n in enumerate(names)]
    if a.retry:
        from bench.common import RESULTS
        res = json.load(open(os.path.join(RESULTS, tag + "_rows.json")))
        again = [i for i, r in zip(items, res) if r.get("error") == "timeout"]
        redo = {str(i): r for i, r in zip(again, pmap_hard("bench.robust:run_one", again, workers=a.workers,
                                                            timeout=a.retry, label="retry"))}
        res = [redo.get(r["item"], r) for r in res]
    else:
        res = pmap_hard("bench.robust:run_one", items, workers=a.workers, timeout=a.timeout, label="robust")
    report(res, items, a.retry or a.timeout, tag)


def report(res, items, timeout, tag="robustness"):
    ok = [r for r in res if r.get("ok")]
    fail = {}
    for r in res:
        if not r.get("ok"):
            k = (r.get("error") or "?").split(":")[0]
            fail[k] = fail.get(k, 0) + 1
    with_rm = [r for r in ok if r.get("to_remove")]
    tets = [r["tet_reduction_pct"] for r in ok if "tet_reduction_pct" in r]
    dd = [abs(r["tip_disp_delta_pct"]) for r in ok if "tip_disp_delta_pct" in r]
    ds = [abs(r["stress_p99_delta_pct"]) for r in ok if "stress_p99_delta_pct" in r]
    secs = [r["seconds"] for r in ok]
    n_in = sum(1 for r in ok if r.get("input_valid"))
    n_valid = sum(1 for r in ok if r.get("output_valid") and r.get("input_valid"))
    within = sum(1 for x in dd if x <= 5)
    worst = [r["worst_case_pct"] for r in ok if r.get("worst_case_pct") is not None]
    within_worst = sum(1 for x in worst if x <= 5)
    before = [r.get("guard_before_pct") or r["worst_case_pct"] for r in ok if r.get("worst_case_pct") is not None]
    within_before = sum(1 for x in before if x <= 5)
    # the same question the stiffness cases alone would answer, without the stress peak the guard also judges
    stiff_before = [r.get("guard_stiffness_before_pct") or r["worst_case_pct"] for r in ok if r.get("worst_case_pct") is not None]
    within_stiff = sum(1 for x in stiff_before if x <= 5)
    # the kernel's job is to remove what was asked; the stress guard may then put some back on purpose
    rm_pairs = [(r["to_remove"] - r.get("kept_by_kernel", 0), r["to_remove"]) for r in with_rm]
    restored = sum(r.get("guard_restored", 0) for r in with_rm if r.get("guard_accepted"))
    asked = sum(r["to_remove"] for r in with_rm)
    ci = {
        "completion_rate": wilson(len(ok), len(items)),
        "output_valid_rate": wilson(n_valid, n_in),
        "feature_removal_success_rate": bootstrap(rm_pairs, lambda s: sum(a for a, _ in s) / max(sum(b for _, b in s), 1)),
        "share_within_5pct_deflection": wilson(within, len(dd)),
        "share_within_5pct_worst_load_case": wilson(within_worst, len(worst)),
        "median_worst_load_case_pct": bootstrap(worst, st.median),
        "share_within_5pct_before_guard": wilson(within_before, len(before)),
        "share_stiffness_within_5pct_before_guard": wilson(within_stiff, len(stiff_before)),
        "median_abs_deflection_change_pct": bootstrap(dd, st.median),
    }
    rep = {
        "dataset": "Fusion 360 Gallery extended STEP, official test split (real designer models)",
        "parts": len(items), "completed": len(ok), "completion_rate": round(len(ok) / len(items), 4),
        "failure_reasons": fail,
        "failed_items": [(r.get("item"), r.get("error")) for r in res if not r.get("ok")][:40],
        "output_valid_rate": round(sum(1 for r in ok if r.get("output_valid") and r.get("input_valid")) / max(sum(1 for r in ok if r.get("input_valid")), 1), 4),
        "invalid_inputs": sum(1 for r in ok if not r.get("input_valid")),
        "parts_with_features_to_remove": len(with_rm),
        "timeout_seconds_per_part": timeout,
        "feature_removal_success_rate": round(sum(a for a, _ in rm_pairs) / max(asked, 1), 4),
        "features_targeted": asked,
        "features_kept_by_kernel": sum(r.get("kept_by_kernel", 0) for r in with_rm),
        "features_restored_by_stress_guard": restored,
        "features_removed_in_final_output": sum(r["removed"] for r in with_rm),
        # with no stress comparison the run falls back to small removals, so the rest of what it targeted stays
        "features_left_by_the_no_check_fallback": asked - sum(r.get("kept_by_kernel", 0) for r in with_rm) - restored - sum(r["removed"] for r in with_rm),
        "parts_without_a_stress_comparison": {"total": sum(1 for r in with_rm if r.get("worst_case_pct") is None),
                                              "shipped_unchanged": sum(1 for r in with_rm if r.get("worst_case_pct") is None and not r["removed"])},
        "median_faces": st.median([r["faces"] for r in ok]) if ok else None,
        "median_face_reduction_pct": st.median([100 * (1 - r["faces_after"] / r["faces"]) for r in with_rm if "faces_after" in r]) if with_rm else None,
        "median_volume_change_pct": st.median([abs(r["volume_delta_pct"]) for r in with_rm if "volume_delta_pct" in r]) if with_rm else None,
        "seconds_per_part": {"median": st.median(secs) if secs else None, "p90": pct(secs, 0.9)},
        "mesh_subset": {"n": len(tets), "median_tet_reduction_pct": st.median(tets) if tets else None,
                        "p10_tet_reduction_pct": pct(tets, 0.1), "p90_tet_reduction_pct": pct(tets, 0.9)},
        "no_stress_check": {"parts": sum(1 for r in ok if r.get("unchecked")),
                            "why": dict(Counter(r["unchecked"] for r in ok if r.get("unchecked")))},
        "meshed_with_curvature_sizing_off": sum(1 for r in ok if r.get("mesh_curvature") == 0 and r.get("worst_case_pct") is not None),
        "stress_guard": {"triggered": sum(1 for r in ok if "guard_accepted" in r),
                         "accepted": sum(1 for r in ok if r.get("guard_accepted")),
                         "undone": sum(1 for r in ok if r.get("guard_undone")),
                         "share_within_5pct_before_guard": round(within_before / max(len(before), 1), 4) if before else None,
                         "share_stiffness_within_5pct_before_guard": round(within_stiff / max(len(stiff_before), 1), 4) if stiff_before else None,
                         "median_features_restored": st.median([r["guard_restored"] for r in ok if r.get("guard_accepted")]) if any(r.get("guard_accepted") for r in ok) else None},
        "stress_subset": {"n": len(dd), "median_abs_deflection_change_pct": st.median(dd) if dd else None,
                          "p90_abs_deflection_change_pct": pct(dd, 0.9),
                          "median_abs_peak_stress_change_pct": st.median(ds) if ds else None,
                          "p90_abs_peak_stress_change_pct": pct(ds, 0.9),
                          "share_within_5pct_deflection": round(within / len(dd), 4) if dd else None,
                          "share_within_5pct_worst_load_case": round(within_worst / len(worst), 4) if worst else None,
                          "median_worst_load_case_pct": st.median(worst) if worst else None,
                          "p90_worst_load_case_pct": pct(worst, 0.9)},
        "ci95": ci,
        "ci_method": "Wilson score interval for rates; 2000-sample part-level bootstrap for the removal rate and medians",
    }
    save(tag + "_rows", res)
    print(save(tag, rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
