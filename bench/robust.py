"""Robustness benchmark: run the FULL pipeline (read, heal, recognize, decide, defeature, validate,
optionally mesh + stress check) on random real-world STEP models and report how often each stage
succeeds, how long it takes, and how much the models are simplified.

    python -m bench.robust --n 500 --mesh 80
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

from bench.common import DATA, pmap_hard, save

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def run_one(args):
    """Full shipped pipeline via Session (rules + GNN, static preset, stress guard when meshing)."""
    name, with_mesh = args
    import shutil
    from defeat import occ
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
        gd = s.metrics.get("guard") or {}
        if gd.get("triggered"):
            out["guard_before_pct"] = gd.get("deflection_delta_pct", gd.get("before_pct"))
            out["guard_accepted"] = bool(gd.get("accepted"))
            out["guard_restored"] = gd.get("restored", 0)
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
    ap.add_argument("--mesh", type=int, default=80, help="how many of the parts also get mesh + stress check")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    test = json.load(open(os.path.join(BASE, "train_test.json")))["test"]
    random.Random(1).shuffle(test)
    items = [(n, i < a.mesh) for i, n in enumerate(test[: a.n])]
    res = pmap_hard("bench.robust:run_one", items, workers=a.workers, timeout=240, label="robust")
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
    rep = {
        "dataset": "Fusion 360 Gallery extended STEP, official test split (real designer models)",
        "parts": len(items), "completed": len(ok), "completion_rate": round(len(ok) / len(items), 4),
        "failure_reasons": fail,
        "failed_items": [(r.get("item"), r.get("error")) for r in res if not r.get("ok")][:40],
        "output_valid_rate": round(sum(1 for r in ok if r.get("output_valid") and r.get("input_valid")) / max(sum(1 for r in ok if r.get("input_valid")), 1), 4),
        "invalid_inputs": sum(1 for r in ok if not r.get("input_valid")),
        "parts_with_features_to_remove": len(with_rm),
        "feature_removal_success_rate": round(sum(r["removed"] for r in with_rm) / max(sum(r["to_remove"] for r in with_rm), 1), 4),
        "median_faces": st.median([r["faces"] for r in ok]) if ok else None,
        "median_face_reduction_pct": st.median([100 * (1 - r["faces_after"] / r["faces"]) for r in with_rm if "faces_after" in r]) if with_rm else None,
        "median_volume_change_pct": st.median([abs(r["volume_delta_pct"]) for r in with_rm if "volume_delta_pct" in r]) if with_rm else None,
        "seconds_per_part": {"median": st.median(secs) if secs else None, "p90": pct(secs, 0.9)},
        "mesh_subset": {"n": len(tets), "median_tet_reduction_pct": st.median(tets) if tets else None,
                        "p10_tet_reduction_pct": pct(tets, 0.1), "p90_tet_reduction_pct": pct(tets, 0.9)},
        "stress_guard": {"triggered": sum(1 for r in ok if "guard_accepted" in r),
                         "accepted": sum(1 for r in ok if r.get("guard_accepted")),
                         "share_within_5pct_before_guard": round(sum(1 for r in ok if "tip_disp_delta_pct" in r and abs(r.get("guard_before_pct", r["tip_disp_delta_pct"])) <= 5) / max(len(dd), 1), 4) if dd else None,
                         "median_features_restored": st.median([r["guard_restored"] for r in ok if r.get("guard_accepted")]) if any(r.get("guard_accepted") for r in ok) else None},
        "stress_subset": {"n": len(dd), "median_abs_deflection_change_pct": st.median(dd) if dd else None,
                          "p90_abs_deflection_change_pct": pct(dd, 0.9),
                          "median_abs_peak_stress_change_pct": st.median(ds) if ds else None,
                          "p90_abs_peak_stress_change_pct": pct(ds, 0.9),
                          "share_within_5pct_deflection": round(sum(1 for x in dd if x <= 5) / len(dd), 4) if dd else None},
    }
    save("robustness_rows", res)
    print(save("robustness", rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
