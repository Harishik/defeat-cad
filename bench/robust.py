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
    name, with_mesh = args
    import time
    from defeat import occ
    from defeat.decide import Policy, decide, default_h
    from defeat.execute import defeature, worker
    from defeat.graph import build_graph
    from defeat.recognize import recognize
    from defeat.validate import geometry_report, mesh_compare
    out = {"stage": "read"}
    t = time.time()
    shape = occ.read_cad(os.path.join(BASE, "breps", "step", name + ".stp"))
    shape = occ.heal(shape)
    out["input_valid"] = occ.is_valid(shape)
    out["solids"] = len(occ.solids_of(shape))
    if out["solids"] == 0:
        return {"ok": False, "error": "no solid in file", **out}
    out["stage"] = "recognize"
    g = build_graph(shape)
    _, feats = recognize(shape, g)   # shipped pipeline: rules + GNN refinement
    out["faces"] = len(g.nodes)
    out["features"] = len(feats)
    out["t_recognize"] = time.time() - t
    pol = Policy("static", default_h(g.char_len))
    d = decide(feats, pol)
    n_rm = sum(1 for v in d.values() if v.action == "remove")
    out["to_remove"] = n_rm
    out["stage"] = "defeature"
    t2 = time.time()
    r = defeature(shape, feats, d, timeout=30)
    out["t_defeature"] = time.time() - t2
    out["removed"] = len(r.removed)
    out["kept_by_kernel"] = len(r.failed)
    out["strategy"] = r.strategy
    out["output_valid"] = occ.is_valid(r.shape)
    if n_rm:
        geo = geometry_report(shape, r.shape)
        out["faces_after"] = geo["faces"][1]
        out["volume_delta_pct"] = geo["volume_delta_pct"]
        out["max_dev_rel"] = geo["deviation"]["max"] / g.char_len
    if with_mesh and r.removed:
        out["stage"] = "mesh"
        m = mesh_compare(shape, r.shape, pol.h, timeout=120)
        if m.get("tet_reduction_pct") is not None:
            out["tet_reduction_pct"] = m["tet_reduction_pct"]
        fd = (m.get("fea") or {}).get("delta") or {}
        if fd.get("tip_disp_delta_pct") is not None:
            out["tip_disp_delta_pct"] = fd["tip_disp_delta_pct"]
            out["stress_p99_delta_pct"] = fd["vm_p99_delta_pct"]
    out["stage"] = "done"
    return out


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
        "output_valid_rate": round(sum(1 for r in ok if r.get("output_valid")) / max(len(ok), 1), 4),
        "parts_with_features_to_remove": len(with_rm),
        "feature_removal_success_rate": round(sum(r["removed"] for r in with_rm) / max(sum(r["to_remove"] for r in with_rm), 1), 4),
        "median_faces": st.median([r["faces"] for r in ok]) if ok else None,
        "median_face_reduction_pct": st.median([100 * (1 - r["faces_after"] / r["faces"]) for r in with_rm if "faces_after" in r]) if with_rm else None,
        "median_volume_change_pct": st.median([abs(r["volume_delta_pct"]) for r in with_rm if "volume_delta_pct" in r]) if with_rm else None,
        "seconds_per_part": {"median": st.median(secs) if secs else None, "p90": pct(secs, 0.9)},
        "mesh_subset": {"n": len(tets), "median_tet_reduction_pct": st.median(tets) if tets else None,
                        "p10_tet_reduction_pct": pct(tets, 0.1), "p90_tet_reduction_pct": pct(tets, 0.9)},
        "stress_subset": {"n": len(dd), "median_abs_deflection_change_pct": st.median(dd) if dd else None,
                          "p90_abs_deflection_change_pct": pct(dd, 0.9),
                          "median_abs_peak_stress_change_pct": st.median(ds) if ds else None,
                          "p90_abs_peak_stress_change_pct": pct(ds, 0.9),
                          "share_within_5pct_deflection": round(sum(1 for x in dd if x <= 5) / len(dd), 4) if dd else None},
    }
    print(save("robustness", rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
