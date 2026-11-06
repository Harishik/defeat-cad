"""Does the stress check's answer depend on the mesh? On validation parts (never test parts):

  noise       : the same unchanged part meshed at h and at 0.9 h. Whatever moves is the check's own noise.
  refinement  : the percent change from defeaturing, measured at h and again at h / 2.
  verdict     : how often both meshes agree on whether the part stays within 5%.

Only parts small enough for the twice finer mesh to fit the direct solver can be checked, so this
covers the smaller parts, not all of them.

    python -m bench.convergence --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

import numpy as np

from bench.common import DATA, bootstrap, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")
MAX_BASE_TETS = 12000    # meshing at h / 2 is what really decides; the shipped solve caps out at 17000
MAX_FINE_TETS = 30000
TOL = 5.0


def _solve(data, h, bc, cap):
    from defeat import fea
    from defeat.validate import mesh_stats
    m = mesh_stats(data, h, keep_mesh=True)
    if not m.get("ok"):
        return {"error": "mesh: " + str(m.get("error"))[:80]}
    if m["tets"] > cap:
        return {"too_big": True, "tets": m["tets"]}
    r = fea.solve(m["mesh"]["p"], m["mesh"]["t"], bc)
    if not r.get("ok"):
        return {"error": r.get("error")}
    return {"cases": r["cases"], "ends": r["ends"], "load_center": r["load_center"], "tets": m["tets"], "dofs": r["dofs"]}


def _changes(a, b):
    return {k: 100 * (b["cases"][k] - a["cases"][k]) / a["cases"][k] for k in a["cases"]}


def run_one(name):
    import shutil
    from defeat import fea, occ
    from defeat.session import Session
    from defeat.validate import step_bytes
    s = Session(os.path.join(BASE, "breps", "step", name + ".stp"))
    try:
        if not any(d.action == "remove" for d in s.decisions.values()):
            return {"skip": "nothing to remove"}
        s.run(with_mesh=False)
        if not s.result.removed:
            return {"skip": "nothing removed"}
        bc = fea.load_case(occ.bbox(s.shape))
        h = s.policy.h
        orig, simple = step_bytes(s.shape), step_bytes(s.result.shape)
        a = _solve(orig, h, bc, MAX_BASE_TETS)
        if a.get("too_big"):
            return {"skip": "too big for the fine mesh"}
        if "error" in a:
            return {"error_at": "a", "detail": a["error"]}
        # every mesh of this part uses the same end regions, so only the mesh changes
        bc = dict(bc, ends=a["ends"], load_center=a["load_center"])
        b = _solve(simple, h, bc, MAX_BASE_TETS)
        a9 = _solve(orig, 0.9 * h, bc, 2 * MAX_BASE_TETS)
        af = _solve(orig, h / 2, bc, MAX_FINE_TETS)
        bf = _solve(simple, h / 2, bc, MAX_FINE_TETS)
        out = {"h": h, "tets": a["tets"], "tets_fine": af.get("tets")}
        if "cases" in a9:
            out["noise"] = {k: abs(100 * (a9["cases"][k] - a["cases"][k]) / a["cases"][k]) for k in a["cases"]}
        if all("cases" in x for x in (b, af, bf)):
            out["change_h"], out["change_fine"] = _changes(a, b), _changes(af, bf)
            out["absolute_shift"] = {k: abs(100 * (af["cases"][k] - a["cases"][k]) / af["cases"][k]) for k in a["cases"]}
        return out
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def _summ(xs):
    if not xs:
        return None
    return {"n": len(xs), "median": round(st.median(xs), 3), "p90": round(float(np.percentile(xs, 90)), 3),
            "within_1pct": round(sum(x <= 1 for x in xs) / len(xs), 3), "median_ci95": bootstrap(xs, st.median)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    res = pmap_hard("bench.convergence:run_one", tr[4000: 4000 + a.n], workers=a.workers, timeout=1800, label="conv")
    ok = [r for r in res if r.get("ok")]
    skipped = {}
    for r in ok:
        if r.get("skip"):
            skipped[r["skip"]] = skipped.get(r["skip"], 0) + 1
    rows = [r for r in ok if "change_h" in r]

    def worst(c):
        return max(abs(v) for v in c.values())
    agree = sum((worst(r["change_h"]) <= TOL) == (worst(r["change_fine"]) <= TOL) for r in rows)
    rep = {"parts": a.n, "failed": sum(1 for r in res if not r.get("ok")), "skipped": skipped,
           "errors": sum(1 for r in ok if r.get("error_at")),
           "parts_compared": len(rows),
           "median_tets_fine": st.median([r["tets_fine"] for r in rows]) if rows else None,
           "noise_same_part_h_vs_0.9h_pct": _summ([max(r["noise"].values()) for r in ok if "noise" in r]),
           "absolute_result_shift_h_vs_fine_pct": _summ([max(r["absolute_shift"].values()) for r in rows]),
           "worst_change_at_h_pct": _summ([worst(r["change_h"]) for r in rows]),
           "worst_change_at_fine_pct": _summ([worst(r["change_fine"]) for r in rows]),
           "error_in_change_h_vs_fine_points": _summ([max(abs(r["change_h"][k] - r["change_fine"][k]) for k in r["change_h"])
                                                      for r in rows]),
           "same_within_5pct_verdict": round(agree / len(rows), 4) if rows else None,
           "same_within_5pct_verdict_ci95": wilson(agree, len(rows)) if rows else None}
    save("convergence_rows", rows)
    print(save("convergence", rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
