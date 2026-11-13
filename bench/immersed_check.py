"""Does the grid stress check agree with the tetrahedral one? On validation parts (never test parts).

For every part with something removed, both checks solve the original and the defeatured part with
the same load cases and end regions:

  tet   : the shipped check, quadratic tetrahedra from gmsh (shipped mesh size, capped like the app)
  grid  : defeat.immersed, the finite cell method on a fixed grid with moment-fitted integration

Reported: how far the grid's percent changes are from the tet ones, and, for a range of grid
tolerances, how often the grid would pass a part the tet check fails at 5%. The grid tolerance used
by the app is picked from this table.

    python -m bench.immersed_check --n 120 --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from bench.common import DATA, bootstrap, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")
TOL = 5.0


def _changes(a, b):
    return {k: 100 * (b["cases"][k] - a["cases"][k]) / a["cases"][k] for k in a["cases"]}


def run_one(name):
    import shutil
    import time
    from bench.convergence import _solve
    from defeat import fea, immersed, occ
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
        t = time.time()
        a = _solve(step_bytes(s.shape), s.policy.h, bc, fea.MAX_TETS)
        if "cases" not in a:
            return {"skip": "tet check unavailable"}
        tb = dict(bc, ends=a["ends"], load_center=a["load_center"])
        b = _solve(step_bytes(s.result.shape), s.policy.h, tb, fea.MAX_TETS)
        if "cases" not in b:
            return {"skip": "tet check unavailable"}
        t_tet = time.time() - t
        t = time.time()
        ga, gb = immersed.compare(s.shape, s.result.shape, bc)
        if not (ga.get("ok") and gb and gb.get("ok")):
            return {"error_at": "grid", "detail": (gb or ga).get("error"), "tet": _changes(a, b)}
        return {"tet": _changes(a, b), "grid": _changes(ga, gb), "tets": a["tets"], "cells": ga["cells"],
                "grid_volume_err_pct": 100 * (ga["volume"] - occ.volume(s.shape)) / occ.volume(s.shape),
                "seconds_tet": t_tet, "seconds_grid": time.time() - t}
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    split = json.load(open(os.path.join(BASE, "train_test.json")))
    tr = split["train"]
    random.Random(0).shuffle(tr)
    names = tr[4000: 4000 + a.n]          # the validation parts bench.robust --val uses
    res = pmap_hard("bench.immersed_check:run_one", names, workers=a.workers, timeout=a.timeout, label="grid")
    both = [r for r in res if "grid" in r]
    worst = lambda c: max(abs(v) for v in c.values())
    diff = [abs(r["grid"][k] - r["tet"][k]) for r in both for k in r["tet"]]
    wt = np.array([worst(r["tet"]) for r in both])
    wg = np.array([worst(r["grid"]) for r in both])
    table = {}
    for tol in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        fail = wt > TOL
        missed = int((fail & (wg <= tol)).sum())
        passed_ok = int((~fail & (wg <= tol)).sum())
        table[str(tol)] = {"tet_fails": int(fail.sum()), "grid_passes_a_tet_fail": missed,
                           "tet_passes_kept_by_grid": wilson(passed_ok, int((~fail).sum())) if (~fail).sum() else None}
    rep = {"parts": a.n, "compared": len(both),
           "skipped": {k: sum(1 for r in res if r.get("skip") == k) for k in {r.get("skip") for r in res if r.get("skip")}},
           "grid_failed": sum(1 for r in res if r.get("error_at") == "grid"),
           "timeouts_or_errors": sum(1 for r in res if r.get("error")),
           "case_abs_diff_pts": {"median": float(np.median(diff)) if diff else None, "p90": float(np.percentile(diff, 90)) if diff else None},
           "worst_case_abs_diff_pts": bootstrap([float(abs(x - y)) for x, y in zip(wg, wt)], lambda v: float(np.median(v))) if len(both) else None,
           "worst_ratio_grid_over_tet_median": float(np.median(wg[wt > 0.5] / wt[wt > 0.5])) if (wt > 0.5).any() else None,
           "grid_volume_err_pct_median": float(np.median([r["grid_volume_err_pct"] for r in both])) if both else None,
           "seconds_grid_median": float(np.median([r["seconds_grid"] for r in both])) if both else None,
           "seconds_tet_median": float(np.median([r["seconds_tet"] for r in both])) if both else None,
           "tolerance_table": table}
    print(json.dumps(rep, indent=1))
    save("immersed_check", rep)
    save("immersed_check_rows", res)


if __name__ == "__main__":
    main()
