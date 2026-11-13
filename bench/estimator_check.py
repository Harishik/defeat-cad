"""Do the one-solve bounds hold? On validation parts (never test parts) and the two sample parts.

For every part with something removed, the simplified part is meshed and solved ONCE; from that solve
defeat.estimator gives, per load case,

  lower : the strain energy of the simplified solution in the filled-in material (a proven lower bound
          on the compliance change when every removed feature is negative)
  est2  : the Antolin-Chanon estimator squared, a bound on the remaining error up to a constant

The original is also meshed and solved, as the referee. Reported: how often lower <= measured holds,
the measured share explained by `lower`, and the constant C that makes lower + C est2 cover the
measured change, so the guard can use the bracket [lower, lower + C est2].

    python -m bench.estimator_check --n 75 --workers 3
    python -m bench.estimator_check --samples
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from bench.common import DATA, ROOT, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def run_one(path):
    import shutil
    from defeat import estimator, fea, occ
    from defeat.session import Session
    from defeat.validate import mesh_stats, step_bytes
    s = Session(path)
    try:
        if not any(d.action == "remove" for d in s.decisions.values()):
            return {"skip": "nothing to remove"}
        s.run(with_mesh=False)
        removed = set(s.result.removed)
        if not removed:
            return {"skip": "nothing removed"}
        bc = fea.load_case(occ.bbox(s.shape))
        h = s.policy.h

        def solve(shape, bc, h0=h, cap=fea.MAX_TETS):
            hf = h0
            for _ in range(8):
                m = mesh_stats(step_bytes(shape), hf, keep_mesh=True)
                if not m.get("ok"):
                    return None, None
                if m["tets"] <= cap:
                    break
                hf *= 1.2
            r = fea.solve(m["mesh"]["p"], m["mesh"]["t"], dict(bc, want_system=True))
            return (m, r) if r.get("ok") else (None, None)

        t = time.time()
        ma, ra = solve(s.shape, bc)                       # referee: the original
        if ra is None:
            return {"skip": "original not solvable"}
        bc = dict(bc, ends=ra["ends"], load_center=ra["load_center"])
        t_orig = time.time() - t
        t = time.time()
        mb, rb = solve(s.result.shape, bc)                # the one solve the bounds need
        if rb is None:
            return {"skip": "simplified not solvable"}
        t_simple = time.time() - t
        t = time.time()
        sysb = rb["system"]
        res = estimator.bounds(sysb["p"], sysb["t"], sysb["U"], sysb["forms"], s.shape, s.result.shape, s.features, removed,
                               bc=bc)
        summ = estimator.summarize(res)
        kinds = list(res["kinds"].values())
        apos = None
        if any(k in ("positive", "mixed") for k in kinds):
            hb = (rb.get("h_fea") or h) / 2
            apos = estimator.removed_material_energy(s.shape, s.result.shape, sysb["p"], sysb["t"],
                                                     {k: sysb["U"][k] for k in ra["cases"]}, hb)
        t_est = time.time() - t
        fine = None
        if os.environ.get("DEFEAT_REFINE"):
            # a finer referee: both parts re-meshed at 1/1.6 of the size, up to 2.5x the app's element cap.
            # The two coarse meshes differ by up to about a percent on their own; a bound can only be
            # judged against a referee closer to the truth than that.
            hr = (ra.get("h_fea") or h) / 1.6
            mf, rf = solve(s.shape, bc, hr, int(2.5 * fea.MAX_TETS))
            mg, rg = solve(s.result.shape, bc, hr, int(2.5 * fea.MAX_TETS))
            if rf is not None and rg is not None:
                fine = {k: 100 * (rf["cases"][k] - rg["cases"][k]) / rg["cases"][k] for k in ra["cases"]}
                fine_tets = [mf["tets"], mg["tets"]]
        cases = {}
        for k in ra["cases"]:
            J, J0 = ra["cases"][k], rb["cases"][k]
            c = {"measured_pct": 100 * (J - J0) / J0, **summ.get(k, {"lower_pct": 0.0, "est_pct": 0.0})}
            if fine is not None:
                c["measured_fine_pct"] = fine[k]
            j0 = res["J0"][k]
            c["pos_pct"] = 100 * apos[k] / j0 if apos else (0.0 if apos is not None or not any(x in ("positive", "mixed") for x in kinds) else None)
            c["general_lower_pct"] = c["lower_pct"] - c["pos_pct"] if c["pos_pct"] is not None else None
            cases[k] = c
        return {"cases": cases, "lower_method": res.get("lower_method"), "load_shift_pct": {k: 100 * v / res["J0"][k] for k, v in res.get("load_shift", {}).items()}, "n_removed": len(removed), "negative": kinds.count("negative"),
                "positive": kinds.count("positive"), "mixed": kinds.count("mixed"),
                "tets": ma["tets"], "tets_fine": fine_tets if fine is not None else None, "seconds_original": t_orig, "seconds_simplified": t_simple, "seconds_estimator": t_est}
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def report(res, tag):
    rows = [r for r in res if "cases" in r]
    pure = [r for r in rows if r["negative"] and not r["positive"] and not r["mixed"]]
    def flat(rs):
        return [c for r in rs for c in r["cases"].values()]
    out = {"parts_compared": len(rows), "all_negative_parts": len(pure),
           "skipped": {k: sum(1 for r in res if r.get("skip") == k) for k in {r.get("skip") for r in res if r.get("skip")}},
           "errors": sum(1 for r in res if r.get("error"))}
    gen = [c for r in rows for c in r["cases"].values() if c.get("general_lower_pct") is not None]
    if gen:
        ok = [c["general_lower_pct"] <= c["measured_pct"] + 0.05 for c in gen]
        out["general_lower_bound"] = {"load_cases": len(gen), "holds": wilson(sum(ok), len(ok)),
                                      "violations": [c for c, k in zip(gen, ok) if not k][:10],
                                      "parts_without_it": sum(1 for r in rows if any(c.get("general_lower_pct") is None for c in r["cases"].values()))}
    for name, rs in (("all_negative", pure), ("any", rows)):
        cs = flat(rs)
        if not cs:
            continue
        noise = 0.05    # percent: below this the two meshes' own difference dominates
        holds = [c["lower_pct"] <= c["measured_pct"] + noise for c in cs]
        sig = [c for c in cs if c["measured_pct"] > 0.5]
        # smallest C with lower + C * est >= measured, per load case (the part of the change the lower bound misses)
        need = [max(0.0, (c["measured_pct"] - c["lower_pct"])) / c["est_pct"] for c in cs if c["est_pct"] > 1e-9]
        out[name] = {"load_cases": len(cs), "lower_bound_holds": wilson(sum(holds), len(holds)),
                     "lower_bound_violations": [c for c, ok in zip(cs, holds) if not ok][:10],
                     "share_explained_by_lower_median": float(np.median([c["lower_pct"] / c["measured_pct"] for c in sig])) if sig else None,
                     "C_needed_p50_p90_p100": [float(np.percentile(need, q)) for q in (50, 90, 100)] if need else None,
                     "upper_over_measured_median_C1": float(np.median([(c["lower_pct"] + c["est_pct"]) / c["measured_pct"] for c in sig])) if sig else None}
    ref = [c for r in rows for c in r["cases"].values() if "measured_fine_pct" in c]
    if ref:
        noise = [abs(c["measured_fine_pct"] - c["measured_pct"]) for c in ref]
        lo = lambda c: c["general_lower_pct"] if c.get("general_lower_pct") is not None else c["lower_pct"]
        strict = [lo(c) <= c["measured_fine_pct"] + 0.05 for c in ref]
        within = [lo(c) <= c["measured_fine_pct"] + abs(c["measured_fine_pct"] - c["measured_pct"]) + 0.05 for c in ref]
        need = [max(0.0, c["measured_fine_pct"] - c["lower_pct"]) / c["est_pct"] for c in ref if c["est_pct"] > 1e-9]
        # Pre-registered before the fresh run: the bound is judged only where the referee itself has
        # converged (coarse and fine measurements within REF_TOL points). Elsewhere the two referee meshes
        # disagree by more than the effect, so neither a hold nor a violation means anything.
        REF_TOL = 0.1
        conv = [c for c in ref if abs(c["measured_fine_pct"] - c["measured_pct"]) <= REF_TOL]
        cstrict = [lo(c) <= c["measured_fine_pct"] + 0.05 for c in conv]
        out["converged_referee"] = {
            "rule": f"|fine - coarse| <= {REF_TOL} pts; bound holds if lower <= fine + 0.05",
            "load_cases": len(conv), "of": len(ref),
            "lower_bound_holds": wilson(sum(cstrict), len(cstrict)) if conv else None,
            "holds_count": int(sum(cstrict)),
            "violations": [c for c, ok in zip(conv, cstrict) if not ok][:12],
            "share_explained_median": float(np.median([lo(c) / c["measured_fine_pct"] for c in conv if c["measured_fine_pct"] > 0.3])) if any(c["measured_fine_pct"] > 0.3 for c in conv) else None}
        out["against_fine_referee"] = {
            "load_cases": len(ref),
            "referee_coarse_vs_fine_pts": {"median": float(np.median(noise)), "p90": float(np.percentile(noise, 90)), "max": float(max(noise))},
            "lower_bound_holds_strict": wilson(sum(strict), len(strict)),
            "lower_bound_holds_within_referee_change": wilson(sum(within), len(within)),
            "violations": [c for c, ok in zip(ref, strict) if not ok][:12],
            "C_needed_p50_p90_p100": [float(np.percentile(need, q)) for q in (50, 90, 100)] if need else None}
    out["seconds_median"] = {k: float(np.median([r[k] for r in rows])) for k in ("seconds_original", "seconds_simplified", "seconds_estimator")} if rows else None
    print(json.dumps(out, indent=1))
    save(tag, out)
    save(tag + "_rows", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--start", type=int, default=0, help="offset into the validation parts (fresh parts for a final check)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--samples", action="store_true", help="the two bundled sample parts instead")
    ap.add_argument("--refine", action="store_true", help="also solve both parts on a finer mesh, as a closer referee")
    ap.add_argument("--only", help="rows file of an earlier run: rerun just its compared and failed parts")
    ap.add_argument("--tag", help="results name (default estimator_check, or _refined with --refine)")
    a = ap.parse_args()
    if a.samples:
        items = [os.path.join(ROOT, "samples", n + ".step") for n in ("l_bracket", "printer_chassis")]
        tag = "estimator_samples"
    else:
        split = json.load(open(os.path.join(BASE, "train_test.json")))
        tr = split["train"]
        random.Random(0).shuffle(tr)
        items = [os.path.join(BASE, "breps", "step", n + ".stp") for n in tr[4000 + a.start: 4000 + a.start + a.n]]   # validation parts (4000-5500)
        tag = "estimator_check"
    if a.only:
        prev = json.load(open(a.only))
        items = [r["item"] for r in prev if "item" in r and ("cases" in r or r.get("error"))]
    if a.refine:
        os.environ["DEFEAT_REFINE"] = "1"      # inherited by the spawned workers
        tag += "_refined"
    tag = a.tag or tag
    res = pmap_hard("bench.estimator_check:run_one", items, workers=a.workers, timeout=a.timeout, label="bounds")
    report(res, tag)


if __name__ == "__main__":
    main()
