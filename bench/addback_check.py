"""Does pricing features with the one-solve estimate make the stress guard put back the right ones?

On validation parts (never test parts), every part with something removed is run twice through the full
app pipeline, identical except for how the guard prices what to put back:

  field     : the shipped price, the original's strain energy around each feature
  estimate  : the one-solve estimate (filled-in energy + Antolin-Chanon estimator) where it applies,
              the field price elsewhere

Only parts where the guard triggers differ. For those, reported per variant: how many features it had to
put back (fewer is better: more of the simplification survives, and both always end within tolerance),
how many trial solves and seconds the guard took, and the final worst load case.

Pre-registered decision (written before the fresh run): "estimate" becomes the default only if, on the
fresh parts, it puts back no more features than "field" in the median AND needs no more trial solves on
average. Otherwise "field" stays and the result is reported as it is.

    python -m bench.addback_check --start 0 --n 75 --tag addback_dev        # development parts
    python -m bench.addback_check --start 75 --n 300 --tag addback_fresh    # scored once
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from bench.common import DATA, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def run_one(path):
    import shutil
    import time
    from defeat import fea
    from defeat.session import Session
    out = {}
    for mode in ("field", "estimate"):
        s = Session(path)
        try:
            if mode == "field" and not any(d.action == "remove" for d in s.decisions.values()):
                return {"skip": "nothing to remove"}
            s.addback = mode
            s._est_used = 0
            t = time.time()
            s.run(with_mesh=True)
            m = s.metrics or {}
            g = m.get("guard") or {}
            out[mode] = {"triggered": bool(g.get("triggered")), "restored": g.get("restored"), "of": g.get("of"),
                         "trials": g.get("trials"), "after_pct": g.get("after_pct"), "before_pct": g.get("before_pct"),
                         "undone": g.get("undone"), "removed": len(m.get("removed") or []), "estimated": s._est_used,
                         "worst_final": fea.worst_case((m.get("fea") or {}).get("delta")) if m.get("fea") else None,
                         "seconds": time.time() - t, "unchecked": m.get("unchecked")}
        finally:
            shutil.rmtree(s.dir, ignore_errors=True)
    if not out["field"]["removed"] and not out["field"]["triggered"]:
        out["note"] = "nothing removed"
    return out


def report(res, tag):
    both = [r for r in res if "field" in r and "estimate" in r]
    trig = [r for r in both if r["field"]["triggered"] and r["estimate"]["triggered"]]
    used = [r for r in trig if r["estimate"]["estimated"]]
    def med(rs, mode, k):
        v = [r[mode][k] for r in rs if r[mode].get(k) is not None]
        return float(np.median(v)) if v else None
    def mean(rs, mode, k):
        v = [r[mode][k] for r in rs if r[mode].get(k) is not None]
        return float(np.mean(v)) if v else None
    out = {"parts": len(res), "ran_both": len(both), "guard_triggered": len(trig), "estimate_used_on": len(used),
           "skipped": sum(1 for r in res if r.get("skip")), "errors": sum(1 for r in res if r.get("error"))}
    for name, rs in (("triggered", trig), ("estimate_used", used)):
        if not rs:
            continue
        fewer = sum(1 for r in rs if (r["estimate"]["restored"] or 0) < (r["field"]["restored"] or 0))
        more = sum(1 for r in rs if (r["estimate"]["restored"] or 0) > (r["field"]["restored"] or 0))
        out[name] = {
            "parts": len(rs),
            "restored_median": {m: med(rs, m, "restored") for m in ("field", "estimate")},
            "restored_mean": {m: mean(rs, m, "restored") for m in ("field", "estimate")},
            "estimate_puts_back_fewer": fewer, "estimate_puts_back_more": more,
            "trials_mean": {m: mean(rs, m, "trials") for m in ("field", "estimate")},
            "seconds_median": {m: med(rs, m, "seconds") for m in ("field", "estimate")},
            "undone": {m: sum(1 for r in rs if r[m].get("undone")) for m in ("field", "estimate")},
            "final_worst_median": {m: med(rs, m, "worst_final") for m in ("field", "estimate")},
            "within_tol_estimate": wilson(sum(1 for r in rs if (r["estimate"]["worst_final"] or 0) <= 5.0 + 1e-9), len(rs))}
    if "triggered" in out:
        t = out["triggered"]
        out["decision"] = ("estimate" if t["restored_median"]["estimate"] <= t["restored_median"]["field"]
                           and t["trials_mean"]["estimate"] <= t["trials_mean"]["field"] else "field")
    print(json.dumps(out, indent=1))
    save(tag, out)
    save(tag + "_rows", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--tag", default="addback_dev")
    a = ap.parse_args()
    split = json.load(open(os.path.join(BASE, "train_test.json")))
    tr = split["train"]
    random.Random(0).shuffle(tr)
    items = [os.path.join(BASE, "breps", "step", n + ".stp") for n in tr[4000 + a.start: 4000 + a.start + a.n]]
    res = pmap_hard("bench.addback_check:run_one", items, workers=a.workers, timeout=a.timeout, label="addback")
    report(res, a.tag)


if __name__ == "__main__":
    main()
