"""Re-run, once, the parts of a saved full-pipeline run that ended with no stress check, through the
current pipeline (which now falls back to the fixed-grid check). It reports coverage only: these
parts have no tetrahedral reference, so the grid's accuracy comes from bench.immersed_check.

    python -m bench.unchecked_rerun --tag robustness_v11
"""
from __future__ import annotations

import argparse
import ast
import json
import os

from bench.common import RESULTS, pmap_hard, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="robustness_v11")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    rows = json.load(open(os.path.join(RESULTS, a.tag + "_rows.json")))
    items = [ast.literal_eval(r["item"]) for r in rows if r.get("to_remove") and r.get("tets_fea") in (None, 0)]
    res = pmap_hard("bench.robust:run_one", items, workers=a.workers, timeout=a.timeout, label="unchecked")
    out = {"source": a.tag, "parts": len(items),
           "checked_now": sum(1 for r in res if r.get("worst_case_pct") is not None),
           "on_grid": sum(1 for r in res if r.get("check_method") == "grid"),
           "still_unchecked": sum(1 for r in res if r.get("unchecked")),
           "rows": res}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    save(a.tag + "_unchecked_rerun", out)


if __name__ == "__main__":
    main()
