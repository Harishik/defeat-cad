"""How often did the kernel leave a feature in place while the old bookkeeping counted it removed?
Removal only (no mesh), on validation parts. The pipeline now credits a feature only when the kernel
deleted all its faces, and retries the rest one by one; the log says how many a single pass left.

    python -m bench.removal_audit --n 300 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from bench.common import DATA, pmap_hard, save, wilson

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def run_one(name):
    import shutil
    from defeat.session import Session
    s = Session(os.path.join(BASE, "breps", "step", name + ".stp"))
    try:
        todo = sum(1 for d in s.decisions.values() if d.action == "remove")
        if not todo:
            return {"todo": 0}
        s.run(with_mesh=False)
        left = 0
        for line in s.result.log:
            m = re.match(r"Single pass left (\d+) features in place", line)
            if m:
                left = int(m.group(1))
        stayed = sum(1 for v in s.result.failed.values() if "left it in place" in v or "unchanged" in v)
        return {"todo": todo, "removed": len(s.result.removed), "failed": len(s.result.failed),
                "single_pass_left": left, "strategy": s.result.strategy, "left_then_kept": stayed}
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    res = pmap_hard("bench.removal_audit:run_one", tr[4000: 4000 + a.n], workers=a.workers, timeout=600, label="audit")
    rows = [r for r in res if r.get("todo")]
    tot = sum(r["todo"] for r in rows)
    over = sum(r.get("single_pass_left", 0) for r in rows if r.get("strategy", "").startswith("single pass"))
    out = {"parts": a.n, "parts_with_removals": len(rows), "features_to_remove": tot,
           "removed_now": sum(r.get("removed", 0) for r in rows),
           "left_in_place_by_a_single_pass": over,
           "parts_affected": wilson(sum(1 for r in rows if r.get("single_pass_left")), len(rows)),
           "old_overcount_share_of_reported": over / max(1, sum(r.get("removed", 0) for r in rows) + over),
           "errors": sum(1 for r in res if r.get("error"))}
    print(json.dumps(out, indent=1))
    save("removal_audit", out)
    save("removal_audit_rows", res)


if __name__ == "__main__":
    main()
