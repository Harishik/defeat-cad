"""DeFeat end to end on real parts from openly licensed collections it was never trained or tuned on.

No face labels exist for these parts, so this measures what a user would see, not recognition
accuracy: does the part open, what is found and targeted, how much the kernel removes or refuses,
whether the output is a valid solid, whether the stress check runs and what it says, and how long
each step takes. Collections (see docs/open-cad-datasets.md for their licenses):

  nist     NIST MBE PMI test models (no restrictions): one STEP file per part
  freecad  FreeCAD parts library (CC-BY 3.0)
  kicad    KiCad 3D models (CC-BY-SA 4.0): molded connector housings and switches

    python -m bench.real_parts --set nist --workers 3
    python -m bench.real_parts --set freecad --n 150 --workers 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np

from bench.common import DATA, pmap_hard, save, wilson

REAL = os.path.join(DATA, "real")


def nist_items():
    """One file per NIST part: the AP242 export where there is one, else the first STEP found."""
    files = sorted(glob.glob(os.path.join(REAL, "nist", "**", "*.stp"), recursive=True))
    best = {}
    for f in files:
        b = os.path.basename(f).lower()
        m = re.match(r"(nist_(?:ctc|ftc|stc)_\d+)", b)
        key = m.group(1) if m else os.path.splitext(b)[0]
        score = (2 if "ap242" in b else 1 if "geometry only" in f.lower() else 0)
        if key not in best or score > best[key][0]:
            best[key] = (score, f)
    return [f for _, f in sorted(best.values(), key=lambda x: x[1])]


def collection(name, n, seed=0):
    if name == "nist":
        items = nist_items()
    else:
        root = os.path.join(REAL, name)
        if name == "freecad":   # engineering parts only: the library also has architecture, medical, sports...
            roots = [os.path.join(root, "FreeCAD-library", d) for d in ("Mechanical Parts", "Industrial Design")]
        else:
            roots = [root]
        items = sorted({f for r in roots for pat in ("*.step", "*.stp", "*.STEP", "*.STP")
                        for f in glob.glob(os.path.join(r, "**", pat), recursive=True)})
        items = [f for f in items if os.path.getsize(f) < 15 * 1024 * 1024]   # very large assemblies are out of scope
    random.Random(seed).shuffle(items) if name != "nist" else None
    return items[:n] if n else items


def run_one(path):
    import shutil
    import time
    from defeat import fea, occ
    from defeat.session import Session
    t0 = time.time()
    try:
        s = Session(path)
    except Exception as e:  # noqa
        return {"opened": False, "error_open": f"{type(e).__name__}: {e}"[:200]}
    try:
        t_open = time.time() - t0
        todo = sum(1 for d in s.decisions.values() if d.action == "remove")
        row = {"opened": True, "seconds_open": t_open, "faces": len(s.graph.nodes), "solids": len(occ.solids_of(s.shape)),
               "features": len(s.features), "kinds": sorted({f.kind for f in s.features}), "targeted": todo}
        if not todo:
            row["nothing_to_remove"] = True
            return row
        t = time.time()
        s.run(with_mesh=True)
        m = s.metrics or {}
        delta = (m.get("fea") or {}).get("delta")
        row.update({"removed": len(m.get("removed") or []), "failed": len(m.get("failed") or {}),
                    "failed_reasons": sorted(set((m.get("failed") or {}).values()))[:5],
                    "valid": occ.is_valid(s.result.shape), "faces_after": len(occ.faces_of(s.result.shape)),
                    "kernel_seconds": (s.result.seconds if s.result else None), "seconds_run": time.time() - t,
                    "unchecked": bool(m.get("unchecked")), "check_method": ((m.get("fea") or {}).get("method") or ("tet" if delta else None)),
                    "worst_case_pct": fea.worst_case(delta) if delta else None,
                    "guard_triggered": bool((m.get("guard") or {}).get("triggered")),
                    "tet_reduction_pct": (m.get("mesh") or {}).get("tet_reduction_pct")})
        return row
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def report(res, items, tag):
    rows = [r for r in res if "opened" in r]
    opened = [r for r in rows if r["opened"]]
    ran = [r for r in opened if "removed" in r]
    checked = [r for r in ran if r.get("worst_case_pct") is not None]
    tgt = sum(r["targeted"] for r in ran)
    out = {
        "parts": len(items), "opened": wilson(len(opened), len(items)) if items else None, "opened_count": len(opened),
        "crashed_or_timed_out": sum(1 for r in res if r.get("error")),
        "nothing_to_remove": sum(1 for r in opened if r.get("nothing_to_remove")),
        "ran": len(ran),
        "valid_output": wilson(sum(1 for r in ran if r["valid"]), len(ran)) if ran else None,
        "targeted_features": tgt, "removed_features": sum(r["removed"] for r in ran),
        "removed_share": (sum(r["removed"] for r in ran) / tgt) if tgt else None,
        "kernel_refusals": sum(r["failed"] for r in ran),
        "refusal_reasons": sorted({x for r in ran for x in r.get("failed_reasons", [])})[:10],
        "stress_checked": wilson(len(checked), len(ran)) if ran else None,
        "within_5pct_after_guard": wilson(sum(1 for r in checked if r["worst_case_pct"] <= 5.0 + 1e-9), len(checked)) if checked else None,
        "guard_triggered": sum(1 for r in ran if r.get("guard_triggered")),
        "median_seconds": {k: float(np.median([r[k] for r in ran if r.get(k) is not None])) if ran else None
                           for k in ("seconds_open", "kernel_seconds", "seconds_run")},
        "feature_kinds_seen": sorted({k for r in opened for k in r.get("kinds", [])}),
    }
    print(json.dumps(out, indent=1))
    save(tag, out)
    save(tag + "_rows", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="nist", choices=["nist", "freecad", "kicad"])
    ap.add_argument("--n", type=int, default=0, help="at most this many parts (0: all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()
    items = collection(a.set, a.n)
    res = pmap_hard("bench.real_parts:run_one", items, workers=a.workers, timeout=a.timeout, label=a.set)
    report(res, items, f"real_{a.set}")


if __name__ == "__main__":
    main()
