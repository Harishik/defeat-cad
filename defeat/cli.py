"""Batch command line:  python -m defeat.cli parts/*.step --preset static --h 3 --out out/"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from .execute import worker
from .report import build_report
from .session import new_session


def main():
    ap = argparse.ArgumentParser(description="DeFeat batch defeaturing")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--preset", default="static", choices=["static", "modal", "thermal", "cfd"])
    ap.add_argument("--h", type=float, default=None, help="target mesh size (mm); default ≈ part diagonal / 40")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-mesh", action="store_true", help="skip gmsh comparison")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in a.files:
        t = time.time()
        s = new_session(f)
        s.set_policy(preset=a.preset, h=a.h, source="cli")
        s.run(with_mesh=not a.no_mesh)
        rep = build_report(s)
        for p in [*s.exports.values(), rep]:
            shutil.copy(p, a.out)
        g, m = s.metrics["geometry"], s.metrics["mesh"] or {}
        row = {"file": os.path.basename(f), "features": len(s.features), "removed": len(s.result.removed),
               "kept_by_kernel": len(s.result.failed), "faces": g["faces"], "volume_delta_pct": round(g["volume_delta_pct"], 3),
               "max_dev_mm": round(g["deviation"]["max"], 3), "tet_reduction_pct": round(m.get("tet_reduction_pct", 0) or 0, 1),
               "seconds": round(time.time() - t, 1)}
        fd = (s.metrics.get("fea") or {}).get("delta") or {}
        row["tip_disp_delta_pct"] = round(fd["tip_disp_delta_pct"], 2) if fd.get("tip_disp_delta_pct") is not None else None
        row["stress_p99_delta_pct"] = round(fd["vm_p99_delta_pct"], 2) if fd.get("vm_p99_delta_pct") is not None else None
        rows.append(row)
        print(f"{row['file']:<30} {row['removed']:>3}/{row['features']:<3} removed  faces {g['faces'][0]}→{g['faces'][1]}  "
              f"tets −{row['tet_reduction_pct']}%  Δvol {row['volume_delta_pct']:+.2f}%  "
              f"Δdeflection {row['tip_disp_delta_pct']}%  Δstress {row['stress_p99_delta_pct']}%  {row['seconds']}s", flush=True)
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=1)
    worker().close()


if __name__ == "__main__":
    main()
