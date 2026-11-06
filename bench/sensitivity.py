"""Which removals change a part's stiffness? On validation parts (never test parts): solve the
original, the full removal, and the full removal with one feature class put back at a time.

    python -m bench.sensitivity --n 400
"""
from __future__ import annotations

import argparse
import json
import os
import random

from bench.common import DATA, pmap_hard, save

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")


def _solve(data, h, bc):
    from defeat import fea
    from defeat.validate import mesh_stats
    st = mesh_stats(data, h, keep_mesh=True)
    if not st.get("ok"):
        return {"ok": False, "error": str(st.get("error"))[:80]}
    return fea.solve_mesh(data, st, h, bc)


def worst(a, b):
    return max(abs(100 * (b["cases"][k] - a["cases"][k]) / a["cases"][k]) for k in a["cases"])


def run_one(name):
    import shutil
    from defeat import fea, occ
    from defeat.decide import Decision
    from defeat.execute import defeature
    from defeat.session import Session
    from defeat.validate import step_bytes
    s = Session(os.path.join(BASE, "breps", "step", name + ".stp"))
    try:
        rm = [f for f in s.features if s.decisions[f.id].action == "remove"]
        if not rm:
            return {"skip": True}
        er = defeature(s.shape, s.features, s.decisions, graph=s.graph)
        if not er.removed:
            return {"skip": True}
        h, bc = s.policy.h, fea.load_case(occ.bbox(s.shape))
        a = _solve(step_bytes(s.shape), h, bc)
        if not a.get("ok"):
            return {"skip": True, "why": a.get("error")}
        fea.lock(bc, a)
        b = _solve(step_bytes(er.shape), h, bc)
        if not b.get("ok"):
            return {"skip": True, "why": b.get("error")}
        v0 = occ.volume(s.shape)
        out = {"worst": worst(a, b), "bend": abs(100 * (b["cases"]["bend_thin"] / a["cases"]["bend_thin"] - 1)),
               "vol_pct": abs(100 * (occ.volume(er.shape) - v0) / v0), "dofs": a["dofs"], "feats": [],
               "failed": len(er.failed), "cases": {k: 100 * (b["cases"][k] / a["cases"][k] - 1) for k in a["cases"]}}
        L = s.graph.char_len
        removed = [f for f in rm if f.id in er.removed]
        for f in removed:
            out["feats"].append({"kind": f.kind, "size_h": f.size / h, "size_L": f.size / L, "source": f.source,
                                 "area": sum(s.graph.nodes[x].area for x in f.faces) / s.graph.total_area,
                                 "faces": len(f.faces)})
        if out["worst"] > 5:
            out["back"] = {}
            for kind in sorted({f.kind for f in removed}):
                dec = {i: (Decision(i, "keep", "") if s.features[i].kind == kind else d) for i, d in s.decisions.items()}
                r = _solve(step_bytes(defeature(s.shape, s.features, dec, graph=s.graph).shape), h, bc)
                if r.get("ok"):
                    out["back"][kind] = worst(a, r)
        return out
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="sensitivity_rows")
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    res = pmap_hard("bench.sensitivity:run_one", tr[4000: 4000 + a.n], workers=a.workers, timeout=1200, label="sens")
    print(save(a.out, [r for r in res if r.get("ok") and not r.get("skip")]))


if __name__ == "__main__":
    main()
