"""Tune the rules+GNN combination thresholds on a VALIDATION set (Fusion train-split parts the GNN
never saw: indices 4000+ of the shuffled train list). The test split is never touched here.

    python -m bench.tune_fusion --n 400
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random

from bench.common import OURS_GROUP, pmap, rule_face_labels, save
from bench.fusion import BASE, GT, load


VETO = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ADD = [0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
OVERRIDE = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]   # 1.01 = off


def one(name):
    """Score every threshold combination on one part with the shipped refine_with_gnn."""
    import copy
    from defeat.datagen import LABELS
    from defeat.gnn import predict_faces
    from defeat.graph import build_graph
    from defeat.recognize import Recognizer, refine_with_gnn
    shape, seg = load(name)
    g = build_graph(shape)
    if len(g.nodes) != len(seg):
        return {"ok": False, "error": "mismatch"}
    gt = [GT[v] for v in seg]
    feats = Recognizer(g).run()
    P = predict_faces(g)
    out = {}
    gnn = [OURS_GROUP.get(LABELS[i], "base") for i in P.argmax(1)]
    for combo in [None] + list(itertools.product(VETO, ADD, OVERRIDE)):
        pred = gnn if combo is None else \
            [OURS_GROUP[k] for k in rule_face_labels(g, refine_with_gnn(g, copy.deepcopy(feats), P, LABELS, *combo))]
        c = []
        for cls in ("fillet", "chamfer"):
            z = list(zip(gt, pred))
            c.append((sum(1 for a, b in z if a == cls and b == cls), sum(1 for a, b in z if a != cls and b == cls),
                      sum(1 for a, b in z if a == cls and b != cls)))
        out["gnn" if combo is None else "%s|%s|%s" % combo] = c
    return {"counts": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="fusion_tuning")
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    val = tr[4000: 4000 + a.n]
    res = [r for r in pmap(one, val, workers=6, timeout=300, label="val") if r.get("ok")]
    best, table = None, []
    for key in res[0]["counts"]:
        f1 = []
        for k in range(2):
            tp = sum(r["counts"][key][k][0] for r in res)
            fp = sum(r["counts"][key][k][1] for r in res)
            fn = sum(r["counts"][key][k][2] for r in res)
            f1.append(round(2 * tp / max(2 * tp + fp + fn, 1), 4))
        if key == "gnn":
            gnn_only = {"fillet_f1": f1[0], "chamfer_f1": f1[1]}
            continue
        veto, add, ov = map(float, key.split("|"))
        table.append({"veto": veto, "add": add, "override": ov, "fillet_f1": f1[0], "chamfer_f1": f1[1],
                      "mean": round(sum(f1) / 2, 4)})
        if best is None or table[-1]["mean"] > best["mean"]:
            best = table[-1]
    from defeat.gnn import MODEL_PATH
    out = {"validation_parts": len(res), "model": os.path.basename(MODEL_PATH), "gnn_only": gnn_only,
           "best": best, "grid": table}
    print(save(a.out, out))
    print("gnn only", gnn_only, "\nbest", best)


if __name__ == "__main__":
    main()
