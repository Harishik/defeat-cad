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

from bench.common import OURS_GROUP, binary_metrics, fuse, pmap, rule_face_labels, save
from bench.fusion import BASE, GT, load


def one(name):
    from defeat.datagen import LABELS
    from defeat.gnn import predict_faces
    from defeat.graph import build_graph
    from defeat.recognize import Recognizer
    shape, seg = load(name)
    g = build_graph(shape)
    if len(g.nodes) != len(seg):
        return {"ok": False, "error": "mismatch"}
    rule = [OURS_GROUP[k] for k in rule_face_labels(g, Recognizer(g).run())]
    P = predict_faces(g)
    return {"gt": [GT[v] for v in seg], "rule": rule, "P": P.tolist()}


def main():
    from defeat.datagen import LABELS
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    val = tr[4000: 4000 + a.n]
    res = [r for r in pmap(one, val, workers=4, timeout=90, label="val") if r.get("ok")]
    best, table = None, []
    for veto, add in itertools.product([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]):
        pairs = []
        for r in res:
            pred = fuse(r["rule"], np.asarray(r["P"]), LABELS, veto=veto, add=add)
            pairs += list(zip(r["gt"], pred))
        f = binary_metrics(pairs, {"fillet"})["f1"]
        c = binary_metrics(pairs, {"chamfer"})["f1"]
        table.append({"veto": veto, "add": add, "fillet_f1": f, "chamfer_f1": c, "mean": round((f + c) / 2, 4)})
        if best is None or table[-1]["mean"] > best["mean"]:
            best = table[-1]
    out = {"validation_parts": len(res), "best": best, "grid": table}
    print(save("fusion_tuning", out))
    print("best", best)


if __name__ == "__main__":
    main()
