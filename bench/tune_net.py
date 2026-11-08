"""Tune how rules and the network combine, on validation parts the network never trained on.

    python -m bench.tune_net
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random

import numpy as np

from bench.common import OURS_GROUP, pmap, rule_face_labels, save
from bench.fusion import BASE, GT, load

VETO = [0.1, 0.3, 0.5, 0.7, 0.9]
ADD = [0.4, 0.6, 0.8]
OVERRIDE = [0.5, 0.7, 0.9]
MACHINED = [0.99, 1.01]
CUT = [0.5, 0.7, 1.01]
STEP = [0.5, 0.7]
GRID = list(itertools.product(VETO, ADD, OVERRIDE, MACHINED, CUT, STEP))


CLASSES = ("fillet", "chamfer", "cut")
FUSION_GROUP = {0: "base", 1: "base", 2: "cut", 3: "cut", 4: "fillet", 5: "chamfer", 6: "base", 7: "base"}


def counts(gt, pred):
    out = {}
    for cls in ("fillet", "chamfer", "cut"):
        z = [(a, "cut" if b in ("cut", "step") else b) for a, b in zip(gt, pred)]
        out[cls] = (sum(1 for a, b in z if a == cls and b == cls), sum(1 for a, b in z if a != cls and b == cls),
                    sum(1 for a, b in z if a == cls and b != cls))
    return out


def one(name):
    import torch
    torch.set_num_threads(1)
    from defeat.datagen import LABELS
    from defeat.graph import build_graph
    from defeat.net import predict_part
    from defeat.recognize import Recognizer, refine_with_net
    shape, seg = load(name)
    g = build_graph(shape)
    if len(g.nodes) != len(seg):
        return {"ok": False, "error": "mismatch"}
    gt = [GT[v] for v in seg]
    feats = Recognizer(g).run()
    P, P8 = predict_part(g)
    res = np.zeros((len(GRID), 3, 3), "int32")
    for i, (veto, add, over, mach, cut, step) in enumerate(GRID):
        f = refine_with_net(g, copy.deepcopy(feats), P, LABELS, veto, add, over, mach, P8=P8, cut=cut, step_veto=step)
        c = counts(gt, [OURS_GROUP[k] for k in rule_face_labels(g, f)])
        res[i] = [c[k] for k in CLASSES]
    # what each source scores alone, for comparison
    alone = {"rules": counts(gt, [OURS_GROUP[k] for k in rule_face_labels(g, feats)]),
             "network": counts(gt, [FUSION_GROUP[int(i)] for i in P8.argmax(1)])}
    return {"res": res, "alone": alone}


def f1(c):
    return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    names = tr[4000:5500][:a.n]
    res = [r for r in pmap(one, names, workers=a.workers, timeout=300, label="tune") if r.get("ok")]
    total = sum(r["res"] for r in res)
    rows = [{"combo": combo, **{c: round(f1(total[i, j]), 4) for j, c in enumerate(CLASSES)}} for i, combo in enumerate(GRID)]
    alone = {}
    for who in ("rules", "network"):
        alone[who] = {c: round(f1([sum(r["alone"][who][c][j] for r in res) for j in range(3)]), 4) for c in CLASSES}
    print(alone)
    rows.sort(key=lambda r: -(r["fillet"] + r["chamfer"] + r["cut"]))
    print(save("net_tuning", {"parts": len(res), "alone": alone, "best": rows[:15], "keys": ["veto", "add", "override", "machined", "cut", "step_veto"]}))
    for r in rows[:10]:
        print(r)


if __name__ == "__main__":
    main()
