"""Build GNN training graphs from real labelled datasets (train splits only).

MFCAD   : full labels  -> base / pocket (slots, passages, pockets) / step / ignore (MFCAD 'chamfer' = bevel cut)
Fusion  : partial      -> fillet / chamfer exact; every other face is "not a blend" (y = -2)

    python -m bench.build_real --mfcad 3000 --fusion 4000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random

from bench.common import DATA, ROOT, pmap

NOT_BLEND = -2
IGNORE = -1


def _graph_sample(shape, y):
    from defeat.datagen import node_features
    from defeat.graph import build_graph
    g = build_graph(shape)
    if len(g.nodes) != len(y) or len(g.nodes) > 600:
        return None
    X, E, EA = node_features(g)
    return {"x": X, "e": E, "ea": EA, "y": y}


def mfcad_one(path):
    from bench.mfcad import FEAT_NAMES, load_labelled
    from defeat.datagen import LABELS
    shape, y = load_labelled(path)
    m = []
    for v in y:
        n = FEAT_NAMES[v]
        m.append(LABELS.index("base") if n == "stock" else IGNORE if n == "chamfer"
                 else LABELS.index("step") if "step" in n else LABELS.index("pocket"))
    d = _graph_sample(shape, m)
    return {"sample": d, "src": "mfcad"} if d else {"ok": False, "error": "skip"}


def fusion_one(name):
    from bench.fusion import load
    from defeat.datagen import LABELS
    shape, seg = load(name)
    y = [LABELS.index("fillet") if v == 4 else LABELS.index("chamfer") if v == 5 else NOT_BLEND for v in seg]
    d = _graph_sample(shape, y)
    return {"sample": d, "src": "fusion"} if d else {"ok": False, "error": "skip"}


def main():
    from bench.mfcad import DIR, split_of
    from bench.fusion import BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfcad", type=int, default=3000)
    ap.add_argument("--fusion", type=int, default=4000)
    a = ap.parse_args()
    mf = [f for f in sorted(glob.glob(os.path.join(DIR, "*.step"))) if split_of(f) == "train"]
    random.Random(0).shuffle(mf)
    fu = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(fu)
    out = []
    for fn, items, lbl in ((mfcad_one, mf[: a.mfcad], "mfcad"), (fusion_one, fu[: a.fusion], "fusion")):
        res = pmap(fn, items, workers=7, timeout=90, label=lbl)
        out += [r["sample"] for r in res if r.get("ok") and r.get("sample")]
    path = os.path.join(ROOT, "models", "dataset_real.pkl")
    pickle.dump(out, open(path, "wb"))
    print(len(out), "graphs ->", path)


if __name__ == "__main__":
    main()
