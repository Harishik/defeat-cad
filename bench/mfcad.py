"""MFCAD benchmark: face-level accuracy of the rule recognizer and the GNN on 15,488 labelled
machining-feature parts (Cao et al. 2020, https://github.com/hducg/MFCAD, MIT license).

    python -m bench.mfcad --n 2000
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random

from bench.common import DATA, OURS_GROUP, fuse, binary_metrics, face_metrics, pmap, rule_face_labels, save

FEAT_NAMES = ['rectangular_through_slot', 'triangular_through_slot',
              'rectangular_passage', 'triangular_passage', '6sides_passage',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step',
              'rectangular_blind_step', 'triangular_blind_step',
              'rectangular_blind_slot',
              'rectangular_pocket', 'triangular_pocket', '6sides_pocket',
              'chamfer', 'stock']
# MFCAD class -> evaluation group (steps are open features with no counterpart class in DeFeat)
GT_GROUP = {i: ("chamfer" if n == "chamfer" else "base" if n == "stock" else "step" if "step" in n else "cut")
            for i, n in enumerate(FEAT_NAMES)}
DIR = os.path.join(DATA, "mfcad", "dataset", "step")


def load_labelled(path):
    from OCP.STEPControl import STEPControl_Reader
    from defeat import occ
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    tr = r.WS().TransferReader()
    faces = occ.faces_of(shape)
    ids = [int(tr.EntityFromShapeResult(f, 1).Name().ToCString()) for f in faces]
    lab = pickle.load(open(path[:-5] + ".face_truth", "rb"))
    return shape, [lab[i] for i in ids]


def split_of(path):
    """Deterministic 80/20 split by file name hash (stable across runs and machines)."""
    import zlib
    return "test" if zlib.crc32(os.path.basename(path).encode()) % 5 == 0 else "train"


def evaluate_one(path):
    from defeat.graph import build_graph
    from defeat.recognize import Recognizer
    shape, y = load_labelled(path)
    g = build_graph(shape)
    feats = Recognizer(g).run()
    rule = [OURS_GROUP[k] for k in rule_face_labels(g, feats)]
    out = {"gt": [GT_GROUP[v] for v in y], "rule": rule, "faces": len(y)}
    try:
        from defeat.gnn import predict_faces
        from defeat.datagen import LABELS
        P = predict_faces(g)
        out["gnn"] = [OURS_GROUP.get(LABELS[i], "base") for i in P.argmax(1)]
        out["fused"] = fuse(rule, P, LABELS)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(DIR, "*.step")))
    files = [f for f in files if split_of(f) == a.split]
    random.Random(0).shuffle(files)
    files = files[: a.n]
    res = pmap(evaluate_one, files, workers=a.workers, timeout=60, label="mfcad")
    ok = [r for r in res if r.get("ok")]
    report = {"dataset": "MFCAD (15,488 machining parts, 16 face classes)", "split": a.split,
              "parts_attempted": len(files), "parts_ok": len(ok),
              "failures": [r for r in res if not r.get("ok")][:20],
              "seconds_per_part": round(sum(r["seconds"] for r in ok) / max(len(ok), 1), 3)}
    for who in ("rule", "gnn", "fused"):
        pairs = [(g, p) for r in ok if who in r for g, p in zip(r["gt"], r[who])]
        if not pairs:
            continue
        report[who] = {
            "feature_vs_stock": binary_metrics(pairs, {"cut", "step", "chamfer", "fillet", "protrusion"}),
            "per_class": face_metrics(pairs, ["base", "cut", "step", "chamfer"]),
            "step_faces_flagged_as_feature": round(
                sum(1 for g, p in pairs if g == "step" and p != "base") / max(1, sum(1 for g, _ in pairs if g == "step")), 4),
        }
    print(save(f"mfcad_{a.split}", report))
    for who in ("rule", "gnn", "fused"):
        if who in report:
            r = report[who]
            print(who, "feature-vs-stock", r["feature_vs_stock"], "\n   per-class", {k: v for k, v in r["per_class"].items() if k != "confusion"})


if __name__ == "__main__":
    main()
