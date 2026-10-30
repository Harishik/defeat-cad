"""Fusion 360 Gallery benchmark: 42,912 real designer models with per-face modelling-operation
labels (ExtrudeSide/End, CutSide/End, Fillet, Chamfer, RevolveSide/End). Uses the official test
split. Fillet and chamfer labels map 1:1 to DeFeat classes; 'Cut' faces are reported as an
indicative comparison only (a design cut is not always a small feature).

    python -m bench.fusion --n 1500
"""
from __future__ import annotations

import argparse
import json
import os
import random

from bench.common import DATA, OURS_GROUP, binary_metrics, f1_ci, face_metrics, pmap, rule_face_labels, save

BASE = os.path.join(DATA, "fusion_ext", "s2.0.1_extended_step")
NAMES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]
GT = {0: "base", 1: "base", 2: "cut", 3: "cut", 4: "fillet", 5: "chamfer", 6: "base", 7: "base"}


def load(name):
    from defeat import occ
    shape = occ.read_cad(os.path.join(BASE, "breps", "step", name + ".stp"))
    seg = [int(v) for v in open(os.path.join(BASE, "breps", "seg", name + ".seg")).read().split()]
    return shape, seg


def evaluate_one(name):
    from defeat import occ
    from defeat.graph import build_graph
    from defeat.recognize import Recognizer
    shape, seg = load(name)
    if len(occ.faces_of(shape)) != len(seg):
        return {"ok": False, "error": "face/label count mismatch"}
    g = build_graph(shape)
    feats = Recognizer(g).run()
    out = {"gt": [GT[v] for v in seg], "rule": [OURS_GROUP[k] for k in rule_face_labels(g, feats)]}
    try:
        from defeat.datagen import LABELS
        from defeat.gnn import predict_faces
        P = predict_faces(g)
        out["gnn"] = [OURS_GROUP.get(LABELS[i], "base") for i in P.argmax(1)]
        # exactly what the app ships: defeat.recognize.refine_with_gnn with the tuned thresholds
        from defeat.recognize import FUSION, refine_with_gnn
        out["fused"] = [OURS_GROUP[k] for k in rule_face_labels(g, refine_with_gnn(g, feats, P, LABELS, **FUSION))]
    except Exception:
        pass
    return out


def _counts(ok, who, cls):
    out = []
    for r in ok:
        if who in r:
            z = list(zip(r["gt"], r[who]))
            out.append((sum(1 for g, p in z if g == cls and p == cls), sum(1 for g, p in z if g != cls and p == cls),
                        sum(1 for g, p in z if g == cls and p != cls)))
    return out


def summarize(ok):
    rep = {}
    for who in ("rule", "gnn", "fused"):
        pairs = [(gt, pr) for r in ok if who in r for gt, pr in zip(r["gt"], r[who])]
        if not pairs:
            continue
        # merge cut-like predictions for the comparison with design cuts
        merged = [(gt, "cut" if pr in ("cut", "step") else pr) for gt, pr in pairs]
        rep[who] = {
            "fillet": {**binary_metrics(pairs, {"fillet"}), "f1_ci95": f1_ci(_counts(ok, who, "fillet"))},
            "chamfer": {**binary_metrics(pairs, {"chamfer"}), "f1_ci95": f1_ci(_counts(ok, who, "chamfer"))},
            "cut_indicative": binary_metrics(merged, {"cut"}),
            "per_class": face_metrics([(gt, pr if pr in ("base", "cut", "fillet", "chamfer") else "base") for gt, pr in merged],
                                      ["base", "cut", "fillet", "chamfer"]),
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    test = json.load(open(os.path.join(BASE, "train_test.json")))["test"]
    random.Random(0).shuffle(test)
    names = test[: a.n]
    res = pmap(evaluate_one, names, workers=a.workers, timeout=90, label="fusion")
    ok = [r for r in res if r.get("ok")]
    report = {"dataset": "Fusion 360 Gallery segmentation, extended STEP (official test split)",
              "parts_attempted": len(names), "parts_ok": len(ok),
              "faces": sum(len(r["gt"]) for r in ok),
              "failure_reasons": {}, "seconds_per_part": round(sum(r["seconds"] for r in ok) / max(len(ok), 1), 3)}
    for r in res:
        if not r.get("ok"):
            k = r.get("error", "?").split(":")[0]
            report["failure_reasons"][k] = report["failure_reasons"].get(k, 0) + 1
    report.update(summarize(ok))
    print(save("fusion_test", report))
    for who in ("rule", "gnn", "fused"):
        if who in report:
            print(who, {k: report[who][k] for k in ("fillet", "chamfer", "cut_indicative")})


if __name__ == "__main__":
    main()
