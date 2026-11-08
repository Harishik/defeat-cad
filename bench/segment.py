"""Per-face segmentation score for the Fusion 360 Gallery's eight modelling labels, the measure used in
the published work on this dataset. Parts, not faces, are resampled for the 95% intervals.

    python -m bench.segment --split val --models models/net_m1.pt,models/net_m2.pt
    python -m bench.segment --split test
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from bench.common import ROOT, save
from bench.train_net import OUT, batches, prepare
from defeat.net import FUSION_LABELS, BRepRecognizer, collate, load_shards

CUT = [2, 3]


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = BRepRecognizer(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def probabilities(model, parts, seed=None, max_faces=6000):
    """Fusion-label probabilities for every face, in part order. With a seed, each part gets a random rotation."""
    rng = np.random.default_rng(seed) if seed is not None else None
    out = []
    for grp in batches(parts, max_faces, False):
        out.append(torch.softmax(model(collate(grp, "cpu", rng))[1], 1).numpy())
    return np.concatenate(out)


def confusion(parts, P):
    """One 8x8 table per part, so the unit of resampling stays the part."""
    C = np.zeros((len(parts), 8, 8), "int64")
    pred = P.argmax(1)
    k = 0
    for i, d in enumerate(parts):
        n = d["x"].shape[0]
        np.add.at(C[i], (d["y88"], pred[k: k + n]), 1)
        k += n
    return C


def _f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def metrics(M):
    """Accuracy, mean IoU and per-class scores from a summed confusion table (rows are the true labels)."""
    tp = np.diag(M).astype("float64")
    fp = M.sum(0) - tp
    fn = M.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    cut_tp = M[np.ix_(CUT, CUT)].sum()
    return {"accuracy": float(tp.sum() / max(M.sum(), 1)), "miou": float(iou.mean()),
            "fillet_f1": float(_f1(tp[4], fp[4], fn[4])), "chamfer_f1": float(_f1(tp[5], fp[5], fn[5])),
            "cut_f1": float(_f1(cut_tp, M[:, CUT].sum() - cut_tp, M[CUT, :].sum() - cut_tp)),
            "iou": {n: round(float(v), 4) for n, v in zip(FUSION_LABELS, iou)}}


def report(C, b=1000, seed=0):
    """Scores with part-level bootstrap intervals."""
    rng = np.random.default_rng(seed)
    full = metrics(C.sum(0))
    keys = ("accuracy", "miou", "fillet_f1", "chamfer_f1", "cut_f1")
    runs = {k: [] for k in keys}
    for _ in range(b):
        w = np.bincount(rng.integers(0, len(C), len(C)), minlength=len(C))
        m = metrics((C * w[:, None, None]).sum(0))
        for k in keys:
            runs[k].append(m[k])
    out = {k: round(full[k], 4) for k in keys}
    out["ci95"] = {k: [round(float(np.percentile(runs[k], 2.5)), 4), round(float(np.percentile(runs[k], 97.5)), 4)] for k in keys}
    out["iou"] = full["iou"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", default="", help="comma-separated checkpoints; default is the shipped ensemble")
    ap.add_argument("--no-ensemble", action="store_true", help="score each model alone only")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    paths = a.models.split(",") if a.models else sorted(glob.glob(os.path.join(ROOT, "models", "recognizer_*.pt")))
    t0 = time.time()
    parts = [prepare(d) for d in load_shards(OUT, {a.split}) if d["src"] == "fusion"]
    if a.limit:
        parts = parts[: a.limit]
    print(f"{len(parts)} {a.split} parts, {sum(d['x'].shape[0] for d in parts)} faces, loaded in {time.time() - t0:.0f}s", flush=True)
    res = {"split": a.split, "parts": len(parts), "faces": int(sum(d["x"].shape[0] for d in parts)), "models": {}}
    plain, turned = [], []
    for path in paths:
        name = os.path.basename(path)
        m = load(path)
        plain.append(probabilities(m, parts))
        turned.append(probabilities(m, parts, seed=7))
        res["models"][name] = {"as_stored": report(confusion(parts, plain[-1])), "rotated": report(confusion(parts, turned[-1]))}
        print(name, {k: res["models"][name]["as_stored"][k] for k in ("accuracy", "miou", "fillet_f1", "chamfer_f1", "cut_f1")},
              "rotated", {k: res["models"][name]["rotated"][k] for k in ("accuracy", "miou")}, flush=True)
    if len(paths) > 1 and not a.no_ensemble:
        res["ensemble"] = {"as_stored": report(confusion(parts, sum(plain) / len(plain))),
                           "rotated": report(confusion(parts, sum(turned) / len(turned)))}
        print("ensemble", res["ensemble"]["as_stored"], "\nrotated", res["ensemble"]["rotated"], flush=True)
    if a.out or a.split == "test":
        print(save(a.out or "segment_test", res))


if __name__ == "__main__":
    main()
