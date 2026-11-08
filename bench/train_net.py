"""Train and compare recognition networks. Model selection uses validation parts only.

    python -m bench.train_net --name full --epochs 40
    python -m bench.train_net --name no_uv --no-uv --limit 12000 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from bench.common import DATA, ROOT
from defeat.datagen import LABELS
from defeat.net import FUSION_LABELS, BRepRecognizer, collate, load_shards

NOT_BLEND = -2
FI, CI = LABELS.index("fillet"), LABELS.index("chamfer")
OUT = os.path.join(DATA, "brep_v2")


def prepare(d):
    """Per-face targets for both heads. Fusion: all eight labels, plus fillet/chamfer on the class head."""
    n = d["x"].shape[0]
    if d["src"] == "fusion":
        y8 = np.asarray(d["y8"], "int64")
        d["y88"] = y8
        d["yy"] = np.where(y8 == 4, FI, np.where(y8 == 5, CI, NOT_BLEND))
    else:
        d["y88"] = np.full(n, -1)
        d["yy"] = np.asarray(d["y"], "int64")
    return d


def batches(parts, max_faces, shuffle, seed=0):
    idx = list(range(len(parts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    cur, nf = [], 0
    for i in idx:
        k = parts[i]["x"].shape[0]
        if cur and nf + k > max_faces:
            yield cur
            cur, nf = [], 0
        cur.append(parts[i])
        nf += k
    if cur:
        yield cur


def loss_fn(l9, l8, b, w8):
    loss = 0.0
    y8 = b["y8"]
    m = y8 >= 0
    if m.any():
        loss = loss + F.cross_entropy(l8[m], y8[m], weight=w8)
    y = b["y"]
    full = y >= 0
    if full.any():
        loss = loss + F.cross_entropy(l9[full], y[full])
    nb = y == NOT_BLEND
    if nb.any():
        p = torch.softmax(l9[nb], 1)
        loss = loss - torch.log((1 - p[:, FI] - p[:, CI]).clamp_min(1e-6)).mean()
    return loss


@torch.no_grad()
def predict(model, parts, device="cpu", max_faces=6000):
    model.eval()
    P9, P8, Y8 = [], [], []
    for grp in batches(parts, max_faces, False):
        b = collate(grp, device, None)
        l9, l8 = model(b)
        P9.append(torch.softmax(l9, 1).cpu().numpy())
        P8.append(torch.softmax(l8, 1).cpu().numpy())
        Y8.append(b["y8"].cpu().numpy())
    return np.concatenate(P9), np.concatenate(P8), np.concatenate(Y8)


def scores(P9, P8, y8):
    pred = P8.argmax(1)
    m = y8 >= 0
    acc = float((pred[m] == y8[m]).mean())
    ious = {}
    for c, name in enumerate(FUSION_LABELS):
        tp = int(((pred == c) & (y8 == c)).sum())
        fp = int(((pred == c) & (y8 != c) & m).sum())
        fn = int(((pred != c) & (y8 == c)).sum())
        ious[name] = round(tp / max(tp + fp + fn, 1), 4)

    def f1(p, g):
        tp, fp, fn = (p & g).sum(), (p & ~g & m).sum(), (~p & g).sum()
        return round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4)
    pcut = np.isin(pred, [2, 3])
    gcut = np.isin(y8, [2, 3])
    out = {"accuracy": round(acc, 4), "miou": round(float(np.mean(list(ious.values()))), 4), "iou": ious,
           "fillet_f1_8": f1(pred == 4, y8 == 4), "chamfer_f1_8": f1(pred == 5, y8 == 5), "cut_f1_8": f1(pcut, gcut)}
    a9 = P9.argmax(1)
    out["fillet_f1_9"] = f1(a9 == FI, y8 == 4)
    out["chamfer_f1_9"] = f1(a9 == CI, y8 == 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="train on this many parts (experiments)")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--attn", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-faces", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-uv", action="store_true")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    # generated parts with a lot of embossed text are many faces of one kind: real parts should set the pace
    tr = [prepare(d) for d in load_shards(OUT, {"train"}) if d["src"] != "synth" or d["x"].shape[0] <= 120]
    va = [prepare(d) for d in load_shards(OUT, {"val"})]
    random.Random(a.seed).shuffle(tr)
    if a.limit:
        tr = tr[: a.limit]
    print(f"train {len(tr)} val {len(va)} loaded in {time.time() - t0:.0f}s", flush=True)
    ys = np.concatenate([d["y88"] for d in tr])
    cnt = np.bincount(ys[ys >= 0], minlength=8).astype("float64")
    w8 = torch.tensor(((cnt.sum() / (cnt + 1)) ** 0.5 / ((cnt.sum() / (cnt + 1)) ** 0.5).mean()), dtype=torch.float32, device=a.device)
    model = BRepRecognizer(h=a.h, layers=a.layers, attn=a.attn, uv=not a.no_uv, ctx=not a.no_ctx, hints=not a.no_hints, dropout=0.1).to(a.device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_batches = sum(1 for _ in batches(tr, a.max_faces, True))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=int(n_batches * 1.03) * a.epochs + 50, pct_start=0.1)
    best, hist = -1, []
    path = os.path.join(ROOT, "models", f"net_{a.name}.pt")
    for ep in range(a.epochs):
        model.train()
        tot, nb, t1 = 0.0, 0, time.time()
        for grp in batches(tr, a.max_faces, True, seed=a.seed + ep):
            b = collate(grp, a.device, None if a.no_rotate else rng)
            l9, l8 = model(b)
            loss = loss_fn(l9, l8, b, w8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        s = scores(*predict(model, va, a.device))
        s.update(epoch=ep, loss=round(tot / nb, 4), seconds=round(time.time() - t1))
        hist.append(s)
        print(json.dumps({k: s[k] for k in ("epoch", "loss", "accuracy", "miou", "fillet_f1_8", "chamfer_f1_8", "cut_f1_8", "fillet_f1_9", "chamfer_f1_9", "seconds")}), flush=True)
        sel = s["accuracy"] + s["miou"] + s["fillet_f1_8"] + s["chamfer_f1_8"]
        if sel > best:
            best = sel
            torch.save({"state": model.state_dict(), "cfg": model.cfg, "epoch": ep, "val": s}, path)
        json.dump({"args": vars(a), "history": hist}, open(os.path.join(ROOT, "bench", "results", f"net_{a.name}.json"), "w"), indent=1)
    print("best", json.dumps(torch.load(path)["val"]), flush=True)


if __name__ == "__main__":
    main()
