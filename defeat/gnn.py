"""Edge-aware graph neural network for face-level feature classification (PyTorch Geometric).

Trained on procedurally generated mold parts plus real labelled CAD (MFCAD, Fusion 360 Gallery
train split, see bench/build_real.py). At runtime it
gives every rule-based feature instance a second, learned opinion: class probabilities,
agreement with the rules and a confidence that the UI shows next to each feature.
"""
from __future__ import annotations

import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

from .datagen import LABELS, node_features

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("DEFEAT_GNN_MODEL") or os.path.join(HERE, "models", "gnn.pt")
METRICS_PATH = os.path.join(HERE, "models", "gnn_metrics.json")

# rule class -> GNN label that counts as agreement
AGREE = {"hole": {"hole"}, "slot": {"hole", "pocket"}, "pocket": {"pocket", "hole"}, "step": {"step", "pocket"}, "boss": {"boss"},
         "rib": {"rib"}, "emboss": {"emboss"}, "fillet": {"fillet"}, "chamfer": {"chamfer", "fillet"},
         "protrusion": {"boss", "rib", "emboss"}}


class FaceGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=96, layers=4, n_cls=len(LABELS)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_cls))

    def forward(self, x, edge_index, edge_attr):
        h = self.inp(x)
        h0 = h
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, edge_index, edge_attr)))
        return self.head(torch.cat([h, h0], dim=1))


def to_data(d):
    return Data(x=torch.tensor(d["x"]), edge_index=torch.tensor(d["e"]), edge_attr=torch.tensor(d["ea"]),
                y=torch.tensor(d["y"], dtype=torch.long))


NOT_BLEND_WEIGHT = float(os.environ.get("DEFEAT_NOT_BLEND_WEIGHT", 1.0))


def _loss(logits, y, w):
    """Cross-entropy on labelled faces (y >= 0), ignore y == -1, and for partially labelled faces
    (y == -2, 'not a fillet or chamfer') push probability mass away from the blend classes. The
    weight on those faces sets the balance between finding every blend and raising no false alarms."""
    full = y >= 0
    loss = F.cross_entropy(logits[full], y[full], weight=w) if full.any() else logits.sum() * 0
    nb = y == -2
    if nb.any():
        p = torch.softmax(logits[nb], 1)
        p_blend = p[:, LABELS.index("fillet")] + p[:, LABELS.index("chamfer")]
        loss = loss + NOT_BLEND_WEIGHT * (-torch.log((1 - p_blend).clamp_min(1e-6))).mean()
    return loss


def train(dataset_path, epochs=60, seed=0, extra_paths=(), hidden=96, layers=4, batch=16, out=None):
    torch.manual_seed(seed)
    random.seed(seed)
    data = [to_data(d) for d in pickle.load(open(dataset_path, "rb"))]
    random.shuffle(data)
    n_test = max(1, len(data) // 5)
    test, tr = data[:n_test], data[n_test:]
    for p in extra_paths:   # real datasets: train split only (held-out evaluation lives in bench/)
        tr += [to_data(d) for d in pickle.load(open(p, "rb"))]
    random.shuffle(tr)
    ys = torch.cat([d.y for d in tr])
    ys = ys[ys >= 0]
    counts = torch.bincount(ys, minlength=len(LABELS)).float()
    w = (counts.sum() / (counts + 1)) ** 0.5
    w = w / w.mean()
    model = FaceGNN(tr[0].x.shape[1], tr[0].edge_attr.shape[1], hidden=hidden, layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(tr, batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot = 0
        for b in loader:
            opt.zero_grad()
            loss = _loss(model(b.x, b.edge_index, b.edge_attr), b.y, w)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} loss {tot / len(loader):.4f}", flush=True)
    metrics = evaluate(model, test)
    metrics.update(n_train=len(tr), n_test=len(test), params=sum(p.numel() for p in model.parameters()),
                   train_faces=int(len(ys)))
    out = out or MODEL_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": model.state_dict(), "in_dim": tr[0].x.shape[1], "edge_dim": tr[0].edge_attr.shape[1],
                "hidden": hidden, "layers": layers}, out)
    json.dump(metrics, open(os.path.splitext(out)[0] + "_metrics.json", "w", encoding="utf-8"), indent=1)
    return metrics


@torch.no_grad()
def evaluate(model, test):
    model.eval()
    P, Y = [], []
    for d in test:
        P.append(model(d.x, d.edge_index, d.edge_attr).argmax(1))
        Y.append(d.y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    k = len(LABELS)
    cm = np.zeros((k, k), int)
    for p, y in zip(P, Y):
        cm[y, p] += 1
    per = {}
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        per[name] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "support": int(cm[i].sum())}
    return {"accuracy": round(float((P == Y).mean()), 4), "per_class": per, "confusion": cm.tolist(), "labels": LABELS}


_MODEL = None


def load():
    global _MODEL
    if _MODEL is None:
        ck = torch.load(MODEL_PATH, map_location="cpu")
        m = FaceGNN(ck["in_dim"], ck["edge_dim"], hidden=ck.get("hidden", 96), layers=ck.get("layers", 4))
        m.load_state_dict(ck["state"])
        m.eval()
        _MODEL = m
    return _MODEL


@torch.no_grad()
def predict_faces(g):
    m = load()
    X, E, EA = node_features(g)
    logits = m(torch.tensor(X), torch.tensor(E), torch.tensor(EA))
    return torch.softmax(logits, 1).numpy()


def predict_features(g, features):
    """Second opinion per feature instance: mean face probabilities, top label, agreement."""
    if not os.path.exists(MODEL_PATH):
        return None
    P = predict_faces(g)
    out = {}
    for f in features:
        p = P[f.faces].mean(0)
        top = int(p.argmax())
        agree_set = AGREE.get(f.kind, {f.kind})
        agree_p = float(sum(p[LABELS.index(l)] for l in agree_set if l in LABELS))
        out[f.id] = {"label": LABELS[top], "p": round(float(p[top]), 3), "agree": agree_p >= 0.5,
                     "agree_p": round(agree_p, 3),
                     "probs": {LABELS[i]: round(float(v), 3) for i, v in enumerate(p) if v > 0.03}}
    return out
