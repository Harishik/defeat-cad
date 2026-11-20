"""Recognition network: geometry grids (UV-Net) + topological walks (BRepNet) + part-wide context.

Each ingredient can be switched off, so bench/train_net.py can measure what every one contributes.
Output: per-face probabilities over DeFeat's classes, and over Fusion 360's eight modelling labels.
"""
from __future__ import annotations

import math
import os
import pickle
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_LABELS = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]


def _pool(src, idx, n, reduce="amax"):
    out = torch.zeros(n, src.shape[1], dtype=src.dtype, device=src.device)
    return out.scatter_reduce(0, idx[:, None].expand_as(src), src, reduce, include_self=False)


class WalkLayer(nn.Module):
    """One BRepNet 'winged edge' convolution: every coedge reads the entities around it, then the
    result is pooled back onto faces and edges."""

    def __init__(self, h, ctx=True, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(13 * h, 3 * h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(3 * h, 3 * h))
        self.nc, self.nf, self.ne = nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h)
        self.ctx = nn.Linear(2 * h, h) if ctx else None
        self.h = h

    def forward(self, f, e, c, b):
        m, n, p, mn, mp = b["cm"], b["cn"], b["cp"], b["cmn"], b["cmp"]
        ce, cf = b["ce"], b["cf"]
        z = torch.cat([c, c[m], c[n], c[p], c[mn], c[mp], f[cf], f[cf[m]], e[ce], e[ce[n]], e[ce[p]], e[ce[mn]], e[ce[mp]]], 1)
        o = self.mlp(z)
        oc, of, oe = o[:, :self.h], o[:, self.h:2 * self.h], o[:, 2 * self.h:]
        f2 = _pool(of, cf, f.shape[0])
        if self.ctx is not None:      # part-wide context, which walks of a few steps can't reach
            g = torch.zeros(b["n_parts"], self.h, device=f.device).index_add_(0, b["fb"], f2)
            g = g / b["fcount"][:, None]
            f2 = f2 + self.ctx(torch.cat([f2, g[b["fb"]]], 1))
        return self.nf(f + f2), self.ne(e + _pool(oe, ce, e.shape[0])), self.nc(c + oc)


class FaceAttention(nn.Module):
    """Self-attention between all faces of a part (the idea behind BrepMFR and BRT)."""

    def __init__(self, h, heads=4, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * h, h))

    def forward(self, f, b):
        P, cnt = b["n_parts"], b["fcount"].long()
        L = int(cnt.max())
        pos = b["fpos"]
        dense = torch.zeros(P, L, f.shape[1], device=f.device)
        dense[b["fb"], pos] = f
        pad = torch.arange(L, device=f.device)[None] >= cnt[:, None]
        out, _ = self.att(dense, dense, dense, key_padding_mask=pad, need_weights=False)
        f = self.n1(f + out[b["fb"], pos])
        return self.n2(f + self.ff(f))


class BRepRecognizer(nn.Module):
    def __init__(self, x_dim=40, h=128, layers=3, n_cls=9, uv=True, walks=True, ctx=True, attn=0, hints=True, dropout=0.1):
        super().__init__()
        self.cfg = dict(x_dim=x_dim, h=h, layers=layers, n_cls=n_cls, uv=uv, walks=walks, ctx=ctx, attn=attn, hints=hints, dropout=dropout)
        self.uv, self.walks, self.hints = uv, walks, hints
        gx = 64 if uv else 0
        if uv:
            self.face_cnn = nn.Sequential(nn.Conv2d(7, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.edge_cnn = nn.Sequential(nn.Conv1d(6, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
                                          nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fin = nn.Sequential(nn.Linear(x_dim + gx, h), nn.ReLU(), nn.Linear(h, h))
        self.ein = nn.Sequential(nn.Linear(10 + gx, h), nn.ReLU(), nn.Linear(h, h))
        self.cin = nn.Linear(1, h)
        self.lay = nn.ModuleList(WalkLayer(h, ctx, dropout) for _ in range(layers))
        self.att = nn.ModuleList(FaceAttention(h, dropout=dropout) for _ in range(attn))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, n_cls))
        self.head8 = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 8))

    def forward(self, b):
        x = b["x"] if self.hints else torch.cat([b["x"][:, :-b["n_hint"]], torch.zeros(b["x"].shape[0], b["n_hint"], device=b["x"].device)], 1)
        if self.uv:
            f = self.fin(torch.cat([x, self.face_cnn(b["uv"])], 1))
            e = self.ein(torch.cat([b["ek"], self.edge_cnn(b["cg"])], 1))
        else:
            f, e = self.fin(x), self.ein(b["ek"])
        f0 = f
        c = self.cin(b["cr"][:, None])
        for lay in self.lay:
            f, e, c = lay(f, e, c, b)
        for a in self.att:
            f = a(f, b)
        z = torch.cat([f, f0], 1)
        return self.head(z), self.head8(z)


# ------------------------------------------------------------------ batching
def rotate(uv, cg, rng):
    """Same random rotation for every grid of a part: the network must not depend on pose."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    uv, cg = uv.copy(), cg.copy()
    for a, t in ((0, 3),):
        uv[:, a:a + 3] = np.einsum("ij,njuv->niuv", q, uv[:, a:a + 3])
        uv[:, t:t + 3] = np.einsum("ij,njuv->niuv", q, uv[:, t:t + 3])
    cg[:, 0:3] = np.einsum("ij,nju->niu", q, cg[:, 0:3])
    cg[:, 3:6] = np.einsum("ij,nju->niu", q, cg[:, 3:6])
    return uv, cg


def collate(parts, device="cpu", rng=None, n_hint=10):
    fo = eo = co = 0
    out = {k: [] for k in ("x", "uv", "cg", "ek", "cf", "ce", "cr", "cn", "cp", "cm", "fb", "fpos", "y", "y8")}
    counts = []
    for i, d in enumerate(parts):
        n, ne, nc = d["x"].shape[0], d["ek"].shape[0], d["cf"].shape[0]
        uv, cg = np.asarray(d["uv"], "float32"), np.asarray(d["cg"], "float32")
        if rng is not None:
            uv, cg = rotate(uv, cg, rng)
        out["x"].append(d["x"]); out["uv"].append(uv); out["cg"].append(cg); out["ek"].append(d["ek"])
        out["cf"].append(d["cf"] + fo); out["ce"].append(d["ce"] + eo); out["cr"].append(d["cr"])
        out["cn"].append(d["cn"] + co); out["cp"].append(d["cp"] + co); out["cm"].append(d["cm"] + co)
        out["fb"].append(np.full(n, i)); out["fpos"].append(np.arange(n))
        out["y"].append(np.asarray(d["yy"], "int64")); out["y8"].append(np.asarray(d["y88"], "int64"))
        counts.append(n)
        fo, eo, co = fo + n, eo + ne, co + nc
    b = {}
    for k, v in out.items():
        a = np.concatenate(v)
        t = torch.from_numpy(a)
        if a.dtype.kind in "iu":
            t = t.long()
        b[k] = t.to(device)
    b["cmn"], b["cmp"] = b["cm"][b["cn"]], b["cm"][b["cp"]]
    b["n_parts"] = len(parts)
    b["fcount"] = torch.tensor(counts, dtype=torch.float32, device=device)
    b["n_hint"] = n_hint
    return b


def load_shards(root, want=None, limit=None, fields=None):
    """All parts in the shard files; a part cut off by a kill at the end of a file is simply ignored."""
    parts = []
    for path in sorted(glob.glob(os.path.join(root, "part_*.pkl"))):
        with open(path, "rb") as fh:
            while True:
                try:
                    d = pickle.load(fh)
                except Exception:
                    break
                if want is None or d["split"] in want:
                    parts.append(d)
                    if limit and len(parts) >= limit:
                        return parts
    return parts


# ------------------------------------------------------------------ at runtime
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENSEMBLE = None


def model_paths():
    env = os.environ.get("DEFEAT_NET")
    return env.split(",") if env else sorted(glob.glob(os.path.join(HERE, "models", "recognizer_*.pt")))


def load_ensemble():
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = []
        for path in model_paths():
            ck = torch.load(path, map_location="cpu")
            m = BRepRecognizer(**ck["cfg"])
            m.load_state_dict(ck["state"])
            m.eval()
            _ENSEMBLE.append(m)
    return _ENSEMBLE


@torch.no_grad()
def predict_part(g):
    """Per-face probabilities, averaged over the ensemble: (DeFeat's classes, Fusion's eight labels). None without a model."""
    from .brepfeat import brep_tensors
    from .datagen import node_features
    cached = getattr(g, "_net_pred", None)
    if cached is not None:
        return cached
    models = load_ensemble()
    if not models or not g.nodes:
        return None
    disk = _cache_path(g)
    if disk and os.path.exists(disk):
        try:
            z = np.load(disk)
            if len(z["p9"]) == len(g.nodes):
                g._net_pred = z["p9"], z["p8"]
                return g._net_pred
        except Exception:
            pass
    d = {"x": node_features(g)[0], "yy": np.full(len(g.nodes), -1), "y88": np.full(len(g.nodes), -1)}
    d.update(brep_tensors(g))
    b = collate([d])
    P9, P8 = 0.0, 0.0
    for m in models:
        l9, l8 = m(b)
        P9, P8 = P9 + torch.softmax(l9, 1), P8 + torch.softmax(l8, 1)
    g._net_pred = (P9 / len(models)).numpy(), (P8 / len(models)).numpy()
    if disk:
        try:
            os.makedirs(os.path.dirname(disk), exist_ok=True)
            np.savez_compressed(disk, p9=g._net_pred[0], p8=g._net_pred[1])
        except Exception:
            pass
    return g._net_pred


def _cache_path(g):
    """Where the app keeps the network's answer for this exact part (faces in their own order), so
    reopening a part or opening a library part skips the network. Off unless DEFEAT_NET_CACHE is set:
    the app sets it, benchmarks never do."""
    root = os.environ.get("DEFEAT_NET_CACHE")
    if not root:
        return None
    try:
        import hashlib
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        h = hashlib.sha1()
        for f in g.faces:
            pr = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, pr)
            c = pr.CentreOfMass()
            h.update(f"{pr.Mass():.9e}:{c.X():.7e}:{c.Y():.7e}:{c.Z():.7e}|".encode())
        h.update(":".join(os.path.basename(p) + str(os.path.getmtime(p)) for p in model_paths()).encode())   # new models, new answers
        return os.path.join(root, h.hexdigest() + ".npz")
    except Exception:
        return None


# rule class -> network label that counts as agreement
AGREE = {"hole": {"hole"}, "slot": {"hole", "pocket"}, "pocket": {"pocket", "hole"}, "step": {"step", "pocket"}, "boss": {"boss"},
         "rib": {"rib"}, "emboss": {"emboss"}, "fillet": {"fillet"}, "chamfer": {"chamfer", "fillet"},
         "protrusion": {"boss", "rib", "emboss"}}


def predict_features(g, features):
    """Second opinion per feature: mean face probabilities, top label, whether it agrees with the rules."""
    from .datagen import LABELS
    pred = predict_part(g)
    if pred is None:
        return None
    P = pred[0]
    out = {}
    for f in features:
        p = P[f.faces].mean(0)
        top = int(p.argmax())
        agree_p = float(sum(p[LABELS.index(k)] for k in AGREE.get(f.kind, {f.kind}) if k in LABELS))
        out[f.id] = {"label": LABELS[top], "p": round(float(p[top]), 3), "agree": agree_p >= 0.5, "agree_p": round(agree_p, 3),
                     "probs": {LABELS[i]: round(float(v), 3) for i, v in enumerate(p) if v > 0.03}}
    return out
