"""Shared helpers: dataset paths, metrics, and a crash/timeout-safe parallel map."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.environ.get("DEFEAT_DATA", os.path.expanduser("~/defeat-data"))
RESULTS = os.path.join(ROOT, "bench", "results")

# our rule / network classes -> evaluation group
OURS_GROUP = {"hole": "cut", "slot": "cut", "pocket": "cut", "step": "step", "chamfer": "chamfer", "fillet": "fillet",
              "boss": "protrusion", "rib": "protrusion", "emboss": "protrusion", "protrusion": "protrusion",
              "base": "base"}


def rule_face_labels(g, feats):
    lab = ["base"] * len(g.nodes)
    for f in feats:
        for fid in f.faces:
            lab[fid] = f.kind
    return lab


def fuse(rule_groups, P, labels, veto=0.4, add=0.5):
    """Rules first; the network may veto a rule fillet/chamfer it finds very unlikely, or add one it is
    very sure about on a face the rules left as body."""
    fi, ci = labels.index("fillet"), labels.index("chamfer")
    out = []
    for r, p in zip(rule_groups, P):
        if r == "fillet" and p[fi] < veto:
            out.append("base")
        elif r == "chamfer" and p[ci] < veto:
            out.append("base")
        elif r == "base" and p[fi] > add:
            out.append("fillet")
        elif r == "base" and p[ci] > add:
            out.append("chamfer")
        else:
            out.append(r)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}


def face_metrics(pairs, classes):
    """pairs: list of (gt_group, pred_group). Per-class P/R/F1/IoU, overall accuracy, macro F1, mIoU."""
    tp, fp, fn = Counter(), Counter(), Counter()
    conf = defaultdict(Counter)
    for gt, pr in pairs:
        conf[gt][pr] += 1
        if gt == pr:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1
    per, ious = {}, []
    for c in classes:
        m = prf(tp[c], fp[c], fn[c])
        denom = tp[c] + fp[c] + fn[c]
        m["iou"] = round(tp[c] / denom, 4) if denom else 0.0
        per[c] = m
        if tp[c] + fn[c]:
            ious.append(m["iou"])
    n = len(pairs)
    acc = sum(tp.values()) / n if n else 0.0
    f1s = [per[c]["f1"] for c in classes if per[c]["support"]]
    return {"faces": n, "accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / max(len(f1s), 1), 4),
            "miou": round(sum(ious) / max(len(ious), 1), 4), "per_class": per,
            "confusion": {k: dict(v) for k, v in conf.items()}}


def binary_metrics(pairs, positive):
    """Feature-vs-stock detection: is this face part of something that should be removed/kept as a feature?"""
    tp = sum(1 for g, p in pairs if g in positive and p in positive)
    fp = sum(1 for g, p in pairs if g not in positive and p in positive)
    fn = sum(1 for g, p in pairs if g in positive and p not in positive)
    tn = len(pairs) - tp - fp - fn
    m = prf(tp, fp, fn)
    m["accuracy"] = round((tp + tn) / len(pairs), 4) if pairs else 0.0
    return m


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (well behaved for small n and rates near 0 or 1)."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def bootstrap(values, stat, b=2000, seed=0):
    """95% percentile bootstrap interval of stat(sample), resampling the list items (parts) with replacement."""
    import random as _r
    if not values:
        return None
    rng = _r.Random(seed)
    n = len(values)
    xs = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return [round(xs[int(0.025 * b)], 4), round(xs[int(0.975 * b) - 1], 4)]


def f1_ci(per_part, b=2000, seed=0):
    """Part-level bootstrap of F1 from per-part (tp, fp, fn) counts: faces of one part are not
    independent, so parts, not faces, are the unit that gets resampled."""
    def f1(sample):
        tp = sum(x[0] for x in sample); fp = sum(x[1] for x in sample); fn = sum(x[2] for x in sample)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return bootstrap(per_part, f1, b, seed)


# ---------------------------------------------------------------- safe parallel map
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _guarded(args):
    fn, item, timeout = args
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout))
    t = time.time()
    try:
        out = fn(item)
        out = out if isinstance(out, dict) else {"value": out}
        out.setdefault("ok", True)
    except _Timeout:
        out = {"ok": False, "error": "timeout"}
    except Exception as e:  # noqa
        out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    out["item"] = str(item)
    out["seconds"] = round(time.time() - t, 3)
    return out


def pmap(fn, items, workers=7, timeout=120, label=""):
    ctx = mp.get_context("spawn")
    res = []
    t0 = time.time()
    with ctx.Pool(workers, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(_guarded, [(fn, it, timeout) for it in items], chunksize=1), 1):
            res.append(r)
            if i % max(1, len(items) // 20) == 0 or i == len(items):
                ok = sum(1 for x in res if x.get("ok"))
                print(f"  {label} {i}/{len(items)}  ok={ok}  {time.time() - t0:.0f}s", flush=True)
    return res


def run_item(fnpath, item):
    """Executed inside a defeat Worker process (killed + respawned on timeout, so even hangs inside
    OpenCASCADE C++ code are contained)."""
    import importlib
    mod, name = fnpath.split(":")
    t = time.time()
    try:
        out = getattr(importlib.import_module(mod), name)(item)
        out = out if isinstance(out, dict) else {"value": out}
        out.setdefault("ok", True)
    except Exception as e:  # noqa
        out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["seconds"] = round(time.time() - t, 3)
    return out


def pmap_hard(fnpath, items, workers=6, timeout=180, label=""):
    """Like pmap, but each worker is a killable process: a task that hangs is killed after `timeout`."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from defeat.execute import Worker
    local = threading.local()
    workers_made = []
    done = []
    lock = threading.Lock()
    t0 = time.time()

    def task(item):
        if not hasattr(local, "w"):
            local.w = Worker(daemon=False)
            with lock:
                workers_made.append(local.w)
        r = local.w.call("bench.common:run_item", fnpath, item, timeout=timeout)
        if r.get("error", "").startswith("kernel timed out"):
            r = {"ok": False, "error": "timeout"}
        elif r.get("error", "").startswith("kernel crashed"):
            r = {"ok": False, "error": "native crash"}
        r["item"] = str(item)
        with lock:
            done.append(r)
            n = len(done)
            if n % max(1, len(items) // 20) == 0 or n == len(items):
                print(f"  {label} {n}/{len(items)}  ok={sum(1 for x in done if x.get('ok'))}  {time.time() - t0:.0f}s", flush=True)
        return r

    try:
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(task, items))
    finally:
        for w in workers_made:
            w.close()
    return res


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, f"{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    return p
