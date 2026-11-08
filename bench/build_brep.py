"""Build the rich training set: all eight Fusion labels, surface and curve grids, coedge pointers.

Runs independent workers, each writing its own files, with a watchdog: a worker stuck inside the CAD
kernel is killed and the part it was on is skipped. Safe to stop and restart at any time.

    python -m bench.build_brep --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import subprocess
import sys
import time

from bench.common import DATA, ROOT

OUT = os.path.join(DATA, "brep_v2")
VAL = (4000, 5500)    # Fusion train-split parts held back for model selection and thresholds


def items(mfcad=6000):
    from bench.fusion import BASE
    from bench.mfcad import DIR, split_of
    split = json.load(open(os.path.join(BASE, "train_test.json")))
    tr = list(split["train"])
    random.Random(0).shuffle(tr)
    te = list(split["test"])
    random.Random(0).shuffle(te)
    mf = [f for f in sorted(glob.glob(os.path.join(DIR, "*.step"))) if split_of(f) == "train"]
    random.Random(0).shuffle(mf)
    # already-scored order preserved: test parts 3000+ have never been looked at
    out = [("fusion", n, "val") for n in tr[VAL[0]:VAL[1]]]
    out += [("mfcad", p, "train") for p in mf[:mfcad]]
    out += [("fusion", n, "train") for n in tr[:VAL[0]] + tr[VAL[1]:]]
    out += [("fusion", n, "test") for n in te]
    return out


def make(src, name, split):
    from defeat.brepfeat import brep_tensors
    from defeat.datagen import LABELS, node_features
    from defeat.graph import build_graph
    if src == "fusion":
        from bench.fusion import load
        shape, seg = load(name)
        y8, y = seg, None
    else:
        from bench.mfcad import FEAT_NAMES, load_labelled
        shape, raw = load_labelled(name)
        y8 = None
        y = [LABELS.index("base") if FEAT_NAMES[v] == "stock" else -1 if FEAT_NAMES[v] == "chamfer"
             else LABELS.index("step") if "step" in FEAT_NAMES[v] else LABELS.index("pocket") for v in raw]
    g = build_graph(shape)
    n = len(y8 if y8 is not None else y)
    if len(g.nodes) != n or n > 600:
        return None
    X, E, EA = node_features(g)
    d = {"name": os.path.basename(name), "src": src, "split": split, "x": X, "e": E, "ea": EA,
         "y8": y8, "y": y}
    d.update(brep_tensors(g))
    return d


def worker(i, n, gen):
    its = items()
    seen = set()
    for f in glob.glob(os.path.join(OUT, "log_*.txt")):
        seen.update(line.split("\t", 1)[1].strip() for line in open(f) if "\t" in line)
    log = open(os.path.join(OUT, f"log_{i}_{gen}.txt"), "a", buffering=1)
    fh = open(os.path.join(OUT, f"part_{i}_{gen}.pkl"), "ab")
    for k, (src, name, split) in enumerate(its):
        if k % n != i or name in seen:
            continue
        log.write(f"start\t{name}\n")
        try:
            d = make(src, name, split)
        except Exception as e:  # noqa
            d = None
            log.write(f"err {type(e).__name__}\t{name}\n")
        if d is not None:
            pickle.dump(d, fh, protocol=4)
            fh.flush()
        log.write(f"done\t{name}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--stall", type=int, default=240)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.worker >= 0:
        return worker(a.worker, a.workers, a.gen)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
    gen = int(time.time())
    def spawn(i, g):
        return subprocess.Popen([sys.executable, "-m", "bench.build_brep", "--workers", str(a.workers), "--worker", str(i), "--gen", str(g)], cwd=ROOT, env=env)
    procs = {i: spawn(i, gen) for i in range(a.workers)}
    total = len(items())
    while procs:
        time.sleep(20)
        now = time.time()
        for i, p in list(procs.items()):
            logs = glob.glob(os.path.join(OUT, f"log_{i}_*.txt"))
            last = max((os.path.getmtime(f) for f in logs), default=now)
            if p.poll() is not None:
                if p.returncode == 0:
                    del procs[i]
                else:   # native crash: the part it was on is logged as started, so it gets skipped
                    gen += 1
                    procs[i] = spawn(i, gen)
            elif now - last > a.stall:
                p.kill()
                p.wait()
                gen += 1
                procs[i] = spawn(i, gen)
        done = 0
        for f in glob.glob(os.path.join(OUT, "log_*.txt")):
            done += sum(1 for line in open(f) if line.startswith("start"))
        print(f"{done}/{total}  workers {len(procs)}", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()


def synth_worker(i, n, count, offset):
    """Generated mold parts with exact per-face labels: the only source of holes, bosses, ribs and embossed text."""
    import signal
    from defeat.brepfeat import brep_tensors
    from defeat.datagen import label_faces, make_part, node_features
    from defeat.graph import build_graph

    def alarm(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, alarm)
    fh = open(os.path.join(OUT, f"part_synth_{i}.pkl"), "ab")
    done = 0
    for seed in range(offset + i, offset + count, n):
        try:
            signal.alarm(60)
            shape, spec = make_part(seed)
            g0, y = label_faces(shape, spec)
            X, E, EA = node_features(g0)
            d = {"name": f"synth_{seed}", "src": "synth", "split": "train", "x": X, "e": E, "ea": EA, "y8": None, "y": y}
            d.update(brep_tensors(g0))
            signal.alarm(0)
            pickle.dump(d, fh, protocol=4)
            fh.flush()
            done += 1
        except Exception:
            signal.alarm(0)
    print("synth worker", i, "wrote", done, flush=True)
