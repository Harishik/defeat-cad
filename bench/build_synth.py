"""Rebuild the synthetic training graphs (the labels come from how each part was generated).

    python -m bench.build_synth --n 300
"""
from __future__ import annotations

import argparse
import os
import pickle

from bench.common import ROOT, pmap


def one(seed):
    from defeat.datagen import sample
    d = sample(seed)
    return {"sample": d} if d else {"ok": False, "error": "skip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    res = pmap(one, list(range(3012, 3012 + a.n)), workers=4, timeout=120, label="synth")
    out = [r["sample"] for r in res if r.get("ok") and r.get("sample")]
    path = os.path.join(ROOT, "models", "dataset.pkl")
    pickle.dump(out, open(path, "wb"))
    print(len(out), "graphs ->", path)


if __name__ == "__main__":
    main()
