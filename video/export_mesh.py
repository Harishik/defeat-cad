"""Export real triangle meshes of a sample part (original + simplified) for the 3D shots in the intro.

    python video/export_mesh.py                 # all showcase parts -> video/public/part_<name>.json
Each triangle carries the kind of feature it belongs to ('' = body), so the film can light up and peel off
exactly the faces DeFeat removes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from defeat.session import Session          # noqa: E402
from defeat.tessellate import tessellate     # noqa: E402


def pack(V, T):
    c = (V.max(0) + V.min(0)) / 2
    s = 2.0 / (V.max(0) - V.min(0)).max()
    return {"p": np.round((V - c) * s, 4).ravel().tolist(), "i": T.ravel().tolist(), "center": c.tolist(), "scale": float(s)}


def export(src, name):
    s = Session(src)
    try:
        s.run(with_mesh=False)
        removed = set(s.result.removed)
        kind_of = {}
        for ft in s.features:
            if ft.id in removed:
                for fc in ft.faces:
                    kind_of[fc] = ft.kind
        V, T, F = tessellate(s.shape)
        orig = pack(V, T)
        kinds = sorted(set(kind_of.values()))
        orig["k"] = [kinds.index(kind_of[f]) + 1 if f in kind_of else 0 for f in F.tolist()]
        orig["kinds"] = kinds
        V2, T2, _ = tessellate(s.result.shape)
        # same frame as the original so the two line up exactly
        simp = {"p": np.round((V2 - np.array(orig["center"])) * orig["scale"], 4).ravel().tolist(), "i": T2.ravel().tolist()}
        out = {"original": orig, "simplified": simp, "faces": [len(s.graph.nodes), None], "removed": len(removed)}
        out["faces"] = [len(s.graph.nodes), int(s.metrics["geometry"]["faces"][1]) if s.metrics else None]
        path = os.path.join(HERE, "public", f"part_{name}.json")
        json.dump(out, open(path, "w"))
        print(f"{name}: tris {len(T)} -> {len(T2)}, removed {len(removed)} features, kinds {kinds}")
    finally:
        shutil.rmtree(s.dir, ignore_errors=True)


SOURCES = {
    "l_bracket": os.path.join(os.path.dirname(HERE), "samples", "l_bracket.step"),
    "printer_chassis": os.path.join(os.path.dirname(HERE), "samples", "printer_chassis.step"),
    **{n: os.path.join(HERE, "parts", f"{n}.step") for n in ("mold_plate", "gearbox_cover", "flanged_hub", "heat_sink", "robot_link")},
}

if __name__ == "__main__":
    for n in (sys.argv[1:] or SOURCES):
        export(SOURCES[n], n)
