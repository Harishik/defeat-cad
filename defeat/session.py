"""A defeaturing session: one uploaded part, its features, decisions, result and exports."""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter

import numpy as np

from . import codegen, occ
from .decide import PRESETS, Policy, decide, default_h
from .execute import defeature
from .graph import build_graph
from .recognize import recognize
from .tessellate import tessellate
from .validate import geometry_report, mesh_compare

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mesh_payload(shape):
    V, T, F = tessellate(shape)
    return {"positions": _b64(V.astype(np.float32)), "indices": _b64(T.astype(np.uint32).ravel()),
            "faceIds": _b64(F.astype(np.uint32)), "nv": int(len(V)), "nt": int(len(T))}


class Session:
    def __init__(self, path: str, name: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.dir = os.path.join(ROOT, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.name = name or os.path.basename(path)
        self.src = os.path.join(self.dir, self.name)
        if os.path.abspath(path) != os.path.abspath(self.src):
            shutil.copy(path, self.src)
        self.lock = threading.Lock()
        self.status = {"stage": "loading", "steps": [], "progress": 0.0, "busy": True}
        self.log = []          # operation log (drives code export + chat context)
        self.result = None
        self.metrics = None
        self.exports = {}
        self.chat = []

        self._step("Reading geometry")
        shape = occ.read_cad(self.src)
        self.shape = occ.heal(shape)
        self._step("Building face-adjacency graph")
        self.graph = build_graph(self.shape)
        self._step(f"Recognizing features on {len(self.graph.nodes)} faces")
        _, self.features = recognize(self.shape, self.graph)
        self.policy = Policy("static", default_h(self.graph.char_len))
        self.decisions = decide(self.features, self.policy)
        self.gnn = None
        try:
            from .gnn import predict_features
            self.gnn = predict_features(self.graph, self.features)
        except Exception:
            self.gnn = None
        self.status.update(stage="ready", busy=False, progress=1.0)

    # ------------------------------------------------------------ status
    def _step(self, msg, progress=None):
        self.status["steps"].append({"t": time.time(), "msg": msg})
        self.status["message"] = msg
        if progress is not None:
            self.status["progress"] = progress

    # ------------------------------------------------------------ views
    def summary(self):
        g = self.graph
        x0, y0, z0, x1, y1, z1 = occ.bbox(self.shape)
        counts = Counter(f.kind for f in self.features)
        rem = Counter(f.kind for f in self.features if self.decisions[f.id].action == "remove")
        return {
            "id": self.id, "name": self.name, "faces": len(g.nodes), "edges": len(g.edges),
            "size": [x1 - x0, y1 - y0, z1 - z0], "volume": occ.volume(self.shape),
            "char_len": g.char_len, "valid": occ.is_valid(self.shape), "solids": len(occ.solids_of(self.shape)),
            "feature_counts": dict(counts), "remove_counts": dict(rem),
            "policy": {"preset": self.policy.preset, "h": self.policy.h,
                       "overrides": self.policy.overrides,
                       "feature_overrides": {str(k): v for k, v in self.policy.feature_overrides.items()},
                       "protected_faces": sorted(self.policy.protected_faces)},
            "presets": {k: {"label": v["label"], "blurb": v["blurb"]} for k, v in PRESETS.items()},
            "features": [self.feature_view(f) for f in self.features],
            "status": self.status, "has_result": self.result is not None,
        }

    def feature_view(self, f):
        d = f.to_dict()
        d["decision"] = self.decisions[f.id].to_dict()
        if self.gnn and f.id in self.gnn:
            d["gnn"] = self.gnn[f.id]
        return d

    def briefing(self) -> str:
        """Compact text description of the part for the LLM."""
        s = self.summary()
        lines = [f"Part '{self.name}': {s['faces']} faces, size {s['size'][0]:.1f} x {s['size'][1]:.1f} x {s['size'][2]:.1f} (model units, assume mm), "
                 f"{s['solids']} solid(s). Target mesh size h={self.policy.h}. Preset: {self.policy.preset}."]
        for f in self.features:
            d = self.decisions[f.id]
            lines.append(f"- #{f.id} {f.kind}: {f.note} -> {d.action.upper()} ({d.reason})")
        return "\n".join(lines)

    # ------------------------------------------------------------ policy
    def set_policy(self, preset=None, h=None, overrides=None, feature_overrides=None,
                   protected_faces=None, merge=True, source="ui"):
        p = self.policy
        if preset:
            p.preset = preset
        if h:
            p.h = float(h)
        if overrides is not None:
            p.overrides = {**p.overrides, **overrides} if merge else dict(overrides)
        if feature_overrides is not None:
            fo = {int(k): v for k, v in feature_overrides.items()}
            p.feature_overrides = {**p.feature_overrides, **fo} if merge else fo
            p.feature_overrides = {k: v for k, v in p.feature_overrides.items() if v in ("keep", "remove")}
        if protected_faces is not None:
            p.protected_faces = set(protected_faces)
        self.decisions = decide(self.features, p)
        self.log.append({"op": "policy", "source": source, "preset": p.preset, "h": p.h,
                         "overrides": dict(p.overrides), "feature_overrides": dict(p.feature_overrides)})
        return self.summary()

    # ------------------------------------------------------------ run
    def run(self, with_mesh=True):
        with self.lock:
            self.status = {"stage": "running", "steps": [], "progress": 0.02, "busy": True}
            try:
                n_rm = sum(1 for d in self.decisions.values() if d.action == "remove")
                self._step(f"Planning removal of {n_rm} features", 0.08)
                er = defeature(self.shape, self.features, self.decisions,
                               progress=lambda m: self._step(m, min(0.55, self.status["progress"] + 0.03)))
                self.result = er
                self._step(f"Rebuilt watertight solid ({er.strategy}, {er.seconds:.1f}s)", 0.6)
                self._step("Measuring geometric deviation", 0.65)
                geo = geometry_report(self.shape, er.shape)
                mesh = None
                if with_mesh:
                    self._step(f"Meshing both models (h = {self.policy.h:g}) and running a static stress check", 0.72)
                    mesh = mesh_compare(self.shape, er.shape, self.policy.h)
                self._step("Writing STEP, code and report", 0.9)
                self.metrics = {"geometry": geo, "mesh": mesh, "fea": (mesh or {}).get("fea"), "removed": er.removed,
                                "failed": {str(k): v for k, v in er.failed.items()},
                                "strategy": er.strategy, "seconds": er.seconds}
                self._export()
                self.log.append({"op": "defeature", "removed": er.removed, "failed": list(er.failed)})
                self._step("Done", 1.0)
                self.status.update(stage="done", busy=False)
            except Exception as e:  # noqa
                self.status.update(stage="error", busy=False, error=f"{type(e).__name__}: {e}")
                raise

    def run_async(self, with_mesh=True):
        t = threading.Thread(target=self._safe_run, args=(with_mesh,), daemon=True)
        self.status = {"stage": "running", "steps": [], "progress": 0.01, "busy": True}
        t.start()

    def _safe_run(self, with_mesh):
        try:
            self.run(with_mesh)
        except Exception:
            import traceback
            traceback.print_exc()

    def _export(self):
        stem = self.name.rsplit(".", 1)[0]
        out_step = os.path.join(self.dir, f"{stem}_defeatured.step")
        occ.write_step(self.result.shape, out_step)
        feats = codegen.feature_signatures(self.graph.faces, self.features, set(self.result.removed))
        tol = 2e-3 * self.graph.char_len
        py = codegen.python_script(self.name, self.policy.preset, self.policy.h, feats, tol)
        cpp = codegen.cpp_program(self.name, self.policy.preset, self.policy.h, feats, tol)
        py_path = os.path.join(self.dir, f"defeature_{stem}.py")
        cpp_path = os.path.join(self.dir, f"defeature_{stem}.cpp")
        open(py_path, "w", encoding="utf-8").write(py)
        open(cpp_path, "w", encoding="utf-8").write(cpp)
        self.exports = {"step": out_step, "py": py_path, "cpp": cpp_path}

    def verify_python_export(self, timeout=120):
        """Round-trip check: run the exported script on the original STEP and compare."""
        stem = self.name.rsplit(".", 1)[0]
        out = os.path.join(self.dir, f"{stem}_roundtrip.step")
        r = subprocess.run([sys.executable, self.exports["py"], self.src, out], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=self.dir)
        if r.returncode != 0 or not os.path.exists(out):
            return {"ok": False, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
        rt = occ.read_cad(out)
        a, b = occ.volume(rt), occ.volume(self.result.shape)
        return {"ok": True, "faces": [len(occ.faces_of(rt)), len(occ.faces_of(self.result.shape))],
                "volume_diff_pct": 100 * abs(a - b) / max(abs(b), 1e-9), "stdout": r.stdout[-1000:]}

    def result_payload(self):
        return mesh_payload(self.result.shape) if self.result else None


SESSIONS: dict[str, Session] = {}


def new_session(path, name=None) -> Session:
    s = Session(path, name)
    SESSIONS[s.id] = s
    return s
