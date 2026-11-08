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

from . import codegen, fea, occ
from .decide import PRESETS, Decision, Policy, decide, default_h, est_volume
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
        self._field = None
        self._finer_orig = None

        self._step("Reading geometry")
        shape = occ.read_cad(self.src)
        self.shape = occ.heal(shape)
        self._step("Building face-adjacency graph")
        self.graph = build_graph(self.shape)
        self._step(f"Recognizing features on {len(self.graph.nodes)} faces")
        _, self.features = recognize(self.shape, self.graph)
        self.policy = Policy("static", default_h(self.graph.char_len))
        self.part_volume = occ.volume(self.shape)
        self.decisions = decide(self.features, self.policy, self.graph, self.part_volume)
        self.net = None
        try:
            from .net import predict_features
            self.net = predict_features(self.graph, self.features)
        except Exception:
            self.net = None
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
                       "protected_faces": sorted(self.policy.protected_faces), "boundary": self.policy.boundary},
            "presets": {k: {"label": v["label"], "blurb": v["blurb"]} for k, v in PRESETS.items()},
            "features": [self.feature_view(f) for f in self.features],
            "status": self.status, "has_result": self.result is not None,
        }

    def feature_view(self, f):
        d = f.to_dict()
        d["decision"] = self.decisions[f.id].to_dict()
        if self.net and f.id in self.net:
            d["net"] = self.net[f.id]
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
                   protected_faces=None, merge=True, source="ui", boundary=None):
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
        if boundary is not None:
            p.boundary = {"fixed": [int(i) for i in boundary.get("fixed") or []], "loaded": [int(i) for i in boundary.get("loaded") or []],
                          "force": [float(v) for v in boundary.get("force") or [0, 0, -100]]}
        self.decisions = decide(self.features, p, self.graph, self.part_volume)
        self.log.append({"op": "policy", "source": source, "preset": p.preset, "h": p.h,
                         "overrides": dict(p.overrides), "feature_overrides": dict(p.feature_overrides)})
        return self.summary()

    @staticmethod
    def _mesh_after_fallback(mesh, er):
        m = dict(mesh or {})
        if not er.removed:
            m["defeatured"] = m.get("original")
            m["tet_reduction_pct"] = 0.0 if (m.get("original") or {}).get("ok") else None
            m["speedup"] = 1.0 if (m.get("original") or {}).get("ok") else None
        else:
            m["defeatured"] = {"ok": False, "error": "the smaller removal was not meshed"}
            m["tet_reduction_pct"] = None
            m["speedup"] = None
        return m

    # ------------------------------------------------------------ run
    def run(self, with_mesh=True):
        with self.lock:
            self.status = {"stage": "running", "steps": [], "progress": 0.02, "busy": True}
            self.unchecked = False
            self.unchecked_why = None
            try:
                if self.policy.verified != bool(with_mesh):  # no stress check to judge it: the volume budget steps in
                    self.policy.verified = bool(with_mesh)
                    self.decisions = decide(self.features, self.policy, self.graph, self.part_volume)
                n_rm = sum(1 for d in self.decisions.values() if d.action == "remove")
                self._step(f"Planning removal of {n_rm} features", 0.08)
                self._end = time.time() + self.run_seconds
                er = defeature(self.shape, self.features, self.decisions, graph=self.graph, budget=0.35 * self.run_seconds,
                               progress=lambda m: self._step(m, min(0.55, self.status["progress"] + 0.03)))
                self.result = er
                self._step(f"Rebuilt watertight solid ({er.strategy}, {er.seconds:.1f}s)", 0.6)
                self._step("Measuring geometric deviation", 0.65)
                geo = geometry_report(self.shape, er.shape)
                mesh = None
                if with_mesh:
                    self._step(f"Meshing both models (h = {self.policy.h:g}) and running a static stress check", 0.72)
                    mesh = mesh_compare(self.shape, er.shape, self.policy.h, want_field=bool(self.guard_tol), analysis=self.policy.preset, boundary=self.policy.boundary,
                                        budget=0.5 * max(self._end - time.time(), 60.0))
                    if not (mesh.get("fea") or {}).get("delta"):
                        # No comparison came back: never leave a physics-based removal unchecked. Fall back to
                        # what is safe without one, the volume budget, and check that smaller removal if the
                        # original itself could be solved. Otherwise say what stopped the check.
                        original = (mesh.get("fea") or {}).get("original") or (mesh.get("original") or {}).get("fea") or {}
                        why = self._unchecked_reason(mesh, original)
                        self._step(f"{why[0]}, so only small removals are kept", 0.75)
                        self.policy.verified = False
                        self.decisions = decide(self.features, self.policy, self.graph, self.part_volume)
                        er = defeature(self.shape, self.features, self.decisions, graph=self.graph, budget=max(self._end - time.time(), 30.0) * 0.3)
                        self.result = er
                        geo = geometry_report(self.shape, er.shape)
                        if original.get("ok") and er.removed:
                            self._step("Stress-checking the smaller removal", 0.77)
                            mesh = mesh_compare(self.shape, er.shape, self.policy.h, want_field=bool(self.guard_tol), analysis=self.policy.preset, boundary=self.policy.boundary,
                                                budget=0.5 * max(self._end - time.time(), 60.0))
                        else:
                            # The first comparison meshed the removal that was just abandoned: its numbers no longer
                            # describe the shipped part. Nothing removed means the part is the original; otherwise
                            # the smaller removal was never meshed, so report no mesh change rather than a stale one.
                            mesh = self._mesh_after_fallback(mesh, er)
                        if not (mesh.get("fea") or {}).get("delta"):
                            self.unchecked = bool(er.removed)
                            self.unchecked_why = why[1] if er.removed else None
                    self._field = ((mesh or {}).get("fea") or {}).get("original", {}).pop("field", None)
                    for side in ("defeatured",):
                        ((mesh or {}).get("fea") or {}).get(side, {}).pop("field", None)
                guard = None
                if with_mesh and self.guard_tol and (mesh or {}).get("fea"):
                    guard = self._stress_guard(er, mesh)
                    if guard and guard.get("accepted"):
                        er, mesh = self.result, guard["mesh"]
                        geo = geometry_report(self.shape, er.shape)
                self._step("Writing STEP, code and report", 0.9)
                self.metrics = {"geometry": geo, "mesh": mesh, "fea": (mesh or {}).get("fea"), "removed": er.removed,
                                "guard": {k: v for k, v in (guard or {}).items() if k != "mesh"} or None,
                                "unchecked": self.unchecked,
                                "unchecked_why": self.unchecked_why,
                                "guard_message": (f"{self.unchecked_why} Only small removals were kept. Run your own check on it."
                                                  if self.unchecked else fea.guard_message(guard, self.policy.preset)),
                                "failed": {str(k): v for k, v in er.failed.items()},
                                "strategy": er.strategy, "seconds": er.seconds}
                self._export()
                self.log.append({"op": "defeature", "removed": er.removed, "failed": list(er.failed)})
                self._step("Done", 1.0)
                self.status.update(stage="done", busy=False)
            except Exception as e:  # noqa
                self.status.update(stage="error", busy=False, error=f"{type(e).__name__}: {e}")
                raise

    guard_tol = 5.0   # % change in any load case the stress guard accepts (None/0 disables it)
    run_seconds = 600.0   # one whole run on one part, on a laptop: every stage below shares this
    guard_evals = 7   # most trial solves it spends looking for the largest removal that passes
    guard_seconds = 240.0
    finer_share = 0.5   # results above this share of the tolerance get a second look on a finer mesh
    finer_ratio = 1.4

    @staticmethod
    def _unchecked_reason(mesh, original):
        """(short status, full sentence) for why no stress comparison came back."""
        if original.get("ok"):
            return ("The simplified part can't be solved with the original's supports",
                    "The simplified part could not be solved with the original's supports and loads, so it has no stress check.")
        err = str((mesh.get("original") or {}).get("error") or original.get("error") or "")
        if "timed out" in err:
            return ("This part is too heavy to stress-check here", "This part was too heavy to stress-check here.")
        return ("This part can't be meshed", "The mesher could not build a volume mesh of this part, so it has no stress check.")

    def _judge(self, delta):
        """The stress peak only means something for a stress analysis; the other presets have their own quantity."""
        return fea.judged(delta, self.guard_tol) if self.policy.preset == "static" else fea.worst_case(delta)

    def _left(self, floor=15.0):
        return max(floor, min(300.0, self._end - time.time()))

    def _finer_delta(self, shape, bc, h_fea):
        """Worst change on a mesh finer by finer_ratio. Coarse meshes can flatter a result near the limit,
        so a verdict close to the tolerance is only trusted once a finer mesh agrees. None if too large."""
        from .execute import worker
        from .validate import step_bytes
        hf = h_fea / self.finer_ratio
        fine = dict(bc, h_fea=hf, want_field=False, max_tets=int(1.6 * fea.MAX_TETS))
        if self._finer_orig is None:
            r = worker().call("defeat.fea:mesh_and_solve", step_bytes(self.shape), hf, fine, timeout=self._left())
            self._finer_orig = (r.get("fea") if r.get("ok") else None) or False
        if not (self._finer_orig and self._finer_orig.get("ok")):
            return None
        r = worker().call("defeat.fea:mesh_and_solve", step_bytes(shape), hf, fine, timeout=self._left())
        if not (r.get("ok") and r.get("fea", {}).get("ok")):
            return None
        return self._judge(fea.compare(self._finer_orig, r["fea"]))

    def _stress_guard(self, er, mesh):
        """Keep the largest removal that stays within tolerance in every load case. Each feature is priced
        from the original's stress solution, the priciest per element saved go back first, and a bisection
        over that order finds how many must return. The output is never knowingly outside tolerance: if
        nothing short of the full part passes, the removal is undone."""
        from .execute import ExecResult, worker
        from .sensitivity import feature_scores, hot_features, restore_order
        from .validate import step_bytes
        t0 = time.time()
        bc = fea.load_case(occ.bbox(self.shape), self.policy.preset)
        if self.policy.boundary.get("fixed"):
            fea.user_case(occ.faces_of(self.shape), self.policy.boundary, bc)
        bc["stat_curvature"] = mesh.get("stat_curvature", 12)
        orig_fea = mesh["fea"]["original"]
        fea.lock(bc, orig_fea)
        field = self._field
        h_fea = orig_fea.get("h_fea") or mesh.get("h") or self.policy.h
        self._finer_orig = None
        delta0 = (mesh.get("fea") or {}).get("delta")
        d0 = self._judge(delta0)
        if d0 is None:
            return {"triggered": False, "worst_case_pct": d0}
        removed = [f for f in self.features if f.id in er.removed]
        finer0 = None
        if d0 <= self.guard_tol:
            if d0 <= self.finer_share * self.guard_tol:
                return {"triggered": False, "worst_case_pct": d0}
            finer0 = self._finer_delta(er.shape, bc, h_fea)
            if finer0 is None or finer0 <= self.guard_tol:
                return {"triggered": False, "worst_case_pct": d0, "finer_pct": finer0}
        first = max(d0, finer0 or 0.0)
        self._step("Stress guard: the result moved past its tolerance, working out what must stay", 0.8)
        scores = feature_scores(removed, self.graph, field, h_fea) if field else {f.id: {"cost": est_volume(f, self.graph), "benefit": 0} for f in removed}
        order = restore_order(removed, scores)
        n = len(order)
        stiff0 = fea.worst_case(delta0)
        peak_led = self.policy.preset == "static" and first > (stiff0 or 0.0) + 1e-9
        peak_before = abs((delta0 or {}).get("vm_peak_delta_pct") or 0.0) if peak_led else None
        if peak_led and field:    # the peak is what moved: what sits on the stress peaks goes back first
            hot = hot_features(removed, self.graph, field, h_fea, bc.get("axis"), orig_fea.get("ends"), ratio=2.0)
            order = [f for f in order if f.id in hot] + [f for f in order if f.id not in hot]

        def trial(k):
            keep = {f.id for f in order[:k]}
            dec = {i: (Decision(i, "keep", "Kept by the stress guard: removing it changed the part's stiffness")
                       if i in keep else d) for i, d in self.decisions.items()}
            for i, why in er.failed.items():   # the kernel already refused these: don't spend the time asking again
                dec[i] = Decision(i, "keep", f"Kept: {why}")
            er2 = defeature(self.shape, self.features, dec, graph=self.graph, budget=0.5 * self._left())
            r = worker().call("defeat.fea:mesh_and_solve", step_bytes(er2.shape), mesh.get("h") or self.policy.h, bc, timeout=self._left())
            if not (r.get("ok") and r.get("fea", {}).get("ok")):
                return None
            new_fea = r.pop("fea")
            delta = fea.compare(orig_fea, new_fea)
            d = self._judge(delta)
            finer = None
            if self.finer_share * self.guard_tol < d <= self.guard_tol:
                finer = self._finer_delta(er2.shape, bc, h_fea)
            self._step(f"Stress guard: kept {k} of {n} → worst load case {d:.1f}%", 0.84)
            ok = d <= self.guard_tol and (finer is None or finer <= self.guard_tol)
            return {"k": k, "delta": d, "finer": finer, "ok": ok, "er": er2, "dec": dec, "fea": new_fea, "delta_all": delta, "mesh_stats": r}

        # first guess from the estimates: how many must return for the rest to be priced within tolerance
        costs = [scores[f.id]["cost"] for f in order]
        total = sum(costs) or 1.0
        target = self.guard_tol / max(first, 1e-9) * total * 0.8
        guess, left = 1, total
        for i, c in enumerate(costs):
            left -= c
            if left <= target:
                guess = i + 1
                break
        lo, hi, best, evals = 0, n, None, 0   # lo: restoring this many fails; hi: restoring this many passes
        k = min(guess, n - 1) if n > 1 else 0
        while k > lo and k < hi and evals < self.guard_evals and time.time() - t0 < self.guard_seconds and time.time() < self._end - 15:
            c = trial(k)
            evals += 1
            if c is not None and c["ok"]:
                best, hi = c, k
            else:
                lo = k
            k = (lo + hi) // 2
        if best is None:      # only the full part passed: nothing to compare, the original stands
            r0 = ExecResult(self.shape)
            r0.strategy = "undone by the stress guard"
            r0.failed = dict(er.failed)
            self.result = r0
            self.decisions = {i: (Decision(i, "keep", "Kept by the stress guard: only the full part stayed within tolerance")
                                  if d.action == "remove" else d) for i, d in self.decisions.items()}
            m2 = dict(mesh)
            m2["defeatured"] = m2["original"]
            m2["tet_reduction_pct"] = 0.0
            m2["speedup"] = 1.0
            same = fea.compare(orig_fea, orig_fea)
            m2["fea"] = dict(mesh["fea"], defeatured=orig_fea, delta=same)
            return {"triggered": True, "accepted": True, "restored": n, "of": n, "before_pct": first, "stiffness_before_pct": stiff0, "after_pct": 0.0,
                    "peak_before_pct": peak_before, "within_tol": True, "undone": True, "mesh": m2, "trials": evals}
        self.result, self.decisions = best["er"], best["dec"]
        m2 = dict(mesh)
        m2["defeatured"] = best["mesh_stats"]
        if m2["original"].get("tets"):
            m2["tet_reduction_pct"] = 100 * (m2["original"]["tets"] - m2["defeatured"]["tets"]) / m2["original"]["tets"]
        m2["fea"] = dict(mesh["fea"], defeatured=best["fea"], delta=best["delta_all"])
        return {"triggered": True, "accepted": True, "restored": best["k"], "of": n, "before_pct": first, "stiffness_before_pct": stiff0, "peak_before_pct": peak_before,
                "after_pct": best["delta"], "peak_after_pct": abs(best["delta_all"].get("vm_peak_delta_pct") or 0.0) if peak_led else None, "finer_pct": best["finer"], "within_tol": True, "undone": False,
                "mesh": m2, "trials": evals}

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
