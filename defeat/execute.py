"""Guarded defeaturing executor.

Every OCCT call runs in a worker process with a timeout (BRepAlgoAPI_Defeaturing can hang on
some fillet faces, OCCT bug 0033561). Strategy: one batched removal first; if it fails or the
result is invalid, fall back to feature-by-feature removal in a safe order, keeping (and
reporting) anything the kernel cannot remove. The output is never an invalid solid.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass, field

from . import occ

VOLUME_LIMIT = 1.0
GROWTH_LIMIT = 2.0
TINY_VOLUME = 0.002      # share of the part's volume
TINY_GROWTH = 0.01       # share of its diagonal
ORDER = ["emboss", "chamfer", "fillet", "hole", "slot", "pocket", "step", "boss", "rib", "protrusion"]


# ---------------------------------------------------------------- worker side
def _remove_faces(brep: bytes, face_ids: list[int], fuzzy: float = 0.0, limits=None):
    """Remove faces; also return where every input face went (from the kernel's own history),
    so later features can be found again exactly even when their neighbours changed.
    limits = (max_volume_change, max_growth): what the removed faces could physically account for."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopTools import TopTools_ListOfShape
    from defeat import occ as _occ
    shape = _occ.from_brep_string(brep)
    faces = _occ.faces_of(shape)
    lst = TopTools_ListOfShape()
    for i in face_ids:
        lst.Append(faces[i])
    d = BRepAlgoAPI_Defeaturing()
    d.SetShape(shape)
    d.AddFacesToRemove(lst)
    d.SetRunParallel(False)
    d.SetToFillHistory(True)
    if fuzzy > 0:
        d.SetFuzzyValue(fuzzy)
    d.Build()
    if not d.IsDone():
        return {"ok": False, "error": "kernel could not rebuild the surrounding faces"}
    res = d.Shape()
    healed = False
    if not _occ.is_valid(res):
        res2 = _occ.heal(res)
        if not _occ.is_valid(res2):
            return {"ok": False, "error": "result was not a valid solid"}
        res, healed = res2, True
    if len(_occ.solids_of(res)) != len(_occ.solids_of(shape)):
        return {"ok": False, "error": "removal split or merged bodies"}
    if len(_occ.faces_of(res)) >= len(faces):
        return {"ok": False, "error": "kernel returned the part unchanged"}
    if limits is not None:
        v0 = abs(_occ.volume(shape))
        dv = abs(_occ.volume(res) - _occ.volume(shape))
        a, b = _occ.bbox(shape), _occ.bbox(res)
        grow = max(max(a[i] - b[i] for i in range(3)), max(b[i + 3] - a[i + 3] for i in range(3)))
        diag = _occ.norm(_occ.sub(a[3:], a[:3]))
        # tiny changes always pass: a deep slot or a sharp corner can beat the estimate and still be harmless
        if (dv > limits[0] and dv > TINY_VOLUME * v0) or (grow > limits[1] and grow > TINY_GROWTH * diag):
            return {"ok": False, "error": "the kernel rebuilt far more material than the feature holds"}
    fmap = None
    if not healed:
        out = _occ.indexed(res, TopAbs_FACE)
        fmap = []
        for f in faces:
            if d.IsDeleted(f):
                fmap.append(-1)
                continue
            mods = d.Modified(f)
            tgt = mods.First() if mods.Size() else f
            fmap.append(out.FindIndex(tgt) - 1)
    return {"ok": True, "brep": _occ.to_brep_string(res), "map": fmap}


def _dispatch(fn, args):
    try:
        if ":" in fn:
            import importlib
            mod, name = fn.split(":")
            f = getattr(importlib.import_module(mod), name)
        else:
            f = globals()[fn]
        return f(*args)
    except Exception as e:  # noqa
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _exit_with_parent():
    """A worker killed on timeout must not leave its own kernel worker running: every worker exits
    as soon as the process that started it is gone (works the same on macOS, Linux and Windows)."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch():
        from multiprocessing.connection import wait
        wait([parent.sentinel])
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def _worker_loop(conn):
    _exit_with_parent()
    while True:
        msg = conn.recv()
        if msg is None:
            break
        fn, args = msg
        try:
            if ":" in fn:
                import importlib
                mod, name = fn.split(":")
                f = getattr(importlib.import_module(mod), name)
            else:
                f = globals()[fn]
            conn.send(f(*args))
        except Exception as e:  # noqa
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})


class Worker:
    """A persistent OCCT worker process that is killed and respawned on timeout."""

    def __init__(self, daemon=True):
        self.proc = None
        self.conn = None
        self.daemon = daemon   # non-daemon workers may start their own kernel worker (benchmarks)

    def _start(self):
        ctx = mp.get_context("spawn")
        a, b = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_loop, args=(b,), daemon=self.daemon)
        self.proc.start()
        self.conn = a

    def call(self, fn, *args, timeout=30.0):
        if mp.current_process().daemon:
            # already inside a guarded pool worker (batch/benchmark runs): it owns the timeout,
            # and daemonic processes may not spawn children, so run in-process
            return _dispatch(fn, args)
        if self.proc is None or not self.proc.is_alive():
            self._start()
        try:
            self.conn.send((fn, args))
            if self.conn.poll(timeout):
                return self.conn.recv()
            err = f"kernel timed out after {timeout:.0f}s"
        except (EOFError, OSError, BrokenPipeError):
            # the native kernel crashed (e.g. a segfault inside OpenCASCADE or gmsh)
            err = "kernel crashed on this geometry"
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc = None
        return {"ok": False, "error": err}

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            try:
                self.conn.send(None)
            except Exception:
                pass
            self.proc.join(1)
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = None


_WORKER: Worker | None = None


def worker() -> Worker:
    global _WORKER
    if _WORKER is None:
        _WORKER = Worker()
    return _WORKER


# ---------------------------------------------------------------- face re-identification
def _signature(face):
    info = occ.surf_info(face)
    return info.kind, occ.centroid(face), occ.area(face)


def remap_faces(old_sigs, new_shape, tol):
    """Map old face ids to new face ids for faces untouched by a removal."""
    new_faces = occ.faces_of(new_shape)
    new_sigs = [_signature(f) for f in new_faces]
    m = {}
    for i, (k, c, a) in enumerate(old_sigs):
        best, bd = None, tol
        for j, (k2, c2, a2) in enumerate(new_sigs):
            if k2 != k or abs(a2 - a) > 0.01 * max(a, 1e-9) + 1e-9:
                continue
            d = occ.norm(occ.sub(c, c2))
            if d < bd:
                best, bd = j, d
        if best is not None:
            m[i] = best
    return m


# ---------------------------------------------------------------- driver
@dataclass
class ExecResult:
    shape: object
    removed: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)    # feature id -> reason
    strategy: str = ""
    seconds: float = 0.0
    log: list = field(default_factory=list)


def _blend_neighbours(ft, graph, todo_by_face, owner_kind):
    """Faces of adjacent blends (fillet/chamfer features being removed, or unclaimed tangent blend
    faces such as corner patches) that the kernel may need to remove together with ft."""
    from .graph import SMOOTH
    extra = set()
    for fid in ft.faces:
        for m, eids in graph.nodes[fid].neighbors.items():
            if m in ft.faces:
                continue
            other = todo_by_face.get(m)
            if other is not None and other.kind in ("fillet", "chamfer"):
                extra |= set(other.faces)
            elif m not in owner_kind and graph.nodes[m].kind != "plane" and any(graph.edges[e].convexity == SMOOTH for e in eids):
                extra.add(m)
    return extra


def _limits(face_ids, size, areas, slack):
    """A feature cannot hold more material than its faces times its size, nor reach further than its
    size. Anything beyond that means the kernel extended neighbouring faces somewhere they should not go."""
    return VOLUME_LIMIT * sum(areas[i] for i in face_ids) * size + slack, GROWTH_LIMIT * size + slack


def defeature(shape, features, decisions, progress=None, timeout=30.0, graph=None) -> ExecResult:
    t0 = time.time()
    todo = [f for f in features if decisions[f.id].action == "remove"]
    todo.sort(key=lambda f: (ORDER.index(f.kind) if f.kind in ORDER else 99, f.size))
    res = ExecResult(shape)
    if not todo:
        res.strategy = "nothing to remove"
        return res
    w = worker()
    brep = occ.to_brep_string(shape)

    def say(msg):
        res.log.append(msg)
        if progress:
            progress(msg)

    all_faces = sorted({fid for f in todo for fid in f.faces})
    areas = [occ.area(f) for f in occ.faces_of(shape)]
    box = occ.bbox(shape)
    diag = max(1.0, occ.norm(occ.sub(box[3:], box[:3])))
    slack = 1e-3 * diag
    vol_slack = slack * sum(areas) * 1e-3
    say(f"Removing {len(todo)} features ({len(all_faces)} faces) in one pass")
    one_by_one = [_limits(f.faces, f.size, areas, 0) for f in todo]
    batch = (sum(l[0] for l in one_by_one) + vol_slack, max(l[1] for l in one_by_one) + slack)
    r = w.call("_remove_faces", brep, all_faces, 0.0, batch, timeout=timeout)
    if r["ok"]:
        res.shape = occ.from_brep_string(r["brep"])
        res.removed = [f.id for f in todo]
        res.strategy = "single pass"
        res.seconds = time.time() - t0
        return res

    say(f"Single pass failed ({r['error']}); switching to feature-by-feature")
    res.strategy = "feature-by-feature"
    cur = shape
    cur_brep = brep
    faces = occ.faces_of(cur)
    sigs = [_signature(f) for f in faces]
    idmap = {i: i for i in range(len(faces))}   # original face id -> current face id
    tol = 1e-3 * diag
    todo_by_face = {fid: f for f in todo for fid in f.faces}
    owner_kind = {fid: f.kind for f in features for fid in f.faces}
    done = set()

    exact_history = True

    def apply(r):
        nonlocal cur, cur_brep, idmap, exact_history
        new = occ.from_brep_string(r["brep"])
        if r.get("map") is not None:   # exact, from the kernel's history
            m = r["map"]
            idmap = {oid: m[cid] for oid, cid in idmap.items() if cid < len(m) and m[cid] >= 0}
        else:                          # healed result: fall back to geometric signatures
            exact_history = False
            cur_sigs = {cid: sigs[oid] for oid, cid in idmap.items()}
            order = sorted(cur_sigs)
            mapping = remap_faces([cur_sigs[c] for c in order], new, tol)
            idmap = {oid: mapping[order.index(cid)] for oid, cid in idmap.items() if order.index(cid) in mapping}
        cur, cur_brep = new, r["brep"]

    for n, ft in enumerate(todo, 1):
        if ft.id in done:
            continue
        say(f"[{n}/{len(todo)}] {ft.kind} #{ft.id}")
        # attempts: the feature alone; then with its tangent blend neighbours; then with a fuzzy tolerance
        attempts = [(set(ft.faces), 0.0)]
        if ft.kind in ("fillet", "chamfer") and graph is not None:
            grp = set(ft.faces) | _blend_neighbours(ft, graph, todo_by_face, owner_kind)
            if grp != set(ft.faces):
                attempts.append((grp, 0.0))
            attempts.append((grp, 1e-4 * diag))
        else:
            attempts.append((set(ft.faces), 1e-4 * diag))
        last = "kernel returned the part unchanged"
        alive = [f for f in ft.faces if f in idmap]
        if not alive and exact_history:
            # every face was deleted by an earlier removal: the feature is already gone
            done.add(ft.id)
            res.removed.append(ft.id)
            continue
        for grp, fz in attempts:
            grp = {f for f in grp if f in idmap} if exact_history else grp
            cur_ids = [idmap.get(f) for f in sorted(grp)]
            if not cur_ids or any(c is None for c in cur_ids):
                last = "geometry changed by an earlier removal"
                continue
            vol, grow = _limits(grp, ft.size, areas, 0)
            r = w.call("_remove_faces", cur_brep, cur_ids, fz, (vol + vol_slack, grow + slack), timeout=timeout)
            if r["ok"]:
                apply(r)
                done.add(ft.id)
                res.removed.append(ft.id)
                # blend features removed as part of the group count as removed too
                for fid in grp:
                    other = todo_by_face.get(fid)
                    if other is not None and other.id not in done and all(x in grp or x not in idmap for x in other.faces):
                        done.add(other.id)
                        res.removed.append(other.id)
                break
            last = r["error"]
        else:
            res.failed[ft.id] = last
    res.shape = cur
    res.seconds = time.time() - t0
    return res
