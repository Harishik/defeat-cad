"""Guarded defeaturing executor.

Every OCCT call runs in a worker process with a timeout (BRepAlgoAPI_Defeaturing can hang on
some fillet faces, OCCT bug 0033561). Strategy: one batched removal first; if it fails or the
result is invalid, fall back to feature-by-feature removal in a safe order, keeping (and
reporting) anything the kernel cannot remove. The output is never an invalid solid.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

from . import occ

ORDER = ["emboss", "chamfer", "fillet", "hole", "slot", "pocket", "step", "boss", "rib", "protrusion"]


# ---------------------------------------------------------------- worker side
def _remove_faces(brep: bytes, face_ids: list[int]):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
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
    d.SetToFillHistory(False)
    d.Build()
    if not d.IsDone():
        return {"ok": False, "error": "kernel could not rebuild the surrounding faces"}
    res = d.Shape()
    if not _occ.is_valid(res):
        res2 = _occ.heal(res)
        if not _occ.is_valid(res2):
            return {"ok": False, "error": "result was not a valid solid"}
        res = res2
    if len(_occ.solids_of(res)) != len(_occ.solids_of(shape)):
        return {"ok": False, "error": "removal split or merged bodies"}
    if len(_occ.faces_of(res)) >= len(faces):
        return {"ok": False, "error": "kernel returned the part unchanged"}
    return {"ok": True, "brep": _occ.to_brep_string(res)}


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


def _worker_loop(conn):
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


def defeature(shape, features, decisions, progress=None, timeout=30.0) -> ExecResult:
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
    say(f"Removing {len(todo)} features ({len(all_faces)} faces) in one pass")
    r = w.call("_remove_faces", brep, all_faces, timeout=timeout)
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
    tol = 1e-3 * max(1.0, occ.norm(occ.sub(occ.bbox(shape)[3:], occ.bbox(shape)[:3])))
    for n, ft in enumerate(todo, 1):
        cur_ids = [idmap.get(f) for f in ft.faces]
        if any(c is None for c in cur_ids):
            res.failed[ft.id] = "geometry changed by an earlier removal"
            continue
        say(f"[{n}/{len(todo)}] {ft.kind} #{ft.id}")
        r = w.call("_remove_faces", cur_brep, cur_ids, timeout=timeout)
        if not r["ok"]:
            res.failed[ft.id] = r["error"]
            continue
        new = occ.from_brep_string(r["brep"])
        cur_sigs = {cid: sigs[oid] for oid, cid in idmap.items()}
        order = sorted(cur_sigs)
        mapping = remap_faces([cur_sigs[c] for c in order], new, tol)
        idmap = {oid: (order.index(cid) in mapping and mapping[order.index(cid)]) for oid, cid in idmap.items()}
        idmap = {k: v for k, v in idmap.items() if v is not False and v is not None}
        cur, cur_brep = new, r["brep"]
        res.removed.append(ft.id)
    res.shape = cur
    res.seconds = time.time() - t0
    return res
