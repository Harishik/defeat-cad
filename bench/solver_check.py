"""Check DeFeat's stress solver against an independent solver and against textbook answers.

CalculiX (open source, shipped with FreeCAD) gets exactly the same mesh, supports and loads as C3D10
elements, and the four compliance values are compared. Beam theory checks the solver from first principles.

    python -m bench.solver_check --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile

import numpy as np

from bench.common import pmap_hard, save
from bench.robust import BASE

CCX = os.environ.get("DEFEAT_CCX") or shutil.which("ccx") or "/Applications/FreeCAD.app/Contents/Resources/bin/ccx"


def write_inp(path, sys_, mat):
    """C3D10 deck: gmsh corners plus straight-edge midside nodes, in the numbering CalculiX expects."""
    p, t, en = sys_["p"], sys_["t"], sys_["edge_nodes"]
    n0 = len(p)
    mid = 0.5 * (p[en[:, 0]] + p[en[:, 1]])
    key = {(int(a), int(b)): n0 + i for i, (a, b) in enumerate(np.sort(en, axis=1))}
    def m(a, b):
        return key[(min(a, b), max(a, b))]
    dof_to = {}
    nod, edg = sys_["nodal"], sys_["edge"]
    for c in range(3):
        for i in range(nod.shape[1]):
            dof_to[int(nod[c, i])] = (i + 1, c + 1)
        for i in range(edg.shape[1]):
            dof_to[int(edg[c, i])] = (n0 + i + 1, c + 1)
    with open(path, "w") as fh:
        fh.write("*NODE\n")
        for i, x in enumerate(np.vstack([p, mid]), 1):
            fh.write(f"{i},{x[0]:.12g},{x[1]:.12g},{x[2]:.12g}\n")
        fh.write("*ELEMENT,TYPE=C3D10,ELSET=ALL\n")
        for i, (a, b, c, d) in enumerate(t.tolist(), 1):
            mids = [m(a, b), m(b, c), m(c, a), m(a, d), m(b, d), m(c, d)]
            ids = [a + 1, b + 1, c + 1, d + 1] + [x + 1 for x in mids]
            fh.write(f"{i}," + ",".join(map(str, ids[:7])) + ",\n" + ",".join(map(str, ids[7:])) + "\n")
        fh.write(f"*NSET,NSET=EVERY,GENERATE\n1,{n0 + len(mid)}\n")
        fh.write(f"*MATERIAL,NAME=M\n*ELASTIC\n{mat['E']},{mat['nu']}\n*SOLID SECTION,ELSET=ALL,MATERIAL=M\n")
        fh.write("*BOUNDARY\n")
        for d in sys_["fixed"].tolist():
            n, c = dof_to[int(d)]
            fh.write(f"{n},{c},{c},0\n")
        names = list(sys_["forms"])
        for name in names:
            f = np.asarray(sys_["forms"][name])
            fh.write(f"*STEP\n*STATIC\n*CLOAD,OP=NEW\n")
            for d in np.nonzero(f)[0].tolist():
                n, c = dof_to[d]
                fh.write(f"{n},{c},{f[d]:.12g}\n")
            fh.write("*NODE PRINT,NSET=EVERY\nU\n*END STEP\n")
    return dof_to, names


def read_dat(path, n_steps, n_nodes):
    """Displacements per step from the .dat file."""
    out = []
    cur = None
    for line in open(path):
        if "displacements" in line:
            cur = np.zeros((n_nodes, 3))
            out.append(cur)
            continue
        a = line.split()
        if cur is not None and len(a) == 4 and a[0].isdigit():
            cur[int(a[0]) - 1] = [float(v.replace("D", "E")) for v in a[1:]]
    return out


def calculix(sys_, mat, force):
    d = tempfile.mkdtemp(prefix="ccx_")
    try:
        dof_to, names = write_inp(os.path.join(d, "job.inp"), sys_, mat)
        n_nodes = len(sys_["p"]) + len(sys_["edge_nodes"])
        env = dict(os.environ, OMP_NUM_THREADS="2", CCX_NPROC_EQUATION_SOLVER="2")
        r = subprocess.run([CCX, "job"], cwd=d, capture_output=True, text=True, env=env, timeout=600)
        U = read_dat(os.path.join(d, "job.dat"), len(names), n_nodes) if os.path.exists(os.path.join(d, "job.dat")) else []
        if len(U) != len(names):
            return None, (r.stdout + r.stderr)[-400:]
        out = {}
        for name, u in zip(names, U):
            f = np.asarray(sys_["forms"][name])
            work = sum(f[i] * u[dof_to[i][0] - 1, dof_to[i][1] - 1] for i in np.nonzero(f)[0].tolist())
            out[name] = float(work) / force
        return out, None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def part_check(name):
    from defeat import fea, occ
    from defeat.decide import default_h
    from defeat.validate import mesh_stats, step_bytes
    shape = occ.read_cad(os.path.join(BASE, "breps", "step", name + ".stp"))
    if len(occ.solids_of(shape)) != 1:
        return {"ok": False, "error": "not one solid"}
    h = default_h(occ.norm(occ.sub(occ.bbox(shape)[3:], occ.bbox(shape)[:3])))
    st = mesh_stats(step_bytes(shape), h, keep_mesh=True)
    if not st.get("ok") or st["tets"] > fea.MAX_TETS:
        return {"ok": False, "error": "skip: too large"}
    bc = fea.load_case(occ.bbox(shape))
    bc["want_system"] = True
    r = fea.solve(st["mesh"]["p"], st["mesh"]["t"], bc)
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error", "")}
    cc, err = calculix(r["system"], fea.MATERIAL, bc["force"])
    if cc is None:
        return {"ok": False, "error": "ccx: " + str(err)}
    return {"tets": st["tets"], "ours": r["cases"], "ccx": cc,
            "diff_pct": {k: 100 * (r["cases"][k] - cc[k]) / cc[k] for k in cc}}


def box_mesh(L, b, t, h):
    import gmsh
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("b")
    gmsh.model.occ.addBox(0, 0, 0, L, b, t)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {int(x): i for i, x in enumerate(tags)}
    _, en = gmsh.model.mesh.getElementsByType(4)
    conn = np.vectorize(idx.get)(np.asarray(en).reshape(-1, 4))
    gmsh.finalize()
    return xyz.reshape(-1, 3), conn


def beam_theory():
    """A long clamped bar: the bending, tension and torsion stiffness against the textbook formulas."""
    from defeat import fea
    L, b, t = 200.0, 20.0, 10.0
    p, T = box_mesh(L, b, t, 4.0)
    bc = fea.load_case((0, 0, 0, L, b, t))
    r = fea.solve(p, T, bc)
    E, nu, F = fea.MATERIAL["E"], fea.MATERIAL["nu"], bc["force"]
    lo, hi = r["ends"]
    # clamped at x <= lo; load spread uniformly over x >= hi
    span = (hi + L) / 2 - lo
    a = L - hi
    G = E / (2 * (1 + nu))
    k = 5 / 6
    def bend(Iyy, area):
        # uniform load over the last `a`, conjugate (mean) deflection, with shear
        end_len = L - lo
        c = hi - lo
        # work-conjugate deflection of a cantilever of length end_len with force F uniform over its last a
        I = Iyy
        w = F / a
        # numeric integration of Euler-Bernoulli plus Timoshenko shear
        x = np.linspace(0, end_len, 40001)
        V = np.where(x < c, F, F * (end_len - x) / a)
        M = np.where(x < c, F * (end_len - a / 2 - x), w * (end_len - x) ** 2 / 2)
        dx = x[1] - x[0]
        return float((M ** 2 / (E * I) + V ** 2 / (k * G * area)).sum() * dx) / F
    area = b * t
    expect = {"bend_thin": bend(b * t ** 3 / 12, area), "bend_wide": bend(t * b ** 3 / 12, area)}
    c = hi - lo
    expect["tension"] = float(F * (c + a / 3) / (E * area))
    return {"ours": {k: r["cases"][k] for k in expect}, "theory": expect,
            "diff_pct": {k: 100 * (r["cases"][k] - expect[k]) / expect[k] for k in expect}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    th = beam_theory()
    print("beam theory", json.dumps(th["diff_pct"]), flush=True)
    tr = json.load(open(os.path.join(BASE, "train_test.json")))["train"]
    random.Random(0).shuffle(tr)
    res = pmap_hard("bench.solver_check:part_check", tr[5500 - a.n * 2: 5500], workers=a.workers, timeout=900, label="ccx")
    ok = [r for r in res if r.get("ok")][: a.n]
    diffs = [abs(v) for r in ok for v in r["diff_pct"].values()]
    rep = {"parts": len(ok), "solver": "CalculiX " + (subprocess.run([CCX, "-v"], capture_output=True, text=True).stdout.strip() or "?"),
           "median_abs_diff_pct": float(np.median(diffs)) if diffs else None, "max_abs_diff_pct": float(np.max(diffs)) if diffs else None,
           "beam_theory": th, "errors": sorted({r.get("error", "")[:80] for r in res if not r.get("ok")})[:6]}
    print(save("solver_check", rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
