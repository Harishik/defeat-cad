"""Export a session as standalone, runnable code (Python/OCP and C++/OCCT).

Faces are located by geometric signature (surface type + centroid + area), not by index,
so the script works on the original STEP regardless of how a CAD system orders faces.
"""
from __future__ import annotations

import datetime
import json

from . import occ


def step_signatures(faces, features, removed_ids, steps):
    """The kernel calls the app applied, in order: every face they name once (type + centroid + area, so
    the script finds it in the ORIGINAL file), and each step as indices into that list. Returns
    (face signatures, steps with a note naming the features each one removes)."""
    uniq = sorted({fid for st in steps for fid in st})
    pos = {fid: i for i, fid in enumerate(uniq)}
    sigs = []
    for fid in uniq:
        f = faces[fid]
        c = occ.centroid(f)
        sigs.append({"type": occ.surf_info(f).kind, "centroid": [round(v, 5) for v in c], "area": round(occ.area(f), 6)})
    owner = {fid: ft for ft in features if ft.id in removed_ids
             for fid in set(ft.faces) | set(getattr(ft, "attached", None) or [])}
    out = []
    for i, st in enumerate(steps):
        kinds = sorted({f"{owner[fid].kind} #{owner[fid].id}" for fid in st if fid in owner})
        note = ", ".join(kinds[:6]) + (f" and {len(kinds) - 6} more" if len(kinds) > 6 else "") or "blend faces"
        out.append({"note": f"step {i + 1}: {note}", "faces": [pos[fid] for fid in st]})
    return sigs, out


def feature_signatures(faces, features, removed_ids):
    out = []
    for ft in features:
        if ft.id not in removed_ids:
            continue
        sigs = []
        for fid in sorted(set(ft.faces) | set(getattr(ft, "attached", None) or [])):   # its own blends go with it
            f = faces[fid]
            c = occ.centroid(f)
            sigs.append({"type": occ.surf_info(f).kind, "centroid": [round(v, 5) for v in c],
                         "area": round(occ.area(f), 6)})
        out.append({"feature": ft.kind, "note": ft.note, "faces": sigs})
    return out


PY_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFeat defeaturing script — generated {date}
Part: {part}    Preset: {preset}    Target mesh size h = {h}

Reproduces the defeatured model from the ORIGINAL STEP file using OpenCASCADE (OCP).
    pip install cadquery-ocp
    python {script} {part} {part_stem}_defeatured.step

Features removed ({n}):
{summary}

The removal is replayed as the app ran it: the same kernel calls in the same order. Each face is found
once in the original file by surface type + centroid + area, then followed through every call by the
kernel's own history, so a face reshaped by an earlier removal is still found.
"""
import sys
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_ListOfShape
from OCP.TopoDS import TopoDS

FACES = {faces}

STEPS = {steps}

TYPES = {{GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
         GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus"}}


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        sys.exit(f"cannot read {{path}}")
    r.TransferRoots()
    return r.OneShape()


def faces_of(shape):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return [TopoDS.Face_s(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def signature(face):
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, p)
    c = p.CentreOfMass()
    kind = TYPES.get(BRepAdaptor_Surface(face, True).GetType(), "other")
    return kind, (c.X(), c.Y(), c.Z()), p.Mass()


def locate(faces, sig, tol):
    """Return the face matching a stored signature (type, centroid, area)."""
    best, bd = None, tol
    for f, (k, c, a) in faces:
        if sig["type"] in ("plane", "cylinder", "cone", "sphere", "torus") and k != sig["type"]:
            continue
        if abs(a - sig["area"]) > 0.02 * sig["area"] + 1e-6:
            continue
        d = sum((x - y) ** 2 for x, y in zip(c, sig["centroid"])) ** 0.5
        if d < bd:
            best, bd = f, d
    return best


def remove(shape, faces):
    lst = TopTools_ListOfShape()
    for f in faces:
        lst.Append(f)
    algo = BRepAlgoAPI_Defeaturing()
    algo.SetShape(shape)
    algo.AddFacesToRemove(lst)
    algo.SetToFillHistory(True)
    algo.Build()
    if not algo.IsDone():
        return None
    out = algo.Shape()
    return (out, algo) if BRepCheck_Analyzer(out).IsValid() else None


def main(src, dst):
    shape = read_step(src)
    tol = {tol}
    faces = [(f, signature(f)) for f in faces_of(shape)]
    current = [locate(faces, s, tol) for s in FACES]   # where each listed face is now (None once deleted)
    result = shape
    for st in STEPS:
        fs = [current[i] for i in st["faces"]]
        if not all(f is not None for f in fs):
            print(f"  skip  {{st['note']}} (not found in this file)")
            continue
        r = remove(result, fs)
        if r is None:
            print(f"  kept  {{st['note']}} (kernel could not remove)")
            continue
        result, algo = r
        for i, f in enumerate(current):
            if f is None:
                continue
            if algo.IsDeleted(f):
                current[i] = None
            else:
                mods = algo.Modified(f)
                if mods.Size():
                    current[i] = TopoDS.Face_s(mods.First())
    # merge faces and edges split by the removals: same shape, fewer mesh constraints
    u = ShapeUpgrade_UnifySameDomain(result, True, True, False)
    u.Build()
    if BRepCheck_Analyzer(u.Shape()).IsValid():
        result = u.Shape()
    w = STEPControl_Writer()
    w.Transfer(result, STEPControl_AsIs)
    w.Write(dst)
    print(f"wrote {{dst}}: {{len(faces)}} -> {{len(faces_of(result))}} faces")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "{part}",
         sys.argv[2] if len(sys.argv) > 2 else "{part_stem}_defeatured.step")
'''

CPP_TEMPLATE = r'''// DeFeat defeaturing program — generated {date}
// Part: {part}   Preset: {preset}   Target mesh size h = {h}
// Build: g++ -std=c++17 defeature.cpp -I$OCCT/include/opencascade -L$OCCT/lib \
//        -lTKernel -lTKMath -lTKBRep -lTKG3d -lTKGeomBase -lTKTopAlgo -lTKBO -lTKShHealing \
//        -lTKDESTEP -lTKXSBase -lTKBool -o defeature        (OCCT >= 7.4)
// Run:   ./defeature {part} {part_stem}_defeatured.step
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <cmath>
#include <iostream>
#include <vector>

struct Sig {{ double cx, cy, cz, area; }};
struct Step {{ const char* note; std::vector<int> faces; }};

// every face the removal names, located once in the original file
static const std::vector<Sig> FACES = {{
{cpp_faces}
}};
// the kernel calls the app applied, in order, as indices into FACES
static const std::vector<Step> STEPS = {{
{cpp_steps}
}};

static TopoDS_Face locate(const TopTools_IndexedMapOfShape& m, const Sig& s, double tol) {{
  TopoDS_Face best; double bd = tol;
  for (int i = 1; i <= m.Extent(); ++i) {{
    TopoDS_Face f = TopoDS::Face(m(i));
    GProp_GProps p; BRepGProp::SurfaceProperties(f, p);
    if (std::abs(p.Mass() - s.area) > 0.02 * s.area + 1e-6) continue;
    gp_Pnt c = p.CentreOfMass();
    double d = std::sqrt(std::pow(c.X() - s.cx, 2) + std::pow(c.Y() - s.cy, 2) + std::pow(c.Z() - s.cz, 2));
    if (d < bd) {{ bd = d; best = f; }}
  }}
  return best;
}}

int main(int argc, char** argv) {{
  const char* src = argc > 1 ? argv[1] : "{part}";
  const char* dst = argc > 2 ? argv[2] : "{part_stem}_defeatured.step";
  STEPControl_Reader r;
  if (r.ReadFile(src) != IFSelect_RetDone) {{ std::cerr << "cannot read " << src << "\n"; return 1; }}
  r.TransferRoots();
  TopoDS_Shape shape = r.OneShape();
  TopTools_IndexedMapOfShape m; TopExp::MapShapes(shape, TopAbs_FACE, m);
  // where each listed face is now: followed through every removal by the kernel's history
  std::vector<TopoDS_Face> current;
  for (const auto& s : FACES) current.push_back(locate(m, s, {tol}));
  TopoDS_Shape result = shape;
  for (const auto& st : STEPS) {{
    TopTools_ListOfShape toRemove; bool found = true;
    for (int i : st.faces) {{ if (current[i].IsNull()) {{ found = false; break; }} toRemove.Append(current[i]); }}
    if (!found) {{ std::cout << "  skip  " << st.note << "\n"; continue; }}
    BRepAlgoAPI_Defeaturing algo;
    algo.SetShape(result);
    algo.AddFacesToRemove(toRemove);
    algo.SetToFillHistory(Standard_True);
    algo.Build();
    if (!algo.IsDone() || !BRepCheck_Analyzer(algo.Shape()).IsValid()) {{ std::cout << "  kept  " << st.note << "\n"; continue; }}
    for (auto& f : current) {{
      if (f.IsNull()) continue;
      if (algo.IsDeleted(f)) {{ f.Nullify(); continue; }}
      const TopTools_ListOfShape& mods = algo.Modified(f);
      if (!mods.IsEmpty()) f = TopoDS::Face(mods.First());
    }}
    result = algo.Shape();
  }}
  // merge faces and edges split by the removals: same shape, fewer mesh constraints
  ShapeUpgrade_UnifySameDomain unify(result, Standard_True, Standard_True, Standard_False);
  unify.Build();
  TopoDS_Shape out = BRepCheck_Analyzer(unify.Shape()).IsValid() ? unify.Shape() : result;
  STEPControl_Writer w; w.Transfer(out, STEPControl_AsIs); w.Write(dst);
  std::cout << "wrote " << dst << "\n";
  return 0;
}}
'''


def _summary(feats):
    from collections import Counter
    c = Counter(f["feature"] for f in feats)
    return "\n".join(f"    - {n} × {k}" for k, n in c.most_common())


def python_script(part_name, preset, h, feats, tol, face_sigs, steps):
    stem = part_name.rsplit(".", 1)[0]
    return PY_TEMPLATE.format(date=datetime.date.today().isoformat(), part=part_name, part_stem=stem,
                              preset=preset, h=h, n=len(feats), summary=_summary(feats),
                              faces=json.dumps(face_sigs, indent=1), steps=json.dumps(steps, indent=1),
                              script=f"defeature_{stem}.py", tol=round(tol, 6))


def cpp_program(part_name, preset, h, face_sigs, steps, tol):
    stem = part_name.rsplit(".", 1)[0]
    faces = [f"  {{{s['centroid'][0]}, {s['centroid'][1]}, {s['centroid'][2]}, {s['area']}}}," for s in face_sigs]
    rows = []
    for st in steps:
        note = st["note"].replace('"', "'")
        rows.append(f'  {{"{note}", {{{", ".join(str(i) for i in st["faces"])}}}}},')
    return CPP_TEMPLATE.format(date=datetime.date.today().isoformat(), part=part_name, part_stem=stem,
                               preset=preset, h=h, cpp_faces="\n".join(faces), cpp_steps="\n".join(rows), tol=round(tol, 6))
