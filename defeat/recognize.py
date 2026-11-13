"""Rule-based feature recognizer producing feature *instances* (groups of faces).

Order: inner-loop regions (holes, slots, pockets, bosses, embosses) -> thin ribs ->
blend chains (fillets) -> chamfers. Each face belongs to at most one feature.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from . import occ
from .graph import CONCAVE, CONVEX, SMOOTH, FaceGraph, build_graph

FEATURE_CLASSES = ["hole", "slot", "pocket", "step", "boss", "rib", "emboss", "fillet", "chamfer", "protrusion"]


@dataclass
class Feature:
    id: int
    kind: str
    faces: list
    params: dict = field(default_factory=dict)
    size: float = 0.0           # governing size in model units (diameter, radius, thickness ...)
    host: int | None = None
    source: str = "rules"
    confidence: float = 0.9
    note: str = ""
    # the feature's own edge blends and chamfers (a boss's tip chamfer, the blend at a hole's bottom):
    # listed as separate features so they can be kept or removed alone, but they must go whenever this
    # feature goes, or the kernel is asked to remove half a boss
    attached: list = field(default_factory=list)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "faces": self.faces, "params": self.params,
                "size": round(self.size, 4), "host": self.host, "source": self.source,
                "confidence": round(self.confidence, 3), "note": self.note, "attached": self.attached}


def _faces_bbox(g: FaceGraph, fids):
    pts = []
    for f in fids:
        x0, y0, z0, x1, y1, z1 = occ.bbox(g.faces[f])
        pts += [(x0, y0, z0), (x1, y1, z1)]
    xs, ys, zs = zip(*pts)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _extent_along(g, fids, origin, n):
    """Signed min/max distance of region vertices from plane (origin, n)."""
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    lo, hi = float("inf"), float("-inf")
    for f in fids:
        for v in occ.sub_shapes(g.faces[f], TopAbs_VERTEX):
            p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(v))
            d = occ.dot(occ.sub((p.X(), p.Y(), p.Z()), origin), n)
            lo, hi = min(lo, d), max(hi, d)
        # sample the face too (vertex-free faces, e.g. full cylinders)
        c = g.nodes[f].centroid
        d = occ.dot(occ.sub(c, origin), n)
        lo, hi = min(lo, d), max(hi, d)
    return lo, hi


def _same_axis(a: occ.SurfInfo, b: occ.SurfInfo, tol):
    if a.axis is None or b.axis is None:
        return False
    if abs(abs(occ.dot(occ.unit(a.axis), occ.unit(b.axis))) - 1) > 1e-3:
        return False
    d = occ.sub(b.loc, a.loc)
    ax = occ.unit(a.axis)
    perp = occ.sub(d, tuple(occ.dot(d, ax) * c for c in ax))
    return occ.norm(perp) < tol


class Recognizer:
    def __init__(self, g: FaceGraph):
        self.g = g
        self.owner: dict[int, int] = {}
        self.features: list[Feature] = []
        self.tol = 1e-3 * g.char_len

    # ------------------------------------------------------------ helpers
    SPLIT_KINDS = {"hole", "slot", "pocket", "step", "boss", "rib", "protrusion"}

    def _add(self, kind, faces, **kw):
        faces = sorted(set(f for f in faces if f not in self.owner))
        attached = []
        if kind in self.SPLIT_KINDS:
            # edge blends and chamfers on a feature are their own (separately removable) features,
            # e.g. keep a large hole but drop its rim chamfer
            def wall(f):  # the curved walls of holes and slots are the feature itself
                n = self.g.nodes[f]
                return kind in ("hole", "slot") and n.kind in ("cylinder", "cone") and n.concave_axis
            core = [f for f in faces if wall(f) or not (self.is_blend(f) or self.is_chamfer_like(f))]
            if core:
                attached = [f for f in faces if f not in core]
                faces = core
        if not faces:
            return None
        ft = Feature(len(self.features), kind, faces, attached=attached, **kw)
        for f in faces:
            self.owner[f] = ft.id
        self.features.append(ft)
        return ft

    def _edge_in_inner_loop_of(self, face, edge_id):
        for loop in self.g.inner_wires.get(face, []):
            if edge_id in loop:
                return True
        return False

    def is_chamfer_like(self, fid):
        """Narrow planar or conical strip meeting its two main neighbours at ~30-60 deg convex edges."""
        g = self.g
        n = g.nodes[fid]
        if n.kind not in ("plane", "cone"):
            return False
        es = sorted((g.edges[e] for eids in n.neighbors.values() for e in eids), key=lambda e: -e.length)
        if len(es) < 2:
            return False
        L = es[0].length
        width = n.area / max(L, 1e-9)
        if width > 0.05 * g.char_len or width > 0.3 * L:
            return False
        lo, hi = math.radians(15), math.radians(75)
        main = [e for e in es if e.convexity != SMOOTH][:2]
        if len(main) < 2 or not all(e.convexity == CONVEX and lo < e.angle < hi for e in main):
            return False
        if n.kind == "cone":
            return True
        # planar: strip (two long edges) or a small corner chamfer whose edges are all ~45 deg
        return all(e.convexity == CONVEX and lo < e.angle < hi for e in es[:3]) or len(es) >= 4

    def is_blend(self, fid):
        n = self.g.nodes[fid]
        if n.kind == "plane":
            return False
        smooth_nbrs = {m for m, eids in n.neighbors.items()
                       if any(self._blend_edge(n, self.g.edges[e]) for e in eids)
                       and not self._same_surface(fid, m)}
        return len(smooth_nbrs) >= 2

    def _blend_edge(self, n, e):
        """A smooth edge along which face n could be blending two faces.

        A cylindrical fillet meets its two parent faces along lines parallel to its axis. A cylinder that
        meets its neighbours smoothly around its circumference (e.g. a flange wall with both rims rounded)
        is a wall bounded by blends, not a blend itself.
        """
        if e.convexity != SMOOTH:
            return False
        if n.kind == "cylinder" and n.info.axis is not None and e.tangent is not None:
            return abs(occ.dot(occ.unit(n.info.axis), e.tangent)) > 0.7
        return True

    def _same_surface(self, a, b):
        na, nb = self.g.nodes[a], self.g.nodes[b]
        if na.kind != nb.kind:
            return False
        if na.kind in ("cylinder", "cone"):
            return abs(na.info.radius - nb.info.radius) < self.tol and _same_axis(na.info, nb.info, self.tol)
        if na.kind == "plane":
            return abs(occ.dot(na.info.normal, nb.info.normal) - 1) < 1e-6
        return False

    def blend_radius(self, fid):
        n = self.g.nodes[fid]
        if n.kind in ("cylinder", "sphere", "cone") and n.info.radius > 0:
            return n.info.radius
        if n.kind == "torus":
            return n.info.radius
        return 1.0 / n.max_curv if n.max_curv > 1e-9 else float("inf")

    # ------------------------------------------------------------ 1. inner-loop regions
    def _flood(self, host, loop):
        g = self.g
        seeds = set()
        for eid in loop:
            e = g.edges[eid]
            other = e.f2 if e.f1 == host else e.f1
            if other != host:
                seeds.add(other)
        region, q = set(seeds), deque(seeds)
        limit = 0.3 * g.total_area
        acc = sum(g.nodes[f].area for f in region)
        while q:
            f = q.popleft()
            for m, eids in g.nodes[f].neighbors.items():
                if m == host or m in region:
                    continue
                if all(self._edge_in_inner_loop_of(m, e) for e in eids):
                    continue  # m hosts this region from the other side (through-feature)
                region.add(m)
                q.append(m)
                acc += g.nodes[m].area
                if acc > limit or len(region) > 400:
                    return None
        return region

    def find_regions(self):
        g = self.g
        regions = []
        for host, loops in g.inner_wires.items():
            for loop in loops:
                if not loop:
                    continue
                reg = self._flood(host, loop)
                if reg:
                    regions.append((host, loop, reg))
        # merge regions that share faces (same physical feature reached from two hosts)
        merged = []
        for host, loop, reg in sorted(regions, key=lambda r: g.nodes[r[0]].area, reverse=True):
            for m in merged:
                if m[2] & reg or any(n in reg for f in m[2] for n in g.nodes[f].neighbors) and m[0] == host:
                    m[2] |= reg
                    m[1].extend(loop)
                    break
            else:
                merged.append([host, list(loop), set(reg)])
        for host, loop, reg in merged:
            self._classify_region(host, loop, reg)

    def _classify_region(self, host, loop, reg):
        g = self.g
        hn = g.nodes[host]
        faces = sorted(reg)
        # protrusion vs depression
        if hn.kind == "plane":
            lo, hi = _extent_along(g, faces, hn.centroid, hn.info.normal)
            protrude = hi > abs(lo)
            height = hi if protrude else -lo
        else:
            conv = [g.edges[e].convexity for e in loop]
            protrude = conv.count(CONCAVE) > conv.count(CONVEX)
            bb = _faces_bbox(g, faces)
            height = min(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        kinds = [g.nodes[f].kind for f in faces]
        cyl = [f for f in faces if g.nodes[f].kind in ("cylinder", "cone") and not self.is_blend(f)]
        blends = [f for f in faces if self.is_blend(f)]
        params = {"height": round(height, 3), "n_faces": len(faces)}
        if not protrude:
            hole_walls = [f for f in faces if g.nodes[f].kind in ("cylinder", "cone") and g.nodes[f].concave_axis]
            blends = [f for f in blends if f not in hole_walls]
            coaxial = hole_walls and all(_same_axis(g.nodes[hole_walls[0]].info, g.nodes[f].info, self.tol * 5)
                                         for f in hole_walls)
            others = [f for f in faces if f not in hole_walls and f not in blends]
            if coaxial and all(g.nodes[f].kind == "plane" for f in others) and len(others) <= 2 * len(hole_walls):
                d = 2 * min(g.nodes[f].info.radius for f in hole_walls)
                params.update(diameter=round(d, 3), depth=round(height, 3), through=len(others) == 0)
                self._add("hole", faces, params=params, size=d, host=host,
                          note=f"Ø{d:.2f} {'through' if not others else 'blind'} hole")
                return
            if len(hole_walls) >= 2 and len(set(round(g.nodes[f].info.radius, 3) for f in hole_walls)) == 1:
                w = 2 * g.nodes[hole_walls[0]].info.radius
                params.update(width=round(w, 3))
                self._add("slot", faces, params=params, size=w, host=host, note=f"slot, width {w:.2f}")
                return
            bb = _faces_bbox(g, faces)
            dims = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
            params.update(min_width=round(dims[1], 3))
            # a pocket is closed on all sides; a step opens onto a side face of the part
            region = set(faces)
            exits = {m for f in faces for m, eids in g.nodes[f].neighbors.items()
                     if m not in region and m != host
                     and not any(self._edge_in_inner_loop_of(m, e) for e in eids)}
            if exits:
                self._add("step", faces, params=dict(params, open=True), size=dims[1], host=host,
                          note=f"step / shoulder, depth {height:.2f}", confidence=0.75)
                return
            self._add("pocket", faces, params=params, size=min(dims[1], height if height > 0 else dims[1]),
                      host=host, note=f"pocket / cut-out, depth {height:.2f}")
            return
        # protrusions
        boss_walls = [f for f in cyl if g.nodes[f].concave_axis is False]
        if boss_walls:
            d = 2 * max(g.nodes[f].info.radius for f in boss_walls)
            params.update(diameter=round(d, 3))
            self._add("boss", faces, params=params, size=d, host=host,
                      note=f"boss Ø{d:.2f}, height {height:.2f}")
            return
        bb = _faces_bbox(g, faces)
        footprint = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])[1:]
        if height < 0.05 * g.char_len and (len(faces) >= 8 or height < 0.25 * footprint[0]):
            self._add("emboss", faces, params=params, size=height, host=host,
                      note=f"emboss / text / logo, height {height:.2f}", confidence=0.85)
            return
        self._add("protrusion", faces, params=params, size=height, host=host,
                  note=f"protrusion, height {height:.2f}", confidence=0.6)

    # ------------------------------------------------------------ 2. thin ribs / gussets
    def find_ribs(self, max_t=None):
        g = self.g
        max_t = max_t or 0.06 * g.char_len
        planes = [n for n in g.nodes if n.kind == "plane" and n.id not in self.owner]
        cand = []
        for c in planes:  # c = cap face
            nb = [m for m, eids in c.neighbors.items()
                  if g.nodes[m].kind == "plane" and all(g.edges[e].convexity == CONVEX for e in eids)]
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    A, B = g.nodes[nb[i]], g.nodes[nb[j]]
                    if occ.dot(A.info.normal, B.info.normal) > -0.999:
                        continue
                    if max(A.area, B.area) > 0.1 * g.total_area or c.area > 0.1 * g.total_area:
                        continue  # walls and floors of the body are not ribs
                    t = abs(occ.dot(occ.sub(B.centroid, A.centroid), A.info.normal))
                    if t < 1e-6 or t > max_t:
                        continue
                    # both side walls must rise from a floor (concave edge) -> rules out outer walls
                    def rises(F):
                        return any(g.edges[e].convexity == CONCAVE for m, eids in F.neighbors.items()
                                   for e in eids if m != c.id)
                    if not (rises(A) and rises(B)):
                        continue
                    cand.append((c.id, A.id, B.id, t))
        groups = []
        for cid, a, b, t in cand:
            fs = {cid, a, b}
            # end faces: planar faces touching both side walls, narrow (~t wide)
            for m in set(g.nodes[a].neighbors) & set(g.nodes[b].neighbors):
                if m == cid or g.nodes[m].kind != "plane" or m in self.owner:
                    continue
                Lm = max((g.edges[e].length for eids in g.nodes[m].neighbors.values() for e in eids), default=1)
                if g.nodes[m].area / max(Lm, 1e-9) <= 1.6 * t:
                    fs.add(m)
            for gr in groups:
                if gr["faces"] & fs:
                    gr["faces"] |= fs
                    gr["t"] = min(gr["t"], t)
                    break
            else:
                groups.append({"faces": fs, "t": t})
        for gr in groups:
            faces = sorted(gr["faces"])
            # rib-root blends belong to the rib
            for f in list(faces):
                for m in g.nodes[f].neighbors:
                    if m not in self.owner and m not in faces and self.is_blend(m):
                        faces.append(m)
            bb = _faces_bbox(g, faces)
            dims = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
            self._add("rib", faces, params={"thickness": round(gr["t"], 3), "height": round(dims[1], 3),
                                            "length": round(dims[2], 3)},
                      size=gr["t"], note=f"rib / gusset, thickness {gr['t']:.2f}", confidence=0.8)

    # ------------------------------------------------------------ 3. blends
    def find_fillets(self):
        g = self.g
        blend = [n.id for n in g.nodes if n.id not in self.owner and self.is_blend(n.id)]
        seen = set()
        for b in blend:
            if b in seen:
                continue
            chain, q = {b}, deque([b])
            while q:
                f = q.popleft()
                for m, eids in g.nodes[f].neighbors.items():
                    if m in chain or m in self.owner or m not in blend:
                        continue
                    if any(g.edges[e].convexity == SMOOTH for e in eids):
                        chain.add(m)
                        q.append(m)
            seen |= chain
            r = min(self.blend_radius(f) for f in chain)
            if r > 0.08 * g.char_len:
                continue  # large sweeping rounds are body shape, not removable detail
            rmax = max(self.blend_radius(f) for f in chain)
            convex = sum(1 for f in chain if g.nodes[f].mean_curv > 0) >= len(chain) / 2
            self._add("fillet", list(chain), params={"radius": round(r, 3), "radius_max": round(rmax, 3),
                                                    "type": "round" if convex else "fillet"},
                      size=r, note=f"{'round' if convex else 'fillet'} R{r:.2f}" + (f"–{rmax:.2f}" if rmax - r > 1e-3 else ""))

    # ------------------------------------------------------------ 4. chamfers
    def find_chamfers(self):
        g = self.g
        lim = 0.05 * g.char_len
        for n in g.nodes:
            if n.id in self.owner or not self.is_chamfer_like(n.id):
                continue
            L = max(g.edges[e].length for eids in n.neighbors.values() for e in eids)
            width = n.area / max(L, 1e-9)
            self._add("chamfer", [n.id], params={"width": round(width, 3), "conical": n.kind == "cone"}, size=width,
                      note=f"{'conical ' if n.kind == 'cone' else ''}chamfer, width {width:.2f}")

    # ------------------------------------------------------------ 5. open cuts (steps, through-slots)
    def find_open_cuts(self, max_region=0.25, max_face=0.12):
        """Depressions that run off the part (no inner loop): faces joined by concave edges.

        A step is a floor + one wall; a through-slot/channel has two walls facing each other.
        Large faces are never pulled in, so the main body of a molded part is left alone.
        """
        g = self.g
        big = max_face * g.total_area
        cand = {n.id for n in g.nodes if n.id not in self.owner and n.kind == "plane" and n.area <= big
                and any(g.edges[e].convexity == CONCAVE for es in n.neighbors.values() for e in es)}
        seen = set()
        for s in sorted(cand):
            if s in seen:
                continue
            comp, q = {s}, deque([s])
            while q:
                f = q.popleft()
                for m, eids in g.nodes[f].neighbors.items():
                    if m in cand and m not in comp and any(g.edges[e].convexity == CONCAVE for e in eids):
                        comp.add(m)
                        q.append(m)
            seen |= comp
            if len(comp) < 2 or sum(g.nodes[f].area for f in comp) > max_region * g.total_area:
                continue
            faces = sorted(comp)
            # a region that spans the part in two directions is the body (e.g. a tray floor), not a feature
            bb = _faces_bbox(g, faces)
            pb = occ.bbox(g.shape)
            spans = sum(1 for k in range(3) if (bb[k + 3] - bb[k]) >= 0.6 * (pb[k + 3] - pb[k]) and (pb[k + 3] - pb[k]) > 1e-9)
            if spans >= 2:
                continue
            planes = [g.nodes[f] for f in faces]
            facing = []
            for i in range(len(planes)):
                for j in range(i + 1, len(planes)):
                    a, b = planes[i], planes[j]
                    if occ.dot(a.info.normal, b.info.normal) < -0.999:
                        d = occ.dot(occ.sub(b.centroid, a.centroid), a.info.normal)
                        # walls of a channel face each other across the same floor
                        floor = (set(a.neighbors) & set(b.neighbors) & set(faces)) - {a.id, b.id}
                        if d > 0 and floor:
                            facing.append(d)
            bb = _faces_bbox(g, faces)
            dims = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
            if facing:
                w = min(facing)
                self._add("slot", faces, params={"width": round(w, 3), "open": True}, size=w,
                          note=f"open slot / channel, width {w:.2f}", confidence=0.75)
            else:
                self._add("step", faces, params={"depth": round(dims[0], 3)}, size=dims[1],
                          note=f"step / shoulder, {dims[1]:.2f} × {dims[0]:.2f}", confidence=0.7)

    def run(self):
        self.find_regions()
        self.find_ribs()
        self.find_chamfers()
        self.find_fillets()
        self.find_open_cuts()
        return self.features


def _machined_with_net(g: FaceGraph, feats, P, labels, t):
    """Steps and cut-outs: the network reads these far better than the rules. Drop a rule step it
    rejects, rename a rule cut-out it calls a step, and add confident groups of faces the rules left
    as body. A group holding a large share of the part is a body, whatever the network thinks."""
    si, pi = labels.index("step"), labels.index("pocket")
    keep = []
    for f in feats:
        p = sum(P[x][si] for x in f.faces) / len(f.faces)
        if f.kind == "step" and p < 1 - t:
            continue
        if f.kind in ("slot", "pocket") and p > t:
            f.kind = "step"
        keep.append(f)
    owned = {x for f in keep for x in f.faces}
    for kind, idx in (("step", si), ("pocket", pi)):
        cand = {n.id for n in g.nodes if n.id not in owned and P[n.id][idx] > t and P[n.id].argmax() == idx}
        while cand:
            s = cand.pop()
            comp, q = {s}, deque([s])
            while q:
                for m in g.nodes[q.popleft()].neighbors:
                    if m in cand:
                        cand.discard(m)
                        comp.add(m)
                        q.append(m)
            faces = sorted(comp)
            area = sum(g.nodes[x].area for x in faces)
            if area > MAX_MACHINED_AREA * g.total_area:
                continue
            L = max(g.edges[e].length for x in faces for es in g.nodes[x].neighbors.values() for e in es)
            w = area / max(L, 1e-9)
            keep.append(Feature(0, kind, faces, params={"width": round(w, 3)}, size=w, source="net",
                                confidence=round(float(sum(P[x][idx] for x in faces) / len(faces)), 3),
                                note=f"{kind}, width {w:.2f} (found by the network)"))
            owned |= comp
    return keep


MAX_MACHINED_AREA = 0.25


def _cuts_from_history(g: FaceGraph, feats, P, P8, labels, cut, step_veto):
    """The eight-label head has seen how real parts were modelled, so it tells design cuts from extrudes
    better than any other cue. Drop a rule step it doesn't see as a cut, and add confident groups of cut
    faces the rules left as body. A group holding a large share of the part is a body, not a feature."""
    pcut = P8[:, 2] + P8[:, 3]
    si, pi, hi = labels.index("step"), labels.index("pocket"), labels.index("hole")
    # the eight-label head has only seen design histories, the other head has seen machined steps: both must reject one
    keep = [f for f in feats if not (f.kind == "step" and sum(max(pcut[x], P[x][si]) for x in f.faces) / len(f.faces) < step_veto)]
    owned = {x for f in keep for x in f.faces}
    cand = {n.id for n in g.nodes if n.id not in owned and pcut[n.id] > cut}
    while cand:
        s = cand.pop()
        comp, q = {s}, deque([s])
        while q:
            for m in g.nodes[q.popleft()].neighbors:
                if m in cand:
                    cand.discard(m)
                    comp.add(m)
                    q.append(m)
        faces = sorted(comp)
        area = sum(g.nodes[x].area for x in faces)
        if area > MAX_MACHINED_AREA * g.total_area:
            continue
        L = max(g.edges[e].length for x in faces for es in g.nodes[x].neighbors.values() for e in es)
        w = area / max(L, 1e-9)
        votes = {k: sum(P[x][i] for x in faces) for k, i in (("step", si), ("pocket", pi), ("hole", hi))}
        kind = max(votes, key=votes.get)
        keep.append(Feature(0, kind, faces, params={"width": round(w, 3)}, size=w, source="net",
                            confidence=round(float(sum(pcut[x] for x in faces) / len(faces)), 3),
                            note=f"{kind}, width {w:.2f} (found by the network)"))
    return keep


def refine_with_net(g: FaceGraph, feats, P, labels, veto=0.4, add=0.5, override=1.01, machined=1.01, P8=None, cut=1.01, step_veto=0.0):
    """Rules + network (thresholds tuned on held-out validation parts, see bench/tune_net.py): drop any
    face of a rule fillet/chamfer the network finds unlikely, add blends the network is confident about, let a very
    confident blend call (> override) take a face back from a non-blend rule feature, and above
    `machined` let it settle steps and cut-outs."""
    fi, ci = labels.index("fillet"), labels.index("chamfer")
    keep = []
    for f in feats:
        if f.kind in ("fillet", "chamfer"):
            # judged face by face: one wrong face shouldn't ride along with good neighbours in a chain
            idx = fi if f.kind == "fillet" else ci
            mine = [x for x in f.faces if P[x][idx] >= veto]
            if not mine:
                continue
            f.faces = mine
            f.confidence = round(float(sum(P[x][idx] for x in mine) / len(mine)), 3)
        elif override <= 1:
            mine = [x for x in f.faces if max(P[x][fi], P[x][ci]) <= override]
            if not mine:
                continue
            f.faces = mine
        keep.append(f)
    owned = {x for f in keep for x in f.faces}
    R = Recognizer(g)
    R.owner = {x: -1 for x in owned}
    for cls, idx in (("fillet", fi), ("chamfer", ci)):
        cand = {n.id for n in g.nodes if n.id not in R.owner and P[n.id][idx] > add and P[n.id].argmax() == idx}
        while cand:
            s = cand.pop()
            comp, q = {s}, deque([s])
            while q:
                x = q.popleft()
                for m in g.nodes[x].neighbors:
                    if m in cand:
                        cand.discard(m)
                        comp.add(m)
                        q.append(m)
            faces = sorted(comp)
            if cls == "fillet":
                # no size limit here: whether a large fillet goes is the analysis's decision, not recognition's
                r = min(R.blend_radius(x) for x in faces)
                ft = Feature(0, "fillet", faces, params={"radius": round(r, 3)}, size=r, source="net",
                             confidence=round(float(sum(P[x][idx] for x in faces) / len(faces)), 3),
                             note=f"fillet R{r:.2f} (found by the network)")
            else:
                L = max(g.edges[e].length for x in faces for es in g.nodes[x].neighbors.values() for e in es)
                w = sum(g.nodes[x].area for x in faces) / max(L, 1e-9)
                ft = Feature(0, "chamfer", faces, params={"width": round(w, 3)}, size=w, source="net",
                             confidence=round(float(sum(P[x][idx] for x in faces) / len(faces)), 3),
                             note=f"chamfer, width {w:.2f} (found by the network)")
            keep.append(ft)
            for x in faces:
                R.owner[x] = -1
    if machined <= 1:
        keep = _machined_with_net(g, keep, P, labels, machined)
    if P8 is not None and (cut <= 1 or step_veto > 0):
        keep = _cuts_from_history(g, keep, P, P8, labels, cut, step_veto)
    for i, f in enumerate(keep):
        f.id = i
    return keep


# rules + network thresholds, tuned by bench/tune_net.py on 1,500 Fusion parts the network never trained on,
# and `machined` also on 500 unused MFCAD train-split parts
FUSION = {"veto": 0.5, "add": 0.8, "override": 0.5, "machined": 0.99, "cut": 0.5, "step_veto": 0.7}


def recognize(shape, g: FaceGraph | None = None, use_net=True):
    g = g or build_graph(shape)
    feats = Recognizer(g).run()
    if use_net:
        try:
            from .datagen import LABELS
            from .net import predict_part
            pp = predict_part(g)
            if pp is not None:
                feats = refine_with_net(g, feats, pp[0], LABELS, P8=pp[1], **FUSION)
        except Exception:
            pass
    return g, feats
