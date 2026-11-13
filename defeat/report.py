"""Self-contained HTML validation report (print to PDF from any browser)."""
from __future__ import annotations

import base64
import datetime
import html
import os

from .decide import PRESETS
from .tessellate import render_png

CLASS_COLORS = {"hole": (0.20, 0.47, 0.96), "slot": (0.35, 0.34, 0.84), "pocket": (0.55, 0.35, 0.85), "step": (0.67, 0.56, 0.41),
                "boss": (0.98, 0.58, 0.10), "rib": (0.20, 0.70, 0.45), "emboss": (0.96, 0.26, 0.45),
                "fillet": (0.10, 0.70, 0.80), "chamfer": (0.95, 0.75, 0.10), "protrusion": (0.6, 0.6, 0.6)}


def _img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def build_report(s) -> str:
    stem = s.name.rsplit(".", 1)[0]
    a, b = os.path.join(s.dir, "r_before.png"), os.path.join(s.dir, "r_after.png")
    cols = {}
    for f in s.features:
        if s.decisions[f.id].action == "remove":
            for fid in f.faces:
                cols[fid] = CLASS_COLORS.get(f.kind, (0.7, 0.7, 0.7))
    render_png(s.shape, a, face_colors=cols, faces=s.graph.faces)
    if s.result:
        render_png(s.result.shape, b)
    m = s.metrics or {}
    geo, mesh = m.get("geometry", {}), m.get("mesh") or {}
    rows = []
    for f in s.features:
        d = s.decisions[f.id]
        st = d.action
        if s.result and f.id in s.result.failed:
            st = "kept (kernel)"
        rows.append(f"<tr><td>#{f.id}</td><td><span class='dot' style='background:rgb({','.join(str(int(255*c)) for c in CLASS_COLORS.get(f.kind,(0.6,0.6,0.6)))})'></span>{f.kind}</td>"
                    f"<td>{html.escape(f.note)}</td><td class='{ 'rm' if st=='remove' else 'kp'}'>{st}</td><td>{html.escape(d.reason)}</td></tr>")

    from .fea import guard_message, worst_case

    def stat(label, val, sub=""):
        return f"<div class='stat'><div class='v'>{val}</div><div class='l'>{label}</div><div class='s'>{sub}</div></div>"

    stats = ""
    if geo:
        stats += stat("Faces", f"{geo['faces'][0]} → {geo['faces'][1]}", f"−{100*(1-geo['faces'][1]/geo['faces'][0]):.0f}%")
        stats += stat("Volume change", f"{geo['volume_delta_pct']:+.2f}%", "")
        stats += stat("Max deviation", f"{geo['deviation']['max']:.2f} mm", f"mean {geo['deviation']['mean']:.2f} mm")
    if mesh.get("tet_reduction_pct") is not None:
        stats += stat("Tetrahedra", f"{mesh['original']['tets']:,} → {mesh['defeatured']['tets']:,}", f"−{mesh['tet_reduction_pct']:.0f}% at h={mesh['h']:g}")
        stats += stat("Mesh quality (min SICN)", f"{mesh['original']['q_min']:.2f} → {mesh['defeatured']['q_min']:.2f}", "higher is better")
    fea = m.get("fea")
    fea_html = ""
    if fea and fea.get("delta"):
        dlt, fo, fd = fea["delta"], fea["original"], fea["defeatured"]
        stats += stat("Tip deflection Δ", f"{dlt['tip_disp_delta_pct']:+.2f}%", f"{fo['tip_disp']:.4f} → {fd['tip_disp']:.4f} mm")
        worst = worst_case(dlt)
        if worst is not None:
            stats += stat("Worst of 4 load cases", f"{worst:.2f}%", "two bending directions, tension, torsion")
        if dlt.get("vm_p99_delta_pct") is not None:
            stats += stat("Peak stress Δ (p99)", f"{dlt['vm_p99_delta_pct']:+.2f}%", f"{fo['vm_p99']:.2f} → {fd['vm_p99']:.2f} MPa")
        stats += stat("Solve time", f"{fo['seconds']:.1f} → {fd['seconds']:.1f} s", f"{fo['dofs']:,} → {fd['dofs']:,} DOFs")
        rows_f = "".join(f"<tr><td>{lbl}</td><td>{fo[k]:.4g}</td><td>{fd[k]:.4g}</td><td>{100*(fd[k]-fo[k])/fo[k]:+.2f}%</td></tr>"
                         for lbl, k in [("Tip deflection (mm)", "tip_disp"), ("Max displacement (mm)", "max_disp"),
                                        ("Stiffness (N/mm)", "stiffness"), ("von Mises p99 (MPa)", "vm_p99"),
                                        ("von Mises p95 (MPa)", "vm_p95"), ("von Mises vol-mean (MPa)", "vm_mean"),
                                        ("Degrees of freedom", "dofs")] if fo.get(k) and fd.get(k))
        guard_note = m.get("guard_message") if fea.get("method") == "grid" else guard_message(m.get("guard"))
        how = ("No volume mesh could be built, so both models were solved on the same fixed grid of quadratic hexahedra, "
               "integrated over the material only (finite cell method with moment-fitted quadrature, inside test by "
               "generalized winding number). The grid gives stiffness changes, not stress peaks."
               if fea.get("method") == "grid" else
               "Both models meshed at the same h and solved with quadratic tetrahedra (scikit-fem). Peak values use the "
               "99th percentile to ignore the singular corners at the clamp.")
        fea_html = (f"<h2>Static stress check</h2><p class='sub'>{html.escape(fea['load_case'])}. {how} "
                    f"This is a comparison between the two models, not a replacement for your full analysis.</p>"
                    + (f"<p><b>Stress guard:</b> {html.escape(guard_note)}</p>" if guard_note else "")
                    + f"<table><tr><th>Quantity</th><th>Original</th><th>Simplified</th><th>Change</th></tr>{rows_f}</table>")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>DeFeat report — {html.escape(s.name)}</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'SF Pro Text','Inter',sans-serif;color:#1d1d1f;max-width:980px;margin:40px auto;padding:0 24px}}
h1{{font-size:34px;letter-spacing:-.02em;margin:0}} h2{{font-size:20px;margin:36px 0 12px}}
.sub{{color:#6e6e73}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.stat{{background:#f5f5f7;border-radius:14px;padding:16px}} .v{{font-size:22px;font-weight:600}} .l{{color:#6e6e73;font-size:12px}} .s{{color:#0071e3;font-size:12px}}
.imgs{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .imgs img{{width:100%;border-radius:14px;background:#fbfbfd}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{padding:7px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}} .rm{{color:#d70015;font-weight:600}} .kp{{color:#248a3d;font-weight:600}}
</style></head><body>
<h1>Defeaturing report</h1><div class='sub'>{html.escape(s.name)} · {PRESETS[s.policy.preset]['label']} · target mesh size h = {s.policy.h:g} mm · {datetime.datetime.now():%Y-%m-%d %H:%M} · DeFeat</div>
<h2>Summary</h2><div class='grid'>{stats}</div>
<h2>Before / after</h2><div class='imgs'><div><img src='{_img(a)}'><div class='sub'>Original — removed features highlighted</div></div>
{"<div><img src='"+_img(b)+"'><div class='sub'>Simplified, simulation-ready</div></div>" if s.result else ""}</div>
{fea_html}
<h2>Feature decisions</h2><table><tr><th>ID</th><th>Class</th><th>Feature</th><th>Action</th><th>Why</th></tr>{''.join(rows)}</table>
<h2>Method</h2><p class='sub'>Features were recognized on the exact B-Rep (face-adjacency graph with convexity), decided against size limits expressed in multiples of the target mesh size, and removed with OpenCASCADE BRepAlgoAPI_Defeaturing in a guarded worker (strategy: {html.escape(m.get('strategy','—'))}). Validity was checked with BRepCheck; both models were meshed with gmsh at the same h.</p>
</body></html>"""
    path = os.path.join(s.dir, f"{stem}_report.html")
    open(path, "w", encoding="utf-8").write(page)
    return path
