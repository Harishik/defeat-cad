"""Assistant benchmarks.

1. Tool calling: plain-English instructions with the expected tool calls: 24 used while building the
   parser, 12 held out, and 60 more held out and written before the model or parser ever saw them.
   Scores the local LLM and the offline keyword parser on exact match, call-level precision/recall,
   and latency.
2. Text-to-CAD: 50 part descriptions with known overall dimensions. Scores valid-solid rate and
   whether the bounding box matches the description (each sorted dimension within 10%).

Rates carry 95% Wilson intervals. Nothing is tuned on the held-out sets.

    python -m bench.assistant
"""
from __future__ import annotations

import os
import statistics as st
import tempfile
import time

from bench.common import ROOT, save, wilson

# (instruction, expected calls as {(name, frozenset of key=value)})
R = "set_class_rule"
COMMANDS = [
    ("Remove the logo", {(R, "emboss", "remove")}),
    ("Get rid of all the fillets", {(R, "fillet", "remove")}),
    ("Keep every hole", {(R, "hole", "keep")}),
    ("Remove the chamfers but keep the ribs", {(R, "chamfer", "remove"), (R, "rib", "keep")}),
    ("Delete the text and the small rounds", {(R, "emboss", "remove"), (R, "fillet", "remove")}),
    ("Keep the screw bosses", {(R, "boss", "keep")}),
    ("Remove all ribs", {(R, "rib", "remove")}),
    ("Drop the vents and pockets", {(R, "pocket", "remove")}),
    ("Remove the slots", {(R, "slot", "remove")}),
    ("Keep the bosses and the ribs, remove the logo", {(R, "boss", "keep"), (R, "rib", "keep"), (R, "emboss", "remove")}),
    ("Run it", {("run_defeaturing",)}),
    ("Simplify the part now", {("run_defeaturing",)}),
    ("Prepare this for a modal analysis", {("set_analysis", "modal")}),
    ("Set it up for a thermal simulation", {("set_analysis", "thermal")}),
    ("I want to do CFD on this", {("set_analysis", "cfd")}),
    ("Use a mesh size of 2 mm", {("set_analysis", "h=2")}),
    ("Static stress analysis with 3 mm elements", {("set_analysis", "static"), ("set_analysis", "h=3")}),
    ("Give me the Python code", {("export", "python")}),
    ("Export the C++ program", {("export", "cpp")}),
    ("Download the simplified STEP file", {("export", "step")}),
    ("Build a flow domain around the part", {("make_fluid_domain",)}),
    ("Keep feature 6", {("set_features", "6", "keep")}),
    ("Remove features 3 and 4", {("set_features", "3", "remove"), ("set_features", "4", "remove")}),
    ("Remove the logo and fillets, keep the holes, then run it",
     {(R, "emboss", "remove"), (R, "fillet", "remove"), (R, "hole", "keep"), ("run_defeaturing",)}),
]

# written after the parser fixes and never used to tune anything: the honest score
HELDOUT = [
    ("Strip off the lettering", {(R, "emboss", "remove")}),
    ("Please leave all the gussets alone", {(R, "rib", "keep")}),
    ("Lose the bevels", {(R, "chamfer", "remove")}),
    ("No fillets please", {(R, "fillet", "remove")}),
    ("Hang on to the bores", {(R, "hole", "keep")}),
    ("Go ahead and process it", {("run_defeaturing",)}),
    ("I'm doing a vibration study", {("set_analysis", "modal")}),
    ("Make the elements 1.5 mm", {("set_analysis", "h=1.5")}),
    ("Can I get the script in Python?", {("export", "python")}),
    ("Remove the standoffs", {(R, "boss", "remove")}),
    ("Get rid of the cutouts and run", {(R, "pocket", "remove"), ("run_defeaturing",)}),
    ("Keep feature #12 no matter what", {("set_features", "12", "keep")}),
]


# a second held-out set, larger, so the score has a usable confidence range. Never used to tune anything.
HELDOUT_2 = [
    ("Take off the engraved text", {(R, "emboss", "remove")}),
    ("Erase the logo from the part", {(R, "emboss", "remove")}),
    ("Remove every chamfer", {(R, "chamfer", "remove")}),
    ("Leave the chamfers as they are", {(R, "chamfer", "keep")}),
    ("Strip out all the rounds", {(R, "fillet", "remove")}),
    ("Keep the fillets", {(R, "fillet", "keep")}),
    ("Get rid of the holes", {(R, "hole", "remove")}),
    ("Don't touch the holes", {(R, "hole", "keep")}),
    ("Remove the pockets", {(R, "pocket", "remove")}),
    ("Keep all pockets", {(R, "pocket", "keep")}),
    ("Fill in the slots", {(R, "slot", "remove")}),
    ("The slots must stay", {(R, "slot", "keep")}),
    ("Delete the ribs", {(R, "rib", "remove")}),
    ("Ribs carry load, keep them", {(R, "rib", "keep")}),
    ("Remove the bosses", {(R, "boss", "remove")}),
    ("Protect the screw posts", {(R, "boss", "keep")}),
    ("Remove the steps", {(R, "step", "remove")}),
    ("Keep the steps", {(R, "step", "keep")}),
    ("Remove the fillets and chamfers", {(R, "fillet", "remove"), (R, "chamfer", "remove")}),
    ("Keep holes and slots", {(R, "hole", "keep"), (R, "slot", "keep")}),
    ("Remove the logo and the chamfers", {(R, "emboss", "remove"), (R, "chamfer", "remove")}),
    ("Keep the ribs and remove the fillets", {(R, "rib", "keep"), (R, "fillet", "remove")}),
    ("Remove pockets, keep bosses", {(R, "pocket", "remove"), (R, "boss", "keep")}),
    ("Delete the fillets, the chamfers and the logo", {(R, "fillet", "remove"), (R, "chamfer", "remove"), (R, "emboss", "remove")}),
    ("Start the simplification", {("run_defeaturing",)}),
    ("Do it", {("run_defeaturing",)}),
    ("Defeature this part", {("run_defeaturing",)}),
    ("Execute the removal", {("run_defeaturing",)}),
    ("Remove the chamfers and then run", {(R, "chamfer", "remove"), ("run_defeaturing",)}),
    ("Keep the holes and go", {(R, "hole", "keep"), ("run_defeaturing",)}),
    ("This is for a modal study", {("set_analysis", "modal")}),
    ("Switch to thermal", {("set_analysis", "thermal")}),
    ("We're running external flow on this", {("set_analysis", "cfd")}),
    ("Set up for structural stress", {("set_analysis", "static")}),
    ("Set the analysis to static", {("set_analysis", "static")}),
    ("Mesh size 5 mm", {("set_analysis", "h=5")}),
    ("Target element size is 0.5 mm", {("set_analysis", "h=0.5")}),
    ("Use 4 mm elements", {("set_analysis", "h=4")}),
    ("Thermal analysis with a 2.5 mm mesh", {("set_analysis", "thermal"), ("set_analysis", "h=2.5")}),
    ("Modal at 3 mm", {("set_analysis", "modal"), ("set_analysis", "h=3")}),
    ("Export as Python", {("export", "python")}),
    ("Write the Python script", {("export", "python")}),
    ("Give me C++", {("export", "cpp")}),
    ("I need the C++ source", {("export", "cpp")}),
    ("Save it as STEP", {("export", "step")}),
    ("Export the result to a STEP file", {("export", "step")}),
    ("Create a fluid domain", {("make_fluid_domain",)}),
    ("Make the enclosure for CFD meshing", {("make_fluid_domain",)}),
    ("Extract the flow volume", {("make_fluid_domain",)}),
    ("Remove feature 7", {("set_features", "7", "remove")}),
    ("Keep features 1 and 2", {("set_features", "1", "keep"), ("set_features", "2", "keep")}),
    ("Feature 9 has to stay", {("set_features", "9", "keep")}),
    ("Delete feature number 15", {("set_features", "15", "remove")}),
    ("Remove #2, #5 and #8", {("set_features", "2", "remove"), ("set_features", "5", "remove"), ("set_features", "8", "remove")}),
    ("Keep feature 3 and remove feature 4", {("set_features", "3", "keep"), ("set_features", "4", "remove")}),
    ("Take away the small rounds and the bevels, then process it", {(R, "fillet", "remove"), (R, "chamfer", "remove"), ("run_defeaturing",)}),
    ("Set it up for vibration with a 1 mm mesh", {("set_analysis", "modal"), ("set_analysis", "h=1")}),
    ("Remove the text, keep the gussets, and export Python", {(R, "emboss", "remove"), (R, "rib", "keep"), ("export", "python")}),
    ("Keep everything with a hole in it", {(R, "hole", "keep")}),
    ("Clean up the bevelled edges", {(R, "chamfer", "remove")}),
]


def normalize(calls):
    out = set()
    for c in calls:
        n, a = c["name"], c.get("arguments") or {}
        if isinstance(a, str):
            import json
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        if n == R:
            out.add((n, a.get("feature_class"), a.get("action")))
        elif n == "set_analysis":
            if a.get("preset"):
                out.add((n, a["preset"]))
            if a.get("mesh_size_mm"):
                out.add((n, f"h={float(a['mesh_size_mm']):g}"))
        elif n == "export":
            out.add((n, a.get("format")))
        elif n == "set_features":
            for i in a.get("feature_ids", []):
                out.add((n, str(i), a.get("action")))
        else:
            out.add((n,))
    return out


def _ci(k, n):
    return [round(v, 4) for v in wilson(k, n)]


def score(rows):
    exact = sum(1 for r in rows if r["pred"] == r["gold"])
    tp = sum(len(r["pred"] & r["gold"]) for r in rows)
    fp = sum(len(r["pred"] - r["gold"]) for r in rows)
    fn = sum(len(r["gold"] - r["pred"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return {"commands": len(rows), "exact_match": round(exact / len(rows), 4),
            "exact_match_ci95": _ci(exact, len(rows)), "call_precision": round(p, 4),
            "call_recall": round(rc, 4), "call_f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0,
            "median_latency_s": round(st.median(r["t"] for r in rows), 2),
            "misses": [{"cmd": r["cmd"], "pred": sorted(map(str, r["pred"])), "gold": sorted(map(str, r["gold"]))}
                       for r in rows if r["pred"] != r["gold"]]}


def tool_calling(cmds, tag):
    from defeat import agent
    from defeat.session import new_session
    s = new_session(os.path.join(ROOT, "samples", "l_bracket.step"))
    sysmsg = agent.SYSTEM.format(briefing=s.briefing())
    res = {}
    rows = []
    for cmd, gold in cmds:
        t = time.time()
        rows.append({"cmd": cmd, "gold": gold, "pred": normalize(agent.parse_offline(cmd)), "t": time.time() - t})
    res["offline_parser"] = score(rows)
    if agent.ollama_available():
        agent.warmup()
        rows = []
        for cmd, gold in cmds:
            t = time.time()
            try:
                m = agent.llm_chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": cmd}],
                                   tools=agent.TOOLS, timeout=180)
                calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
                         for c in (m.get("tool_calls") or [])]
            except Exception:
                calls = []
            rows.append({"cmd": cmd, "gold": gold, "pred": normalize(calls), "t": time.time() - t})
            print(f"  llm {tag} {len(rows)}/{len(cmds)}", flush=True)
        res[agent.CHAT_MODEL] = score(rows)
    return res


PARTS = [
    ("A rectangular plate 100 x 60 x 5 mm with four 6 mm holes 8 mm from each corner", (5, 60, 100)),
    ("A 60 mm diameter flange, 10 mm thick, with a 20 mm bore and six 6 mm bolt holes on a 44 mm circle", (10, 60, 60)),
    ("An L-bracket: 80 x 40 x 5 mm base with a 50 mm tall, 5 mm thick wall on one long edge", (40, 55, 80)),
    ("A cylinder 30 mm in diameter and 50 mm long with a 10 mm hole through its axis", (30, 30, 50)),
    ("A 50 mm cube with 3 mm fillets on all edges", (50, 50, 50)),
    ("A spacer ring: outer diameter 40 mm, inner diameter 25 mm, 8 mm thick", (8, 40, 40)),
    ("A rectangular box 120 x 80 x 40 mm, hollowed out with 3 mm walls and an open top", (40, 80, 120)),
    ("A 70 x 70 x 6 mm plate with a 30 mm center hole and 2 mm chamfers on the outer edges", (6, 70, 70)),
    ("A stepped shaft: 20 mm diameter for 40 mm, then 12 mm diameter for 30 mm", (20, 20, 70)),
    ("A 90 x 30 x 8 mm bar with a 60 x 10 mm slot through the middle", (8, 30, 90)),
    ("A hex prism 24 mm across flats and 15 mm tall with an 8 mm hole", (15, 24, 27.7)),
    ("A 100 x 50 x 4 mm cover with two 30 x 8 mm rectangular vents and 2 mm rounded corners", (4, 50, 100)),
    # below: held out, never used to tune the prompt or choose the model
    ("A solid cylinder 25 mm in diameter and 100 mm long", (25, 25, 100)),
    ("A 40 x 40 x 40 mm cube", (40, 40, 40)),
    ("A flat bar 200 x 20 x 3 mm", (3, 20, 200)),
    ("A disc 80 mm in diameter and 5 mm thick", (5, 80, 80)),
    ("A sphere 50 mm in diameter", (50, 50, 50)),
    ("A tube with 30 mm outer diameter, 24 mm inner diameter, 120 mm long", (30, 30, 120)),
    ("A cone 40 mm in base diameter tapering to 20 mm at the top, 30 mm tall", (30, 40, 40)),
    ("A square plate 50 x 50 x 10 mm with a 20 mm hole in the center", (10, 50, 50)),
    ("A 150 x 100 x 10 mm plate with a 40 mm center hole", (10, 100, 150)),
    ("A 60 x 40 x 20 mm block with a 10 mm through hole along the 20 mm direction", (20, 40, 60)),
    ("A 100 x 100 x 8 mm plate with four 10 mm holes on a 70 x 70 mm pattern", (8, 100, 100)),
    ("A 120 x 40 x 6 mm strip with five 5 mm holes equally spaced along its centerline", (6, 40, 120)),
    ("A washer: 20 mm outer diameter, 10 mm hole, 2 mm thick", (2, 20, 20)),
    ("A flange 100 mm in diameter and 12 mm thick with a 40 mm bore and eight 8 mm holes on a 75 mm bolt circle", (12, 100, 100)),
    ("A hub: 60 mm diameter disc 10 mm thick with a 30 mm diameter boss 20 mm tall on top", (30, 60, 60)),
    ("A shaft with a 30 mm diameter for 50 mm, then 20 mm diameter for 50 mm, then 10 mm diameter for 50 mm", (30, 30, 150)),
    ("A 50 x 50 x 50 mm cube with a 20 mm hole through the center", (50, 50, 50)),
    ("A 80 x 60 x 30 mm box with 5 mm fillets on the vertical edges", (30, 60, 80)),
    ("A 100 x 60 x 10 mm plate with 3 mm chamfers on all top edges", (10, 60, 100)),
    ("A 40 x 40 x 10 mm plate with 5 mm rounded corners", (10, 40, 40)),
    ("A T-section beam 100 mm long, flange 40 x 5 mm, web 30 x 5 mm", (35, 40, 100)),
    ("An L-angle 100 mm long with 30 mm legs, 3 mm thick", (30, 30, 100)),
    ("A U-channel 200 mm long, 50 mm wide, 25 mm tall, with 3 mm walls", (25, 50, 200)),
    ("A 100 x 80 x 30 mm open-top tray with 2 mm walls", (30, 80, 100)),
    ("A 60 x 60 x 40 mm box, hollow, 4 mm walls, open on top", (40, 60, 60)),
    ("A 100 x 50 x 20 mm block with a 60 x 30 x 10 mm pocket in the top face", (20, 50, 100)),
    ("A 80 x 80 x 15 mm plate with a 40 x 40 mm square through hole in the middle", (15, 80, 80)),
    ("A 120 x 30 x 10 mm bar with a 80 x 8 mm slot through it", (10, 30, 120)),
    ("A 100 x 100 x 5 mm plate with three parallel 60 x 5 mm slots", (5, 100, 100)),
    ("A 60 x 40 x 30 mm block with a 20 x 10 mm step cut along one long top edge", (30, 40, 60)),
    ("A hexagonal prism 30 mm across flats and 50 mm long", (30, 34.6, 50)),
    ("An octagonal plate 60 mm across flats, 8 mm thick", (8, 60, 60)),
    ("A triangular prism with a 40 mm equilateral base, 60 mm long", (34.6, 40, 60)),
    ("A 120 x 60 x 8 mm base plate with a 40 mm tall, 8 mm thick rib along its centerline", (48, 60, 120)),
    ("A bracket: 60 x 60 x 6 mm base and a 60 x 40 x 6 mm upright wall at one edge", (46, 60, 60)),
    ("A spool: 40 mm diameter end discs 5 mm thick, joined by a 20 mm diameter core 30 mm long", (40, 40, 40)),
    ("A cylinder 40 mm in diameter and 60 mm long with a 25 mm diameter blind hole 40 mm deep", (40, 40, 60)),
    ("A cylinder 50 mm in diameter and 20 mm long with 2 mm chamfers on both ends", (20, 50, 50)),
]


def text_to_cad():
    from defeat import agent, occ
    if not agent.ollama_available():
        return None
    rows = []
    for desc, dims in PARTS:
        out = os.path.join(tempfile.mkdtemp(), "p.step")
        t = time.time()
        try:
            r = agent.text_to_cad(desc, out)
        except Exception as e:  # noqa
            r = {"ok": False, "attempts": [{"error": str(e)}]}
        row = {"prompt": desc, "valid": r["ok"], "attempts": len(r.get("attempts", [])), "t": time.time() - t}
        if r["ok"]:
            x0, y0, z0, x1, y1, z1 = occ.bbox(occ.read_cad(out))
            got = sorted([x1 - x0, y1 - y0, z1 - z0])
            want = sorted(dims)
            row["bbox"] = [round(v, 2) for v in got]
            row["bbox_ok"] = all(abs(gv - wv) <= 0.1 * wv + 0.5 for gv, wv in zip(got, want))
        rows.append(row)
        print(f"  cad {len(rows)}/{len(PARTS)} valid={row['valid']} bbox_ok={row.get('bbox_ok')}", flush=True)
    n = len(rows)
    return {"model": agent.CHAT_MODEL, "prompts": n,
            "valid_solid_rate": round(sum(r["valid"] for r in rows) / n, 4),
            "valid_solid_ci95": _ci(sum(r["valid"] for r in rows), n),
            "dimension_match_ci95": _ci(sum(1 for r in rows if r.get("bbox_ok")), n),
            "first_try_rate": round(sum(1 for r in rows if r["valid"] and r["attempts"] == 1) / n, 4),
            "dimension_match_rate": round(sum(1 for r in rows if r.get("bbox_ok")) / n, 4),
            "median_seconds": round(st.median(r["t"] for r in rows), 1), "rows": rows}


def main():
    rep = {"tool_calling_dev": tool_calling(COMMANDS, "dev"), "tool_calling_heldout": tool_calling(HELDOUT + HELDOUT_2, "heldout")}
    rep["text_to_cad"] = text_to_cad()
    print(save("assistant", rep))
    for split in ("tool_calling_dev", "tool_calling_heldout"):
        for k, v in rep[split].items():
            print(split, k, {x: v[x] for x in ("exact_match", "call_precision", "call_recall", "call_f1", "median_latency_s")})
    if rep["text_to_cad"]:
        t = rep["text_to_cad"]
        print("text-to-cad", {x: t[x] for x in ("valid_solid_rate", "first_try_rate", "dimension_match_rate", "median_seconds")})


if __name__ == "__main__":
    main()
