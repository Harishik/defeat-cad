"""Assistant benchmarks.

1. Tool calling: 24 plain-English instructions with the expected tool calls. Scores the local LLM and
   the offline keyword parser on exact match, call-level precision/recall, and latency.
2. Text-to-CAD: 12 part descriptions with known overall dimensions. Scores valid-solid rate and
   whether the bounding box matches the description (each sorted dimension within 10%).

    python -m bench.assistant
"""
from __future__ import annotations

import os
import statistics as st
import tempfile
import time

from bench.common import ROOT, save

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


def score(rows):
    exact = sum(1 for r in rows if r["pred"] == r["gold"])
    tp = sum(len(r["pred"] & r["gold"]) for r in rows)
    fp = sum(len(r["pred"] - r["gold"]) for r in rows)
    fn = sum(len(r["gold"] - r["pred"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return {"commands": len(rows), "exact_match": round(exact / len(rows), 4), "call_precision": round(p, 4),
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
            "first_try_rate": round(sum(1 for r in rows if r["valid"] and r["attempts"] == 1) / n, 4),
            "dimension_match_rate": round(sum(1 for r in rows if r.get("bbox_ok")) / n, 4),
            "median_seconds": round(st.median(r["t"] for r in rows), 1), "rows": rows}


def main():
    rep = {"tool_calling_dev": tool_calling(COMMANDS, "dev"), "tool_calling_heldout": tool_calling(HELDOUT, "heldout")}
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
