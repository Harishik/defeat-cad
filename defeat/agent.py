"""Chat operator: a local open-source LLM (Ollama) turns plain English into tool calls on the
exact engine. The LLM never touches geometry directly; every change goes through the same
functions the UI buttons use, and a keyword parser keeps things working offline."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time

import httpx

from .decide import PRESETS
from .recognize import FEATURE_CLASSES

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.environ.get("DEFEAT_CHAT_MODEL", "qwen3.5:4b")
CAD_MODEL = os.environ.get("DEFEAT_CAD_MODEL", "hf.co/yuvit-batra/qwen2.5-coder-7b-cadquery-gguf:Q4_K_M")

CLASS_ENUM = FEATURE_CLASSES

TOOLS = [
    {"type": "function", "function": {
        "name": "set_class_rule",
        "description": "Keep or remove a whole class of recognized features, optionally only below a size limit in mm "
                       "(hole/boss: diameter, fillet: radius, chamfer/slot/pocket: width, rib: thickness, emboss: height).",
        "parameters": {"type": "object", "properties": {
            "feature_class": {"type": "string", "enum": CLASS_ENUM},
            "action": {"type": "string", "enum": ["keep", "remove", "auto"],
                       "description": "auto = go back to the preset rule"},
            "max_size_mm": {"type": "number", "description": "only remove features smaller than this"}},
            "required": ["feature_class", "action"]}}},
    {"type": "function", "function": {
        "name": "set_features",
        "description": "Keep or remove specific features by their id number (#id in the part briefing).",
        "parameters": {"type": "object", "properties": {
            "feature_ids": {"type": "array", "items": {"type": "integer"}},
            "action": {"type": "string", "enum": ["keep", "remove", "auto"]}},
            "required": ["feature_ids", "action"]}}},
    {"type": "function", "function": {
        "name": "set_analysis",
        "description": "Choose the analysis type preset and/or the target mesh element size h in mm.",
        "parameters": {"type": "object", "properties": {
            "preset": {"type": "string", "enum": list(PRESETS)},
            "mesh_size_mm": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "run_defeaturing",
        "description": "Execute the defeaturing with the current decisions, mesh both models and compare.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "export",
        "description": "Prepare downloads: the simplified STEP file or the code that reproduces it.",
        "parameters": {"type": "object", "properties": {
            "format": {"type": "string", "enum": ["step", "python", "cpp", "report"]}},
            "required": ["format"]}}},
    {"type": "function", "function": {
        "name": "create_part",
        "description": "Create a NEW part from a text description (text-to-CAD). Use only when the user asks to make or design a new part.",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string"}}, "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "make_fluid_domain",
        "description": "Build a CFD flow domain: a box around the part with the part subtracted.",
        "parameters": {"type": "object", "properties": {
            "padding_pct": {"type": "number", "description": "padding around the part, % of its size"}}}}},
]

SYSTEM = """You operate DeFeat, a CAD defeaturing tool for CAE simulation prep. The user may not know CAD.
You can only change the model by calling tools. Prefer tool calls over talk. Use several tool calls in one reply when the
request has several parts. Sizes are in mm. 'Logo', 'text', 'lettering', 'engraving' mean feature_class emboss.
'Rounds' and 'blends' mean fillet. 'Screw bosses' mean boss. 'Mounting holes' are usually the larger holes: keep them
with set_class_rule(hole, keep) unless sizes are given. When the user asks to run, prepare, simplify or 'do it', call
run_defeaturing after setting rules. If the user only asks a question, answer briefly in plain English (2-3 sentences)
using the part briefing. Never invent features that are not in the briefing.

Words to classes: logo, text, lettering, engraving, label, branding -> emboss. rounds, blends, edge rounds -> fillet.
bevels, bevelled edges -> chamfer. bores, drilled holes, bolt holes, mounting holes -> hole. gussets, webs, stiffeners -> rib.
standoffs, screw posts, bosses -> boss. vents, cut-outs, windows, pockets -> pocket. slots -> slot. steps, ledges -> step.
Action words: lose, drop, strip, wipe, erase, fill, plug, close up, flatten, take out, "can go", "out" -> remove.
keep, leave, stay, protect, "hang on to", "don't touch", "must stay", "in" -> keep.
Analysis words: stress, structural, static -> static. vibration, natural frequencies, modal -> modal. heat, thermal -> thermal.
airflow, external flow, CFD -> cfd. A bare "at 3 mm", "h = 3", "elements 3 mm" is set_analysis(mesh_size_mm=3).
Call run_defeaturing ONLY when asked to run, go, start, process, execute, do it or clean it up now: setting a rule alone does not
run anything. Call export only when asked for a file or code. Numbered features use set_features, not set_class_rule; list
every id named. Make one call for each thing asked, and nothing that wasn't asked.
Examples:
"Remove the chamfers but keep the ribs" -> set_class_rule(chamfer, remove); set_class_rule(rib, keep)
"Modal at 3 mm" -> set_analysis(preset=modal, mesh_size_mm=3)
"Remove #2 and keep #5" -> set_features([2], remove); set_features([5], keep)
"Keep the holes and go" -> set_class_rule(hole, keep); run_defeaturing
"Save it as STEP" -> export(step)
"Make a 40 mm cube" -> create_part("a 40 mm cube")

PART BRIEFING
{briefing}
"""


# ---------------------------------------------------------------- LLM
def ollama_available(timeout=1.5) -> bool:
    try:
        return httpx.get(f"{OLLAMA}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def ollama_models():
    try:
        return [m["name"] for m in httpx.get(f"{OLLAMA}/api/tags", timeout=2).json().get("models", [])]
    except Exception:
        return []


def llm_chat(messages, tools=None, model=None, timeout=180, options=None):
    body = {"model": model or CHAT_MODEL, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0.1, "num_ctx": 8192, **(options or {})}, "keep_alive": "30m"}
    if tools:
        body["tools"] = tools
    r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]


def warmup():
    try:
        llm_chat([{"role": "user", "content": "hi"}], timeout=120, options={"num_predict": 1})
    except Exception:
        pass


# ---------------------------------------------------------------- offline parser
_CLASS_WORDS = {
    "emboss": ["logo", "text", "letter", "emboss", "engrav", "label", "lettering", "branding"],
    "fillet": ["fillet", "round", "blend", "radius", "radii"],
    "chamfer": ["chamfer", "bevel"],
    "hole": ["hole", "bore", "drill"],
    "boss": ["boss", "standoff", "screw post", "post"],
    "rib": ["rib", "gusset", "web", "stiffener"],
    "slot": ["slot"],
    "pocket": ["pocket", "cut-out", "cutout", "vent", "window"],
    "step": ["step", "ledge"],
}


_RUN = re.compile(r"\b(run|go|go ahead|do it|execute|apply|start|process it|kick it off|defeature|defeaturing)\b"
                  r"|\b(simplify|clean)\b.*\b(now|it|part|model|this)\b|\bsimplification\b")
_FALSE_RUN = re.compile(r"\b(thermal|cfd|modal|static|stress|flow|analysis|study|simulation) run\b|\b(we're|we are|i'm|am|it's|is) (running|going)\b|\brunning\b")
_EXPORT = re.compile(r"\b(python|\.py|c\+\+|cpp|step|report|export|download|save)\b")


def wants_run(text: str) -> bool:
    t = text.lower()
    return bool(_RUN.search(t)) and not (_FALSE_RUN.search(t) and not re.search(r"\b(then (run|go|start)|and run|run it)\b", t))


def sanitize(calls, text):
    """Keep the LLM to what was actually said: numbers must appear in the request, a run or an export
    must have been asked for, and a clear request for a run is never dropped."""
    nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}
    t = text.lower()
    out, seen = [], set()
    for c in calls:
        name, a = c["name"], c.get("arguments") or {}
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        a = dict(a)
        if name == "set_features":
            a["feature_ids"] = [i for i in a.get("feature_ids", []) if float(i) in nums]
            if not a["feature_ids"]:
                continue
        elif name == "set_class_rule":
            if a.get("max_size_mm") is not None and float(a["max_size_mm"]) not in nums:
                a.pop("max_size_mm")
        elif name == "set_analysis":
            if a.get("mesh_size_mm") is not None and float(a["mesh_size_mm"]) not in nums:
                a.pop("mesh_size_mm")
            if a.get("preset") and a.get("mesh_size_mm") and not stated_preset(t):
                a.pop("preset")
            if not a.get("preset") and not a.get("mesh_size_mm"):
                continue
        elif name == "export":
            if not _EXPORT.search(t):
                continue
        elif name == "run_defeaturing":
            if not wants_run(t):
                continue
        key = json.dumps([name, a], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append({"name": name, "arguments": a})
    if wants_run(t) and not any(c["name"] == "run_defeaturing" for c in out) and not any(c["name"] == "create_part" for c in out):
        out.append({"name": "run_defeaturing", "arguments": {}})
    return out


def plan_calls(messages, text, tools=None, timeout=150):
    """The LLM proposes, the checks in sanitize() dispose. Returns (calls, reply, via)."""
    try:
        m = llm_chat(messages, tools=tools or TOOLS, timeout=timeout)
    except Exception:
        m = {}
    calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", {})} for tc in (m.get("tool_calls") or [])]
    calls = sanitize(calls, text)
    if calls:
        return calls, (m.get("content") or "").strip(), CHAT_MODEL
    # the model was busy, timed out or called nothing: the keyword parser still understands most requests
    return parse_offline(text), (m.get("content") or "").strip(), "keyword parser"


_KEEP = r"\b(keep|preserve|don'?t (remove|touch)|do not (remove|touch)|leave|stays?|must stay|remain|protect|hang on|hold on)\b"
_REMOVE = (r"\b(remove|delete|drop|strip|get rid|clean|suppress|simplify|lose|wipe|erase|fill|plug|close up|flatten"
           r"|take (out|off|away)|can go|out|no|without)\b")
_DOMAIN = r"\b(fluid|flow) (domain|volume|region)\b|\benclosure\b"


def stated_preset(t: str):
    """The analysis the request names, if any."""
    for preset in PRESETS:
        if preset in t or (preset == "static" and "stress" in t) or (preset == "modal" and "vibration" in t) \
                or (preset == "cfd" and ("flow" in t or "cfd" in t)):
            return preset
    return None


def parse_offline(text: str):
    t = text.lower()
    calls = []
    domain = bool(re.search(_DOMAIN, t))
    preset = stated_preset(t)
    if preset and not domain:     # a fluid domain is geometry: asking for one doesn't change the analysis
        calls.append({"name": "set_analysis", "arguments": {"preset": preset}})
    m = (re.search(r"(?:mesh|elements?)\s*(?:size)?\s*(?:is|of|=|to|at)?\s*(\d+(?:\.\d+)?)\s*mm", t)
         or re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:elements?|mesh)", t)
         or (preset and re.search(r"\b(?:at|with)\s*(\d+(?:\.\d+)?)\s*mm", t)))
    if m:
        calls.append({"name": "set_analysis", "arguments": {"mesh_size_mm": float(m.group(1))}})
    # split into clauses so "remove X but keep Y" works; a clause with no verb carries on the last one,
    # and "keep them" points back at what the clause before it named
    act, last_classes = None, []
    ids_all = {"keep": [], "remove": []}
    for clause in re.split(r"\bbut\b|,|;|\band\b(?=\s+(?:keep|remove|delete|drop|strip|preserve))|\.", t):
        if re.search(_KEEP, clause):
            act = "keep"
        elif re.search(_REMOVE, clause):
            act = "remove"
        found = [cls for cls, words in _CLASS_WORDS.items() if any(w in clause for w in words)]
        if not act:
            last_classes = found or last_classes
            continue
        if not found and re.search(r"\b(them|those|these|it)\b", clause):
            found = last_classes
        size = re.search(r"(?:under|below|smaller than|less than|<)\s*(\d+(?:\.\d+)?)\s*mm", clause)
        for cls in found:
            args = {"feature_class": cls, "action": act}
            if size and act == "remove":
                args["max_size_mm"] = float(size.group(1))
            calls.append({"name": "set_class_rule", "arguments": args})
        last_classes = found or last_classes
        ids = re.findall(r"#(\d+)", clause) or (re.findall(r"\d+", clause.split("feature", 1)[1])
                                                 if re.search(r"\bfeatures?\s+(?:number\s+|no\.?\s+)?#?\d", clause) else [])
        ids_all[act] += [int(i) for i in ids]
    for act2, ids in ids_all.items():
        if ids:
            calls.append({"name": "set_features", "arguments": {"feature_ids": ids, "action": act2}})
    if wants_run(t):
        calls.append({"name": "run_defeaturing", "arguments": {}})
    for fmt, words in {"python": ["python", ".py"], "cpp": ["c++", "cpp"], "step": ["step file", "download step"]}.items():
        if any(w in t for w in words) or (fmt == "step" and re.search(r"\b(save|export|download)\b.*\bstep\b", t)):
            calls.append({"name": "export", "arguments": {"format": fmt}})
    if re.search(r"\b(make|create|design|model|build)\b.*\b(part|bracket|plate|flange|block|box|housing|mount)\b", t) \
            and "defeature" not in t and "domain" not in t and "around" not in t:
        calls = [{"name": "create_part", "arguments": {"description": text}}]
    if domain or re.search(r"\b(extract|make|build|create)\b.*\b(flow|fluid)\b", t):
        calls = [c for c in calls if c["name"] != "set_analysis" or "mesh_size_mm" in c["arguments"]]
        calls.append({"name": "make_fluid_domain", "arguments": {}})
    return calls


# ---------------------------------------------------------------- text-to-CAD
CAD_SYSTEM = ("You are a CAD design assistant. Given a description of a mechanical part, respond with a complete "
              "CadQuery (Python) script that builds it. Use millimeters. The script must import cadquery as cq and "
              "assign the final single-solid model to a variable named `result`. Respond with only the code.")

CHEATSHEET = """You write CadQuery 2.4 Python. Rules:
- import cadquery as cq ; final solid in a variable named result ; millimeters ; Z is up.
- Valid Workplane methods: box(l,w,h), cylinder(h,r), circle(r), rect(w,h), polygon(n,d), polyline(pts), close(),
  extrude(d), revolve(deg), hole(d, depth=None), cboreHole(d, cbD, cbDepth), cskHole(d, cskD, cskAngle),
  slot2D(length, d), pushPoints([(x,y),...]), rarray(xs, ys, nx, ny), polarArray(r, startAng, ang, n),
  faces('>Z'), faces('<Z'), edges('|Z'), workplane(), center(x,y), translate((x,y,z)), union(other), cut(other),
  fillet(r), chamfer(d), shell(t), text(txt, size, height).
- NO method named cutHole, cutCylinder, makeHole. To drill several holes:
  result = result.faces('>Z').workplane().pushPoints([(x1,y1),(x2,y2)]).hole(d)
- Bolt circle: .faces('>Z').workplane().polarArray(R, 0, 360, n).hole(d)   (polarArray ALWAYS takes 4 args)
- Rectangular cut-outs / vents / slots through a plate:
  result = result.faces('>Z').workplane().pushPoints([(0,20),(0,-20)]).rect(30, 8).cutThruAll()
- Rounded slot: .faces('>Z').workplane().slot2D(30, 8).cutThruAll()
- Fillet vertical edges BEFORE drilling holes: .edges('|Z').fillet(r)
- Keep it simple and robust. Respond with ONLY the code in one ```python block.
Example:
```python
import cadquery as cq
result = (cq.Workplane('XY').box(100, 60, 5)
          .edges('|Z').fillet(4)
          .faces('>Z').workplane().pushPoints([(-42,-22),(42,-22),(-42,22),(42,22)]).hole(6)
          .faces('>Z').workplane().hole(20))
```"""

RUNNER = r'''
import sys, cadquery as cq
ns = {"cq": cq, "__name__": "__cad__"}
exec(open(sys.argv[1], encoding="utf-8").read(), ns)
r = ns.get("result")
if r is None:
    raise SystemExit("script did not define `result`")
shape = r.val() if hasattr(r, "val") else r
if hasattr(shape, "wrapped") and not shape.isValid():
    raise SystemExit("result is not a valid solid")
cq.exporters.export(cq.Workplane().add(shape), sys.argv[2])
print("ok")
'''

_BANNED = re.compile(r"\b(import\s+(os|sys|subprocess|socket|shutil|requests|httpx|urllib)|__import__|open\(|eval\(|exec\()")


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_cadquery(code: str, out_step: str, timeout=60):
    if _BANNED.search(code):
        return False, "generated code tried to use a blocked module or call"
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, code.encode())
    os.close(fd)
    runner = path + ".runner.py"
    open(runner, "w", encoding="utf-8").write(RUNNER)
    try:
        r = subprocess.run([sys.executable, "-I", runner, path, out_step], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        if r.returncode == 0 and os.path.exists(out_step):
            return True, ""
        err = (r.stderr or r.stdout).strip().splitlines()
        return False, "\n".join(err[-6:])
    except subprocess.TimeoutExpired:
        return False, f"script timed out after {timeout}s"
    finally:
        for p in (path, runner):
            try:
                os.unlink(p)
            except OSError:
                pass


# Tested patterns (tests/test_pipeline.py runs every one): the shapes a bare prompt most often gets wrong.
EXAMPLES = [
    ("sphere ball", "A sphere 50 mm in diameter", "result = cq.Workplane('XY').sphere(25)"),
    ("tube pipe ring washer spacer sleeve outer inner", "A tube, 30 mm outer diameter, 24 mm inner diameter, 120 mm long",
     "result = cq.Workplane('XY').circle(15).circle(12).extrude(120)"),
    ("cone tapered truncated taper frustum", "A cone 40 mm at the base, 20 mm at the top, 30 mm tall",
     "result = cq.Workplane('XZ').polyline([(0, 0), (20, 0), (10, 30), (0, 30)]).close().revolve(360, (0, 0, 0), (0, 1, 0))"),
    ("shaft stepped sections diameter for", "A stepped shaft: 20 mm diameter for 40 mm, then 12 mm diameter for 30 mm",
     "result = cq.Workplane('XY').circle(10).extrude(40).faces('>Z').workplane().circle(6).extrude(30)"),
    ("hex hexagon hexagonal prism flats nut pentagon octagon polygon across corners",
     "A hex prism 24 mm across flats, 15 mm tall. polygon(n, d) takes the across-CORNERS diameter: for a hexagon across flats f, d = f / cos(30 deg)",
     "import math\nresult = cq.Workplane('XY').polygon(6, 24 / math.cos(math.radians(30))).extrude(15)"),
    ("triangle triangular prism equilateral", "A triangular prism with a 40 mm equilateral base, 60 mm long",
     "h = 40 * 3 ** 0.5 / 2\nresult = cq.Workplane('XY').polyline([(0, 0), (40, 0), (20, h)]).close().extrude(60)"),
    ("box tray open top hollow walls hollowed shell enclosure", "An open-top box 120 x 80 x 40 mm with 3 mm walls",
     "result = cq.Workplane('XY').box(120, 80, 40).faces('>Z').shell(-3)"),
    ("closed hollow sealed box", "A closed hollow box 100 x 60 x 40 mm with 4 mm walls",
     "result = cq.Workplane('XY').box(100, 60, 40).cut(cq.Workplane('XY').box(92, 52, 32))"),
    ("l-bracket l-angle angle legs upright wall bracket", "An L-angle 100 mm long with 30 mm legs, 3 mm thick",
     "result = cq.Workplane('XY').polyline([(0, 0), (30, 0), (30, 3), (3, 3), (3, 30), (0, 30)]).close().extrude(100)"),
    ("base plate wall upright rib along centerline edge", "A 120 x 60 x 8 mm base plate with a 40 mm tall, 8 mm thick rib along its centerline. Height above the base adds to its thickness.",
     "base = cq.Workplane('XY').box(120, 60, 8, centered=(True, True, False))\nrib = cq.Workplane('XY').box(120, 8, 40, centered=(True, True, False)).translate((0, 0, 8))\nresult = base.union(rib)"),
    ("t-section t-beam tee flange web i-beam", "A T-section 100 mm long, flange 40 x 5 mm, web 30 x 5 mm",
     "result = cq.Workplane('XY').polyline([(-20, 30), (20, 30), (20, 35), (-20, 35)]).close().extrude(100).union(cq.Workplane('XY').polyline([(-2.5, 0), (2.5, 0), (2.5, 30), (-2.5, 30)]).close().extrude(100))"),
    ("u-channel channel c-section z-profile profile", "A U-channel 200 mm long, 50 mm wide, 25 mm tall, with 3 mm walls",
     "result = cq.Workplane('XY').polyline([(-25, 0), (25, 0), (25, 25), (22, 25), (22, 3), (-22, 3), (-22, 25), (-25, 25)]).close().extrude(200)"),
    ("spool bobbin end discs core flanges joined", "A spool: 40 mm diameter end discs 5 mm thick, joined by a 20 mm diameter core 30 mm long",
     "result = cq.Workplane('XY').circle(20).extrude(5).faces('>Z').workplane().circle(10).extrude(30).faces('>Z').workplane().circle(20).extrude(5)"),
    ("flange bolt circle bore holes on a circle disc", "A flange 60 mm in diameter, 10 mm thick, 20 mm bore, six 6 mm holes on a 44 mm circle",
     "result = cq.Workplane('XY').circle(30).extrude(10).faces('>Z').workplane().hole(20).faces('>Z').workplane().polarArray(22, 0, 360, 6).hole(6)"),
    ("grid pattern holes corners equally spaced along centerline", "A 100 x 100 x 8 mm plate with four 10 mm holes on a 70 x 70 mm pattern",
     "result = cq.Workplane('XY').box(100, 100, 8).faces('>Z').workplane().rarray(70, 70, 2, 2).hole(10)"),
    ("pocket blind recess top face", "A 100 x 50 x 20 mm block with a 60 x 30 x 10 mm pocket in the top face",
     "result = cq.Workplane('XY').box(100, 50, 20).faces('>Z').workplane().rect(60, 30).cutBlind(-10)"),
    ("square through hole slot slots parallel window", "A 100 x 100 x 5 mm plate with three parallel 60 x 5 mm slots",
     "result = cq.Workplane('XY').box(100, 100, 5).faces('>Z').workplane().pushPoints([(0, -25), (0, 0), (0, 25)]).rect(60, 5).cutThruAll()"),
    ("step cut along edge notch rabbet", "A 60 x 40 x 30 mm block with a 20 x 10 mm step cut along one long top edge",
     "result = cq.Workplane('XY').box(60, 40, 30).cut(cq.Workplane('XY').box(60, 20, 10).translate((0, 10, 10)))"),
    ("chamfer chamfers both ends fillet fillets rounded ends cylinder", "A cylinder 50 mm in diameter and 20 mm long with 2 mm chamfers on both ends",
     "result = cq.Workplane('XY').circle(25).extrude(20).faces('>Z or <Z').edges().chamfer(2)"),
    ("blind hole deep cylinder", "A cylinder 40 mm in diameter and 60 mm long with a 25 mm blind hole 40 mm deep",
     "result = cq.Workplane('XY').circle(20).extrude(60).faces('>Z').workplane().hole(25, 40)"),
    ("boss hub disc on top standing", "A hub: 60 mm diameter disc 10 mm thick with a 30 mm diameter boss 20 mm tall on top. The overall height adds up to 30 mm.",
     "result = cq.Workplane('XY').circle(30).extrude(10).faces('>Z').workplane().circle(15).extrude(20)"),
    ("cross plus shaped bars right angles", "A cross of two 100 x 30 mm bars at right angles, 8 mm thick",
     "result = cq.Workplane('XY').box(100, 30, 8).union(cq.Workplane('XY').box(30, 100, 8))"),
]


def _examples_for(description, k=3):
    """The few tested patterns closest to this request, by shared words."""
    words = set(re.findall(r"[a-z0-9-]+", description.lower()))
    ranked = sorted(EXAMPLES, key=lambda e: -len(words & set(e[0].split())))
    out = []
    for keys, desc, code in ranked[:k]:
        if words & set(keys.split()):
            out.append(f"# {desc}\nimport cadquery as cq\n{code}")
    return "\n\n".join(out)


SPEC_SYSTEM = """Read the description of a mechanical part and answer with JSON only:
{"overall_mm": [a, b, c], "holes": n}
overall_mm is the part's whole bounding box in mm as three numbers, adding up heights of stacked sections and walls.
For a round part use its diameter twice. "holes" counts separate round holes asked for, 0 if none or if not stated.
Examples:
"A 120 x 60 x 8 mm base plate with a 40 mm tall rib" -> {"overall_mm": [120, 60, 48], "holes": 0}
"A 60 mm flange 10 mm thick with a 20 mm bore and six 6 mm holes" -> {"overall_mm": [60, 60, 10], "holes": 7}
"A stepped shaft: 20 mm for 40 mm, then 12 mm for 30 mm" -> {"overall_mm": [20, 20, 70], "holes": 0}"""


def _spec(description, model):
    """What the part should measure, read from the request."""
    try:
        reply = llm_chat([{"role": "system", "content": SPEC_SYSTEM}, {"role": "user", "content": description}], model=model,
                         timeout=180, options={"temperature": 0.0, "num_predict": 80})
        m = re.search(r"\{.*\}", reply.get("content", ""), re.S)
        d = json.loads(m.group(0))
        dims = sorted(float(x) for x in d.get("overall_mm", []))
        return {"dims": dims if len(dims) == 3 and min(dims) > 0 else None, "holes": int(d.get("holes") or 0)}
    except Exception:
        return {"dims": None, "holes": 0}


def _measure(step_path):
    """The kernel's own numbers for what was built, including DeFeat's hole count."""
    from . import occ
    from .graph import build_graph
    from .recognize import Recognizer
    shape = occ.read_cad(step_path)
    x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
    g = build_graph(shape, curvature=False)
    holes = sum(1 for f in Recognizer(g).run() if f.kind == "hole")
    return {"dims": sorted([x1 - x0, y1 - y0, z1 - z0]), "holes": holes, "solids": len(occ.solids_of(shape)), "volume": abs(occ.volume(shape))}


def _problems(spec, m):
    out = []
    if m["solids"] != 1:
        out.append(f"the result is {m['solids']} separate solids, it must be one connected solid")
    if spec.get("dims"):
        for got, want in zip(m["dims"], spec["dims"]):
            if abs(got - want) > 0.02 * want + 0.2:
                out.append(f"it should measure {spec['dims']} mm overall (sorted), but it measures {[round(v, 2) for v in m['dims']]} mm")
                break
    if spec.get("holes") and m["holes"] < spec["holes"]:
        out.append(f"it should have {spec['holes']} round holes, but only {m['holes']} were found")
    return out


def text_to_cad(description: str, out_step: str, model=None, retries=3, repairs=3, log=None):
    """The LLM writes CadQuery from tested examples. We run it in a sandbox and fix errors (inner loop),
    then measure the solid with the CAD kernel, compare it with what was asked and send any mismatch back
    (outer loop). The verify-and-repair idea follows CADSmith (2026) and CADCodeVerify (2025)."""
    model = model or CHAT_MODEL
    system = CAD_SYSTEM if model == CAD_MODEL else CHEATSHEET
    near = _examples_for(description)
    if near and model != CAD_MODEL:
        system += "\n\nClosest tested patterns:\n```python\n" + near + "\n```"
    spec = _spec(description, model)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": description}]
    attempts, best = [], None
    code, fixes, repair = "", 0, 0
    tmp = out_step + ".try.step"
    while fixes <= retries + repairs:
        t = time.time()
        reply = llm_chat(msgs, model=model, timeout=600, options={"temperature": 0.2, "num_predict": 1200})
        code = _extract_code(reply.get("content", ""))
        ok, err = run_cadquery(code, tmp)
        fixes += 1
        issues = None
        if ok:
            try:
                issues = _problems(spec, _measure(tmp))
            except Exception as e:  # noqa
                ok, err = False, f"the result could not be measured: {e}"
        attempts.append({"model": model, "ok": ok, "error": err, "issues": issues, "seconds": round(time.time() - t, 1)})
        if ok and (best is None or len(issues) < len(best["issues"])):
            os.replace(tmp, out_step)
            best = {"code": code, "issues": issues}
        if ok and not issues:
            if log:
                log(f"Attempt {fixes}: valid solid, dimensions match")
            break
        if ok:
            repair += 1
            if log:
                log(f"Attempt {fixes}: measured and corrected: {issues[0]}")
            if repair > repairs:
                break
            msgs += [{"role": "assistant", "content": code},
                     {"role": "user", "content": "I built and measured that with the CAD kernel: " + "; ".join(issues) + f". The request was: {description} Fix the script. Respond with only the corrected code."}]
        else:
            if log:
                log(f"Attempt {fixes}: error, retrying")
            msgs += [{"role": "assistant", "content": code},
                     {"role": "user", "content": f"That script failed:\n{err}\nFix it. Respond with only the corrected code."}]
    try:
        os.unlink(tmp)
    except OSError:
        pass
    if best is None:
        return {"ok": False, "code": code, "attempts": attempts, "model": model, "spec": spec}
    return {"ok": True, "code": best["code"], "attempts": attempts, "model": model, "spec": spec, "unresolved": best["issues"]}
