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
    "boss": ["boss", "standoff"],
    "rib": ["rib", "gusset", "web"],
    "slot": ["slot"],
    "pocket": ["pocket", "cut-out", "cutout", "vent", "window"],
}


def parse_offline(text: str):
    t = text.lower()
    calls = []
    for preset in PRESETS:
        if preset in t or (preset == "static" and "stress" in t) or (preset == "modal" and "vibration" in t) \
                or (preset == "cfd" and ("flow" in t or "cfd" in t)):
            calls.append({"name": "set_analysis", "arguments": {"preset": preset}})
            break
    m = (re.search(r"(?:mesh|element)\s*(?:size)?\s*(?:of|=|to)?\s*(\d+(?:\.\d+)?)\s*mm", t)
         or re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:elements?|mesh)", t))
    if m:
        calls.append({"name": "set_analysis", "arguments": {"mesh_size_mm": float(m.group(1))}})
    # split into clauses so "remove X but keep Y" works
    for clause in re.split(r"\bbut\b|,|;|\band\b(?=\s+(?:keep|remove|delete|drop|strip|preserve))|\.", t):
        act = None
        if re.search(r"\b(keep|preserve|don't remove|do not remove|leave)\b", clause):
            act = "keep"
        elif re.search(r"\b(remove|delete|drop|strip|get rid|clean|suppress|simplify)\b", clause):
            act = "remove"
        if not act:
            continue
        size = re.search(r"(?:under|below|smaller than|less than|<)\s*(\d+(?:\.\d+)?)\s*mm", clause)
        for cls, words in _CLASS_WORDS.items():
            if any(w in clause for w in words):
                args = {"feature_class": cls, "action": act}
                if size and act == "remove":
                    args["max_size_mm"] = float(size.group(1))
                calls.append({"name": "set_class_rule", "arguments": args})
        ids = re.findall(r"#(\d+)", clause) or (re.findall(r"\d+", clause.split("feature", 1)[1])
                                                 if re.search(r"\bfeatures?\s+#?\d", clause) else [])
        if ids:
            calls.append({"name": "set_features", "arguments": {"feature_ids": [int(i) for i in ids], "action": act}})
    if re.search(r"\b(run|go|do it|execute|apply|start)\b", t) or re.search(r"\bsimplify\b.*\b(now|it|part|model)\b", t):
        calls.append({"name": "run_defeaturing", "arguments": {}})
    for fmt, words in {"python": ["python", ".py"], "cpp": ["c++", "cpp"], "step": ["step file", "download step"]}.items():
        if any(w in t for w in words):
            calls.append({"name": "export", "arguments": {"format": fmt}})
    if re.search(r"\b(make|create|design|model|build)\b.*\b(part|bracket|plate|flange|block|box|housing|mount)\b", t) \
            and "defeature" not in t and "domain" not in t and "around" not in t:
        calls = [{"name": "create_part", "arguments": {"description": text}}]
    if "fluid domain" in t or "flow domain" in t or "enclosure" in t:
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


def text_to_cad(description: str, out_step: str, model=None, retries=3, log=None):
    """LLM writes CadQuery; we execute it sandboxed, validate, and feed errors back."""
    model = model or CHAT_MODEL
    system = CAD_SYSTEM if model == CAD_MODEL else CHEATSHEET
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": description}]
    attempts = []
    for k in range(retries + 1):
        t = time.time()
        reply = llm_chat(msgs, model=model, timeout=600, options={"temperature": 0.2, "num_predict": 1200})
        code = _extract_code(reply.get("content", ""))
        ok, err = run_cadquery(code, out_step)
        attempts.append({"model": model, "ok": ok, "error": err, "seconds": round(time.time() - t, 1)})
        if log:
            log(f"Attempt {k + 1}: {'valid solid' if ok else 'error, retrying'}")
        if ok:
            return {"ok": True, "code": code, "attempts": attempts, "model": model}
        msgs += [{"role": "assistant", "content": code},
                 {"role": "user", "content": f"That script failed:\n{err}\nFix it. Respond with only the corrected code."}]
    return {"ok": False, "code": code, "attempts": attempts, "model": model}
