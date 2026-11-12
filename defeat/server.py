"""FastAPI backend for the DeFeat web app."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, agent, occ
from .decide import PRESETS
from .session import ROOT, SESSIONS, mesh_payload, new_session

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(HERE, "samples")
WEB = os.path.join(HERE, "web", "dist")

app = FastAPI(title="DeFeat", version=__version__)
# The app is served from the same origin it talks to. Set DEFEAT_CORS to a comma-separated list of origins
# only when another site must call this API.
_CORS = [o.strip() for o in os.environ.get("DEFEAT_CORS", "").split(",") if o.strip()]
if _CORS:
    app.add_middleware(CORSMiddleware, allow_origins=_CORS, allow_methods=["*"], allow_headers=["*"])
MAX_UPLOAD_MB = float(os.environ.get("DEFEAT_MAX_UPLOAD_MB", "200"))


def get(sid):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s


# ---------------------------------------------------------------- parts
@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "llm": agent.ollama_available(), "models": agent.ollama_models(),
            "chat_model": agent.CHAT_MODEL, "cad_model": agent.CAD_MODEL}


@app.get("/api/samples")
def samples():
    out = []
    for f in sorted(os.listdir(SAMPLES)):
        if f.lower().endswith((".step", ".stp")):
            out.append({"name": f, "title": f.rsplit(".", 1)[0].replace("_", " ").title(),
                        "kb": round(os.path.getsize(os.path.join(SAMPLES, f)) / 1024)})
    return out


_PREVIEWS: dict[str, dict] = {}


@app.get("/api/preview/{name}")
def preview(name: str):
    """Lightweight mesh + which faces would be removed, for the welcome-page dot renders (cached)."""
    name = os.path.basename(name)
    if name in _PREVIEWS:
        return _PREVIEWS[name]
    p = os.path.join(SAMPLES, name)
    if not os.path.exists(p):
        raise HTTPException(404)
    from .decide import Policy, decide, default_h
    from .graph import build_graph
    from .recognize import recognize
    shape = occ.heal(occ.read_cad(p))
    g = build_graph(shape)
    _, feats = recognize(shape, g)   # the same rules + network the session uses, so the card matches the run
    d = decide(feats, Policy("static", default_h(g.char_len)), g, occ.volume(shape))
    removable = sorted({fid for f in feats if d[f.id].action == "remove" for fid in f.faces})
    out = {"mesh": mesh_payload(shape), "removable": removable, "features": len(feats),
           "faces": len(g.nodes), "remove": sum(1 for f in feats if d[f.id].action == "remove")}
    _PREVIEWS[name] = out
    return out


@app.post("/api/sample/{name}")
def open_sample(name: str):
    p = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return new_session(p).summary()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "part.step")
    if not name.lower().endswith((".step", ".stp", ".iges", ".igs", ".brep")):
        raise HTTPException(400, "Please upload a STEP, IGES or BREP file")
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(p, "wb") as fh:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"The file is larger than {MAX_UPLOAD_MB:g} MB")
            fh.write(chunk)
    try:
        s = new_session(p, name)
    except Exception as e:  # noqa
        raise HTTPException(400, f"Could not read this file: {e}")
    return s.summary()


@app.get("/api/session/{sid}")
def session_info(sid: str):
    return get(sid).summary()


@app.get("/api/session/{sid}/mesh/{which}")
def session_mesh(sid: str, which: str):
    s = get(sid)
    if which == "original":
        if not hasattr(s, "_mesh_orig"):
            s._mesh_orig = mesh_payload(s.shape)
        return s._mesh_orig
    if which == "result":
        if not s.result:
            raise HTTPException(404, "no result yet")
        return s.result_payload()
    raise HTTPException(400)


class PolicyIn(BaseModel):
    preset: str | None = None
    h: float | None = None
    overrides: dict | None = None
    feature_overrides: dict | None = None
    protected_faces: list | None = None
    boundary: dict | None = None
    merge: bool = True


@app.post("/api/session/{sid}/policy")
def session_policy(sid: str, body: PolicyIn):
    s = get(sid)
    if body.preset and body.preset not in PRESETS:
        raise HTTPException(400, "unknown preset")
    return s.set_policy(body.preset, body.h, body.overrides, body.feature_overrides, body.protected_faces, body.merge, boundary=body.boundary)


@app.post("/api/session/{sid}/run")
def session_run(sid: str, mesh: bool = True):
    s = get(sid)
    if s.status.get("busy"):
        raise HTTPException(409, "already running")
    s.run_async(with_mesh=mesh)
    return {"ok": True}


@app.get("/api/session/{sid}/status")
def session_status(sid: str):
    s = get(sid)
    return {"status": s.status, "has_result": s.result is not None}


@app.get("/api/session/{sid}/result")
def session_result(sid: str):
    s = get(sid)
    if not s.result:
        raise HTTPException(404, "no result yet")
    return {"metrics": s.metrics, "summary": s.summary(),
            "code": {"python": open(s.exports["py"], encoding="utf-8").read(), "cpp": open(s.exports["cpp"], encoding="utf-8").read()}}


@app.get("/api/session/{sid}/download/{kind}")
def session_download(sid: str, kind: str):
    s = get(sid)
    if kind == "original":
        return FileResponse(s.src, filename=s.name)
    if kind == "report":
        from .report import build_report
        path = build_report(s)
        return FileResponse(path, filename=os.path.basename(path))
    if kind not in s.exports:
        raise HTTPException(404, "run defeaturing first")
    return FileResponse(s.exports[kind], filename=os.path.basename(s.exports[kind]))


@app.post("/api/session/{sid}/verify")
def session_verify(sid: str):
    s = get(sid)
    if not s.result:
        raise HTTPException(404, "run defeaturing first")
    return s.verify_python_export()


# ---------------------------------------------------------------- chat
class ChatIn(BaseModel):
    message: str
    session: str | None = None


JOBS: dict[str, dict] = {}


def _fluid_domain(s, pad_pct=50.0):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    shape = s.result.shape if s.result else s.shape
    x0, y0, z0, x1, y1, z1 = occ.bbox(shape)
    p = pad_pct / 100.0
    dx, dy, dz = (x1 - x0) * p, (y1 - y0) * p, (z1 - z0) * p
    box = BRepPrimAPI_MakeBox(gp_Pnt(x0 - dx, y0 - dy, z0 - dz), gp_Pnt(x1 + dx * 3, y1 + dy, z1 + dz)).Shape()
    dom = BRepAlgoAPI_Cut(box, shape).Shape()
    stem = s.name.rsplit(".", 1)[0]
    path = os.path.join(s.dir, f"{stem}_fluid_domain.step")
    occ.write_step(dom, path)
    return path


def _create_part_job(job_id, description):
    job = JOBS[job_id]
    try:
        out = os.path.join(ROOT, f"gen_{job_id}.step")
        os.makedirs(ROOT, exist_ok=True)
        job["steps"].append("Writing CadQuery code")
        r = agent.text_to_cad(description, out, log=lambda m: job["steps"].append(m))
        job["code"] = r.get("code")
        job["model"] = r.get("model")
        if r["ok"]:
            job["steps"].append("Recognizing features on the new part")
            ns = new_session(out, "generated_part.step")
            job["session"] = ns.id
            job["done"] = True
        else:
            job["error"] = (r["attempts"][-1].get("error") or "could not build a valid solid")[-400:]
            job["done"] = True
    except Exception as e:  # noqa
        job["error"] = str(e)
        job["done"] = True


NOUN = {"hole": "holes", "slot": "slots", "pocket": "pockets", "step": "steps", "boss": "bosses", "rib": "ribs",
        "emboss": "text / logo embosses", "fillet": "fillets", "chamfer": "chamfers", "protrusion": "protrusions"}


def execute_calls(s, calls):
    """Apply tool calls to the session. Returns (reply lines, UI actions)."""
    lines, actions = [], []
    for c in calls:
        name, a = c["name"], c.get("arguments") or {}
        if isinstance(a, str):
            try:
                import json
                a = json.loads(a)
            except Exception:
                a = {}
        try:
            if name == "set_class_rule" and s:
                cls, act = a.get("feature_class"), a.get("action")
                if act == "auto":
                    ov = dict(s.policy.overrides)
                    ov.pop(cls, None)
                    s.set_policy(overrides=ov, merge=False, source="chat")
                    lines.append(f"{NOUN[cls].capitalize()} back on the preset rule.")
                elif act == "keep" and a.get("max_size_mm"):
                    lim = float(a["max_size_mm"]) * 0.999
                    s.set_policy(overrides={cls: {"remove_below": 0, "max_size": lim}}, source="chat")
                    lines.append(f"Keeping {NOUN[cls]} of {a['max_size_mm']:g} mm and larger; smaller ones still go.")
                elif act == "keep":
                    s.set_policy(overrides={cls: {"remove": "never"}}, source="chat")
                    lines.append(f"Keeping all {NOUN[cls]}.")
                elif a.get("max_size_mm"):
                    s.set_policy(overrides={cls: {"remove_below": 0, "max_size": float(a["max_size_mm"])}}, source="chat")
                    lines.append(f"Removing {NOUN[cls]} smaller than {a['max_size_mm']:g} mm.")
                else:
                    s.set_policy(overrides={cls: {"remove": "always"}}, source="chat")
                    lines.append(f"Removing all {NOUN[cls]}.")
                actions.append({"type": "policy"})
            elif name == "set_features" and s:
                ids, act = a.get("feature_ids", []), a.get("action")
                fo = {i: (act if act != "auto" else None) for i in ids}
                s.set_policy(feature_overrides=fo, source="chat")
                lines.append(f"{'Keeping' if act == 'keep' else 'Removing' if act == 'remove' else 'Resetting'} "
                             + ", ".join(f"#{i}" for i in ids) + ".")
                actions.append({"type": "policy"})
                actions.append({"type": "focus", "features": ids})
            elif name == "set_analysis" and s:
                s.set_policy(preset=a.get("preset"), h=a.get("mesh_size_mm"), source="chat")
                if a.get("preset"):
                    lines.append(f"Preset set to {PRESETS[a['preset']]['label']}.")
                if a.get("mesh_size_mm"):
                    lines.append(f"Target mesh size h = {a['mesh_size_mm']:g} mm.")
                actions.append({"type": "policy"})
            elif name == "run_defeaturing" and s:
                if not s.status.get("busy"):
                    s.run_async()
                lines.append("Running the defeaturing now — I'll mesh both versions so you can compare.")
                actions.append({"type": "run"})
            elif name == "export" and s:
                fmt = {"python": "py", "cpp": "cpp", "step": "step", "report": "report"}.get(a.get("format"), "step")
                actions.append({"type": "download", "format": fmt})
                lines.append({"py": "The Python script", "cpp": "The C++ program", "step": "The simplified STEP",
                              "report": "The report"}[fmt] + " is ready in Results (after a run).")
            elif name == "make_fluid_domain" and s:
                path = _fluid_domain(s, float(a.get("padding_pct") or 50))
                s.exports["fluid"] = path
                actions.append({"type": "download", "format": "fluid"})
                lines.append("Built a CFD flow domain around the part (part subtracted from an enclosure).")
            elif name == "create_part":
                jid = f"{int(time.time() * 1000):x}"
                JOBS[jid] = {"steps": [], "done": False, "description": a.get("description", "")}
                threading.Thread(target=_create_part_job, args=(jid, a.get("description", "")), daemon=True).start()
                actions.append({"type": "create_job", "job": jid})
                lines.append("Designing that part — writing CadQuery code, then checking it builds a valid solid.")
        except Exception as e:  # noqa
            lines.append(f"Couldn't apply {name}: {e}")
    return lines, actions


@app.post("/api/chat")
def chat(body: ChatIn):
    s = SESSIONS.get(body.session) if body.session else None
    text = body.message.strip()
    via, calls, reply = "offline parser", [], ""
    t0 = time.time()
    if agent.ollama_available():
        try:
            briefing = s.briefing() if s else "No part loaded yet. The user can upload a STEP file or ask you to create a new part."
            msgs = [{"role": "system", "content": agent.SYSTEM.format(briefing=briefing)}]
            if s:
                msgs += s.chat[-6:]
            msgs.append({"role": "user", "content": text})
            tools = agent.TOOLS if s else [t for t in agent.TOOLS if t["function"]["name"] == "create_part"]
            calls, reply, via = agent.plan_calls(msgs, text, tools=tools)
        except Exception:
            calls = []
    if not calls and not reply:
        calls = agent.parse_offline(text)
        via = "offline parser"
    lines, actions = execute_calls(s, calls)
    if lines:
        reply = " ".join(lines) if not reply else reply + "\n\n" + " ".join(lines)
    if not reply:
        reply = "I can remove or keep feature types (\"drop the logo and fillets under 1 mm\"), pick an analysis type, run the defeaturing, export STEP / Python / C++, or design a new part from a description."
    if s:
        s.chat += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    return {"reply": reply, "calls": calls, "actions": actions, "via": via, "seconds": round(time.time() - t0, 1),
            "summary": s.summary() if s else None}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    return j


@app.get("/api/session/{sid}/download_fluid")
def dl_fluid(sid: str):
    s = get(sid)
    if "fluid" not in s.exports:
        raise HTTPException(404)
    return FileResponse(s.exports["fluid"], filename=os.path.basename(s.exports["fluid"]))


@app.on_event("startup")
def _warm():
    threading.Thread(target=agent.warmup, daemon=True).start()


# ---------------------------------------------------------------- web app
if os.path.isdir(WEB):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = os.path.realpath(os.path.join(WEB, path))
        if path and f.startswith(os.path.realpath(WEB) + os.sep) and os.path.isfile(f):
            return FileResponse(f)
        # the page names hashed assets, so it must be revalidated or an update keeps showing the old UI
        return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-cache"})
