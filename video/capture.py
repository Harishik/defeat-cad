"""Capture 1920x1080 app states (+ element boxes) and dot-cloud data for the videos.

    python video/capture.py            # app must be running on :8000
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "public", "shots")
URL = "http://127.0.0.1:8000"
W, H = 1920, 1080
boxes = {}


def box(page, name, selector):
    el = page.locator(selector).first
    b = el.bounding_box()
    if b:
        boxes[name] = {k: round(v, 1) for k, v in b.items()}


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, name + ".png"), timeout=240_000)
    print("shot", name)


def open_sample(page, title):
    page.goto(URL)
    page.get_by_text(title, exact=True).click(timeout=240_000)
    page.wait_for_selector(".part-name", timeout=240_000)
    # clicking a card below the fold scrolls the document; the workspace must start at the top
    page.evaluate("window.scrollTo(0, 0); document.scrollingElement.scrollTop = 0")
    page.wait_for_timeout(4000)


def points(name, n=9000, edge_share=0.3):
    """Area-weighted surface dots + evenly spaced sharp-edge dots, normalised, with removable flags."""
    d = json.load(urllib.request.urlopen(f"{URL}/api/preview/{name}"))
    m = d["mesh"]
    P = np.frombuffer(base64.b64decode(m["positions"]), np.float32).reshape(-1, 3)
    T = np.frombuffer(base64.b64decode(m["indices"]), np.uint32).reshape(-1, 3)
    F = np.frombuffer(base64.b64decode(m["faceIds"]), np.uint32)
    rem = np.isin(F, d["removable"])
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    w = area.copy()
    ra, ba = area[rem].sum(), area[~rem].sum()
    if ra > 0:
        w[rem] *= max(5, 0.22 * ba / (0.78 * ra))
    rng = np.random.default_rng(1)
    ns = int(n * (1 - edge_share))
    idx = rng.choice(len(T), ns, p=w / w.sum())
    u, v = rng.random(ns), rng.random(ns)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + (b[idx] - a[idx]) * u[:, None] + (c[idx] - a[idx]) * v[:, None]
    flags = rem[idx].astype(int)
    # sharp edges: triangle edges whose adjacent normals differ by > 28 deg
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    edges = {}
    for t, tri in enumerate(T):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((tri[i], tri[j])))
            edges.setdefault(key, []).append(t)
    segs = []
    for (i, j), ts in edges.items():
        if len(ts) == 1 or np.dot(nrm[ts[0]], nrm[ts[1]]) < np.cos(np.radians(28)):
            segs.append((P[i], P[j]))
    L = np.array([np.linalg.norm(q - p) for p, q in segs])
    ne = n - ns
    cum = np.cumsum(L)
    r = (np.arange(ne) + 0.5) / ne * cum[-1]
    k = np.searchsorted(cum, r)
    f = (r - np.concatenate([[0], cum[:-1]])[k]) / np.maximum(L[k], 1e-9)
    ep = np.array([segs[q][0] + f_ * (segs[q][1] - segs[q][0]) for q, f_ in zip(k, f)])
    allp = np.vstack([ep, pts])
    flags = np.concatenate([np.zeros(ne, int), flags])
    ctr = (allp.max(0) + allp.min(0)) / 2
    s = 2.0 / (allp.max(0) - allp.min(0)).max()
    allp = (allp - ctr) * s
    out = {"p": np.round(allp, 4).ravel().tolist(), "r": flags.tolist(), "faces": d["faces"],
           "features": d["features"], "remove": d["remove"]}
    path = os.path.join(HERE, "public", f"points_{name.split('.')[0]}.json")
    json.dump(out, open(path, "w"))
    print("points", path, len(flags))


def main():
    os.makedirs(OUT, exist_ok=True)
    points("l_bracket.step")
    points("printer_chassis.step", n=11000)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="dark")
        page = ctx.new_page()
        page.set_default_timeout(240_000)

        page.goto(URL)
        page.wait_for_timeout(2200)
        shot(page, "welcome")
        box(page, "welcome.command", ".command")
        box(page, "welcome.kbar", ".kbar")
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(2500)
        shot(page, "welcome_cards")

        open_sample(page, "L Bracket")
        shot(page, "features")
        for n, sel in (("run", "button.run"), ("side", ".side"), ("frow", ".frow"), ("chips", ".legend"),
                       ("presets", ".seg"), ("h", ".hctl"), ("kbtn", ".kbtn")):
            box(page, "features." + n, sel)
        page.locator("button.chip", has_text="Text & logos").click()
        page.wait_for_timeout(900)
        box(page, "isolate.chip", "button.chip.on")
        shot(page, "isolate")
        page.keyboard.press("Escape")
        page.locator("button.chip", has_text="Holes").click()
        page.wait_for_timeout(900)
        shot(page, "isolate_holes")
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
        page.keyboard.press("Meta+k")
        page.wait_for_timeout(500)
        page.keyboard.type("modal")
        page.wait_for_timeout(600)
        shot(page, "cmdk")
        box(page, "cmdk", ".cmdk")
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        page.locator("button.run").click()
        page.wait_for_selector(".loader", timeout=60_000)
        page.wait_for_timeout(1800)
        shot(page, "loader")
        page.wait_for_selector(".results", timeout=400_000)
        page.wait_for_timeout(1600)
        shot(page, "results")
        box(page, "results", ".results")
        box(page, "results.stats", ".results .stats")
        box(page, "results.fea", ".results .fea")
        box(page, "results.dl", ".results .downloads")
        box(page, "results.code", ".results .code-card")
        page.locator(".results .icon-btn").click()
        page.wait_for_timeout(1000)
        page.mouse.move(10, 600)
        page.get_by_role("button", name="Original").click()
        page.wait_for_timeout(1500)
        shot(page, "view_original")
        page.get_by_role("button", name="Simplified").click()
        page.wait_for_timeout(1500)
        shot(page, "view_simplified")
        page.get_by_role("button", name="Compare").click()
        page.wait_for_timeout(1200)
        shot(page, "compare")

        open_sample(page, "Printer Chassis")
        shot(page, "chassis")
        page.get_by_text("Assistant", exact=True).click()
        page.wait_for_timeout(700)
        shot(page, "chat_empty")
        page.locator(".chat-input input").fill("Prep this for a modal analysis, keep the screw bosses, drop the logo and run it")
        page.wait_for_timeout(300)
        shot(page, "chat_typed")
        box(page, "chat.input", ".chat-input")
        page.locator(".chat-input input").press("Enter")
        page.wait_for_selector(".msg.assistant .calls", timeout=400_000)
        page.wait_for_timeout(1500)
        shot(page, "chat_reply")
        box(page, "chat.side", ".side")
        try:
            page.wait_for_selector(".results", timeout=400_000)
            page.wait_for_timeout(1500)
            shot(page, "chat_results")
        except Exception:
            pass
        sid = page.evaluate("fetch('/api/samples').then(()=>1)")
        ctx.close()

        # report page (light, it's a printable document)
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, color_scheme="light")
        page = ctx.new_page()
        rep = os.path.join(os.path.dirname(HERE), "out", "l_bracket_report.html")
        if os.path.exists(rep):
            page.goto("file://" + rep)
            page.wait_for_timeout(800)
            shot(page, "report")
        b.close()
    json.dump(boxes, open(os.path.join(HERE, "public", "boxes.json"), "w"), indent=1)
    print("boxes", len(boxes))


if __name__ == "__main__":
    main()
